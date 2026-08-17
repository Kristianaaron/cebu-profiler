"""Worker-only llama.cpp RPC lifecycle for capture workloads.

The lifecycle owns only the remote transient worker.  All external commands
cross an injected argv runner, which keeps imports and unit tests side-effect
free.  A capture coordinator must remeasure the worker after every capture.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.canary_constants import WORKER_TRANSIENT_UNIT
from model_atlas.llamacpp_rpc_runtime import (
    DEFAULT_TOOLCHAIN_ROOT,
    EXPECTED_RPC_SERVER_SHA256,
    PINNED_COMMIT,
    LlamaCppRpcRuntimeConfig,
)

__all__ = [
    "WorkerRpcLaunchEvidence",
    "WorkerRpcLifecycleError",
    "WorkerRpcSystemdLifecycle",
]


class WorkerRpcLifecycleError(RuntimeError):
    """Sanitized failure at the remote process boundary."""


class ArgvRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str: ...


class WorkerRpcLaunchEvidence(BaseModel):
    """Immutable, independently measured identity of the worker process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    worker_pid: int = Field(gt=0)
    worker_start_ticks: int = Field(gt=0)
    worker_argv: tuple[str, ...]
    worker_exe_path: str = Field(pattern=r"^/")
    worker_exe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    worker_unit: str = Field(min_length=1)
    evidence_kind: Literal["measured"] = "measured"


class WorkerRpcSystemdLifecycle:
    """Start, attest, remeasure, and stop only the remote RPC worker."""

    def __init__(
        self,
        config: LlamaCppRpcRuntimeConfig,
        *,
        worker_ssh_target: str,
        runner: ArgvRunner,
        toolchain_root: Path = DEFAULT_TOOLCHAIN_ROOT,
        worker_unit: str = WORKER_TRANSIENT_UNIT,
        ssh_argv: Sequence[str] = ("ssh", "-o", "BatchMode=yes"),
    ) -> None:
        if not worker_ssh_target or not worker_unit:
            raise ValueError("worker target and unit are required")
        if not toolchain_root.is_absolute():
            raise ValueError("toolchain_root must be absolute")
        if not ssh_argv or any(not value for value in ssh_argv):
            raise ValueError("ssh_argv must contain non-empty values")
        self._config = config
        self._target = worker_ssh_target
        self._runner = runner
        self._toolchain_root = toolchain_root
        self._worker_unit = worker_unit
        self._ssh_argv = tuple(ssh_argv)
        self._started = False
        self._launch_evidence: WorkerRpcLaunchEvidence | None = None

    @property
    def launch_evidence(self) -> WorkerRpcLaunchEvidence | None:
        return self._launch_evidence

    @staticmethod
    def _start_argv(unit: str, worker_argv: Sequence[str]) -> tuple[str, ...]:
        return (
            "systemd-run",
            "--user",
            "--unit",
            unit,
            "--service-type=exec",
            "--property=Restart=no",
            "--property=KillMode=control-group",
            "--",
        ) + tuple(worker_argv)

    def _remote(self, argv: Sequence[str]) -> tuple[str, ...]:
        return self._ssh_argv + ("--", self._target) + tuple(argv)

    def _run(self, argv: Sequence[str]) -> str:
        try:
            return self._runner(self._remote(argv))
        except Exception as exc:  # noqa: BLE001 - sanitize the remote boundary
            raise WorkerRpcLifecycleError("worker RPC service command failed") from exc

    def _pid(self) -> int:
        output = self._run(
            (
                "systemctl",
                "--user",
                "show",
                "--property=MainPID",
                "--value",
                self._worker_unit,
            )
        ).strip()
        if not output.isdecimal() or int(output) <= 0:
            raise WorkerRpcLifecycleError("worker RPC MainPID unavailable")
        return int(output)

    def _measure(self, pid: int) -> WorkerRpcLaunchEvidence:
        exe_path = self._run(("readlink", "-f", f"/proc/{pid}/exe")).strip()
        if not exe_path.startswith("/"):
            raise WorkerRpcLifecycleError("worker RPC executable path unavailable")

        proc_exe = f"/proc/{pid}/exe"
        hash_fields = self._run(("sha256sum", proc_exe)).strip().split()
        valid_hash = (
            len(hash_fields) >= 2
            and len(hash_fields[0]) == 64
            and all(character in "0123456789abcdef" for character in hash_fields[0])
            and hash_fields[1] == proc_exe
        )
        if not valid_hash:
            raise WorkerRpcLifecycleError("worker RPC executable hash unavailable")

        raw_stat = self._run(("cat", f"/proc/{pid}/stat")).strip().split()
        if len(raw_stat) < 22 or not raw_stat[21].isdecimal() or int(raw_stat[21]) <= 0:
            raise WorkerRpcLifecycleError("worker RPC process start identity unavailable")
        start_ticks = int(raw_stat[21])
        raw_cmdline = self._run(("cat", f"/proc/{pid}/cmdline"))
        argv = tuple(value for value in raw_cmdline.split("\0") if value)
        commit = self._run(("git", "-C", str(self._toolchain_root), "rev-parse", "HEAD")).strip()

        if exe_path != str(self._config.worker_rpc_server_path):
            raise WorkerRpcLifecycleError("worker RPC executable path mismatch")
        if hash_fields[0] != EXPECTED_RPC_SERVER_SHA256:
            raise WorkerRpcLifecycleError("worker RPC executable hash mismatch")
        if argv != self._config.worker_argv():
            raise WorkerRpcLifecycleError("worker RPC command line mismatch")
        if commit != PINNED_COMMIT:
            raise WorkerRpcLifecycleError("worker llama.cpp commit mismatch")

        return WorkerRpcLaunchEvidence(
            worker_pid=pid,
            worker_start_ticks=start_ticks,
            worker_argv=argv,
            worker_exe_path=exe_path,
            worker_exe_sha256=hash_fields[0],
            worker_git_commit=commit,
            worker_unit=self._worker_unit,
        )

    def start(self) -> WorkerRpcLaunchEvidence:
        """Start the worker-only unit and return its measured launch identity."""
        if self._started:
            raise WorkerRpcLifecycleError("worker RPC lifecycle is already started")
        worker_started = False
        try:
            self._run(self._start_argv(self._worker_unit, self._config.worker_argv()))
            worker_started = True
            evidence = self._measure(self._pid())
            self._launch_evidence = evidence
            self._started = True
            return evidence
        except Exception as exc:  # noqa: BLE001 - cleanup a partial start
            if worker_started:
                launched_pid = (
                    self._launch_evidence.worker_pid if self._launch_evidence else None
                )
                self._stop_and_verify(launched_pid)
            self._started = False
            self._launch_evidence = None
            if isinstance(exc, WorkerRpcLifecycleError):
                raise
            raise WorkerRpcLifecycleError("worker RPC transient start failed") from exc

    def remeasure_after_capture(self) -> WorkerRpcLaunchEvidence:
        """Fail unless the exact launch process and identity still exist."""
        launch = self._launch_evidence
        if not self._started or launch is None:
            raise WorkerRpcLifecycleError("worker RPC lifecycle is not started")
        current_pid = self._pid()
        if current_pid != launch.worker_pid:
            raise WorkerRpcLifecycleError("worker RPC MainPID changed during capture")
        current = self._measure(current_pid)
        if current != launch:
            raise WorkerRpcLifecycleError("worker RPC identity changed during capture")
        return current

    def _stop_and_verify(self, pid: int | None) -> None:
        self._run(("systemctl", "--user", "stop", self._worker_unit))
        output = self._run(
            (
                "systemctl",
                "--user",
                "show",
                "--property=MainPID",
                "--value",
                self._worker_unit,
            )
        ).strip()
        if output != "0":
            raise WorkerRpcLifecycleError("worker RPC unit did not quiesce")
        if pid is not None:
            self._run(("test", "!", "-e", f"/proc/{pid}"))

    def stop(self) -> None:
        """Stop the remote worker unit without ever addressing a head unit."""
        if not self._started:
            return
        try:
            launched_pid = self._launch_evidence.worker_pid if self._launch_evidence else None
            self._stop_and_verify(launched_pid)
        finally:
            self._started = False
            self._launch_evidence = None
