"""Concrete, fail-closed execution of one pinned GLM-5.2 candidate evaluation.

The driver coordinates an already-authorized :class:`CandidateEvalPlan` with
the two-Spark llama.cpp RPC lifecycle.  Every external boundary is injectable,
so unit tests execute no subprocess, service, HTTP request, network operation,
or GPU workload.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_atlas.evaluation.eval_lab import EvalLabAdapter
from model_atlas.evaluation.glm52_candidate_eval import (
    CandidateEvalPlan,
    CandidateEvalResult,
    CandidateEvalTaskEvidence,
    build_task_evidence,
    parse_candidate_eval_runs,
)
from model_atlas.fit_telemetry import CanaryPhase, CanaryStep
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    LlamaCppRpcRuntimeConfig,
)
from model_atlas.runtime_canary_driver import HttpResponse, RuntimeLaunchEvidence
from model_atlas.two_node_canary_executor import RuntimeProcessIds

_MAX_STDOUT_BYTES = 1024 * 1024
_MAX_EXECUTABLE_BYTES = 64 * 1024 * 1024
_RUN_ID = frozenset("0123456789abcdef")


class CandidateEvalExecutionError(RuntimeError):
    """A sanitized candidate evaluation boundary failure."""


@dataclass(frozen=True)
class EvalLabCommandResult:
    """Non-secret output from the pinned Eval Lab command."""

    returncode: int
    stdout: bytes


class EvalLabCommandRunner(Protocol):
    def run(
        self, argv: Sequence[str], *, cwd: Path, timeout_seconds: float
    ) -> EvalLabCommandResult: ...


class CandidateHttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse: ...


class CandidateRuntimeLifecycle(Protocol):
    @property
    def launch_evidence(self) -> RuntimeLaunchEvidence | None: ...

    def start(self, step: CanaryStep) -> RuntimeProcessIds: ...

    def measure_post_run_worker(self, pids: RuntimeProcessIds) -> RuntimeLaunchEvidence: ...

    def stop(self) -> None: ...


class RuntimeQuiescenceVerifier(Protocol):
    def verify(self, pids: RuntimeProcessIds) -> None: ...


class SubprocessEvalLabRunner:
    """Production subprocess boundary; stderr is never captured or disclosed."""

    def run(
        self, argv: Sequence[str], *, cwd: Path, timeout_seconds: float
    ) -> EvalLabCommandResult:
        try:
            completed = subprocess.run(
                tuple(argv),
                cwd=cwd,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CandidateEvalExecutionError("Eval Lab command execution failed") from exc
        if len(completed.stdout) > _MAX_STDOUT_BYTES:
            raise CandidateEvalExecutionError("Eval Lab stdout exceeded its safe bound")
        return EvalLabCommandResult(returncode=completed.returncode, stdout=completed.stdout)


class ArgvRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str: ...


class SystemdRuntimeQuiescenceVerifier:
    """Independently prove both transient units and launch PIDs are gone."""

    def __init__(
        self,
        *,
        worker_ssh_target: str,
        runner: ArgvRunner,
        worker_unit: str = "atlas-glm52-rpc-worker",
        head_unit: str = "atlas-glm52-rpc-head",
        ssh_argv: Sequence[str] = ("ssh", "-o", "BatchMode=yes"),
    ) -> None:
        if not worker_ssh_target or not worker_unit or not head_unit:
            raise ValueError("worker target and runtime units are required")
        if not ssh_argv or any(not value for value in ssh_argv):
            raise ValueError("SSH argv must contain non-empty values")
        self._target = worker_ssh_target
        self._runner = runner
        self._worker_unit = worker_unit
        self._head_unit = head_unit
        self._ssh_argv = tuple(ssh_argv)

    def _run(self, argv: Sequence[str], *, worker: bool) -> str:
        command = (
            self._ssh_argv + ("--", self._target) + tuple(argv) if worker else tuple(argv)
        )
        try:
            return self._runner(command)
        except Exception as exc:  # noqa: BLE001 - sanitize command output
            raise CandidateEvalExecutionError("runtime quiescence command failed") from exc

    def verify(self, pids: RuntimeProcessIds) -> None:
        for unit, pid, worker in (
            (self._head_unit, pids.head_server_pid, False),
            (self._worker_unit, pids.worker_rpc_pid, True),
        ):
            main_pid = self._run(
                ("systemctl", "--user", "show", "--property=MainPID", "--value", unit),
                worker=worker,
            ).strip()
            if main_pid != "0":
                raise CandidateEvalExecutionError("runtime unit did not quiesce")
            self._run(("test", "!", "-e", f"/proc/{pid}"), worker=worker)


def _stable_executable_sha256(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise CandidateEvalExecutionError("Eval Lab executable path is not canonical")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_EXECUTABLE_BYTES
        ):
            raise CandidateEvalExecutionError("Eval Lab executable is not a bounded file")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > _MAX_EXECUTABLE_BYTES:
                raise CandidateEvalExecutionError("Eval Lab executable exceeded its safe bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if size != before.st_size or identity_before != identity_after:
            raise CandidateEvalExecutionError("Eval Lab executable changed during measurement")
        return digest.hexdigest()
    except OSError as exc:
        raise CandidateEvalExecutionError("Eval Lab executable cannot be measured") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class CandidateEvalExecutionResult(BaseModel):
    """Measured, typed evidence for a successful candidate evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    execution_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eval_lab_revision: Literal["a20da6c6b9cbf872f7c083bffe66afde40c2c8f2"] = (
        "a20da6c6b9cbf872f7c083bffe66afde40c2c8f2"
    )
    eval_lab_cwd: str = Field(pattern=r"^/")
    endpoint_health_verified: Literal[True] = True
    launch_evidence: RuntimeLaunchEvidence
    post_run_evidence: RuntimeLaunchEvidence
    quiescence_verified: Literal[True] = True
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stdout_size_bytes: int = Field(ge=0, le=_MAX_STDOUT_BYTES)
    task_evidence: tuple[CandidateEvalTaskEvidence, ...] = Field(min_length=1)
    candidate_result: CandidateEvalResult
    evidence_kind: Literal["measured"] = "measured"

    @model_validator(mode="after")
    def _bind_content(self) -> CandidateEvalExecutionResult:
        if self.launch_evidence != self.post_run_evidence:
            raise ValueError("runtime identity changed during candidate evaluation")
        if self.task_evidence != self.candidate_result.task_evidence:
            raise ValueError("execution task evidence differs from parsed result")
        payload = self.model_dump(mode="json", exclude={"execution_sha256"})
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.execution_sha256 is not None and self.execution_sha256 != expected:
            raise ValueError("candidate execution digest differs from canonical content")
        object.__setattr__(self, "execution_sha256", expected)
        return self


def _snapshot_run_dirs(runs_root: Path, *, require_exists: bool) -> frozenset[str]:
    if not runs_root.is_absolute() or runs_root.is_symlink():
        raise CandidateEvalExecutionError("Eval Lab runs root must be absolute and symlink-free")
    if not runs_root.exists():
        if require_exists:
            raise CandidateEvalExecutionError("Eval Lab did not create its runs root")
        return frozenset()
    if not runs_root.is_dir() or runs_root.resolve() != runs_root:
        raise CandidateEvalExecutionError("Eval Lab runs root is not a stable directory")
    names: set[str] = set()
    for item in runs_root.iterdir():
        if item.is_symlink():
            raise CandidateEvalExecutionError("Eval Lab runs root contains a symlink")
        if item.is_dir():
            names.add(item.name)
    return frozenset(names)


def _stdout_run_ids(stdout: bytes, tasks: tuple[str, ...]) -> tuple[str, ...]:
    if len(stdout) > _MAX_STDOUT_BYTES:
        raise CandidateEvalExecutionError("Eval Lab stdout exceeded its safe bound")
    try:
        payload = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateEvalExecutionError("Eval Lab stdout is not valid JSON") from exc
    if not isinstance(payload, list) or len(payload) != len(tasks):
        raise CandidateEvalExecutionError("Eval Lab stdout does not identify every task run")
    run_ids: list[str] = []
    for expected_task, row in zip(tasks, payload, strict=True):
        if not isinstance(row, dict):
            raise CandidateEvalExecutionError("Eval Lab stdout run summary is invalid")
        run_id = row.get("run_id")
        if (
            row.get("task_id") != expected_task
            or not isinstance(run_id, str)
            or len(run_id) != 12
            or any(character not in _RUN_ID for character in run_id)
        ):
            raise CandidateEvalExecutionError("Eval Lab stdout run identity is invalid")
        run_ids.append(run_id)
    if len(run_ids) != len(set(run_ids)):
        raise CandidateEvalExecutionError("Eval Lab stdout run identities are ambiguous")
    return tuple(run_ids)


class CandidateEvalExecutor:
    """Execute one immutable candidate plan against one measured two-node runtime."""

    def __init__(
        self,
        config: LlamaCppRpcRuntimeConfig,
        *,
        lifecycle: CandidateRuntimeLifecycle,
        transport: CandidateHttpTransport,
        command_runner: EvalLabCommandRunner,
        quiescence: RuntimeQuiescenceVerifier,
        eval_lab_cwd: Path,
    ) -> None:
        if not eval_lab_cwd.is_absolute():
            raise ValueError("Eval Lab cwd must be absolute")
        self._config = config
        self._lifecycle = lifecycle
        self._transport = transport
        self._command_runner = command_runner
        self._quiescence = quiescence
        self._cwd = eval_lab_cwd

    def _check_plan(self, plan: CandidateEvalPlan) -> None:
        request = plan.eval_request
        if plan.plan_sha256 is None:
            raise CandidateEvalExecutionError("candidate evaluation plan digest is incomplete")
        try:
            cwd = self._cwd.resolve(strict=True)
        except OSError as exc:
            raise CandidateEvalExecutionError("pinned Eval Lab cwd is unavailable") from exc
        if self._cwd.is_symlink() or cwd != self._cwd or Path(request.eval_lab_root) != cwd:
            raise CandidateEvalExecutionError("candidate evaluation cwd differs from its plan")
        expected_endpoint = f"http://{self._config.api_host}:{self._config.api_port}/v1"
        endpoint = str(request.endpoint.endpoint_url).rstrip("/")
        expected = (
            plan.candidate_artifact_path == str(self._config.artifact_path)
            and plan.candidate_artifact_sha256 == self._config.artifact_sha256
            and plan.runtime_config_sha256 == self._config.canonical_sha256()
            and endpoint == expected_endpoint
            and request.endpoint.config_sha256 == self._config.canonical_sha256()
        )
        if not expected:
            raise CandidateEvalExecutionError("candidate endpoint differs from runtime contract")
        if _stable_executable_sha256(Path(plan.argv[0])) != plan.eval_lab_executable_sha256:
            raise CandidateEvalExecutionError("pinned Eval Lab executable drifted")
        try:
            emitted = EvalLabAdapter(executable=plan.argv[0]).emit_argv(request)
        except (OSError, ValueError) as exc:
            raise CandidateEvalExecutionError("pinned Eval Lab inputs drifted") from exc
        if emitted.eval_lab_revision != plan.eval_lab_revision or emitted.argv != plan.argv:
            raise CandidateEvalExecutionError("pinned Eval Lab argv drifted")

    def _verify_launch(
        self, evidence: RuntimeLaunchEvidence | None, pids: RuntimeProcessIds
    ) -> RuntimeLaunchEvidence:
        if evidence is None:
            raise CandidateEvalExecutionError("runtime launch measurement is missing")
        expected = (
            evidence.head_pid == pids.head_server_pid
            and evidence.worker_pid == pids.worker_rpc_pid
            and evidence.head_argv == self._config.head_argv()
            and evidence.worker_argv == self._config.worker_argv()
            and evidence.head_exe_path == str(self._config.llama_server_path)
            and evidence.head_exe_sha256 == EXPECTED_LLAMA_SERVER_SHA256
            and evidence.worker_exe_path == str(self._config.worker_rpc_server_path)
            and evidence.worker_exe_sha256 == EXPECTED_RPC_SERVER_SHA256
        )
        if not expected:
            raise CandidateEvalExecutionError("runtime launch measurement drifted")
        return evidence

    def _verify_health(self, plan: CandidateEvalPlan) -> None:
        base_url = str(plan.eval_request.endpoint.endpoint_url).rstrip("/")
        try:
            health = self._transport.request("GET", base_url.removesuffix("/v1") + "/health")
            models = self._transport.request("GET", base_url + "/models")
            model_payload = json.loads(models.body)
        except Exception as exc:  # noqa: BLE001 - sanitize HTTP details
            raise CandidateEvalExecutionError("candidate endpoint health request failed") from exc
        data = model_payload.get("data") if isinstance(model_payload, dict) else None
        model_ids: set[str] = set()
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    model_ids.add(row["id"])
        if (
            health.status != 200
            or models.status != 200
            or len(health.body) > _MAX_STDOUT_BYTES
            or len(models.body) > _MAX_STDOUT_BYTES
            or plan.eval_request.model_name not in model_ids
        ):
            raise CandidateEvalExecutionError("candidate endpoint health verification failed")

    def execute(self, plan: CandidateEvalPlan) -> CandidateEvalExecutionResult:
        self._check_plan(plan)
        assert plan.plan_sha256 is not None
        runs_root = Path(plan.eval_request.runs_root)
        before = _snapshot_run_dirs(runs_root, require_exists=False)
        step = CanaryStep(
            step_id="candidate-eval-4k",
            phase=CanaryPhase.CONTEXT_RESTART,
            context_tokens=self._config.context_size,
            max_output_tokens=plan.parameters.max_tokens,
            repeats=1,
            discard_warmup_repeats=0,
            restart_runtime=True,
        )
        pids: RuntimeProcessIds | None = None
        launch: RuntimeLaunchEvidence | None = None
        post_run: RuntimeLaunchEvidence | None = None
        command: EvalLabCommandResult | None = None
        failure: CandidateEvalExecutionError | None = None
        stop_failure: Exception | None = None
        quiescence_failure: Exception | None = None
        try:
            pids = self._lifecycle.start(step)
            launch = self._verify_launch(self._lifecycle.launch_evidence, pids)
            self._verify_health(plan)
            timeout = plan.parameters.timeout_seconds * len(plan.eval_request.tasks) + 60.0
            command = self._command_runner.run(plan.argv, cwd=self._cwd, timeout_seconds=timeout)
            if (
                isinstance(command.returncode, bool)
                or not isinstance(command.returncode, int)
                or not isinstance(command.stdout, bytes)
                or len(command.stdout) > _MAX_STDOUT_BYTES
            ):
                raise CandidateEvalExecutionError("Eval Lab command result is invalid")
            if _stable_executable_sha256(Path(plan.argv[0])) != plan.eval_lab_executable_sha256:
                raise CandidateEvalExecutionError("pinned Eval Lab executable drifted")
            if command.returncode != 0:
                failure = CandidateEvalExecutionError("Eval Lab command exited nonzero")
        except CandidateEvalExecutionError as exc:
            failure = exc
        except Exception as exc:  # noqa: BLE001 - sanitize injected boundaries
            failure = CandidateEvalExecutionError("candidate evaluation execution failed")
            failure.__cause__ = exc
        finally:
            if pids is not None:
                try:
                    post_run = self._lifecycle.measure_post_run_worker(pids)
                    if launch is None or post_run != launch:
                        raise CandidateEvalExecutionError(
                            "runtime identity changed during candidate evaluation"
                        )
                except Exception as exc:  # noqa: BLE001 - cleanup still required
                    if failure is None:
                        failure = CandidateEvalExecutionError(
                            "post-evaluation worker attestation failed"
                        )
                        failure.__cause__ = exc
                try:
                    self._lifecycle.stop()
                except Exception as exc:  # noqa: BLE001 - independently check quiescence
                    stop_failure = exc
                try:
                    self._quiescence.verify(pids)
                except Exception as exc:  # noqa: BLE001 - sanitize quiescence details
                    quiescence_failure = exc
        if stop_failure is not None or quiescence_failure is not None:
            raise CandidateEvalExecutionError("runtime stop or quiescence is uncertain") from (
                stop_failure or quiescence_failure
            )
        if failure is not None:
            raise failure
        if pids is None or launch is None or post_run is None or command is None:
            raise CandidateEvalExecutionError("candidate evaluation evidence is incomplete")

        tasks = tuple(plan.eval_request.tasks)
        run_ids = _stdout_run_ids(command.stdout, tasks)
        after = _snapshot_run_dirs(runs_root, require_exists=True)
        new_dirs = after - before
        if new_dirs != frozenset(run_ids):
            raise CandidateEvalExecutionError("Eval Lab run-directory discovery is ambiguous")
        try:
            evidence = tuple(
                build_task_evidence(task_id, runs_root / run_id)
                for task_id, run_id in zip(tasks, run_ids, strict=True)
            )
            parsed = parse_candidate_eval_runs(plan, evidence)
        except Exception as exc:  # noqa: BLE001 - normalize evidence boundary
            raise CandidateEvalExecutionError("Eval Lab run evidence validation failed") from exc
        return CandidateEvalExecutionResult(
            plan_sha256=plan.plan_sha256,
            eval_lab_cwd=str(self._cwd),
            launch_evidence=launch,
            post_run_evidence=post_run,
            stdout_sha256=hashlib.sha256(command.stdout).hexdigest(),
            stdout_size_bytes=len(command.stdout),
            task_evidence=evidence,
            candidate_result=parsed,
        )


__all__ = [
    "CandidateEvalExecutionError",
    "CandidateEvalExecutionResult",
    "CandidateEvalExecutor",
    "CandidateHttpTransport",
    "CandidateRuntimeLifecycle",
    "EvalLabCommandResult",
    "EvalLabCommandRunner",
    "RuntimeQuiescenceVerifier",
    "SubprocessEvalLabRunner",
    "SystemdRuntimeQuiescenceVerifier",
]
