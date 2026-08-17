"""Trap-safe two-node maintenance coordination.

The coordinator owns no model/runtime implementation.  It only brackets an
operator-supplied payload with a deterministic production drain and best-effort
restoration.  Every external command goes through ``CommandRunner`` so tests
never touch real services.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.ops.maintenance_watch import extract_shard_progress


def _now() -> datetime:
    return datetime.now(UTC)


def _publish_json_exclusive(path: Path, encoded: bytes) -> None:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise MaintenanceFailure("maintenance receipt path is invalid")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent = os.open("/", flags)
    temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = -1
    try:
        for component in path.parent.parts[1:]:
            following = os.open(component, flags, dir_fd=parent)
            os.close(parent)
            parent = following
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise MaintenanceFailure("maintenance receipt write failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.fsync(parent)
        os.unlink(temporary, dir_fd=parent)
        os.fsync(parent)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent)
            os.fsync(parent)
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


class CommandResult(BaseModel):
    """Sanitized command result; output is never serialized into the receipt."""

    model_config = ConfigDict(frozen=True)
    returncode: int
    stdout: str = ""


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str]) -> CommandResult:
        """Run argv directly (never through a shell)."""


class _ChildProcess(Protocol):
    pid: int

    def communicate(self) -> tuple[str, str | None]: ...

    def wait(self, timeout: float | None = None) -> int: ...


class _ProcessFactory(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        stdout: int,
        stderr: int,
        text: bool,
        start_new_session: bool,
    ) -> _ChildProcess: ...


class SubprocessCommandRunner:
    """Production runner that reaps its child group before propagating signals."""

    def __init__(
        self,
        *,
        process_factory: _ProcessFactory | None = None,
        killpg: Callable[[int, int], None] = os.killpg,
        wait_timeout_seconds: float = 10.0,
        group_poll_attempts: int = 10,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if wait_timeout_seconds <= 0 or group_poll_attempts < 1:
            raise ValueError("wait timeout and group poll attempts must be positive")
        self._process_factory = process_factory or self._popen
        self._killpg = killpg
        self._wait_timeout = wait_timeout_seconds
        self._group_poll_attempts = group_poll_attempts
        self._sleep = sleep

    @staticmethod
    def _popen(
        argv: Sequence[str],
        *,
        stdout: int,
        stderr: int,
        text: bool,
        start_new_session: bool,
    ) -> _ChildProcess:
        return subprocess.Popen(
            list(argv),
            stdout=stdout,
            stderr=stderr,
            text=text,
            start_new_session=start_new_session,
        )

    def _wait_leader(self, process: _ChildProcess) -> bool:
        while True:
            try:
                process.wait(timeout=self._wait_timeout)
                return True
            except MaintenanceInterrupted:
                continue
            except subprocess.TimeoutExpired:
                return False

    def _group_exists(self, process_group: int) -> bool:
        while True:
            try:
                self._killpg(process_group, 0)
            except MaintenanceInterrupted:
                continue
            except ProcessLookupError:
                return False
            return True

    def _wait_for_group_exit(self, process_group: int) -> bool:
        for attempt in range(self._group_poll_attempts):
            if not self._group_exists(process_group):
                return True
            if attempt + 1 < self._group_poll_attempts:
                while True:
                    try:
                        self._sleep(self._wait_timeout / self._group_poll_attempts)
                    except MaintenanceInterrupted:
                        continue
                    break
        return False

    def _reap_group(self, process: _ChildProcess) -> None:
        group = process.pid
        self._signal_group(group, signal.SIGTERM)
        leader_reaped = self._wait_leader(process)
        if self._group_exists(group):
            self._signal_group(group, signal.SIGKILL)
            if not leader_reaped:
                leader_reaped = self._wait_leader(process)
            if not self._wait_for_group_exit(group):
                raise ProcessGroupQuiescenceError("payload process group could not be quiesced")
        # The group may already be gone, but the direct child still must be reaped.
        if not leader_reaped and not self._wait_leader(process):
            raise ProcessGroupQuiescenceError("payload process leader could not be reaped")

    def _signal_group(self, process_group: int, sig: signal.Signals) -> None:
        while True:
            try:
                self._killpg(process_group, sig)
            except MaintenanceInterrupted:
                continue
            except ProcessLookupError:
                return
            return

    def run(self, argv: Sequence[str]) -> CommandResult:
        process = self._process_factory(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _stderr = process.communicate()
        except BaseException:
            try:
                self._reap_group(process)
            except ProcessGroupQuiescenceError:
                raise
            except BaseException as exc:
                raise ProcessGroupQuiescenceError(
                    "payload process group cleanup could not be proven"
                ) from exc
            raise
        returncode = process.wait()
        return CommandResult(returncode=returncode, stdout=stdout)


class BinaryHash(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    sha256: str | None
    readable: bool


class StateSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    target: str
    kind: Literal["user_unit", "container"]
    active: bool
    observed_at: datetime


class ActionReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)
    action_id: str
    target: str
    requested: bool
    executed: bool
    success: bool
    started_at: datetime
    finished_at: datetime
    evidence: str = ""


class MaintenanceReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    run_id: str
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    previous_states: list[StateSnapshot] = Field(default_factory=list)
    binary_hashes: list[BinaryHash] = Field(default_factory=list)
    actions: list[ActionReceipt] = Field(default_factory=list)
    restoration_evidence: dict[str, bool] = Field(default_factory=dict)
    manual_intervention_required: bool = False
    success: bool = False
    failure: str | None = None


class MaintenanceConfig(BaseModel):
    """Fixed, non-secret maintenance targets.

    Unit-file paths are retained only for receipt/config compatibility.  The
    maintenance coordinator never removes unit files: transient units can be
    stopped safely, while deleting persistent units is not rollback-safe.
    """

    model_config = ConfigDict(frozen=True)
    gateway_unit: str = "hermes-gateway.service"
    vision_adapter_unit: str = "dsv4-vision-adapter.service"
    qwen_unit: str = "qwen35-vision.service"
    dsv4_container: str = "deepseek-v4-flash-vllm-dspark-1"
    dsv4_stop_script: Path = Path(
        "/home/kristianaaron/ai-lab/repos/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/"
        "stop-deepseek-v4-flash-dspark.sh"
    )
    dsv4_start_script: Path = Path(
        "/home/kristianaaron/ai-lab/repos/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/"
        "start-deepseek-v4-flash-dspark.sh"
    )
    head_runtime_unit: str = HEAD_TRANSIENT_UNIT
    worker_rpc_unit: str = WORKER_TRANSIENT_UNIT
    worker_ssh_target: str = "10.77.0.2"
    head_runtime_unit_file: Path | None = None
    worker_rpc_unit_file: Path | None = None
    binary_paths: tuple[Path, ...] = ()
    journal_dir: Path
    receipt_path: Path
    # Optional shard count for the DSV4 model being restored, so the live UX can
    # show a real "N / M shards" progress bar. When unset, no shard events are
    # emitted (honest: the coordinator does not know the model layout).
    dsv4_model_shards: int | None = None

    def all_binary_paths(self) -> tuple[Path, ...]:
        return (self.dsv4_stop_script, self.dsv4_start_script, *self.binary_paths)


class MaintenanceInterrupted(RuntimeError):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"signal:{signum}")


class MaintenanceFailure(RuntimeError):
    pass


class ProcessGroupQuiescenceError(RuntimeError):
    """The payload process group survived termination; operator action is required."""


class MaintenanceCoordinator:
    """Execute and always unwind a two-node maintenance window."""

    def __init__(
        self,
        config: MaintenanceConfig,
        runner: CommandRunner,
        *,
        execute: bool = False,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.config = config
        self.runner = runner
        self.execute = execute
        self.clock = clock
        self._states: dict[str, StateSnapshot] = {}
        self._actions: list[ActionReceipt] = []
        self._restored: set[str] = set()
        self._runtime_drained: set[str] = set()
        self._restore_completed = False
        self._restoration_evidence: dict[str, bool] = {}
        self._restoring = False

    def _command(self, argv: Sequence[str]) -> CommandResult:
        return self.runner.run(argv)

    def _unit_active(self, unit: str, *, remote: bool = False) -> bool:
        command = ["systemctl", "--user", "is-active", "--quiet", unit]
        if remote:
            command = ["ssh", self.config.worker_ssh_target, *command]
        result = self._command(command)
        if result.returncode == 0:
            return True
        # 3 = inactive; 4 = unknown/not-found. A unit that does not exist is
        # genuinely not running, so it is not a consumer to drain or restore.
        if result.returncode in (3, 4):
            return False
        raise MaintenanceFailure("unit liveness is unknown")

    def _container_active(self, name: str) -> bool:
        result = self._command(["docker", "inspect", "--format", "{{.State.Running}}", name])
        if result.returncode != 0:
            raise MaintenanceFailure("container liveness is unknown")
        state = result.stdout.strip()
        if state == "true":
            return True
        if state == "false":
            return False
        raise MaintenanceFailure("container liveness is malformed")

    def _snapshot(self) -> None:
        observed = self.clock()
        units = (
            ("gateway", self.config.gateway_unit, False),
            ("vision_adapter", self.config.vision_adapter_unit, False),
            ("qwen", self.config.qwen_unit, False),
            ("head_runtime", self.config.head_runtime_unit, False),
            ("worker_rpc", self.config.worker_rpc_unit, True),
        )
        for key, unit, remote in units:
            self._states[key] = StateSnapshot(
                target=unit,
                kind="user_unit",
                active=self._unit_active(unit, remote=remote),
                observed_at=observed,
            )
        self._states["dsv4"] = StateSnapshot(
            target=self.config.dsv4_container,
            kind="container",
            active=self._container_active(self.config.dsv4_container),
            observed_at=observed,
        )

    @staticmethod
    def _hash_binary(path: Path) -> BinaryHash:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return BinaryHash(path=str(path), sha256=None, readable=False)
        return BinaryHash(path=str(path), sha256=digest.hexdigest(), readable=True)

    def _record(
        self,
        action_id: str,
        target: str,
        *,
        requested: bool,
        command: Sequence[str] | None,
        output_path: Path | None = None,
    ) -> bool:
        started = self.clock()
        executed = bool(self.execute and requested and command is not None)
        success = True
        evidence = "dry_run" if requested and not self.execute else "not_previously_active"
        if executed:
            assert command is not None
            try:
                result = self._command(command)
                success = result.returncode == 0
                evidence = "return_code=0" if success else f"return_code={result.returncode}"
                if output_path is not None and success:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(result.stdout, encoding="utf-8")
                    evidence = (
                        f"journal_sha256={hashlib.sha256(result.stdout.encode()).hexdigest()}"
                    )
            except MaintenanceInterrupted:
                if not self._restoring:
                    raise
                success = False
                evidence = "interrupted_during_restore"
            except ProcessGroupQuiescenceError:
                raise
            except BaseException as exc:
                if not self._restoring:
                    raise
                success = False
                evidence = f"exception_type={type(exc).__name__}"
        self._actions.append(
            ActionReceipt(
                action_id=action_id,
                target=target,
                requested=requested,
                executed=executed,
                success=success,
                started_at=started,
                finished_at=self.clock(),
                evidence=evidence,
            )
        )
        return success

    def _required(
        self, action_id: str, target: str, command: Sequence[str], *, active: bool
    ) -> None:
        if not self._record(action_id, target, requested=active, command=command):
            raise MaintenanceFailure(f"{action_id}:return_code")

    def _emit(
        self,
        *,
        phase: str,
        status: str,
        service: str | None = None,
        method: str | None = None,
        detail: str | None = None,
        shard_current: int | None = None,
        shard_total: int | None = None,
    ) -> None:
        """Append a typed maintenance-lifecycle event for live UX / observability.

        Best-effort by design: a failure to write the diagnostics stream never
        fails a maintenance run. Events carry phase+status and optional
        service/method/detail, appended to ``maintenance-events.jsonl`` in the
        journal dir so a UI or ``maintenance-watch`` can tail drain -> produce
        -> restore in real time.
        """
        event: dict[str, object] = {
            "ts": self.clock().isoformat(timespec="microseconds") + "Z",
            "phase": phase,
            "status": status,
        }
        if service is not None:
            event["service"] = service
        if method is not None:
            event["method"] = method
        if detail is not None:
            event["detail"] = detail
        if shard_current is not None:
            event["shard_current"] = shard_current
        if shard_total is not None:
            event["shard_total"] = shard_total
        try:
            self.config.journal_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.journal_dir / "maintenance-events.jsonl"
            encoded = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
            with open(path, "ab") as fh:
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:  # noqa: BLE001 — diagnostics never fail maintenance
            pass

    def report_shard_progress(self, text: str) -> None:
        """Feed model-loader output to emit per-shard progress events.

        Only emits when a real ``N/M`` shard-progress line is present (honest:
        never fabricates loading progress). Used by a call site that tails the
        restored model's own loader output.
        """
        progress = extract_shard_progress(text)
        if progress is not None:
            current, total = progress
            self._emit(
                phase="restore",
                status="shard_loaded",
                service="dsv4",
                shard_current=current,
                shard_total=total,
            )

    def _acquire(self) -> None:
        # External dependency order is part of the safety contract.
        for key, unit in (
            ("gateway", self.config.gateway_unit),
            ("vision_adapter", self.config.vision_adapter_unit),
            ("qwen", self.config.qwen_unit),
        ):
            self._required(
                f"stop_{key}",
                unit,
                ["systemctl", "--user", "stop", unit],
                active=self._states[key].active,
            )
            self._emit(phase="drain", status="release", service=key)
        self._required(
            "stop_dsv4",
            str(self.config.dsv4_stop_script),
            [str(self.config.dsv4_stop_script)],
            active=self._states["dsv4"].active,
        )
        self._emit(phase="drain", status="release", service="dsv4")
        self._preserve_and_stop_runtime(
            key="head_runtime",
            unit=self.config.head_runtime_unit,
            remote=False,
            unit_file=self.config.head_runtime_unit_file,
        )
        self._preserve_and_stop_runtime(
            key="worker_rpc",
            unit=self.config.worker_rpc_unit,
            remote=True,
            unit_file=self.config.worker_rpc_unit_file,
        )

    def verify_drained(self) -> None:
        """Authoritatively re-probe all consumers; no caller-supplied state."""
        active = (
            self._unit_active(self.config.gateway_unit),
            self._unit_active(self.config.vision_adapter_unit),
            self._unit_active(self.config.qwen_unit),
            self._container_active(self.config.dsv4_container),
            self._unit_active(self.config.head_runtime_unit),
            self._unit_active(self.config.worker_rpc_unit, remote=True),
        )
        if any(active):
            raise MaintenanceFailure("drain verification failed")
        self._emit(phase="drain", status="complete", detail="all consumers quiesced")

    def _preserve_and_stop_runtime(
        self,
        *,
        key: Literal["head_runtime", "worker_rpc"],
        unit: str,
        remote: bool,
        unit_file: Path | None,
    ) -> None:
        del unit_file  # unit deletion is deliberately unsupported
        if key in self._runtime_drained:
            return
        journal_path = self.config.journal_dir / f"{key}.journal.log"
        journal_command = [
            "journalctl",
            "--user",
            "-u",
            unit,
            "--no-pager",
            "--output=short-iso-precise",
        ]
        if remote:
            journal_command = ["ssh", self.config.worker_ssh_target, *journal_command]
        self._record(
            f"preserve_{key}_journal",
            unit,
            requested=True,
            command=journal_command,
            output_path=journal_path,
        )
        stop_command = ["systemctl", "--user", "stop", unit]
        if remote:
            stop_command = ["ssh", self.config.worker_ssh_target, *stop_command]
        # Only issue the stop when the transient unit is currently active. If it
        # is already inactive/not-found (a normal state before the payload brings
        # up the canary), requesting a stop would return rc 5 (already stopped)
        # and fail the whole transaction for no reason.
        previously_active = self._unit_active(unit, remote=remote)
        self._record(
            f"stop_{key}", unit, requested=previously_active, command=stop_command
        )
        if not previously_active:
            self._runtime_drained.add(key)
            return
        self._runtime_drained.add(key)
        self._emit(phase="drain", status="release", service=key)

    def _restore_action(
        self, key: str, action_id: str, target: str, command: Sequence[str]
    ) -> None:
        if key in self._restored:
            return
        previously_active = self._states[key].active
        self._record(
            action_id,
            target,
            requested=previously_active,
            command=command,
        )
        self._restored.add(key)
        self._emit(phase="restore", status="load", service=key)

    def restore(self) -> None:
        """Best-effort, idempotent rollback; every transition is attempted."""
        if self._restore_completed:
            return
        self._restoring = True
        self._emit(phase="restore", status="start")
        try:
            # Experimental consumers are always drained head-first, then worker.
            # A payload may have started these exact transient units after the
            # acquisition drain, so restoration must stop them again.
            self._runtime_drained.discard("head_runtime")
            self._runtime_drained.discard("worker_rpc")
            self._preserve_and_stop_runtime(
                key="head_runtime",
                unit=self.config.head_runtime_unit,
                remote=False,
                unit_file=self.config.head_runtime_unit_file,
            )
            self._preserve_and_stop_runtime(
                key="worker_rpc",
                unit=self.config.worker_rpc_unit,
                remote=True,
                unit_file=self.config.worker_rpc_unit_file,
            )
            if self.execute:
                try:
                    runtime_active = self._unit_active(self.config.head_runtime_unit)
                    worker_active = self._unit_active(self.config.worker_rpc_unit, remote=True)
                except BaseException as exc:
                    raise ProcessGroupQuiescenceError(
                        "experimental runtime quiescence is unknown"
                    ) from exc
                if runtime_active or worker_active:
                    raise ProcessGroupQuiescenceError("experimental runtime did not quiesce")
            # Restore a pre-existing runtime in dependency order: worker first.
            self._restore_action(
                "worker_rpc",
                "restore_worker_rpc",
                self.config.worker_rpc_unit,
                [
                    "ssh",
                    self.config.worker_ssh_target,
                    "systemctl",
                    "--user",
                    "start",
                    self.config.worker_rpc_unit,
                ],
            )
            self._restore_action(
                "head_runtime",
                "restore_head_runtime",
                self.config.head_runtime_unit,
                ["systemctl", "--user", "start", self.config.head_runtime_unit],
            )
            self._restore_action(
                "dsv4",
                "restore_dsv4",
                str(self.config.dsv4_start_script),
                [str(self.config.dsv4_start_script)],
            )
            if self.config.dsv4_model_shards:
                self._emit(
                    phase="restore",
                    status="shard_plan",
                    service="dsv4",
                    shard_total=self.config.dsv4_model_shards,
                )
            for key, unit in (
                ("qwen", self.config.qwen_unit),
                ("vision_adapter", self.config.vision_adapter_unit),
                ("gateway", self.config.gateway_unit),
            ):
                self._restore_action(
                    key,
                    f"restore_{key}",
                    unit,
                    ["systemctl", "--user", "start", unit],
                )
        finally:
            self._restoring = False

        if self.execute:
            self._restoration_evidence = {
                "dsv4": self._container_active(self.config.dsv4_container)
                == self._states["dsv4"].active,
                "qwen": self._unit_active(self.config.qwen_unit) == self._states["qwen"].active,
                "vision_adapter": self._unit_active(self.config.vision_adapter_unit)
                == self._states["vision_adapter"].active,
                "gateway": self._unit_active(self.config.gateway_unit)
                == self._states["gateway"].active,
                "head_runtime": self._unit_active(self.config.head_runtime_unit)
                == self._states["head_runtime"].active,
                "worker_rpc": self._unit_active(self.config.worker_rpc_unit, remote=True)
                == self._states["worker_rpc"].active,
            }
        self._restore_completed = True
        self._emit(phase="restore", status="complete", detail="services restored")

    def run(
        self,
        payload: Sequence[str] | None = None,
        *,
        payload_scope: Callable[[], AbstractContextManager[None]] | None = None,
        payload_action: Callable[[], None] | None = None,
    ) -> MaintenanceReceipt:
        if payload is not None and payload_action is not None:
            raise ValueError("payload command and in-process action are mutually exclusive")
        started = self.clock()
        run_id = f"maintenance-{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
        failure: str | None = None
        manual_intervention = False
        self._snapshot()
        hashes = [self._hash_binary(path) for path in self.config.all_binary_paths()]
        try:
            self._emit(phase="drain", status="start")
            self._acquire()
            self.verify_drained()
            produce_method = (
                Path(str(payload[0])).name
                if payload
                else "in-process-action"
                if payload_action is not None
                else None
            )
            self._emit(phase="produce", status="start", method=produce_method)
            if payload:
                scope = payload_scope() if payload_scope is not None else nullcontext()
                with scope:
                    if not self._record(
                        "operator_payload",
                        "operator_payload",
                        requested=True,
                        command=list(payload),
                    ):
                        raise MaintenanceFailure("operator_payload:return_code")
            elif payload_action is not None:
                scope = payload_scope() if payload_scope is not None else nullcontext()
                with scope:
                    action_started = self.clock()
                    try:
                        payload_action()
                    except BaseException:
                        self._actions.append(
                            ActionReceipt(
                                action_id="operator_payload",
                                target="in_process_operator_payload",
                                requested=True,
                                executed=True,
                                success=False,
                                started_at=action_started,
                                finished_at=self.clock(),
                                evidence="callback_failed",
                            )
                        )
                        raise
                    self._actions.append(
                        ActionReceipt(
                            action_id="operator_payload",
                            target="in_process_operator_payload",
                            requested=True,
                            executed=True,
                            success=True,
                            started_at=action_started,
                            finished_at=self.clock(),
                            evidence="callback_completed",
                        )
                    )
            self._emit(phase="produce", status="complete")
        except ProcessGroupQuiescenceError as exc:
            failure = self._failure_label(exc)
            manual_intervention = True
        except BaseException as exc:
            failure = self._failure_label(exc)
        finally:
            if not manual_intervention:
                try:
                    self.restore()
                except ProcessGroupQuiescenceError as exc:
                    restore_failure = self._failure_label(exc)
                    failure = f"{failure};restore={restore_failure}" if failure else restore_failure
                    manual_intervention = True
                except BaseException as exc:
                    restore_failure = self._failure_label(exc)
                    failure = f"{failure};restore={restore_failure}" if failure else restore_failure

        action_success = all(action.success for action in self._actions)
        evidence_success = all(self._restoration_evidence.values())
        self._emit(
            phase="maintenance",
            status="complete",
            detail=f"success={failure is None and action_success and evidence_success}",
        )
        receipt = MaintenanceReceipt(
            run_id=run_id,
            dry_run=not self.execute,
            started_at=started,
            finished_at=self.clock(),
            previous_states=list(self._states.values()),
            binary_hashes=hashes,
            actions=list(self._actions),
            restoration_evidence=dict(self._restoration_evidence),
            manual_intervention_required=manual_intervention,
            success=failure is None and action_success and evidence_success,
            failure=failure,
        )
        _publish_json_exclusive(
            self.config.receipt_path,
            receipt.model_dump_json(indent=2).encode("utf-8"),
        )
        return receipt

    @staticmethod
    def _failure_label(exc: BaseException) -> str:
        # Never persist arbitrary exception text: command/library exceptions may
        # contain argv, environment fragments, or credentials.
        if isinstance(exc, MaintenanceInterrupted):
            return f"MaintenanceInterrupted:signal:{exc.signum}"
        return type(exc).__name__


SignalHandler = Callable[[int, FrameType | None], object] | int | None


def install_signal_traps() -> dict[int, SignalHandler]:
    """Install SIGINT/SIGTERM traps and return handlers for later restoration."""

    previous: dict[int, SignalHandler] = {}

    def interrupt(signum: int, _frame: object) -> None:
        raise MaintenanceInterrupted(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    return previous


def restore_signal_traps(previous: dict[int, SignalHandler]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def receipt_contains_secret_keys(receipt: MaintenanceReceipt) -> bool:
    """Defense-in-depth assertion for serialized audit output."""
    forbidden = ("token", "secret", "password", "credential", "environment", "env")
    payload = json.loads(receipt.model_dump_json())

    def visit(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                any(word in str(key).lower() for word in forbidden) or visit(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(visit(item) for item in value)
        return False

    return visit(payload)
