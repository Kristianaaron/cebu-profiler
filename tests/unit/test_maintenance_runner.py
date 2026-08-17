from __future__ import annotations

import signal
import subprocess
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.ops.maintenance import (
    CommandResult,
    MaintenanceConfig,
    MaintenanceCoordinator,
    MaintenanceFailure,
    MaintenanceInterrupted,
    ProcessGroupQuiescenceError,
    SubprocessCommandRunner,
    receipt_contains_secret_keys,
)


class FakeRunner:
    def __init__(
        self,
        *,
        active: set[str] | None = None,
        fail_mutation: int | None = None,
        interrupt_payload: bool = False,
        interrupt_command: tuple[str, ...] | None = None,
    ) -> None:
        self.active = set(active or ())
        self.fail_mutation = fail_mutation
        self.interrupt_payload = interrupt_payload
        self.interrupt_command = interrupt_command
        self.calls: list[tuple[str, ...]] = []
        self.mutations: list[tuple[str, ...]] = []

    @staticmethod
    def _inner(argv: tuple[str, ...]) -> tuple[str, ...]:
        return argv[2:] if argv[:1] == ("ssh",) else argv

    def run(self, argv: Sequence[str]) -> CommandResult:
        command = tuple(argv)
        self.calls.append(command)
        inner = self._inner(command)
        if inner[:4] == ("systemctl", "--user", "is-active", "--quiet"):
            return CommandResult(returncode=0 if inner[4] in self.active else 3)
        if inner[:3] == ("docker", "inspect", "--format"):
            return CommandResult(
                returncode=0,
                stdout="true\n" if inner[-1] in self.active else "false\n",
            )
        if inner[:1] == ("journalctl",):
            self.mutations.append(command)
            return self._mutation_result()
        if command == ("operator-command",) and self.interrupt_payload:
            self.mutations.append(command)
            raise MaintenanceInterrupted(15)
        if command == self.interrupt_command:
            self.mutations.append(command)
            raise MaintenanceInterrupted(2)

        self.mutations.append(command)
        result = self._mutation_result()
        if result.returncode != 0:
            return result
        if inner[:3] == ("systemctl", "--user", "stop"):
            self.active.discard(inner[3])
        elif inner[:3] == ("systemctl", "--user", "start"):
            self.active.add(inner[3])
        elif str(inner[0]).endswith("stop-deepseek-v4-flash-dspark.sh"):
            self.active.discard("deepseek-v4-flash-vllm-dspark-1")
        elif str(inner[0]).endswith("start-deepseek-v4-flash-dspark.sh"):
            self.active.add("deepseek-v4-flash-vllm-dspark-1")
        return result

    def _mutation_result(self) -> CommandResult:
        number = len(self.mutations)
        if self.fail_mutation == number:
            return CommandResult(returncode=41)
        return CommandResult(returncode=0, stdout="safe journal\n")


def config(tmp_path: Path) -> MaintenanceConfig:
    stop = tmp_path / "stop-deepseek-v4-flash-dspark.sh"
    start = tmp_path / "start-deepseek-v4-flash-dspark.sh"
    binary = tmp_path / "runtime-bin"
    stop.write_text("stop", encoding="utf-8")
    start.write_text("start", encoding="utf-8")
    binary.write_text("runtime", encoding="utf-8")
    return MaintenanceConfig(
        dsv4_stop_script=stop,
        dsv4_start_script=start,
        binary_paths=(binary,),
        head_runtime_unit_file=tmp_path / HEAD_TRANSIENT_UNIT,
        worker_rpc_unit_file=Path("/tmp") / WORKER_TRANSIENT_UNIT,
        journal_dir=tmp_path / "journals",
        receipt_path=tmp_path / "receipt.json",
    )


ALL_PRODUCTION = {
    "hermes-gateway.service",
    "dsv4-vision-adapter.service",
    "qwen35-vision.service",
    "deepseek-v4-flash-vllm-dspark-1",
}


def action_ids(coordinator_receipt: object) -> list[str]:
    return [item.action_id for item in coordinator_receipt.actions]  # type: ignore[attr-defined]


def test_exact_stop_and_rollback_order(tmp_path: Path) -> None:
    runner = FakeRunner(active=ALL_PRODUCTION)
    receipt = MaintenanceCoordinator(config(tmp_path), runner, execute=True).run(
        ["operator-command"]
    )

    ids = action_ids(receipt)
    assert ids[:9] == [
        "stop_gateway",
        "stop_vision_adapter",
        "stop_qwen",
        "stop_dsv4",
        "preserve_head_runtime_journal",
        "stop_head_runtime",
        "preserve_worker_rpc_journal",
        "stop_worker_rpc",
        "operator_payload",
    ]
    assert ids[9:] == [
        "preserve_head_runtime_journal",
        "stop_head_runtime",
        "preserve_worker_rpc_journal",
        "stop_worker_rpc",
        "restore_worker_rpc",
        "restore_head_runtime",
        "restore_dsv4",
        "restore_qwen",
        "restore_vision_adapter",
        "restore_gateway",
    ]
    assert receipt.success
    assert all(receipt.restoration_evidence.values())


@pytest.mark.parametrize("failure_number", range(1, 14))
def test_partial_failure_at_every_mutating_transition_restores(
    tmp_path: Path, failure_number: int
) -> None:
    runner = FakeRunner(active=ALL_PRODUCTION, fail_mutation=failure_number)
    receipt = MaintenanceCoordinator(config(tmp_path), runner, execute=True).run(
        ["operator-command"]
    )

    assert not receipt.success
    ids = action_ids(receipt)
    assert ids.index("preserve_head_runtime_journal") < ids.index("stop_head_runtime")
    assert ids.index("stop_head_runtime") < ids.index("preserve_worker_rpc_journal")
    assert ids.index("restore_dsv4") < ids.index("restore_qwen")
    assert ids.index("restore_qwen") < ids.index("restore_vision_adapter")
    assert ids.index("restore_vision_adapter") < ids.index("restore_gateway")


def test_signal_like_exception_runs_full_restore(tmp_path: Path) -> None:
    runner = FakeRunner(active=ALL_PRODUCTION, interrupt_payload=True)
    receipt = MaintenanceCoordinator(config(tmp_path), runner, execute=True).run(
        ["operator-command"]
    )

    assert receipt.failure == "MaintenanceInterrupted:signal:15"
    assert "restore_gateway" in action_ids(receipt)
    assert runner.active >= ALL_PRODUCTION


def test_restore_starts_only_previously_active_production_services(tmp_path: Path) -> None:
    runner = FakeRunner(active={"qwen35-vision.service"})
    receipt = MaintenanceCoordinator(config(tmp_path), runner, execute=True).run()
    actions = {item.action_id: item for item in receipt.actions}

    assert actions["restore_qwen"].executed
    assert not actions["restore_dsv4"].executed
    assert not actions["restore_vision_adapter"].executed
    assert not actions["restore_gateway"].executed


def test_restore_recreates_exact_preexisting_runtime_state(tmp_path: Path) -> None:
    active = ALL_PRODUCTION | {
        HEAD_TRANSIENT_UNIT,
        WORKER_TRANSIENT_UNIT,
    }
    runner = FakeRunner(active=active)
    receipt = MaintenanceCoordinator(config(tmp_path), runner, execute=True).run()
    actions = {item.action_id: item for item in receipt.actions}

    assert actions["restore_worker_rpc"].executed
    assert actions["restore_head_runtime"].executed
    assert runner.active >= active
    assert receipt.restoration_evidence["worker_rpc"]
    assert receipt.restoration_evidence["head_runtime"]


def test_second_signal_during_restore_does_not_abort_later_actions(tmp_path: Path) -> None:
    runner = FakeRunner(
        active=ALL_PRODUCTION,
        interrupt_payload=True,
        interrupt_command=(
            "ssh",
            "10.77.0.2",
            "systemctl",
            "--user",
            "stop",
            WORKER_TRANSIENT_UNIT,
        ),
    )
    receipt = MaintenanceCoordinator(config(tmp_path), runner, execute=True).run(
        ["operator-command"]
    )

    ids = action_ids(receipt)
    assert "stop_worker_rpc" in ids
    assert "restore_dsv4" in ids
    assert "restore_gateway" in ids
    assert not receipt.success


def test_restore_is_idempotent(tmp_path: Path) -> None:
    runner = FakeRunner(active=ALL_PRODUCTION)
    coordinator = MaintenanceCoordinator(config(tmp_path), runner, execute=True)
    coordinator.run()
    mutation_count = len(runner.mutations)
    coordinator.restore()
    assert len(runner.mutations) == mutation_count


def test_dry_run_has_no_mutating_commands(tmp_path: Path) -> None:
    runner = FakeRunner(active=ALL_PRODUCTION)
    receipt = MaintenanceCoordinator(config(tmp_path), runner).run(["operator-command"])

    assert receipt.dry_run
    assert not runner.mutations
    assert all(not action.executed for action in receipt.actions)


def test_receipt_excludes_output_and_exception_secrets(tmp_path: Path) -> None:
    class SecretExceptionRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            command = tuple(argv)
            if command == ("operator-command",):
                raise RuntimeError("TOKEN=do-not-persist PASSWORD=hunter2")
            return super().run(argv)

    receipt = MaintenanceCoordinator(
        config(tmp_path), SecretExceptionRunner(active=ALL_PRODUCTION), execute=True
    ).run(["operator-command"])
    serialized = receipt.model_dump_json().lower()
    assert "do-not-persist" not in serialized
    assert "hunter2" not in serialized
    assert not receipt_contains_secret_keys(receipt)


def test_journal_is_written_before_optional_unit_removal(tmp_path: Path) -> None:
    runner = FakeRunner(active=ALL_PRODUCTION)
    cfg = config(tmp_path)
    MaintenanceCoordinator(cfg, runner, execute=True).run()

    assert (cfg.journal_dir / "head_runtime.journal.log").read_text() == "safe journal\n"
    assert (cfg.journal_dir / "worker_rpc.journal.log").read_text() == "safe journal\n"


def test_unit_is_not_removed_when_journal_cannot_be_persisted(tmp_path: Path) -> None:
    runner = FakeRunner(active=ALL_PRODUCTION)
    cfg = config(tmp_path).model_copy(update={"journal_dir": tmp_path / "blocked"})
    cfg.journal_dir.write_text("not a directory", encoding="utf-8")
    receipt = MaintenanceCoordinator(cfg, runner, execute=True).run()

    ids = action_ids(receipt)
    assert "remove_head_runtime_unit" not in ids
    assert "remove_worker_rpc_unit" not in ids
    assert not receipt.success


def test_payload_scope_is_never_entered_when_acquisition_fails(tmp_path: Path) -> None:
    entered: list[str] = []

    @contextmanager
    def scope():  # type: ignore[no-untyped-def]
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    runner = FakeRunner(active=ALL_PRODUCTION, fail_mutation=1)
    MaintenanceCoordinator(config(tmp_path), runner, execute=True).run(
        ["operator-command"], payload_scope=scope
    )
    assert entered == []


def test_stop_success_but_autorestart_blocks_scope_and_payload(tmp_path: Path) -> None:
    entered: list[str] = []

    @contextmanager
    def scope():  # type: ignore[no-untyped-def]
        entered.append("entered")
        yield

    class AutoRestartRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            result = super().run(argv)
            command = tuple(argv)
            if command == ("systemctl", "--user", "stop", "hermes-gateway.service"):
                self.active.add("hermes-gateway.service")
            return result

    runner = AutoRestartRunner(active=ALL_PRODUCTION)
    receipt = MaintenanceCoordinator(config(tmp_path), runner, execute=True).run(
        ["operator-command"], payload_scope=scope
    )
    assert entered == []
    assert "operator_payload" not in action_ids(receipt)
    assert receipt.failure == "MaintenanceFailure"


def test_payload_scope_wraps_only_payload_and_exits_before_restore(tmp_path: Path) -> None:
    events: list[str] = []

    @contextmanager
    def scope():  # type: ignore[no-untyped-def]
        events.append("lease-enter")
        try:
            yield
        finally:
            events.append("lease-exit")

    class OrderedRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            if tuple(argv) == ("operator-command",):
                events.append("payload")
            if str(tuple(argv)[0]).endswith("start-deepseek-v4-flash-dspark.sh"):
                events.append("restore-begins")
            return super().run(argv)

    MaintenanceCoordinator(
        config(tmp_path), OrderedRunner(active=ALL_PRODUCTION), execute=True
    ).run(["operator-command"], payload_scope=scope)
    assert events.index("lease-enter") < events.index("payload") < events.index("lease-exit")
    assert events.index("lease-exit") < events.index("restore-begins")


def test_subprocess_runner_terminates_and_reaps_group_before_interrupt_propagates() -> None:
    events: list[str] = []

    class Child:
        pid = 444
        waits = 0

        def communicate(self) -> tuple[str, str | None]:
            events.append("communicate")
            raise MaintenanceInterrupted(15)

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            events.append(f"wait-{self.waits}")
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["payload"], timeout)
            return 0

    def factory(*_args: object, **kwargs: object) -> Child:
        assert kwargs["start_new_session"] is True
        return Child()

    alive = True

    def killpg(pid: int, sig: int) -> None:
        nonlocal alive
        events.append(f"kill-{pid}-{sig.name if isinstance(sig, signal.Signals) else sig}")
        if sig == signal.SIGKILL:
            alive = False
        if sig == 0 and not alive:
            raise ProcessLookupError

    runner = SubprocessCommandRunner(process_factory=factory, killpg=killpg, wait_timeout_seconds=1)
    with pytest.raises(MaintenanceInterrupted):
        runner.run(["payload"])
    assert events == [
        "communicate",
        "kill-444-SIGTERM",
        "wait-1",
        "kill-444-0",
        "kill-444-SIGKILL",
        "wait-2",
        "kill-444-0",
    ]


def test_subprocess_runner_waits_leader_even_when_group_is_already_gone() -> None:
    events: list[str] = []

    class Child:
        pid = 777

        def communicate(self) -> tuple[str, str | None]:
            raise MaintenanceInterrupted(2)

        def wait(self, timeout: float | None = None) -> int:
            events.append("wait")
            return 0

    def gone(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    runner = SubprocessCommandRunner(
        process_factory=lambda *_args, **_kwargs: Child(), killpg=gone, wait_timeout_seconds=1
    )
    with pytest.raises(MaintenanceInterrupted):
        runner.run(["payload"])
    assert events == ["wait"]


@pytest.mark.parametrize(
    ("kind", "returncode", "stdout"),
    [
        ("ssh", 255, ""),
        ("dbus", 1, ""),
        ("docker_error", 1, ""),
        ("docker_malformed", 0, "maybe\n"),
    ],
)
def test_unknown_liveness_fails_closed_before_scope(
    tmp_path: Path, kind: str, returncode: int, stdout: str
) -> None:
    entered: list[str] = []

    @contextmanager
    def scope():  # type: ignore[no-untyped-def]
        entered.append("entered")
        yield

    class UnknownRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            command = tuple(argv)
            inner = self._inner(command)
            if kind == "ssh" and command[:1] == ("ssh",):
                return CommandResult(returncode=returncode, stdout=stdout)
            if kind == "dbus" and inner[:4] == (
                "systemctl",
                "--user",
                "is-active",
                "--quiet",
            ):
                return CommandResult(returncode=returncode, stdout=stdout)
            if kind.startswith("docker") and inner[:3] == ("docker", "inspect", "--format"):
                return CommandResult(returncode=returncode, stdout=stdout)
            return super().run(argv)

    with pytest.raises(MaintenanceFailure, match="liveness"):
        MaintenanceCoordinator(config(tmp_path), UnknownRunner(), execute=True).run(
            ["operator-command"], payload_scope=scope
        )
    assert entered == []


def test_persistent_payload_group_skips_restoration_and_marks_manual_intervention(
    tmp_path: Path,
) -> None:
    class PersistentRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            if tuple(argv) == ("operator-command",):
                raise ProcessGroupQuiescenceError("still alive")
            return super().run(argv)

    receipt = MaintenanceCoordinator(
        config(tmp_path), PersistentRunner(active=ALL_PRODUCTION), execute=True
    ).run(["operator-command"])
    assert receipt.manual_intervention_required
    assert receipt.failure == "ProcessGroupQuiescenceError"
    assert not any(action.action_id.startswith("restore_") for action in receipt.actions)


def test_payload_worker_that_survives_stop_blocks_production_restore(tmp_path: Path) -> None:
    class StubbornWorkerRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            command = tuple(argv)
            inner = self._inner(command)
            if command == ("operator-command",):
                self.active.add(WORKER_TRANSIENT_UNIT)
            if inner == ("systemctl", "--user", "stop", WORKER_TRANSIENT_UNIT):
                self.calls.append(command)
                self.mutations.append(command)
                return CommandResult(returncode=0)
            return super().run(argv)

    receipt = MaintenanceCoordinator(
        config(tmp_path), StubbornWorkerRunner(active=ALL_PRODUCTION), execute=True
    ).run(["operator-command"])
    assert receipt.manual_intervention_required
    assert "ProcessGroupQuiescenceError" in (receipt.failure or "")
    assert not any(action.action_id.startswith("restore_") for action in receipt.actions)


def test_second_signal_during_reap_is_suppressed_until_group_is_gone_then_restore_runs(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Child:
        pid = 818
        waits = 0

        def communicate(self) -> tuple[str, str | None]:
            raise MaintenanceInterrupted(15)

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            events.append(f"wait-{self.waits}")
            if self.waits == 1:
                raise MaintenanceInterrupted(2)
            return 0

    term_calls = 0

    def killpg(_pid: int, sig: int) -> None:
        nonlocal term_calls
        events.append(f"kill-{sig}")
        if sig == signal.SIGTERM:
            term_calls += 1
            if term_calls == 1:
                raise MaintenanceInterrupted(2)
        if sig == 0:
            raise ProcessLookupError

    payload_runner = SubprocessCommandRunner(
        process_factory=lambda *_args, **_kwargs: Child(), killpg=killpg, wait_timeout_seconds=1
    )

    class IntegratedRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            if tuple(argv) == ("operator-command",):
                return payload_runner.run(argv)
            if str(tuple(argv)[0]).endswith("start-deepseek-v4-flash-dspark.sh"):
                events.append("restore")
            return super().run(argv)

    receipt = MaintenanceCoordinator(
        config(tmp_path), IntegratedRunner(active=ALL_PRODUCTION), execute=True
    ).run(["operator-command"])
    assert receipt.failure == "MaintenanceInterrupted:signal:15"
    assert events[:4] == ["kill-15", "kill-15", "wait-1", "wait-2"]
    assert events.index("kill-0") < events.index("restore")


@pytest.mark.parametrize("failure_site", ["term_permission", "group_probe", "leader_wait"])
def test_uncertain_payload_cleanup_requires_manual_intervention_without_restore(
    tmp_path: Path, failure_site: str
) -> None:
    class Child:
        pid = 909

        def communicate(self) -> tuple[str, str | None]:
            raise MaintenanceInterrupted(15)

        def wait(self, timeout: float | None = None) -> int:
            if failure_site == "leader_wait":
                raise OSError("wait failed")
            return 0

    def killpg(_pid: int, sig: int) -> None:
        if failure_site == "term_permission" and sig == signal.SIGTERM:
            raise PermissionError("denied")
        if failure_site == "group_probe" and sig == 0:
            raise OSError("probe failed")
        if sig == 0:
            raise ProcessLookupError

    payload_runner = SubprocessCommandRunner(
        process_factory=lambda *_args, **_kwargs: Child(), killpg=killpg, wait_timeout_seconds=1
    )

    class IntegratedRunner(FakeRunner):
        def run(self, argv: Sequence[str]) -> CommandResult:
            if tuple(argv) == ("operator-command",):
                return payload_runner.run(argv)
            return super().run(argv)

    receipt = MaintenanceCoordinator(
        config(tmp_path), IntegratedRunner(active=ALL_PRODUCTION), execute=True
    ).run(["operator-command"])
    assert receipt.manual_intervention_required
    assert receipt.failure == "ProcessGroupQuiescenceError"
    assert not any(action.action_id.startswith("restore_") for action in receipt.actions)
