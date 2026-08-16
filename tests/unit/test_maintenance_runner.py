from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from model_atlas.ops.maintenance import (
    CommandResult,
    MaintenanceConfig,
    MaintenanceCoordinator,
    MaintenanceInterrupted,
    receipt_contains_secret_keys,
)


class FakeRunner:
    def __init__(
        self,
        *,
        active: set[str] | None = None,
        fail_mutation: int | None = None,
        interrupt_payload: bool = False,
    ) -> None:
        self.active = set(active or ())
        self.fail_mutation = fail_mutation
        self.interrupt_payload = interrupt_payload
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
        head_runtime_unit_file=tmp_path / "atlas-glm52-runtime.service",
        worker_rpc_unit_file=Path("/tmp/atlas-glm52-rpc.service"),
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
    assert ids[:5] == [
        "stop_gateway",
        "stop_vision_adapter",
        "stop_qwen",
        "stop_dsv4",
        "operator_payload",
    ]
    assert ids[5:] == [
        "preserve_head_runtime_journal",
        "stop_head_runtime",
        "remove_head_runtime_unit",
        "reload_after_head_runtime_removal",
        "preserve_worker_rpc_journal",
        "stop_worker_rpc",
        "remove_worker_rpc_unit",
        "reload_after_worker_rpc_removal",
        "restore_dsv4",
        "restore_qwen",
        "restore_vision_adapter",
        "restore_gateway",
    ]
    assert receipt.success
    assert all(receipt.restoration_evidence.values())


@pytest.mark.parametrize("failure_number", range(1, 18))
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
