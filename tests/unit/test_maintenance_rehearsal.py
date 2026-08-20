import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_maintenance_runner import ALL_PRODUCTION, FakeRunner, config  # noqa: E402

from model_atlas.ops.maintenance import MaintenanceCoordinator


def _events(cfg):
    p = cfg.journal_dir / "maintenance-events.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_full_cycle_rehearsal_emits_ordered_lifecycle(tmp_path: Path) -> None:
    captured = []
    runner = FakeRunner(active=set(ALL_PRODUCTION))
    cfg = config(tmp_path)
    coordinator = MaintenanceCoordinator(cfg, runner, execute=True)
    coordinator.run(payload_action=lambda: captured.append("produce-ran"))

    evs = _events(cfg)
    order = [e["phase"] for e in evs]
    phases = [(e["phase"], e["status"]) for e in evs]
    assert "drain" in order and "produce" in order and "restore" in order
    assert ("maintenance", "complete") in phases
    assert ("drain", "complete") in phases
    assert order.index("produce") < order.index("restore")
    assert captured == ["produce-ran"]


def test_rehearsal_restore_brings_services_back(tmp_path: Path) -> None:
    runner = FakeRunner(active=set(ALL_PRODUCTION))
    cfg = config(tmp_path)
    coordinator = MaintenanceCoordinator(cfg, runner, execute=True)
    coord_result = coordinator.run(payload_action=lambda: None)
    assert ALL_PRODUCTION.issubset(runner.active), f"not all restored: {ALL_PRODUCTION - runner.active}"
    assert all(coord_result.restoration_evidence.values()), coord_result.restoration_evidence
