import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_maintenance_runner import ALL_PRODUCTION, FakeRunner, config  # noqa: E402

from model_atlas.ops.maintenance import MaintenanceCoordinator


def _events(cfg) -> list[dict]:
    p = cfg.journal_dir / "maintenance-events.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_debug_event_surfaces_payload_scope_failure(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    runner = FakeRunner(active=set(ALL_PRODUCTION))

    class _Boom:
        def __enter__(self):
            raise RuntimeError("SIMULATED_PREPRODUCE_REASON_xyz")

        def __exit__(self, *a):
            return False

    coordinator = MaintenanceCoordinator(cfg, runner, execute=True)

    try:
        coordinator.run(["python", "operator-payload"], payload_scope=lambda: _Boom())
    except RuntimeError:
        pass  # expected: the simulated failure propagates

    evs = _events(cfg)
    debug = [e for e in evs if e.get("status") == "debug"]
    assert debug, f"expected a maintenance.debug event; tail={evs[-4:]}"
    detail = debug[-1].get("detail", "")
    assert "SIMULATED_PREPRODUCE_REASON_xyz" in detail, f"missing reason; got {detail}"
    assert "RuntimeError" in detail, f"missing type; got {detail}"
