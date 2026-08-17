import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_maintenance_runner import ALL_PRODUCTION, FakeRunner, config  # noqa: E402

from model_atlas.ops.maintenance import MaintenanceCoordinator


def _events(cfg) -> list[dict[str, object]]:
    path: Path = cfg.journal_dir / "maintenance-events.jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _phases(events: list[dict[str, object]]) -> list[str]:
    return [f"{e['phase']}.{e['status']}" for e in events]


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(n == h for h in it) for n in needle)


def test_emits_full_lifecycle_in_order(tmp_path: Path) -> None:
    runner = FakeRunner(active=set(ALL_PRODUCTION))
    cfg = config(tmp_path)
    MaintenanceCoordinator(cfg, runner, execute=True).run(["operator-command"])

    events = _events(cfg)
    assert events, "expected a maintenance-events.jsonl stream"
    phases = _phases(events)

    expected = [
        "drain.start",
        "drain.complete",
        "produce.start",
        "produce.complete",
        "restore.start",
        "restore.complete",
        "maintenance.complete",
    ]
    assert _is_subsequence(expected, phases), f"missing ordering: {phases}"

    # the produce event names the payload program
    produce = [e for e in events if e["phase"] == "produce" and e["status"] == "start"]
    assert produce and produce[0].get("method") == "operator-command"


def test_drain_reports_dsv4_release(tmp_path: Path) -> None:
    runner = FakeRunner(active=set(ALL_PRODUCTION))
    cfg = config(tmp_path)
    MaintenanceCoordinator(cfg, runner, execute=True).run()
    events = _events(cfg)
    releases = [
        e.get("service")
        for e in events
        if e["phase"] == "drain" and e["status"] == "release"
    ]
    assert "dsv4" in releases


def test_restore_reports_dsv4_load(tmp_path: Path) -> None:
    runner = FakeRunner(active=set(ALL_PRODUCTION))
    cfg = config(tmp_path)
    MaintenanceCoordinator(cfg, runner, execute=True).run()
    events = _events(cfg)
    restores = [
        e.get("service")
        for e in events
        if e["phase"] == "restore" and e["status"] == "load"
    ]
    assert "dsv4" in restores


def test_events_emitted_even_in_dry_run(tmp_path: Path) -> None:
    runner = FakeRunner(active=set(ALL_PRODUCTION))
    cfg = config(tmp_path)
    MaintenanceCoordinator(cfg, runner).run(["operator-command"])  # dry-run
    assert _events(cfg)
