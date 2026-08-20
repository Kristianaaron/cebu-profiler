import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_maintenance_runner import ALL_PRODUCTION, FakeRunner, config  # noqa: E402

from model_atlas.ops.maintenance import MaintenanceCoordinator


def test_preflight_ok_when_all_ready(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    coord = MaintenanceCoordinator(cfg, FakeRunner(active=set(ALL_PRODUCTION)), execute=True)
    # artifact: the start script path exists; pass a real file + matching sha
    art = cfg.dsv4_start_script
    art.chmod(0o755)
    import hashlib
    sha = hashlib.sha256(art.read_bytes()).hexdigest()
    lease = tmp_path / "lease.json"
    lease.write_text("{}")
    res = coord.preflight(
        artifact=art,
        artifact_sha256=sha,
        lease=lease,
        requires=(cfg.dsv4_start_script,),  # only the one we chmod'd
        plan_compiles=lambda: True,
    )
    assert res["ok"] is True, res


def test_preflight_blocks_missing_lease(tmp_path: Path) -> None:
    """The exact silent-killer: missing lease must BLOCK (no drain)."""
    cfg = config(tmp_path)
    coord = MaintenanceCoordinator(cfg, FakeRunner(active=set(ALL_PRODUCTION)), execute=True)
    missing = tmp_path / "nope-lease.json"
    res = coord.preflight(artifact=cfg.dsv4_start_script, lease=missing)
    assert res["ok"] is False
    kinds = {b["kind"] for b in res["blockers"]}
    assert "lease" in kinds


def test_preflight_blocks_missing_backend(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    coord = MaintenanceCoordinator(cfg, FakeRunner(active=set(ALL_PRODUCTION)), execute=True)
    res = coord.preflight(
        artifact=cfg.dsv4_start_script,
        lease=tmp_path / "ok-lease.json",
        requires=(tmp_path / "does-not-exist.bin",),
    )
    lease = tmp_path / "ok-lease.json"
    lease.touch()
    res = coord.preflight(artifact=cfg.dsv4_start_script, lease=lease,
                          requires=(tmp_path / "does-not-exist.bin",))
    assert res["ok"] is False
    assert any(b["kind"] == "backend" for b in res["blockers"])


def test_preflight_emits_event(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    coord = MaintenanceCoordinator(cfg, FakeRunner(active=set(ALL_PRODUCTION)), execute=True)
    coord.preflight(artifact=cfg.dsv4_start_script, lease=tmp_path / "ev-lease.json")
    p = cfg.journal_dir / "maintenance-events.jsonl"
    import json
    if p.exists():
        evs = [json.loads(l) for l in p.read_text().splitlines()]
        assert any(e.get("phase") == "preflight" for e in evs)
