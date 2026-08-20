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


def _shard_loaded(cfg):
    return [
        e for e in _events(cfg)
        if e["phase"] == "restore" and e["status"] == "shard_loaded"
    ]


def test_report_shard_progress_emits_shard_loaded(tmp_path: Path) -> None:
    """Feeding a vLLM-style loader line to report_shard_progress emits a
    shard_loaded event with the correct N/M (so the modal's per-shard bar can
    tick 23/48 -> ... )."""
    cfg = config(tmp_path)
    coordinator = MaintenanceCoordinator(cfg, FakeRunner(active=set(ALL_PRODUCTION)), execute=True)
    coordinator.report_shard_progress("Loading safetensors checkpoint shards: 23/48")
    loaded = _shard_loaded(cfg)
    assert loaded, "expected a shard_loaded event"
    assert loaded[-1]["shard_current"] == 23
    assert loaded[-1]["shard_total"] == 48
