import json
from pathlib import Path

from model_atlas.dashboard import (
    _maintenance_payload,
    build_dashboard_data,
    render_dashboard,
)


def _write_events(journal: Path) -> str:
    journal.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ts": "t", "phase": "drain", "status": "release", "service": "dsv4"},
        {"ts": "t", "phase": "drain", "status": "release", "service": "qwen"},
        {
            "ts": "t",
            "phase": "produce",
            "status": "start",
            "method": "run_glm52_compression_maintenance.py",
        },
        {"ts": "t", "phase": "restore", "status": "load", "service": "dsv4"},
        {
            "ts": "t",
            "phase": "restore",
            "status": "shard_loaded",
            "service": "dsv4",
            "shard_current": 23,
            "shard_total": 57,
        },
    ]
    path = journal / "maintenance-events.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return str(path)


def test_maintenance_payload_reads_event_stream(tmp_path: Path, monkeypatch) -> None:
    journal = tmp_path / "j"
    _write_events(journal)
    monkeypatch.setenv("ATLAS_MAINTENANCE_JOURNAL_DIR", str(journal))
    payload = _maintenance_payload()
    assert payload["present"] is True
    assert payload["phase"] == "restore"
    assert payload["released"] == ["dsv4", "qwen"]
    assert payload["shard_current"] == 23
    assert payload["shard_total"] == 57
    assert "23/57 shards" in payload["status"]
    assert isinstance(payload["elapsed_seconds"], int)
    assert isinstance(payload["eta_remaining_seconds"], int)
    assert payload["estimated_total_seconds"] > 0


def test_maintenance_absent_without_events(tmp_path: Path, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("ATLAS_MAINTENANCE_JOURNAL_DIR", str(empty))
    payload = _maintenance_payload()
    assert payload["present"] is False


def test_dashboard_renders_maintenance_surface() -> None:
    html = render_dashboard(build_dashboard_data())
    assert 'data-tab="maintenance"' in html
    assert "panel-maintenance" in html
    assert '"maintenance"' in html
    # time affordances rendered into the modal script
    assert "mt-el" in html and "mt-left" in html
    assert "remaining" in html and "expected:" in html
