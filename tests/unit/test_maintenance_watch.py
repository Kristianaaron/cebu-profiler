import json
from pathlib import Path

from model_atlas.ops.maintenance_watch import (
    extract_shard_progress,
    read_events,
    render_maintenance_status,
    shard_bar,
)


def _line(phase: str, status: str, **extra: object) -> str:
    ev = {"ts": "2026-01-01T00:00:00.000000Z", "phase": phase, "status": status, **extra}
    return json.dumps(ev, sort_keys=True)


def test_empty_stream() -> None:
    assert render_maintenance_status([]) == "no maintenance events yet"


def test_drain_phase_lists_released() -> None:
    lines = [
        _line("drain", "start"),
        _line("drain", "release", service="gateway"),
        _line("drain", "release", service="dsv4"),
    ]
    status = render_maintenance_status(lines)
    assert "Draining services" in status
    assert "dsv4" in status and "gateway" in status


def test_produce_phase_shows_method_and_complete() -> None:
    running = [_line("produce", "start", method="run_glm52_compression_maintenance.py")]
    assert "Producing derivative" in render_maintenance_status(running)
    done = running + [_line("produce", "complete")]
    assert "complete" in render_maintenance_status(done)


def test_restore_phase_lists_loaded_services() -> None:
    lines = [
        _line("restore", "start"),
        _line("restore", "load", service="dsv4"),
        _line("restore", "load", service="qwen"),
    ]
    status = render_maintenance_status(lines)
    assert "Restoring / loading services" in status
    assert "dsv4" in status and "qwen" in status


def test_maintenance_complete_shows_result() -> None:
    lines = [_line("maintenance", "complete", detail="success=True")]
    status = render_maintenance_status(lines)
    assert "Maintenance" in status
    assert "success=True" in status


def test_read_events_parses_and_skips_garbage(tmp_path: Path) -> None:
    path = tmp_path / "maintenance-events.jsonl"
    body = (
        "{not json}\n"
        + _line("drain", "start")
        + "\n"
        + _line("drain", "release", service="dsv4")
        + "\n"
    )
    path.write_text(body, encoding="utf-8")
    events = read_events(path)
    assert len(events) == 2
    assert [e["status"] for e in events] == ["start", "release"]


def test_read_events_missing_file(tmp_path: Path) -> None:
    assert read_events(tmp_path / "nope.jsonl") == []


def test_shard_bar_renders_current_total() -> None:
    bar = shard_bar(12, 24)
    assert "12/24 shards" in bar
    assert "[" in bar and "]" in bar


def test_shard_bar_full() -> None:
    assert shard_bar(24, 24) == "[########################] 24/24 shards"


def test_extract_shard_progress_vllm_line() -> None:
    line = "Loading safetensors checkpoint shards: 40%|████        | 23/57 [00:12<00:00, 4.62it/s]"
    assert extract_shard_progress(line) == (23, 57)


def test_extract_shard_progress_absent() -> None:
    assert extract_shard_progress("model weights loaded") is None


def test_render_restore_shows_shard_bar() -> None:
    lines = [
        _line("restore", "start"),
        _line("restore", "load", service="dsv4"),
        _line("restore", "shard_loaded", shard_current=23, shard_total=57),
    ]
    status = render_maintenance_status(lines)
    assert "Restoring / loading services" in status
    assert "23/57 shards" in status
    assert "[" in status and "]" in status
