"""Render the maintenance lifecycle event stream as a live human/Ux status.

The coordinator appends typed events to ``maintenance-events.jsonl``:
``drain.start / drain.release.<svc> / drain.complete``,
``produce.start(.<method>) / produce.complete``,
``restore.start / restore.load.<svc> / restore.complete``,
``maintenance.complete``. This module turns any tail of that stream into a
compact, current-phase status line so a UI or ``maintenance-watch`` can show the
user exactly what the pipeline is doing right now (draining, producing, or
restoring/loading).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PHASE_LABEL = {
    "drain": "Draining services",
    "produce": "Producing derivative",
    "restore": "Restoring / loading services",
    "maintenance": "Maintenance",
}

_BAR_WIDTH = 24


def _int_values(records: Iterable[dict[str, Any]], key: str) -> Iterable[int]:
    return (int(r[key]) for r in records if isinstance(r.get(key), int))


def _latest(records: Iterable[dict[str, Any]], phase: str, status: str) -> dict[str, Any] | None:
    found = None
    for rec in records:
        if rec.get("phase") == phase and rec.get("status") == status:
            found = rec
    return found


def shard_bar(current: int, total: int) -> str:
    """Render a terminal-style progress bar like ``[##########........] 12/24``."""
    if total <= 0 or current < 0:
        return f"0/{total} shards"
    filled = _BAR_WIDTH if current >= total else (current * _BAR_WIDTH) // total
    bar = "#" * filled + "." * (_BAR_WIDTH - filled)
    return f"[{bar}] {current}/{total} shards"


_SHARD_RE = re.compile(
    r"(?i)(shard|checkpoint|safetensors).{0,40}?(\d+)\s*(?:/|of)\s*(\d+)"
)


def extract_shard_progress(text: str) -> tuple[int, int] | None:
    """Pull the latest ``current/total`` shard progress from vLLM-style output.

    Mirrors the terminal loading line a runtime prints per weight shard, e.g.
    ``Loading safetensors checkpoint shards: 40%|████        | 23/57`` →
    ``(23, 57)``. Returns ``None`` when no shard progress is present.
    """
    latest: tuple[int, int] | None = None
    for match in _SHARD_RE.finditer(text):
        current = int(match.group(2))
        total = int(match.group(3))
        if total > 0 and 0 <= current <= total:
            latest = (current, total)
    return latest


def render_maintenance_status(raw: Iterable[str]) -> str:
    """Render a status line from raw JSONL lines. Returns 'no maintenance
    events yet' when the stream is empty."""
    records: list[dict[str, Any]] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    if not records:
        return "no maintenance events yet"

    # Identify the furthest-progressed phase in stream order.
    order = ("drain", "produce", "restore", "maintenance")
    current = next(
        (p for p in reversed(order) if any(r["phase"] == p for r in records)),
        "drain",
    )

    released = sorted(
        {r["service"] for r in records if r["phase"] == "drain" and r["status"] == "release"}
    )
    loaded = sorted(
        {r["service"] for r in records if r["phase"] == "restore" and r["status"] == "load"}
    )
    produce = _latest(records, "produce", "start")
    done = _latest(records, "maintenance", "complete")

    parts = [PHASE_LABEL[current]]
    if current == "drain":
        parts.append(f"released: {', '.join(released) or '(none active)'}")
    elif current == "produce":
        suffix = "complete" if _latest(records, "produce", "complete") else "running…"
        parts.append(f"{(produce or {}).get('method', '?')} {suffix}")
    elif current == "restore":
        parts.append(f"loaded: {', '.join(loaded) or '(working…)'}")
        shard_events = [
            r
            for r in records
            if r["phase"] == "restore" and r["status"] == "shard_loaded"
        ]
        if shard_events:
            total = max(_int_values(shard_events, "shard_total"), default=0)
            cur = max(_int_values(shard_events, "shard_current"), default=0)
            if total > 0:
                parts.append(shard_bar(cur, total))
    elif done is not None:
        detail = str(done.get("detail", ""))
        parts.append(f"result: {detail}")
    return " | ".join(parts)


def read_events(path: Path) -> list[dict[str, Any]]:
    """Parse a maintenance-events.jsonl file into a list of event dicts."""
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            rec = _try_json(ln)
            if rec is not None:
                events.append(rec)
    return events


def _try_json(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


__all__ = [
    "PHASE_LABEL",
    "extract_shard_progress",
    "read_events",
    "render_maintenance_status",
    "shard_bar",
]
