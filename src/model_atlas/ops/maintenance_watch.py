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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PHASE_LABEL = {
    "drain": "Draining services",
    "produce": "Producing derivative",
    "restore": "Restoring / loading services",
    "maintenance": "Maintenance",
}


def _latest(records: Iterable[dict[str, Any]], phase: str, status: str) -> dict[str, Any] | None:
    found = None
    for rec in records:
        if rec.get("phase") == phase and rec.get("status") == status:
            found = rec
    return found


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
    current = next((p for p in order if any(r["phase"] == p for r in records)), "drain")

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


__all__ = ["PHASE_LABEL", "read_events", "render_maintenance_status"]
