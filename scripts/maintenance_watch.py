#!/usr/bin/env python3
"""Live visualization of the Atlas maintenance drain/run/restore lifecycle.

Tails ``<journal-dir>/maintenance-events.jsonl`` (written by the maintenance
coordinator) and renders the current phase — draining services, producing the
derivative (modelopt / width-slice), then restoring/loading DSV4 + sharding —
updating in place so an operator can watch a scheduled window run.

Examples:
    maintenance-watch --journal-dir /path/to/journals          # live tail
    maintenance-watch --journal-dir ... --interval 0.5         # poll faster
    # replay the full stream once and exit, useful headless/CI:
    maintenance-watch --journal-dir ... --once
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from model_atlas.ops.maintenance_watch import render_maintenance_status


def _offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _tail_lines(path: Path, offset: int) -> tuple[str, int]:
    if not path.exists():
        return "", 0
    size = path.stat().st_size
    if size < offset:
        offset = 0  # file was truncated/rotated
    with open(path, encoding="utf-8") as fh:
        fh.seek(offset)
        chunk = fh.read()
    return chunk, size


def poll(journal_dir: Path, *, once: bool, interval: float) -> int:
    path = journal_dir / "maintenance-events.jsonl"
    offset = _offset(path)
    while True:
        try:
            chunk, offset = _tail_lines(path, offset)
        except OSError:
            chunk = ""
        status = render_maintenance_status(chunk.splitlines())
        sys.stdout.write("\r\033[K" + status)
        sys.stdout.flush()
        if once:
            sys.stdout.write("\n")
            return 0
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="replay current stream once and exit")
    parser.add_argument("--interval", type=float, default=0.5, help="poll seconds (default 0.5)")
    args = parser.parse_args()
    if not args.journal_dir.is_dir():
        print(f"journal dir not found: {args.journal_dir}", file=sys.stderr)
        return 1
    return poll(args.journal_dir, once=args.once, interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
