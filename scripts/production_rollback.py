#!/usr/bin/env python3
"""Production rollback helper for the DeepSeek vLLM service.

NOT a substitute for the operator's own stop/start procedure — this prints the
exact freeze/stop/restart commands for the maintenance window. It never
executes a stop itself. Use only when the operator has scheduled the window.
"""

from __future__ import annotations

import subprocess
import sys


def find_pids() -> list[tuple[str, str]]:
    """List running GPU processes (vLLM worker / llama-server) — read-only."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] nvidia-smi query failed: {exc}")
        return []
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2:
            rows.append((p[0], p[1]))
    return rows


def print_runbook() -> None:
    pids = find_pids()
    print("Active GPU processes (from nvidia-smi):")
    if not pids:
        print("  none")
    for pid, name in pids:
        print(f"  pid={pid} {name}")
    print(
        "\nMAINTENANCE WINDOW — operator MUST perform the following, never this script:\n"
    )
    for pid, name in pids:
        print(f"  # freeze {name} (pid {pid}) input, then SIGTERM (graceful):")
        print(f"  kill -TERM {pid}   # graceful; wait for clean drain")
        print("  # confirm gone:")
        freed = " || echo 'freed'"
        q = f"  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '{pid}'"
        print(f"{q} {freed.strip()}")
    print(
        "\nSet up the two-node experiment (section 4 of docs/glm52-runbook.md), run it,\n"
        "record measured frontier, then RESTART the production service with the\n"
        "operator's original DeepSeek vLLM launch command (2-rank TP).\n"
    )


def main() -> None:
    if "--restart" in sys.argv:
        print(
            "[ABORT] This helper never restarts production services itself. "
            "Use the operator's original DeepSeek vLLM launch command."
        )
        return
    print_runbook()


if __name__ == "__main__":
    main()
