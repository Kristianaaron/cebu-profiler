#!/usr/bin/env python3
"""The sole real-execution entry point: drain, lease, canary, restore."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.ops.maintenance import (
    MaintenanceConfig,
    MaintenanceCoordinator,
    SubprocessCommandRunner,
    install_signal_traps,
    restore_signal_traps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("canary_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canary_args = args.canary_args[1:] if args.canary_args[:1] == ["--"] else args.canary_args
    canary_script = Path(__file__).with_name("run_two_node_canary.py")
    payload_script = Path(__file__).with_name("run_bound_canary_payload.py")
    payload = (
        sys.executable,
        str(payload_script),
        "--lease",
        str(args.lease),
        "--plan-sha256",
        args.plan_sha256,
        "--artifact",
        args.artifact,
        "--artifact-sha256",
        args.artifact_sha256,
        "--head-unit",
        HEAD_TRANSIENT_UNIT,
        "--worker-unit",
        WORKER_TRANSIENT_UNIT,
        "--",
        sys.executable,
        str(canary_script),
        "--execute",
        "--maintenance-lease",
        str(args.lease),
        "--artifact",
        args.artifact,
        "--artifact-sha256",
        args.artifact_sha256,
        *canary_args,
    )
    config = MaintenanceConfig(
        journal_dir=args.journal_dir,
        receipt_path=args.receipt,
        head_runtime_unit=HEAD_TRANSIENT_UNIT,
        worker_rpc_unit=WORKER_TRANSIENT_UNIT,
    )
    previous = install_signal_traps()
    try:
        receipt = MaintenanceCoordinator(
            config, SubprocessCommandRunner(), execute=args.execute
        ).run(payload if args.execute else None)
    finally:
        restore_signal_traps(previous)
    print(receipt.model_dump_json())
    return 0 if receipt.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
