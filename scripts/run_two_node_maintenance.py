#!/usr/bin/env python3
"""Dry-run-first entry point for a trap-safe two-node maintenance window."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_atlas.ops.maintenance import (
    MaintenanceConfig,
    MaintenanceCoordinator,
    SubprocessCommandRunner,
    install_signal_traps,
    receipt_contains_secret_keys,
    restore_signal_traps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true", help="perform mutations; default is dry-run"
    )
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--head-unit-file", type=Path)
    parser.add_argument("--worker-unit-file", type=Path)
    parser.add_argument("--binary", action="append", type=Path, default=[])
    parser.add_argument("payload", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = args.payload
    if payload and payload[0] == "--":
        payload = payload[1:]
    config = MaintenanceConfig(
        journal_dir=args.journal_dir,
        receipt_path=args.receipt,
        head_runtime_unit_file=args.head_unit_file,
        worker_rpc_unit_file=args.worker_unit_file,
        binary_paths=tuple(args.binary),
    )
    previous = install_signal_traps()
    try:
        receipt = MaintenanceCoordinator(
            config,
            SubprocessCommandRunner(),
            execute=args.execute,
        ).run(payload or None)
    finally:
        restore_signal_traps(previous)
    if receipt_contains_secret_keys(receipt):
        raise RuntimeError("receipt schema unexpectedly contains secret-bearing keys")
    print(receipt.model_dump_json(indent=2))
    return 0 if receipt.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
