#!/usr/bin/env python3
"""Internal payload: issue a lease only while MaintenanceCoordinator runs it."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from model_atlas.canary_lease import (
    CanaryLeaseBinding,
    remove_active_lease,
    write_active_lease,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--head-unit", required=True)
    parser.add_argument("--worker-unit", required=True)
    parser.add_argument("payload", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = args.payload[1:] if args.payload[:1] == ["--"] else args.payload
    if not payload:
        raise RuntimeError("bound canary payload is required")
    binding = CanaryLeaseBinding(
        plan_sha256=args.plan_sha256,
        artifact_path=args.artifact,
        artifact_sha256=args.artifact_sha256,
        head_unit=args.head_unit,
        worker_unit=args.worker_unit,
    )
    handle = write_active_lease(args.lease, binding)
    try:
        return subprocess.run(payload, check=False).returncode
    finally:
        remove_active_lease(handle)


if __name__ == "__main__":
    raise SystemExit(main())
