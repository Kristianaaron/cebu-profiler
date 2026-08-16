#!/usr/bin/env python3
"""Emit one strict local telemetry JSON object for a two-node canary sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_atlas.fit_telemetry import NodeRole, ProcessRole
from model_atlas.telemetry_probe import TelemetryProbeConfig, collect_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=NodeRole, required=True)
    parser.add_argument("--process-role", type=ProcessRole, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--sample-set-id", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--rdma-interface", required=True)
    parser.add_argument("--disk-device", required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--sys-class-net", type=Path, default=Path("/sys/class/net"))
    parser.add_argument("--diskstats-path", type=Path, default=Path("/proc/diskstats"))
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = collect_sample(
        TelemetryProbeConfig(
            node=args.node,
            process_role=args.process_role,
            pid=args.pid,
            sample_set_id=args.sample_set_id,
            phase_id=args.phase_id,
            context_tokens=args.context_tokens,
            rdma_interface=args.rdma_interface,
            disk_device=args.disk_device,
            proc_root=args.proc_root,
            sys_class_net=args.sys_class_net,
            diskstats_path=args.diskstats_path,
            gpu_index=args.gpu_index,
        )
    )
    print(sample.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
