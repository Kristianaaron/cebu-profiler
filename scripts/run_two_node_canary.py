#!/usr/bin/env python3
"""Execute the two-Spark canary only after an approved maintenance window.

Without ``--execute`` this prints the pinned plan and argv only; it never
hashes artifacts, opens HTTP, SSHes, or starts/stops a service.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from model_atlas.fit_telemetry import (
    CandidateBinding,
    TwoNodeTelemetryCollector,
    build_base_canary_plan,
)
from model_atlas.llamacpp_rpc_runtime import LlamaCppRpcRuntimeAdapter, LlamaCppRpcRuntimeConfig
from model_atlas.runtime_canary_driver import (
    CanaryRequestClient,
    LoopbackHttpTransport,
    SystemdUserRuntimeLifecycle,
)
from model_atlas.two_node_canary_executor import (
    JsonlEvidenceStore,
    SshWorkerHashProbe,
    SubprocessTelemetryRunner,
    TwoNodeCanaryExecutor,
)


class _ArgvRunner:
    def __call__(self, argv: Sequence[str]) -> str:
        result = subprocess.run(list(argv), check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("external command failed")
        return result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true", help="perform the real canary; default is dry-run"
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument(
        "--telemetry-probe",
        type=Path,
        default=Path(__file__).with_name("collect_two_node_telemetry.py"),
    )
    parser.add_argument("--rdma-interface")
    parser.add_argument("--disk-device")
    parser.add_argument("--worker-ssh-target", default="10.77.0.2")
    parser.add_argument("--worker-host", default="169.254.200.197")
    parser.add_argument(
        "--toolchain-root",
        type=Path,
        default=Path("/home/kristianaaron/tmp/atlas-toolchains/llama.cpp"),
    )
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=Path(
            "/home/kristianaaron/tmp/atlas-toolchains/llama.cpp/build-atlas/bin/llama-server"
        ),
    )
    parser.add_argument(
        "--worker-rpc-server",
        type=Path,
        default=Path(
            "/home/kristianaaron/tmp/atlas-toolchains/llama.cpp/build-atlas/bin/ggml-rpc-server"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = LlamaCppRpcRuntimeConfig(
        artifact_path=args.artifact,
        artifact_sha256=args.artifact_sha256,
        llama_server_path=args.llama_server,
        worker_rpc_server_path=args.worker_rpc_server,
        worker_host=args.worker_host,
    )
    plan = build_base_canary_plan(
        CandidateBinding(
            artifact_path=str(config.artifact_path),
            artifact_sha256=config.artifact_sha256,
            runtime_config_sha256=config.canonical_sha256(),
            llama_server_sha256="86d791cf2ba2332b75b1589eece04a29488cf37d7fce871584c929fc85f644bb",
            worker_rpc_server_sha256="6b448f515e4f674c99c37ce20fd82bde9cbb28c0b2bd1fd9b0e16db3ee81ce76",
            head_argv=config.head_argv(),
            worker_argv=config.worker_argv(),
        )
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "head_argv": config.head_argv(),
                    "worker_argv": config.worker_argv(),
                    "plan": plan.model_dump(mode="json"),
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.rdma_interface or not args.disk_device:
        raise RuntimeError("--rdma-interface and --disk-device are required with --execute")

    argv_runner = _ArgvRunner()
    requests = CanaryRequestClient(config, transport=LoopbackHttpTransport())
    lifecycle = SystemdUserRuntimeLifecycle(
        config,
        worker_ssh_target=args.worker_ssh_target,
        runner=argv_runner,
        health_ready=requests.health_ready,
    )
    worker = SshWorkerHashProbe(
        ssh_target=args.worker_ssh_target,
        worker_host=config.worker_host,
        rpc_server_path=config.worker_rpc_server_path,
        toolchain_root=args.toolchain_root,
        runner=argv_runner,
    )
    telemetry = TwoNodeTelemetryCollector(
        probe_argv=(
            str(args.telemetry_probe),
            "--rdma-interface",
            args.rdma_interface,
            "--disk-device",
            args.disk_device,
        ),
        worker_ssh_target=args.worker_ssh_target,
        runner=SubprocessTelemetryRunner(),
    )
    result = TwoNodeCanaryExecutor(
        runtime=LlamaCppRpcRuntimeAdapter(config, toolchain_root=args.toolchain_root),
        worker_attestation=worker,
        lifecycle=lifecycle,
        requests=requests,
        telemetry=telemetry,
        evidence=JsonlEvidenceStore(args.evidence),
    ).execute(plan)
    print(result.receipt.model_dump_json())
    return 0 if result.receipt.runtime_claim_validated else 1


if __name__ == "__main__":
    raise SystemExit(main())
