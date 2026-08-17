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
from typing import cast

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.canary_lease import CanaryLeaseBinding, require_active_lease
from model_atlas.fit_telemetry import (
    CandidateBinding,
    TwoNodeTelemetryCollector,
    build_base_canary_plan,
)
from model_atlas.llamacpp_rpc_runtime import LlamaCppRpcRuntimeAdapter, LlamaCppRpcRuntimeConfig
from model_atlas.ops.maintenance import (
    MaintenanceConfig,
    MaintenanceCoordinator,
    SubprocessCommandRunner,
)
from model_atlas.runtime_canary_driver import (
    CanaryRequestClient,
    LoopbackHttpTransport,
    SystemdUserRuntimeLifecycle,
)
from model_atlas.telemetry_python import TelemetryPythonConfig, verify_telemetry_python
from model_atlas.two_node_canary_executor import (
    JsonlEvidenceStore,
    RuntimeContract,
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
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--execute", action="store_true", help="perform the real canary; default is dry-run"
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--producer-run-id")
    parser.add_argument("--producer-plan-id")
    parser.add_argument("--producer-recipe-sha256")
    parser.add_argument("--producer-profile-id")
    parser.add_argument("--producer-recommendation-id")
    parser.add_argument("--producer-handoff-sha256")
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument(
        "--telemetry-probe",
        type=Path,
        default=Path(__file__).with_name("collect_two_node_telemetry.py"),
    )
    parser.add_argument("--rdma-interface")
    parser.add_argument("--disk-device")
    parser.add_argument(
        "--telemetry-python",
        type=Path,
        default=Path("/home/kristianaaron/ai-lab/venvs/vllm/bin/python"),
    )
    parser.add_argument("--telemetry-python-sha256")
    parser.add_argument("--maintenance-lease", type=Path)
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
            producer_run_id=args.producer_run_id,
            producer_plan_id=args.producer_plan_id,
            producer_recipe_sha256=args.producer_recipe_sha256,
            producer_profile_id=args.producer_profile_id,
            producer_recommendation_id=args.producer_recommendation_id,
            producer_handoff_sha256=args.producer_handoff_sha256,
        )
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "head_argv": config.head_argv(),
                    "worker_argv": config.worker_argv(),
                    "plan_sha256": plan.canonical_sha256(),
                    "plan": plan.model_dump(mode="json"),
                },
                sort_keys=True,
            )
        )
        return 0
    if (
        not args.rdma_interface
        or not args.disk_device
        or not args.telemetry_python_sha256
        or args.maintenance_lease is None
    ):
        raise RuntimeError(
            "--rdma-interface, --disk-device, --telemetry-python-sha256, and "
            "--maintenance-lease are required with --execute"
        )
    # Re-probe live production/transient state at the execution boundary.  A
    # valid lease alone never permits a canary while a consumer is still active.
    MaintenanceCoordinator(
        MaintenanceConfig(
            journal_dir=args.evidence.parent,
            receipt_path=args.evidence.with_suffix(".maintenance.json"),
            head_runtime_unit=HEAD_TRANSIENT_UNIT,
            worker_rpc_unit=WORKER_TRANSIENT_UNIT,
            worker_ssh_target=args.worker_ssh_target,
        ),
        SubprocessCommandRunner(),
        execute=True,
    ).verify_drained()
    require_active_lease(
        args.maintenance_lease,
        CanaryLeaseBinding(
            plan_sha256=plan.canonical_sha256(),
            artifact_path=str(config.artifact_path),
            artifact_sha256=config.artifact_sha256,
            head_unit=HEAD_TRANSIENT_UNIT,
            worker_unit=WORKER_TRANSIENT_UNIT,
        ),
    )

    argv_runner = _ArgvRunner()
    telemetry_python = TelemetryPythonConfig(
        interpreter=args.telemetry_python,
        interpreter_sha256=args.telemetry_python_sha256,
        worker_ssh_target=args.worker_ssh_target,
    )
    verify_telemetry_python(telemetry_python, runner=argv_runner)
    requests = CanaryRequestClient(config, transport=LoopbackHttpTransport())
    lifecycle = SystemdUserRuntimeLifecycle(
        config,
        worker_ssh_target=args.worker_ssh_target,
        runner=argv_runner,
        health_ready=requests.health_ready,
        worker_unit=WORKER_TRANSIENT_UNIT,
        head_unit=HEAD_TRANSIENT_UNIT,
    )
    worker = SshWorkerHashProbe(
        ssh_target=args.worker_ssh_target,
        worker_host=config.worker_host,
        rpc_server_path=config.worker_rpc_server_path,
        toolchain_root=args.toolchain_root,
        runner=argv_runner,
    )
    telemetry = TwoNodeTelemetryCollector(
        probe_argv=telemetry_python.probe_argv(
            args.telemetry_probe,
            "--rdma-interface",
            args.rdma_interface,
            "--disk-device",
            args.disk_device,
        ),
        worker_ssh_target=args.worker_ssh_target,
        runner=SubprocessTelemetryRunner(),
    )
    result = TwoNodeCanaryExecutor(
        runtime=cast(
            RuntimeContract,
            LlamaCppRpcRuntimeAdapter(config, toolchain_root=args.toolchain_root),
        ),
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
