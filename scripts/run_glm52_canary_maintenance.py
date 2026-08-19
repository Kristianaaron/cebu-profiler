#!/usr/bin/env python3
"""The sole real-execution entry point: drain, lease, canary, restore."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.canary_lease import CanaryLeaseBinding, active_lease_scope
from model_atlas.fit_telemetry import CandidateBinding, build_base_canary_plan
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    LlamaCppRpcRuntimeConfig,
)
from model_atlas.ops.maintenance import (
    MaintenanceConfig,
    MaintenanceCoordinator,
    SubprocessCommandRunner,
    install_signal_traps,
    restore_signal_traps,
)
from model_atlas.runtime_artifact_handoff import load_verified_compression_handoff
from model_atlas.runtime_canary_handoff import load_verified_runtime_canary_handoff

_RESERVED_CANARY_OPTIONS = {
    "--execute",
    "--maintenance-lease",
    "--artifact",
    "--artifact-sha256",
    "--evidence",
    "--result",
    "--producer-run-id",
    "--producer-plan-id",
    "--producer-recipe-sha256",
    "--producer-profile-id",
    "--producer-recommendation-id",
    "--producer-handoff-sha256",
    "--worker-host",
    "--llama-server",
    "--worker-rpc-server",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--canary-result", type=Path, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--compression-result", type=Path)
    parser.add_argument("--artifact")
    parser.add_argument("--artifact-sha256")
    parser.add_argument("canary_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    has_result = args.compression_result is not None
    has_raw = args.artifact is not None or args.artifact_sha256 is not None
    if has_result == has_raw:
        raise RuntimeError(
            "provide exactly one artifact mode: --compression-result or "
            "--artifact with --artifact-sha256"
        )
    if has_result:
        handoff = load_verified_compression_handoff(args.compression_result)
        artifact = handoff.artifact_path
        artifact_sha256 = handoff.artifact_sha256
    else:
        if args.artifact is None or args.artifact_sha256 is None:
            raise RuntimeError("raw artifact mode requires path and sha256")
        artifact = args.artifact
        artifact_sha256 = args.artifact_sha256
        handoff = None
    canary_args = args.canary_args[1:] if args.canary_args[:1] == ["--"] else args.canary_args
    overridden = sorted(
        {
            argument.split("=", 1)[0]
            for argument in canary_args
            if argument.split("=", 1)[0].startswith("--")
            and any(
                option.startswith(argument.split("=", 1)[0]) for option in _RESERVED_CANARY_OPTIONS
            )
        }
    )
    if overridden:
        raise RuntimeError(
            "canary passthrough may not override bound options: " + ", ".join(overridden)
        )
    runtime_config = LlamaCppRpcRuntimeConfig(
        artifact_path=Path(artifact),
        artifact_sha256=artifact_sha256,
    )
    plan = build_base_canary_plan(
        CandidateBinding(
            artifact_path=str(runtime_config.artifact_path),
            artifact_sha256=runtime_config.artifact_sha256,
            runtime_config_sha256=runtime_config.canonical_sha256(),
            llama_server_sha256=EXPECTED_LLAMA_SERVER_SHA256,
            worker_rpc_server_sha256=EXPECTED_RPC_SERVER_SHA256,
            head_argv=runtime_config.head_argv(),
            worker_argv=runtime_config.worker_argv(),
            producer_run_id=handoff.producer_run_id if handoff is not None else None,
            producer_plan_id=handoff.producer_plan_id if handoff is not None else None,
            producer_recipe_sha256=(
                handoff.producer_recipe_sha256 if handoff is not None else None
            ),
            producer_profile_id=handoff.producer_profile_id if handoff is not None else None,
            producer_recommendation_id=(
                handoff.producer_recommendation_id if handoff is not None else None
            ),
            producer_handoff_sha256=(handoff.handoff_sha256 if handoff is not None else None),
        )
    )
    derived_plan_sha256 = plan.canonical_sha256()
    if args.plan_sha256 is not None and args.plan_sha256 != derived_plan_sha256:
        raise RuntimeError("supplied plan sha256 differs from the canonical canary plan")
    plan_sha256 = derived_plan_sha256
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "artifact": artifact,
                    "artifact_sha256": artifact_sha256,
                    "plan_sha256": plan_sha256,
                    "plan": plan.model_dump(mode="json"),
                    "head_argv": runtime_config.head_argv(),
                    "worker_argv": runtime_config.worker_argv(),
                    "evidence": str(args.evidence),
                    "canary_result": str(args.canary_result),
                },
                sort_keys=True,
            )
        )
        return 0
    canary_script = Path(__file__).with_name("run_two_node_canary.py")
    payload = (
        sys.executable,
        str(canary_script),
        "--execute",
        "--maintenance-lease",
        str(args.lease),
        "--artifact",
        artifact,
        "--artifact-sha256",
        artifact_sha256,
        "--evidence",
        str(args.evidence),
        "--result",
        str(args.canary_result),
        *(
            (
                "--producer-run-id",
                handoff.producer_run_id,
                "--producer-plan-id",
                handoff.producer_plan_id,
                "--producer-recipe-sha256",
                handoff.producer_recipe_sha256,
                "--producer-profile-id",
                handoff.producer_profile_id,
                "--producer-recommendation-id",
                handoff.producer_recommendation_id,
                "--producer-handoff-sha256",
                handoff.handoff_sha256,
            )
            if handoff is not None
            else ()
        ),
        *canary_args,
    )
    config = MaintenanceConfig(
        journal_dir=args.journal_dir,
        receipt_path=args.receipt,
        head_runtime_unit=HEAD_TRANSIENT_UNIT,
        worker_rpc_unit=WORKER_TRANSIENT_UNIT,
        dsv4_model_shards=48,
    )
    previous = install_signal_traps()
    try:
        binding = CanaryLeaseBinding(
            plan_sha256=plan_sha256,
            artifact_path=artifact,
            artifact_sha256=artifact_sha256,
            head_unit=HEAD_TRANSIENT_UNIT,
            worker_unit=WORKER_TRANSIENT_UNIT,
        )
        receipt = MaintenanceCoordinator(
            config, SubprocessCommandRunner(), execute=args.execute
        ).run(
            payload if args.execute else None,
            payload_scope=(
                partial(active_lease_scope, args.lease, binding) if args.execute else None
            ),
        )
    finally:
        restore_signal_traps(previous)
    if receipt.success:
        load_verified_runtime_canary_handoff(
            args.canary_result,
            expected_plan=plan,
            require_evaluation_ready=True,
        )
    print(receipt.model_dump_json())
    return 0 if receipt.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
