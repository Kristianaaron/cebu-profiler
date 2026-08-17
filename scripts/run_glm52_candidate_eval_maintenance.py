#!/usr/bin/env python3
"""Dry-run or execute the reviewed GLM candidate evaluation maintenance transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import subprocess
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.canary_lease import (
    CanaryLeaseBinding,
    active_lease_scope,
    require_active_lease,
)
from model_atlas.evaluation.glm52_candidate_eval import (
    CandidateEvalPlan,
    build_glm52_candidate_eval_plan,
)
from model_atlas.evaluation.glm52_candidate_eval_driver import (
    CandidateEvalExecutionResult,
    CandidateEvalExecutor,
    GitEvalLabCheckoutVerifier,
    SubprocessEvalLabRunner,
    SystemdRuntimeQuiescenceVerifier,
)
from model_atlas.fit_telemetry import CanaryPlan, CandidateBinding, build_base_canary_plan
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
from model_atlas.recommend.policy import AtlasProfile
from model_atlas.runtime_artifact_handoff import (
    CompressionHandoff,
    load_verified_compression_handoff,
)
from model_atlas.runtime_canary_driver import LoopbackHttpTransport, SystemdUserRuntimeLifecycle
from model_atlas.runtime_canary_handoff import (
    RuntimeCanaryHandoff,
    load_verified_runtime_canary_handoff,
)

_WORKER_SSH_TARGET = "10.77.0.2"
_MAX_PROFILE_BYTES = 4 * 1024 * 1024
_MAX_RESULT_BYTES = 16 * 1024 * 1024


class _ArgvRunner:
    def __call__(self, argv: Sequence[str]) -> str:
        completed = subprocess.run(
            tuple(argv),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=120.0,
        )
        if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
            raise RuntimeError("external runtime identity command failed")
        return completed.stdout


def _open_directory(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RuntimeError("directory path must be absolute and canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular(path: Path, *, limit: int) -> bytes:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise RuntimeError("input path must be absolute and symlink-free")
    parent = _open_directory(path.parent)
    descriptor = os.open(
        path.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > limit:
            raise RuntimeError("input is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if len(payload) != before.st_size or identity(before) != identity(after):
            raise RuntimeError("input changed during bounded read")
        return payload
    finally:
        os.close(descriptor)
        os.close(parent)


def _profile_tokenizer(path: Path, handoff: CompressionHandoff) -> str:
    encoded = _read_regular(path, limit=_MAX_PROFILE_BYTES)
    if hashlib.sha256(encoded).hexdigest() != handoff.producer_profile_sha256:
        raise RuntimeError("profile bytes differ from the compression lineage")
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise RuntimeError("profile must be a JSON object")
    profile = AtlasProfile.from_dict(payload)
    if profile.profile_id_of() != handoff.producer_profile_id or profile.execution is None:
        raise RuntimeError("profile identity differs from the compression lineage")
    return profile.execution.tokenizer_hash


def _canary_plan(handoff: CompressionHandoff, config: LlamaCppRpcRuntimeConfig) -> CanaryPlan:
    return build_base_canary_plan(
        CandidateBinding(
            artifact_path=handoff.artifact_path,
            artifact_sha256=handoff.artifact_sha256,
            runtime_config_sha256=config.canonical_sha256(),
            llama_server_sha256=EXPECTED_LLAMA_SERVER_SHA256,
            worker_rpc_server_sha256=EXPECTED_RPC_SERVER_SHA256,
            head_argv=config.head_argv(),
            worker_argv=config.worker_argv(),
            producer_run_id=handoff.producer_run_id,
            producer_plan_id=handoff.producer_plan_id,
            producer_recipe_sha256=handoff.producer_recipe_sha256,
            producer_profile_id=handoff.producer_profile_id,
            producer_recommendation_id=handoff.producer_recommendation_id,
            producer_handoff_sha256=handoff.handoff_sha256,
        )
    )


def _build_plan(
    args: argparse.Namespace,
) -> tuple[
    CompressionHandoff,
    RuntimeCanaryHandoff,
    LlamaCppRpcRuntimeConfig,
    CandidateEvalPlan,
]:
    compression = load_verified_compression_handoff(args.compression_result)
    config = LlamaCppRpcRuntimeConfig(
        artifact_path=Path(compression.artifact_path),
        artifact_sha256=compression.artifact_sha256,
    )
    canary = load_verified_runtime_canary_handoff(
        args.canary_result,
        expected_plan=_canary_plan(compression, config),
        require_evaluation_ready=True,
    )
    tokenizer = _profile_tokenizer(args.profile, compression)
    plan = build_glm52_candidate_eval_plan(
        compression_handoff=compression,
        runtime_canary_handoff=canary,
        eval_output_root=args.operation_root,
        verified_tokenizer_sha256=tokenizer,
    )
    return compression, canary, config, plan


def _validate_paths(args: argparse.Namespace) -> None:
    paths = (
        args.compression_result,
        args.canary_result,
        args.profile,
        args.operation_root,
        args.journal_dir,
        args.receipt,
        args.lease,
        args.result,
    )
    if any(not path.is_absolute() for path in paths) or len(set(paths)) != len(paths):
        raise RuntimeError("operation paths must be absolute and mutually distinct")
    directory_paths = (
        args.operation_root,
        args.journal_dir,
        args.receipt.parent,
        args.lease.parent,
        args.result.parent,
        args.compression_result.parent,
        args.canary_result.parent,
        args.profile.parent,
    )
    directory_identities: list[tuple[int, int]] = []
    for path in directory_paths:
        descriptor = _open_directory(path)
        try:
            measured = os.fstat(descriptor)
            directory_identities.append((measured.st_dev, measured.st_ino))
        finally:
            os.close(descriptor)
    operation_identity, journal_identity = directory_identities[:2]
    if operation_identity == journal_identity:
        raise RuntimeError("operation root and maintenance journal must be separate directories")
    output_slots = {
        (*identity, path.name)
        for path, identity in zip(
            (args.receipt, args.lease, args.result),
            directory_identities[2:5],
            strict=True,
        )
    }
    if len(output_slots) != 3:
        raise RuntimeError("maintenance output paths physically alias")
    if args.result.parent != args.operation_root:
        raise RuntimeError("evaluation result must be a direct child of the operation root")
    for path in (args.receipt, args.lease, args.result):
        parent = _open_directory(path.parent)
        try:
            try:
                os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise RuntimeError("maintenance receipt, lease, and result must be fresh paths")
        finally:
            os.close(parent)


def _publish_result(path: Path, result: CandidateEvalExecutionResult) -> None:
    payload = result.model_dump_json(indent=2).encode()
    if len(payload) > _MAX_RESULT_BYTES:
        raise RuntimeError("candidate evaluation result exceeds its bound")
    parent = _open_directory(path.parent)
    temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("candidate evaluation result write failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.fsync(parent)
        os.unlink(temporary, dir_fd=parent)
        os.fsync(parent)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent)
            os.fsync(parent)
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    loaded = CandidateEvalExecutionResult.model_validate_json(
        _read_regular(path, limit=_MAX_RESULT_BYTES)
    )
    if loaded != result:
        raise RuntimeError("published candidate evaluation result failed verification")


def _execute_in_process(args: argparse.Namespace, expected_plan_sha256: str) -> None:
    compression, _canary, config, plan = _build_plan(args)
    assert plan.plan_sha256 is not None
    if expected_plan_sha256 != plan.plan_sha256:
        raise RuntimeError("candidate evaluation plan changed after dispatch")
    binding = CanaryLeaseBinding(
        plan_sha256=plan.plan_sha256,
        artifact_path=compression.artifact_path,
        artifact_sha256=compression.artifact_sha256,
        head_unit=HEAD_TRANSIENT_UNIT,
        worker_unit=WORKER_TRANSIENT_UNIT,
    )
    require_active_lease(args.lease, binding, expected_coordinator_pid=os.getpid())
    maintenance = MaintenanceConfig(
        journal_dir=args.journal_dir,
        receipt_path=args.receipt,
        worker_ssh_target=_WORKER_SSH_TARGET,
        head_runtime_unit=HEAD_TRANSIENT_UNIT,
        worker_rpc_unit=WORKER_TRANSIENT_UNIT,
    )
    MaintenanceCoordinator(maintenance, SubprocessCommandRunner(), execute=True).verify_drained()
    runner = _ArgvRunner()
    transport = LoopbackHttpTransport()
    lifecycle = SystemdUserRuntimeLifecycle(
        config,
        worker_ssh_target=_WORKER_SSH_TARGET,
        runner=runner,
        health_ready=lambda: (
            transport.request("GET", f"http://{config.api_host}:{config.api_port}/health").status
            == 200
        ),
    )
    execution = CandidateEvalExecutor(
        config,
        lifecycle=lifecycle,
        transport=transport,
        command_runner=SubprocessEvalLabRunner(),
        quiescence=SystemdRuntimeQuiescenceVerifier(
            worker_ssh_target=_WORKER_SSH_TARGET,
            runner=runner,
            worker_unit=WORKER_TRANSIENT_UNIT,
            head_unit=HEAD_TRANSIENT_UNIT,
        ),
        checkout_verifier=GitEvalLabCheckoutVerifier(),
        eval_lab_cwd=Path(plan.eval_request.eval_lab_root),
    ).execute(plan)
    _publish_result(args.result, execution)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--compression-result", type=Path, required=True)
    parser.add_argument("--canary-result", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--operation-root", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_paths(args)
    compression, _canary, _config, plan = _build_plan(args)
    assert plan.plan_sha256 is not None
    if args.expected_plan_sha256 is not None and args.expected_plan_sha256 != plan.plan_sha256:
        raise RuntimeError("supplied plan digest differs from canonical content")
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "plan_sha256": plan.plan_sha256,
                    "compression_handoff_sha256": compression.handoff_sha256,
                    "runtime_canary_handoff_sha256": plan.runtime_canary_handoff_sha256,
                    "artifact_path": compression.artifact_path,
                    "artifact_sha256": compression.artifact_sha256,
                    "argv": plan.argv,
                    "plan": plan.model_dump(mode="json"),
                },
                sort_keys=True,
            )
        )
        return 0
    config = MaintenanceConfig(
        journal_dir=args.journal_dir,
        receipt_path=args.receipt,
        worker_ssh_target=_WORKER_SSH_TARGET,
        head_runtime_unit=HEAD_TRANSIENT_UNIT,
        worker_rpc_unit=WORKER_TRANSIENT_UNIT,
    )
    binding = CanaryLeaseBinding(
        plan_sha256=plan.plan_sha256,
        artifact_path=compression.artifact_path,
        artifact_sha256=compression.artifact_sha256,
        head_unit=HEAD_TRANSIENT_UNIT,
        worker_unit=WORKER_TRANSIENT_UNIT,
    )
    previous = install_signal_traps()
    try:
        receipt = MaintenanceCoordinator(config, SubprocessCommandRunner(), execute=True).run(
            payload_scope=partial(active_lease_scope, args.lease, binding),
            payload_action=partial(_execute_in_process, args, plan.plan_sha256),
        )
    finally:
        restore_signal_traps(previous)
    if receipt.success:
        loaded = CandidateEvalExecutionResult.model_validate_json(
            _read_regular(args.result, limit=_MAX_RESULT_BYTES)
        )
        if loaded.plan_sha256 != plan.plan_sha256:
            raise RuntimeError("candidate evaluation result is bound to a different plan")
    print(receipt.model_dump_json())
    return 0 if receipt.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
