#!/usr/bin/env python3
"""Maintenance-scoped GLM candidate + identity capture and KLD/CKA gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.canary_lease import (
    CanaryLeaseBinding,
    active_lease_scope,
    require_active_lease,
)
from model_atlas.evaluation.capture_metrics import evaluate_capture_pair
from model_atlas.evaluation.llamacpp_capture import build_capture_argv, finalize_capture
from model_atlas.glm52_capture_plan import Glm52CapturePlan, build_glm52_capture_plan
from model_atlas.llamacpp_rpc_runtime import LlamaCppRpcRuntimeConfig
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
from model_atlas.worker_rpc_lifecycle import WorkerRpcSystemdLifecycle

_MAX_JSON_BYTES = 16 * 1024 * 1024
_CHUNK = 4 * 1024 * 1024


def _read_bounded_regular(path: Path) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON_BYTES:
            raise RuntimeError("capture input must be a bounded regular file")
        digest = bytearray()
        while len(digest) <= _MAX_JSON_BYTES:
            chunk = os.read(descriptor, min(_CHUNK, _MAX_JSON_BYTES + 1 - len(digest)))
            if not chunk:
                break
            digest.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(digest) != before.st_size
            or len(digest) > _MAX_JSON_BYTES
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise RuntimeError("capture input changed during bounded read")
        return bytes(digest)
    finally:
        os.close(descriptor)


def _profile(path: Path, handoff: CompressionHandoff) -> tuple[AtlasProfile, str, Path]:
    encoded = _read_bounded_regular(path)
    profile_sha = hashlib.sha256(encoded).hexdigest()
    if profile_sha != handoff.producer_profile_sha256:
        raise RuntimeError("profile bytes differ from the compression authorization")
    try:
        value = json.loads(encoded)
        profile = AtlasProfile.from_dict(value)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("profile schema is invalid") from exc
    if profile.profile_id_of() != handoff.producer_profile_id or profile.execution is None:
        raise RuntimeError("profile identity differs from compression lineage")
    tokenizer = Path(profile.execution.checkpoint_path) / "tokenizer.json"
    return profile, profile_sha, tokenizer


def _plan(args: argparse.Namespace) -> tuple[Glm52CapturePlan, CompressionHandoff]:
    handoff = load_verified_compression_handoff(args.compression_result)
    profile, profile_sha, tokenizer = _profile(args.profile, handoff)
    execution = profile.execution
    assert execution is not None
    plan = build_glm52_capture_plan(
        work_root=args.work_root,
        model_path=Path(handoff.artifact_path),
        model_sha256=handoff.artifact_sha256,
        source_manifest_sha256=execution.source_manifest_digest,
        profile_tokenizer_path=tokenizer,
        profile_tokenizer_sha256=execution.tokenizer_hash,
        producer_artifact_sha256=handoff.evidence_sha256,
        recipe_sha256=handoff.producer_recipe_sha256,
        plan_id=handoff.producer_plan_id,
        run_id=handoff.producer_run_id,
        profile_id=handoff.producer_profile_id,
        profile_sha256=profile_sha,
        recommendation_id=handoff.producer_recommendation_id,
        compression_handoff_sha256=handoff.handoff_sha256,
    )
    if args.plan_sha256 is not None and args.plan_sha256 != plan.plan_sha256:
        raise RuntimeError("supplied capture plan differs from canonical content")
    return plan, handoff


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_paths(
    args: argparse.Namespace, plan: Glm52CapturePlan, handoff: CompressionHandoff
) -> None:
    controls = {
        "work root": args.work_root,
        "journal directory": args.journal_dir,
        "maintenance receipt": args.maintenance_receipt,
        "lease": args.lease,
        "result": args.result,
    }
    for name, path in controls.items():
        if not path.is_absolute() or path.is_symlink():
            raise RuntimeError(f"{name} must be an absolute non-symlink path")
    items = list(controls.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if _overlap(left, right):
                raise RuntimeError(f"{left_name} overlaps {right_name}")
    model = Path(handoff.artifact_path)
    run_dir = model.parents[2]
    tokenizer = Path(plan.candidate.profile_tokenizer_path)
    protected = {
        "compression result": args.compression_result,
        "profile": args.profile,
        "runtime run directory": run_dir,
        "source checkpoint": tokenizer.parent,
    }
    for control_name, control in controls.items():
        for protected_name, protected_path in protected.items():
            if _overlap(control, protected_path):
                raise RuntimeError(f"{control_name} overlaps protected {protected_name}")
    if args.execute and args.work_root.exists():
        raise RuntimeError("capture work root must not exist before execution")


def _exclusive_json(path: Path, payload: object) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("capture result write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest


def _write_model_evidence(plan: Glm52CapturePlan) -> None:
    path = Path(plan.model_evidence_path)
    encoded = json.dumps(
        plan.model_evidence.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != plan.model_evidence_sha256:
        raise RuntimeError("capture model evidence digest drifted")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("model evidence write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_text(argv: Sequence[str]) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - exact non-shell argv
            tuple(argv),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("capture process boundary failed") from exc
    return completed.stdout


def _run_native(argv: tuple[str, ...]) -> None:
    try:
        subprocess.run(  # noqa: S603 - exact reviewed non-shell argv
            argv,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("native capture failed") from exc


def _execute_payload(args: argparse.Namespace) -> int:
    if not args.execute or not args.payload or not args.expected_plan_sha256:
        raise RuntimeError("capture payload identity is incomplete")
    plan, handoff = _plan(args)
    _validate_paths(args, plan, handoff)
    if plan.plan_sha256 != args.expected_plan_sha256:
        raise RuntimeError("capture plan drifted after maintenance preflight")
    assert plan.plan_sha256 is not None
    binding = CanaryLeaseBinding(
        plan_sha256=plan.plan_sha256,
        artifact_path=handoff.artifact_path,
        artifact_sha256=handoff.artifact_sha256,
        head_unit=HEAD_TRANSIENT_UNIT,
        worker_unit=WORKER_TRANSIENT_UNIT,
    )
    require_active_lease(args.lease, binding)
    MaintenanceCoordinator(
        MaintenanceConfig(journal_dir=args.journal_dir, receipt_path=args.maintenance_receipt),
        SubprocessCommandRunner(),
        execute=True,
    ).verify_drained()
    args.work_root.mkdir(mode=0o700)
    _write_model_evidence(plan)
    runtime = LlamaCppRpcRuntimeConfig(
        artifact_path=Path(handoff.artifact_path), artifact_sha256=handoff.artifact_sha256
    )
    lifecycle = WorkerRpcSystemdLifecycle(
        runtime, worker_ssh_target=args.worker_ssh_target, runner=_run_text
    )
    launch = lifecycle.start()
    try:
        _run_native(build_capture_argv(plan.candidate, common_argv=plan.common_argv))
        candidate = finalize_capture(plan.candidate)
        after_candidate = lifecycle.remeasure_after_capture()
        _run_native(build_capture_argv(plan.identity_control, common_argv=plan.common_argv))
        identity = finalize_capture(plan.identity_control)
        after_identity = lifecycle.remeasure_after_capture()
    finally:
        lifecycle.stop()
    report = evaluate_capture_pair(
        reference_root=Path(plan.identity_control.output_dir),
        reference=identity,
        candidate_root=Path(plan.candidate.output_dir),
        candidate=candidate,
    )
    if report.identity_control_passed is not True:
        raise RuntimeError("capture identity control failed")
    report_path = args.work_root / "identity-metrics.json"
    report_sha = _exclusive_json(report_path, report.model_dump(mode="json"))
    result = {
        "schema_version": 1,
        "status": "completed",
        "quality_claim": False,
        "capture_plan_sha256": plan.plan_sha256,
        "compression_handoff_sha256": handoff.handoff_sha256,
        "worker_launch": launch.model_dump(mode="json"),
        "worker_after_candidate": after_candidate.model_dump(mode="json"),
        "worker_after_identity": after_identity.model_dump(mode="json"),
        "candidate_capture_id": candidate.capture_id,
        "identity_capture_id": identity.capture_id,
        "identity_control_passed": True,
        "metric_report_id": report.report_id,
        "metric_report_path": str(report_path),
        "metric_report_sha256": report_sha,
    }
    _exclusive_json(args.result, result)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--payload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compression-result", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--maintenance-receipt", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--expected-plan-sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-ssh-target", default="10.77.0.2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.payload:
        return _execute_payload(args)
    plan, handoff = _plan(args)
    _validate_paths(args, plan, handoff)
    public = {
        "execute": args.execute,
        "capture_plan_sha256": plan.plan_sha256,
        "quality_claim": False,
        "compression_handoff_sha256": handoff.handoff_sha256,
        "candidate_request_id": plan.candidate.request_id,
        "identity_request_id": plan.identity_control.request_id,
        "candidate_request": plan.candidate.model_dump(mode="json"),
        "identity_request": plan.identity_control.model_dump(mode="json"),
    }
    if not args.execute:
        print(json.dumps(public, sort_keys=True))
        return 0
    assert plan.plan_sha256 is not None
    payload = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--execute",
        "--payload",
        "--compression-result",
        str(args.compression_result),
        "--profile",
        str(args.profile),
        "--work-root",
        str(args.work_root),
        "--journal-dir",
        str(args.journal_dir),
        "--maintenance-receipt",
        str(args.maintenance_receipt),
        "--lease",
        str(args.lease),
        "--result",
        str(args.result),
        "--expected-plan-sha256",
        plan.plan_sha256,
        "--worker-ssh-target",
        args.worker_ssh_target,
    )
    binding = CanaryLeaseBinding(
        plan_sha256=plan.plan_sha256,
        artifact_path=handoff.artifact_path,
        artifact_sha256=handoff.artifact_sha256,
        head_unit=HEAD_TRANSIENT_UNIT,
        worker_unit=WORKER_TRANSIENT_UNIT,
    )
    previous = install_signal_traps()
    try:
        receipt = MaintenanceCoordinator(
            MaintenanceConfig(journal_dir=args.journal_dir, receipt_path=args.maintenance_receipt),
            SubprocessCommandRunner(),
            execute=True,
        ).run(payload, payload_scope=partial(active_lease_scope, args.lease, binding))
    finally:
        restore_signal_traps(previous)
    if not receipt.success:
        raise RuntimeError("capture maintenance transaction failed")
    print(json.dumps(public, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
