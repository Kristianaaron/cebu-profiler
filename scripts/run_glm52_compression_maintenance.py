#!/usr/bin/env python3
"""Authorize and run the GLM-5.2 mixed-GGUF producer inside maintenance.

Dry-run is the default and performs only the deterministic recommendation and
preview.  Real execution is bracketed by :class:`MaintenanceCoordinator`; the
child payload must prove that it is the direct child of the live coordinator
and that the profile and recipe identities still match the preflight preview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.canary_lease import (
    CanaryLeaseBinding,
    active_lease_scope,
    require_active_lease,
)
from model_atlas.ops.maintenance import (
    MaintenanceConfig,
    MaintenanceCoordinator,
    SubprocessCommandRunner,
    install_signal_traps,
    restore_signal_traps,
)
from model_atlas.recommend import CompressionIntent, RecommendationService, RecTarget
from model_atlas.recommend.policy import AtlasProfile, method_spec

METHOD = "llamacpp-gguf-mixed"
TERMINAL = {
    "completed",
    "completed_with_warnings",
    "failed_terminal",
    "failed_recoverable",
    "cancelled",
}


def _read_regular_file(path: Path, *, limit: int = 16 * 1024 * 1024) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("profile must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(encoded) > limit or len(encoded) != before.st_size:
            raise ValueError("profile exceeds bounded read limit")
        if identity_before != identity_after:
            raise RuntimeError("profile changed during bounded read")
        return encoded
    finally:
        os.close(descriptor)


def _profile_identity(path: Path) -> tuple[AtlasProfile, str]:
    encoded = _read_regular_file(path)
    raw = json.loads(encoded)
    if not isinstance(raw, dict):
        raise ValueError("profile JSON must be an object")
    return AtlasProfile.from_dict(raw), hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    """Bounded profile digest retained as a narrow compatibility helper."""

    return hashlib.sha256(_read_regular_file(path)).hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_control_paths(args: argparse.Namespace, profile: AtlasProfile) -> None:
    controls = {
        "result": args.result,
        "maintenance receipt": args.maintenance_receipt,
        "maintenance lease": args.lease,
        "journal directory": args.journal_dir,
    }
    items = list(controls.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if _paths_overlap(left, right):
                raise ValueError(f"{left_name} overlaps {right_name}")
    execution = profile.execution
    if execution is None:
        raise ValueError("profile lacks executable source identity")
    protected = {
        "profile": args.profile,
        "profiles directory": args.profiles_dir,
        "work root": args.work_root,
        "source checkpoint": Path(execution.checkpoint_path).resolve(strict=True),
    }
    for control_name, control in controls.items():
        for protected_name, protected_path in protected.items():
            if _paths_overlap(control, protected_path):
                raise ValueError(f"{control_name} overlaps protected {protected_name}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("incomplete result write")
        os.fsync(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--payload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, default=Path("profiles"))
    parser.add_argument("--work-root", type=Path, default=Path("controlplane_runs"))
    parser.add_argument("--memory-target-gib", type=float, default=115.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--maintenance-receipt", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--expected-recipe-sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--expected-profile-sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--expected-profile-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--expected-recommendation-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--expected-plan-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--expected-run-id", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def _preview(
    args: argparse.Namespace,
) -> tuple[RecommendationService, dict[str, Any], dict[str, Any]]:
    if not args.profile.is_absolute() or not args.profile.is_file() or args.profile.is_symlink():
        raise ValueError("profile must be an absolute regular non-symlink file")
    if args.memory_target_gib <= 0 or args.poll_seconds <= 0:
        raise ValueError("memory target and poll interval must be positive")
    try:
        args.profile.relative_to(args.profiles_dir)
    except ValueError as exc:
        raise ValueError("profile must be contained by profiles-dir") from exc
    service = RecommendationService(
        profile_root=args.profiles_dir,
        work_root=args.work_root,
        # Preflight construction must never auto-resume a persisted run. New
        # starts below use the service's joined dedicated-worker fallback;
        # persisted runs use the synchronous verified ControlPlane resume.
        supervised_executor=False,
    )
    try:
        profile = service.import_profile(args.profile)
        authorization = service.authorize(
            profile.profile_id_of(),
            RecTarget(memory_target_gib=args.memory_target_gib),
            intent=CompressionIntent.QUANTIZE_ONLY,
        )
        derivatives = sorted(
            method
            for method in authorization["authorized_methods"]
            if not method_spec(method).planning_only
        )
        if derivatives != [METHOD]:
            raise RuntimeError("GLM mixed-GGUF method is not the sole authorized derivative")
        preview = service.preview_selection(authorization["token"], [METHOD])
        readiness = preview.get("readiness") or {}
        if not all(
            readiness.get(key) is True
            for key in ("verified_plan", "pins_pass", "intent_satisfied", "executable")
        ):
            raise RuntimeError("GLM mixed-GGUF preview is not executable")
        if preview.get("actual_families") != ["quantization"]:
            raise RuntimeError("preview does not contain exactly the quantization family")
        return service, authorization, preview
    except BaseException:
        service.shutdown(wait=True)
        raise


def _public_preview(
    args: argparse.Namespace,
    authorization: dict[str, Any],
    preview: dict[str, Any],
    *,
    profile_sha256: str,
) -> dict[str, Any]:
    return {
        "mode": "execute" if args.execute else "dry_run",
        "profile_path": str(args.profile),
        "profile_sha256": profile_sha256,
        "profile_id": authorization["profile_id"],
        "recommendation_id": authorization["recommendation_id"],
        "method": METHOD,
        "intent": "quantize_only",
        "preview_id": preview["preview_id"],
        "recipe_id": preview["recipe_id"],
        "recipe_sha256": preview["recipe_sha256"],
        "plan_id": preview["plan_id"],
        "run_id": preview["run_id"],
        "readiness": preview["readiness"],
        "runtime_claim": "artifact_only_unvalidated",
    }


def _binding(
    args: argparse.Namespace, recipe_sha256: str, profile_sha256: str
) -> CanaryLeaseBinding:
    return CanaryLeaseBinding(
        plan_sha256=recipe_sha256,
        artifact_path=str(args.profile),
        artifact_sha256=profile_sha256,
        head_unit=HEAD_TRANSIENT_UNIT,
        worker_unit=WORKER_TRANSIENT_UNIT,
    )


def _run_payload(args: argparse.Namespace) -> int:
    if not args.execute:
        raise RuntimeError("payload mode requires --execute")
    expected = (
        args.expected_recipe_sha256,
        args.expected_profile_sha256,
        args.expected_profile_id,
        args.expected_recommendation_id,
        args.expected_plan_id,
        args.expected_run_id,
    )
    if any(not value for value in expected):
        raise RuntimeError("payload identity is incomplete")
    profile, profile_sha256 = _profile_identity(args.profile)
    _validate_control_paths(args, profile)
    if profile_sha256 != args.expected_profile_sha256:
        raise RuntimeError("profile bytes drifted after maintenance preflight")
    binding = _binding(args, args.expected_recipe_sha256, profile_sha256)
    require_active_lease(args.lease, binding)

    # The lease is necessary but not sufficient: independently prove the live
    # production drain from inside the payload immediately before dispatch.
    MaintenanceCoordinator(
        MaintenanceConfig(journal_dir=args.journal_dir, receipt_path=args.maintenance_receipt),
        SubprocessCommandRunner(),
        execute=True,
    ).verify_drained()

    service, authorization, preview = _preview(args)
    try:
        if _sha256_file(args.profile) != args.expected_profile_sha256:
            raise RuntimeError("profile bytes drifted during payload authorization")
        live_identity = (
            authorization["profile_id"],
            authorization["recommendation_id"],
            preview["recipe_sha256"],
            preview["plan_id"],
            preview["run_id"],
        )
        expected_identity = (
            args.expected_profile_id,
            args.expected_recommendation_id,
            args.expected_recipe_sha256,
            args.expected_plan_id,
            args.expected_run_id,
        )
        if live_identity != expected_identity:
            raise RuntimeError("authorization identity drifted after maintenance preflight")
        run_id = str(preview["run_id"])
        job_path = args.work_root / "runs" / run_id / "job.json"
        if job_path.exists():
            engine = service.plane.engine_for(run_id)
            if (
                engine.compiled.plan_id != preview["plan_id"]
                or engine.compiled.recipe_sha256 != preview["recipe_sha256"]
            ):
                raise RuntimeError("persisted run identity differs from authorized preview")
            current = service.job_status(run_id)
            state = str(current.get("status", ""))
            if state in {"failed_terminal", "cancelled"}:
                raise RuntimeError("persisted run is not safely resumable")
            # Resume even a completed job: this is the engine's authoritative
            # DONE-output/CAS hash verification path, not merely a status read.
            service.plane.resume(run_id)
        else:
            started = service.start_authorized(
                authorization["token"],
                preview["preview_id"],
                preview["hash"],
                [METHOD],
                {},
                plan_id=preview["plan_id"],
                recipe_sha256=preview["recipe_sha256"],
            )
            if str(started["run_id"]) != run_id:
                raise RuntimeError("started run identity differs from authorized preview")
        while True:
            status = service.job_status(run_id)
            state = str(status.get("status", ""))
            if state in TERMINAL:
                service.shutdown(wait=True)
                if state in {"completed", "completed_with_warnings"}:
                    # Re-enter the verified resume path immediately before
                    # publication, then demand that the durable state remains
                    # successful. This catches missing/corrupt DONE outputs.
                    service.plane.resume(run_id)
                    verified = service.job_status(run_id)
                    state = str(verified.get("status", ""))
                    if state not in {"completed", "completed_with_warnings"}:
                        raise RuntimeError("post-run integrity verification did not complete")
                if _sha256_file(args.profile) != args.expected_profile_sha256:
                    raise RuntimeError("profile bytes drifted before result publication")
                result = _public_preview(
                    args,
                    authorization,
                    preview,
                    profile_sha256=args.expected_profile_sha256,
                )
                result.update(
                    {
                        "status": state,
                        "outputs": service.job_output(run_id),
                        "lineage": service.run_lineage(run_id),
                    }
                )
                _atomic_json(args.result, result)
                return 0 if state in {"completed", "completed_with_warnings"} else 1
            time.sleep(args.poll_seconds)
    finally:
        service.shutdown(wait=True)


def main() -> int:
    args = parse_args()
    if args.profile.is_symlink():
        raise ValueError("profile must not be a symlink")
    args.profile = args.profile.resolve(strict=True)
    args.profiles_dir = args.profiles_dir.resolve()
    args.work_root = args.work_root.resolve()
    args.journal_dir = args.journal_dir.resolve()
    args.maintenance_receipt = args.maintenance_receipt.resolve()
    args.result = args.result.resolve()
    args.lease = args.lease.resolve()
    profile, profile_sha256 = _profile_identity(args.profile)
    _validate_control_paths(args, profile)
    for path, label in (
        (args.result, "result"),
        (args.maintenance_receipt, "maintenance receipt"),
    ):
        if path.exists() or path.is_symlink():
            raise ValueError(f"{label} path already exists; use a new attempt path")
    if args.payload:
        return _run_payload(args)
    if args.lease.exists() or args.lease.is_symlink():
        raise ValueError("maintenance lease path already exists; use a new attempt path")

    service, authorization, preview = _preview(args)
    try:
        if _sha256_file(args.profile) != profile_sha256:
            raise RuntimeError("profile bytes drifted during maintenance preflight")
        public = _public_preview(
            args,
            authorization,
            preview,
            profile_sha256=profile_sha256,
        )
    finally:
        service.shutdown(wait=True)
    if not args.execute:
        _atomic_json(args.result, public)
        print(json.dumps(public, sort_keys=True))
        return 0

    recipe_sha256 = str(preview["recipe_sha256"])
    binding = _binding(args, recipe_sha256, profile_sha256)
    payload = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--execute",
        "--payload",
        "--profile",
        str(args.profile),
        "--profiles-dir",
        str(args.profiles_dir),
        "--work-root",
        str(args.work_root),
        "--memory-target-gib",
        str(args.memory_target_gib),
        "--poll-seconds",
        str(args.poll_seconds),
        "--journal-dir",
        str(args.journal_dir),
        "--maintenance-receipt",
        str(args.maintenance_receipt),
        "--result",
        str(args.result),
        "--lease",
        str(args.lease),
        "--expected-recipe-sha256",
        recipe_sha256,
        "--expected-profile-sha256",
        profile_sha256,
        "--expected-profile-id",
        str(authorization["profile_id"]),
        "--expected-recommendation-id",
        str(authorization["recommendation_id"]),
        "--expected-plan-id",
        str(preview["plan_id"]),
        "--expected-run-id",
        str(preview["run_id"]),
    )
    config = MaintenanceConfig(
        journal_dir=args.journal_dir,
        receipt_path=args.maintenance_receipt,
        head_runtime_unit=HEAD_TRANSIENT_UNIT,
        worker_rpc_unit=WORKER_TRANSIENT_UNIT,
    )
    previous = install_signal_traps()
    try:
        receipt = MaintenanceCoordinator(config, SubprocessCommandRunner(), execute=True).run(
            payload,
            payload_scope=partial(active_lease_scope, args.lease, binding),
        )
    finally:
        restore_signal_traps(previous)
    print(receipt.model_dump_json())
    return 0 if receipt.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
