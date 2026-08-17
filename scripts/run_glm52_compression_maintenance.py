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
import re
import stat
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

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


def _open_directory_chain(path: Path) -> int:
    if not path.is_absolute():
        raise RuntimeError("CAS run directory must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
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


def _open_cas_slot(run_dir: Path, sha256: str) -> int:
    directory = _open_directory_chain(run_dir)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        objects = os.open("objects", flags, dir_fd=directory)
        try:
            prefix = os.open(sha256[:2], flags, dir_fd=objects)
            try:
                return os.open(
                    f"{sha256}.blob",
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=prefix,
                )
            finally:
                os.close(prefix)
        finally:
            os.close(objects)
    finally:
        os.close(directory)


def _descriptor_identity(descriptor: int) -> tuple[str, int, tuple[int, ...]]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("published model artifact is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    measured = 0
    while True:
        chunk = os.read(descriptor, 4 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        measured += len(chunk)
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
    if identity_before != identity_after or measured != before.st_size:
        raise RuntimeError("published model artifact changed during verification")
    return digest.hexdigest(), measured, identity_before


class _RuntimeArtifactPin(NamedTuple):
    run_dir: Path
    sha256: str
    size_bytes: int
    identity: tuple[int, ...]


def _reverify_runtime_artifact(pin: _RuntimeArtifactPin) -> None:
    descriptor = _open_cas_slot(pin.run_dir, pin.sha256)
    try:
        measured_sha, measured_size, identity = _descriptor_identity(descriptor)
    finally:
        os.close(descriptor)
    if measured_sha != pin.sha256 or measured_size != pin.size_bytes or identity != pin.identity:
        raise RuntimeError("model.gguf CAS output drifted during result publication")


def _verified_runtime_artifact(
    service: RecommendationService,
    run_id: str,
    outputs: dict[str, Any],
) -> tuple[dict[str, Any], _RuntimeArtifactPin]:
    """Resolve the sole GGUF output to its verified absolute CAS path.

    Runtime validation consumes a filesystem path, not an in-memory blob.  The
    handoff therefore binds the engine-published relative CAS reference to the
    exact run directory and independently re-hashes the large file without
    materializing it in memory.
    """

    raw = outputs.get("outputs")
    if outputs.get("run_id") != run_id or not isinstance(raw, list):
        raise RuntimeError("completed run output metadata is malformed")
    if len(raw) != 2 or any(not isinstance(item, dict) for item in raw):
        raise RuntimeError("completed run output set differs from the reviewed stage contract")
    model_hits = [
        item for item in raw if item.get("stage") == METHOD and item.get("name") == "model.gguf"
    ]
    evidence_hits = [
        item
        for item in raw
        if item.get("stage") == METHOD and item.get("name") == f"{METHOD}.evidence.json"
    ]
    if len(model_hits) != 1 or len(evidence_hits) != 1:
        raise RuntimeError("completed run output set is not the reviewed model/evidence pair")
    output = model_hits[0]
    evidence = evidence_hits[0]
    relpath = output.get("relpath")
    expected_sha = output.get("sha256")
    expected_size = output.get("size_bytes")
    if (
        not isinstance(relpath, str)
        or not relpath
        or not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
    ):
        raise RuntimeError("model.gguf output reference is incomplete")
    evidence_sha = evidence.get("sha256")
    evidence_size = evidence.get("size_bytes")
    evidence_relpath = evidence.get("relpath")
    if (
        not isinstance(evidence_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence_sha) is None
        or not isinstance(evidence_size, int)
        or isinstance(evidence_size, bool)
        or evidence_size <= 0
        or not isinstance(evidence_relpath, str)
    ):
        raise RuntimeError("stage evidence output reference is incomplete")
    relative = Path(relpath)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("model.gguf output reference escapes the run directory")
    expected_relative = Path("objects") / expected_sha[:2] / f"{expected_sha}.blob"
    if relative != expected_relative:
        raise RuntimeError("model.gguf output reference is not its canonical CAS slot")
    expected_evidence_relative = Path("objects") / evidence_sha[:2] / f"{evidence_sha}.blob"
    if Path(evidence_relpath) != expected_evidence_relative:
        raise RuntimeError("stage evidence output reference is not its canonical CAS slot")
    engine = service.plane.engine_for(run_id)
    run_dir = Path(engine.run_dir)
    if not run_dir.is_absolute() or run_dir.name != run_id or run_dir.parent.name != "runs":
        raise RuntimeError("engine run directory does not match the completed run")
    artifact = run_dir / relative
    descriptor = _open_cas_slot(run_dir, expected_sha)
    try:
        measured_sha, measured_size, identity = _descriptor_identity(descriptor)
    finally:
        os.close(descriptor)
    if measured_sha != expected_sha or measured_size != expected_size:
        raise RuntimeError("model.gguf CAS output identity differs from the job record")
    evidence_descriptor = _open_cas_slot(run_dir, evidence_sha)
    try:
        measured_evidence_sha, measured_evidence_size, _evidence_identity = _descriptor_identity(
            evidence_descriptor
        )
    finally:
        os.close(evidence_descriptor)
    if measured_evidence_sha != evidence_sha or measured_evidence_size != evidence_size:
        raise RuntimeError("stage evidence CAS output identity differs from the job record")
    contract = {
        "path": str(artifact),
        "sha256": measured_sha,
        "size_bytes": measured_size,
        "stage": METHOD,
        "logical_name": "model.gguf",
        "relpath": relpath,
        "runtime_validated": False,
        "evidence": {
            "stage": METHOD,
            "logical_name": f"{METHOD}.evidence.json",
            "sha256": evidence_sha,
            "size_bytes": evidence_size,
            "relpath": evidence_relpath,
        },
    }
    return contract, _RuntimeArtifactPin(run_dir, measured_sha, measured_size, identity)


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
    parser = argparse.ArgumentParser(allow_abbrev=False)
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
                outputs = service.job_output(run_id)
                if not isinstance(outputs, dict):
                    raise RuntimeError("completed run output metadata is not an object")
                lineage = service.run_lineage(run_id)
                runtime_artifact, artifact_pin = _verified_runtime_artifact(
                    service, run_id, outputs
                )
                result.update(
                    {
                        "status": state,
                        "outputs": outputs,
                        "runtime_artifact": runtime_artifact,
                        "lineage": lineage,
                    }
                )
                _atomic_json(args.result, result)
                try:
                    _reverify_runtime_artifact(artifact_pin)
                except BaseException:
                    args.result.unlink(missing_ok=True)
                    directory = os.open(args.result.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                    raise
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
