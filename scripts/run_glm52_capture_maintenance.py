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
from contextlib import suppress
from functools import partial
from pathlib import Path

from model_atlas.canary_constants import HEAD_TRANSIENT_UNIT, WORKER_TRANSIENT_UNIT
from model_atlas.canary_lease import (
    CanaryLeaseBinding,
    active_lease_scope,
    require_active_lease,
)
from model_atlas.evaluation.capture_metrics import evaluate_capture_pair
from model_atlas.evaluation.llamacpp_capture import (
    build_capture_argv,
    finalize_capture,
    preflight_capture_request,
)
from model_atlas.glm52_capture_plan import (
    CAPTURE_BINARY_SHA256,
    CAPTURE_BUILD_CONTRACT,
    CAPTURE_BUILD_CONTRACT_SHA256,
    Glm52CapturePlan,
    build_glm52_capture_plan,
)
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
_MAX_LIBRARY_BYTES = 512 * 1024 * 1024
_CHUNK = 4 * 1024 * 1024
_WORKER_SSH_TARGET = "10.77.0.2"


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


def _open_directory_chain(path: Path) -> int:
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


def _directory_identity(descriptor: int) -> tuple[int, int]:
    value = os.fstat(descriptor)
    return value.st_dev, value.st_ino


def _assert_directory_path_identity(path: Path, expected: tuple[int, int]) -> None:
    descriptor = _open_directory_chain(path)
    try:
        if _directory_identity(descriptor) != expected:
            raise RuntimeError("capture work root path identity drifted")
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
        try:
            canonical = (
                path.resolve(strict=True)
                if path.exists()
                else path.parent.resolve(strict=True) / path.name
            )
        except OSError as exc:
            raise RuntimeError(f"{name} parent must be an existing canonical directory") from exc
        if path != canonical:
            raise RuntimeError(f"{name} must not traverse a symlinked ancestor")
    for name, path in {
        "compression result": args.compression_result,
        "profile": args.profile,
    }.items():
        if not path.is_absolute() or path.is_symlink() or path != path.resolve(strict=True):
            raise RuntimeError(f"{name} must be a canonical non-symlink file")
    items = list(controls.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if _overlap(left, right):
                raise RuntimeError(f"{left_name} overlaps {right_name}")
    model = Path(handoff.artifact_path).resolve(strict=True)
    run_dir = model.parents[2]
    tokenizer = Path(plan.candidate.profile_tokenizer_path).resolve(strict=True)
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


def _operation_sha256(args: argparse.Namespace, plan: Glm52CapturePlan) -> str:
    payload = {
        "capture_plan_sha256": plan.plan_sha256,
        "compression_result": str(args.compression_result),
        "profile": str(args.profile),
        "work_root": str(args.work_root),
        "journal_dir": str(args.journal_dir),
        "maintenance_receipt": str(args.maintenance_receipt),
        "lease": str(args.lease),
        "result": str(args.result),
        "worker_ssh_target": _WORKER_SSH_TARGET,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _exclusive_json_at(parent: int, name: str, payload: object) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if Path(name).name != name:
        raise RuntimeError("publication name must be a single path component")
    temporary = f".{name}.tmp-{os.getpid()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent,
    )
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("capture result write was incomplete")
        os.fsync(descriptor)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent)
        raise
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
        try:
            os.fsync(parent)
        except BaseException:
            os.unlink(name, dir_fd=parent)
            os.fsync(parent)
            raise
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent)
    return digest


def _bounded_sha256(path: Path) -> str:
    return hashlib.sha256(_read_bounded_regular(path)).hexdigest()


def _bounded_sha256_at(parent: int, name: str) -> str:
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JSON_BYTES:
            raise RuntimeError("published JSON is not a bounded regular file")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, _CHUNK):
            digest.update(chunk)
            total += len(chunk)
        if total != info.st_size:
            raise RuntimeError("published JSON changed during verification")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _write_model_evidence(plan: Glm52CapturePlan, root_fd: int) -> None:
    path = Path(plan.model_evidence_path)
    encoded = json.dumps(
        plan.model_evidence.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != plan.model_evidence_sha256:
        raise RuntimeError("capture model evidence digest drifted")
    descriptor = os.open(
        path.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=root_fd,
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


def _prepare_runtime_libraries(root_fd: int) -> int:
    contract_bytes = _read_bounded_regular(CAPTURE_BUILD_CONTRACT)
    if hashlib.sha256(contract_bytes).hexdigest() != CAPTURE_BUILD_CONTRACT_SHA256:
        raise RuntimeError("capture build contract bytes drifted")
    contract = json.loads(contract_bytes)
    library_root = Path(contract["library_root"])
    libraries = contract["libraries"]
    library_sizes = contract.get("library_sizes")
    if (
        not isinstance(libraries, dict)
        or not libraries
        or not isinstance(library_sizes, dict)
        or set(library_sizes) != set(libraries)
    ):
        raise RuntimeError("capture library contract is invalid")
    source_root_fd = _open_directory_chain(library_root)
    os.mkdir("runtime-libs", 0o700, dir_fd=root_fd)
    library_fd = os.open(
        "runtime-libs",
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    try:
        for name, expected_sha in sorted(libraries.items()):
            expected_size = library_sizes.get(name)
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(expected_sha, str)
                or not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size <= 0
                or expected_size > _MAX_LIBRARY_BYTES
            ):
                raise RuntimeError("capture library contract is invalid")
            source = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=source_root_fd,
            )
            target = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o500,
                dir_fd=library_fd,
            )
            digest = hashlib.sha256()
            try:
                before = os.fstat(source)
                if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                    raise RuntimeError("capture library is not the contracted regular file")
                total = 0
                while total < expected_size:
                    chunk = os.read(source, min(_CHUNK, expected_size - total))
                    if not chunk:
                        raise RuntimeError("capture library ended before its contracted size")
                    digest.update(chunk)
                    if os.write(target, chunk) != len(chunk):
                        raise OSError("capture library copy was incomplete")
                    total += len(chunk)
                if os.read(source, 1):
                    raise RuntimeError("capture library exceeds its contracted size")
                after = os.fstat(source)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise RuntimeError("capture library changed during private copy")
                os.fsync(target)
                os.lseek(target, 0, os.SEEK_SET)
                target_digest = hashlib.sha256()
                target_total = 0
                while chunk := os.read(target, _CHUNK):
                    target_digest.update(chunk)
                    target_total += len(chunk)
            finally:
                os.close(source)
                os.close(target)
            if (
                digest.hexdigest() != expected_sha
                or target_digest.hexdigest() != expected_sha
                or target_total != expected_size
            ):
                raise RuntimeError("capture library bytes drifted during private copy")
            prefix, version = name.split(".so.", 1)
            soname = f"{prefix}.so.{version.split('.', 1)[0]}"
            basename = f"{prefix}.so"
            os.link(name, soname, src_dir_fd=library_fd, dst_dir_fd=library_fd)
            os.link(name, basename, src_dir_fd=library_fd, dst_dir_fd=library_fd)
        os.fsync(library_fd)
        os.fchmod(library_fd, 0o500)
        os.set_inheritable(library_fd, True)
        os.close(source_root_fd)
        return library_fd
    except BaseException:
        os.close(source_root_fd)
        os.close(library_fd)
        raise


def _run_native(argv: tuple[str, ...], *, library_fd: int) -> None:
    descriptor = os.open(
        argv[0],
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o111 == 0:
            raise RuntimeError("capture executable is not a regular executable")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _CHUNK):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            digest.hexdigest() != CAPTURE_BINARY_SHA256
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise RuntimeError("capture executable bytes differ from the reviewed pin")
        pid = os.fork()
        if pid == 0:  # pragma: no cover - exercised only by the real maintenance payload
            try:
                null = os.open(os.devnull, os.O_RDWR)
                os.dup2(null, 1)
                os.dup2(null, 2)
                environment = {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "LANG": "C",
                    "LD_LIBRARY_PATH": f"/proc/self/fd/{library_fd}",
                    "PATH": "/usr/bin:/bin",
                }
                os.execve(descriptor, argv, environment)
            except BaseException:
                os._exit(127)
        _, status = os.waitpid(pid, 0)
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise RuntimeError("native capture failed")
    finally:
        os.close(descriptor)


def _execute_payload(args: argparse.Namespace) -> int:
    if (
        not args.execute
        or not args.payload
        or not args.expected_plan_sha256
        or not args.expected_operation_sha256
    ):
        raise RuntimeError("capture payload identity is incomplete")
    plan, handoff = _plan(args)
    _validate_paths(args, plan, handoff)
    if plan.plan_sha256 != args.expected_plan_sha256:
        raise RuntimeError("capture plan drifted after maintenance preflight")
    operation_sha256 = _operation_sha256(args, plan)
    if operation_sha256 != args.expected_operation_sha256:
        raise RuntimeError("capture operation paths drifted after maintenance preflight")
    assert plan.plan_sha256 is not None
    binding = CanaryLeaseBinding(
        plan_sha256=operation_sha256,
        artifact_path=handoff.artifact_path,
        artifact_sha256=handoff.artifact_sha256,
        head_unit=HEAD_TRANSIENT_UNIT,
        worker_unit=WORKER_TRANSIENT_UNIT,
    )
    require_active_lease(args.lease, binding)
    MaintenanceCoordinator(
        MaintenanceConfig(
            journal_dir=args.journal_dir,
            receipt_path=args.maintenance_receipt,
            worker_ssh_target=_WORKER_SSH_TARGET,
        ),
        SubprocessCommandRunner(),
        execute=True,
    ).verify_drained()
    args.work_root.mkdir(mode=0o700)
    root_fd = _open_directory_chain(args.work_root)
    root_identity = _directory_identity(root_fd)
    result_parent_fd = _open_directory_chain(args.result.parent)
    _write_model_evidence(plan, root_fd)
    os.set_inheritable(root_fd, True)
    library_fd = _prepare_runtime_libraries(root_fd)
    root_anchor = Path(f"/proc/self/fd/{root_fd}")
    candidate_anchor = root_anchor / Path(plan.candidate.output_dir).name
    identity_anchor = root_anchor / Path(plan.identity_control.output_dir).name
    runtime = LlamaCppRpcRuntimeConfig(
        artifact_path=Path(handoff.artifact_path), artifact_sha256=handoff.artifact_sha256
    )
    lifecycle = WorkerRpcSystemdLifecycle(
        runtime, worker_ssh_target=_WORKER_SSH_TARGET, runner=_run_text
    )
    launch = lifecycle.start()
    try:
        _assert_directory_path_identity(args.work_root, root_identity)
        preflight_capture_request(plan.candidate)
        _assert_directory_path_identity(args.work_root, root_identity)
        candidate_argv = build_capture_argv(
            plan.candidate,
            common_argv=plan.common_argv,
            execution_output_dir=str(candidate_anchor),
        )
        _run_native(candidate_argv, library_fd=library_fd)
        _assert_directory_path_identity(args.work_root, root_identity)
        candidate = finalize_capture(plan.candidate, root_override=candidate_anchor)
        after_candidate = lifecycle.remeasure_after_capture()
        _assert_directory_path_identity(args.work_root, root_identity)
        preflight_capture_request(plan.identity_control)
        _assert_directory_path_identity(args.work_root, root_identity)
        identity_argv = build_capture_argv(
            plan.identity_control,
            common_argv=plan.common_argv,
            execution_output_dir=str(identity_anchor),
        )
        _run_native(identity_argv, library_fd=library_fd)
        _assert_directory_path_identity(args.work_root, root_identity)
        identity = finalize_capture(plan.identity_control, root_override=identity_anchor)
        after_identity = lifecycle.remeasure_after_capture()
    finally:
        lifecycle.stop()
        os.close(library_fd)
    _assert_directory_path_identity(args.work_root, root_identity)
    report = evaluate_capture_pair(
        reference_root=identity_anchor,
        reference=identity,
        candidate_root=candidate_anchor,
        candidate=candidate,
    )
    if report.identity_control_passed is not True:
        raise RuntimeError("capture identity control failed")
    report_path = args.work_root / "identity-metrics.json"
    report_sha = _exclusive_json_at(root_fd, report_path.name, report.model_dump(mode="json"))
    candidate_manifest_path = candidate_anchor / "capture-manifest.json"
    identity_manifest_path = identity_anchor / "capture-manifest.json"
    published_candidate_manifest = Path(plan.candidate.output_dir) / "capture-manifest.json"
    published_identity_manifest = Path(plan.identity_control.output_dir) / "capture-manifest.json"
    _assert_directory_path_identity(args.work_root, root_identity)
    candidate_manifest_sha = _bounded_sha256(candidate_manifest_path)
    identity_manifest_sha = _bounded_sha256(identity_manifest_path)
    result = {
        "schema_version": 1,
        "status": "completed",
        "quality_claim": False,
        "capture_plan_sha256": plan.plan_sha256,
        "capture_operation_sha256": operation_sha256,
        "compression_handoff_sha256": handoff.handoff_sha256,
        "worker_launch": launch.model_dump(mode="json"),
        "worker_after_candidate": after_candidate.model_dump(mode="json"),
        "worker_after_identity": after_identity.model_dump(mode="json"),
        "candidate_capture_id": candidate.capture_id,
        "candidate_manifest_path": str(published_candidate_manifest),
        "candidate_manifest_sha256": candidate_manifest_sha,
        "identity_capture_id": identity.capture_id,
        "identity_manifest_path": str(published_identity_manifest),
        "identity_manifest_sha256": identity_manifest_sha,
        "identity_control_passed": True,
        "metric_report_id": report.report_id,
        "metric_report_path": str(report_path),
        "metric_report_sha256": report_sha,
    }
    _exclusive_json_at(result_parent_fd, args.result.name, result)
    try:
        _assert_directory_path_identity(args.work_root, root_identity)
        if (
            _bounded_sha256_at(root_fd, report_path.name) != report_sha
            or _bounded_sha256(candidate_manifest_path) != candidate_manifest_sha
            or _bounded_sha256(identity_manifest_path) != identity_manifest_sha
        ):
            raise RuntimeError("capture publication inputs drifted")
        repeated = evaluate_capture_pair(
            reference_root=identity_anchor,
            reference=identity,
            candidate_root=candidate_anchor,
            candidate=candidate,
        )
        if repeated.report_id != report.report_id or repeated.identity_control_passed is not True:
            raise RuntimeError("capture metrics drifted during result publication")
        _assert_directory_path_identity(args.work_root, root_identity)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(args.result.name, dir_fd=result_parent_fd)
            os.fsync(result_parent_fd)
        os.close(root_fd)
        os.close(result_parent_fd)
        raise
    os.close(root_fd)
    os.close(result_parent_fd)
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
    parser.add_argument("--expected-operation-sha256", default="", help=argparse.SUPPRESS)
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
        "capture_operation_sha256": _operation_sha256(args, plan),
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
    operation_sha256 = _operation_sha256(args, plan)
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
        "--expected-operation-sha256",
        operation_sha256,
    )
    binding = CanaryLeaseBinding(
        plan_sha256=operation_sha256,
        artifact_path=handoff.artifact_path,
        artifact_sha256=handoff.artifact_sha256,
        head_unit=HEAD_TRANSIENT_UNIT,
        worker_unit=WORKER_TRANSIENT_UNIT,
    )
    previous = install_signal_traps()
    try:
        receipt = MaintenanceCoordinator(
            MaintenanceConfig(
                journal_dir=args.journal_dir,
                receipt_path=args.maintenance_receipt,
                worker_ssh_target=_WORKER_SSH_TARGET,
            ),
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
