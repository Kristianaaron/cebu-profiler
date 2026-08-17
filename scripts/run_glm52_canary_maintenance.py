#!/usr/bin/env python3
"""The sole real-execution entry point: drain, lease, canary, restore."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple

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

_COMPRESSION_METHOD = "llamacpp-gguf-mixed"
_MAX_RESULT_BYTES = 16 * 1024 * 1024
_RESERVED_CANARY_OPTIONS = {
    "--execute",
    "--maintenance-lease",
    "--artifact",
    "--artifact-sha256",
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


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"compression result lacks {key}")
    return value


def _read_bounded_regular(path: Path, limit: int = _MAX_RESULT_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise RuntimeError("compression result must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
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
        if len(payload) > limit or len(payload) != before.st_size:
            raise RuntimeError("compression result exceeds its bounded read")
        if identity_before != identity_after:
            raise RuntimeError("compression result changed during read")
        return payload
    finally:
        os.close(descriptor)


def _open_directory_chain(path: Path) -> int:
    if not path.is_absolute():
        raise RuntimeError("runtime artifact run directory must be absolute")
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


def _descriptor_identity(descriptor: int) -> tuple[str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("runtime artifact must be a regular file")
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, 4 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
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
    ) or size != before.st_size:
        raise RuntimeError("runtime artifact changed during verification")
    return digest.hexdigest(), size


class _CompressionHandoff(NamedTuple):
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    evidence_sha256: str
    evidence_size_bytes: int
    evidence_relpath: str
    producer_run_id: str
    producer_plan_id: str
    producer_recipe_sha256: str
    producer_profile_id: str
    producer_recommendation_id: str
    handoff_sha256: str


def _artifact_from_compression_result(path: Path) -> _CompressionHandoff:
    payload = json.loads(_read_bounded_regular(path))
    if not isinstance(payload, dict):
        raise RuntimeError("compression result must be a JSON object")
    if payload.get("status") not in {"completed", "completed_with_warnings"}:
        raise RuntimeError("compression result is not successful")
    if payload.get("method") != _COMPRESSION_METHOD:
        raise RuntimeError("compression result method is not the reviewed producer")
    if payload.get("runtime_claim") != "artifact_only_unvalidated":
        raise RuntimeError("compression result runtime claim is not artifact-only")
    artifact = payload.get("runtime_artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError("compression result lacks a runtime artifact")
    raw_path = artifact.get("path")
    sha256 = artifact.get("sha256")
    size = artifact.get("size_bytes")
    relpath = artifact.get("relpath")
    artifact_evidence = artifact.get("evidence")
    producer_run_id = _required_string(payload, "run_id")
    producer_plan_id = _required_string(payload, "plan_id")
    producer_recipe_sha256 = _required_string(payload, "recipe_sha256")
    producer_profile_id = _required_string(payload, "profile_id")
    producer_recommendation_id = _required_string(payload, "recommendation_id")
    lineage = {
        "producer_run_id": producer_run_id,
        "producer_plan_id": producer_plan_id,
        "producer_recipe_sha256": producer_recipe_sha256,
        "producer_profile_id": producer_profile_id,
        "producer_recommendation_id": producer_recommendation_id,
    }
    if (
        artifact.get("stage") != _COMPRESSION_METHOD
        or artifact.get("logical_name") != "model.gguf"
        or artifact.get("runtime_validated") is not False
        or not isinstance(raw_path, str)
        or not raw_path.startswith("/")
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not isinstance(artifact_evidence, dict)
    ):
        raise RuntimeError("compression runtime artifact contract is malformed")
    evidence_sha = artifact_evidence.get("sha256")
    evidence_size = artifact_evidence.get("size_bytes")
    evidence_relpath = artifact_evidence.get("relpath")
    if (
        artifact_evidence.get("stage") != _COMPRESSION_METHOD
        or artifact_evidence.get("logical_name") != f"{_COMPRESSION_METHOD}.evidence.json"
        or not isinstance(evidence_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence_sha) is None
        or not isinstance(evidence_size, int)
        or isinstance(evidence_size, bool)
        or evidence_size <= 0
        or not isinstance(evidence_relpath, str)
    ):
        raise RuntimeError("compression evidence artifact contract is malformed")
    artifact_path = Path(raw_path)
    expected_relative = Path("objects") / sha256[:2] / f"{sha256}.blob"
    expected_evidence_relative = Path("objects") / evidence_sha[:2] / f"{evidence_sha}.blob"
    if relpath != str(expected_relative) or len(artifact_path.parents) < 3:
        raise RuntimeError("compression runtime artifact is not in its canonical CAS slot")
    if evidence_relpath != str(expected_evidence_relative):
        raise RuntimeError("compression evidence artifact is not in its canonical CAS slot")
    run_dir = artifact_path.parents[2]
    run_id = producer_run_id
    if (
        run_dir.name != run_id
        or run_dir.parent.name != "runs"
        or artifact_path != run_dir / expected_relative
    ):
        raise RuntimeError("compression runtime artifact path disagrees with run lineage")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("run_id") != run_id:
        raise RuntimeError("compression output metadata disagrees with run lineage")
    raw_outputs = outputs.get("outputs")
    if not isinstance(raw_outputs, list) or len(raw_outputs) != 2:
        raise RuntimeError("compression output set differs from the reviewed stage contract")
    model_hits = [
        item
        for item in raw_outputs
        if isinstance(item, dict)
        and item.get("stage") == _COMPRESSION_METHOD
        and item.get("name") == "model.gguf"
        and item.get("sha256") == sha256
        and item.get("size_bytes") == size
        and item.get("relpath") == relpath
    ]
    evidence_hits = [
        item
        for item in raw_outputs
        if isinstance(item, dict)
        and item.get("stage") == _COMPRESSION_METHOD
        and item.get("name") == f"{_COMPRESSION_METHOD}.evidence.json"
        and item.get("sha256") == evidence_sha
        and item.get("size_bytes") == evidence_size
        and item.get("relpath") == evidence_relpath
    ]
    if len(model_hits) != 1 or len(evidence_hits) != 1:
        raise RuntimeError("compression output set is not the reviewed model/evidence pair")
    descriptor = _open_cas_slot(run_dir, sha256)
    try:
        measured_sha, measured_size = _descriptor_identity(descriptor)
    finally:
        os.close(descriptor)
    if measured_sha != sha256 or measured_size != size:
        raise RuntimeError("compression runtime artifact bytes drifted")
    evidence_descriptor = _open_cas_slot(run_dir, evidence_sha)
    try:
        measured_evidence_sha, measured_evidence_size = _descriptor_identity(evidence_descriptor)
    finally:
        os.close(evidence_descriptor)
    if measured_evidence_sha != evidence_sha or measured_evidence_size != evidence_size:
        raise RuntimeError("compression evidence artifact bytes drifted")
    handoff_payload = {
        **lineage,
        "artifact_path": str(artifact_path),
        "artifact_sha256": measured_sha,
        "artifact_size_bytes": measured_size,
        "artifact_relpath": relpath,
        "evidence_sha256": evidence_sha,
        "evidence_size_bytes": evidence_size,
        "evidence_relpath": evidence_relpath,
    }
    encoded = json.dumps(handoff_payload, sort_keys=True, separators=(",", ":")).encode()
    return _CompressionHandoff(
        artifact_path=str(artifact_path),
        artifact_sha256=measured_sha,
        artifact_size_bytes=measured_size,
        evidence_sha256=evidence_sha,
        evidence_size_bytes=evidence_size,
        evidence_relpath=evidence_relpath,
        producer_run_id=producer_run_id,
        producer_plan_id=producer_plan_id,
        producer_recipe_sha256=producer_recipe_sha256,
        producer_profile_id=producer_profile_id,
        producer_recommendation_id=producer_recommendation_id,
        handoff_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
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
        handoff = _artifact_from_compression_result(args.compression_result)
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
            producer_handoff_sha256=(
                handoff.handoff_sha256 if handoff is not None else None
            ),
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
    print(receipt.model_dump_json())
    return 0 if receipt.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
