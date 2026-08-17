"""Durable, content-bound handoff for a measured two-Spark canary."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_atlas.fit_telemetry import CanaryPlan, StopReason
from model_atlas.schemas.evidence import EvidenceKind
from model_atlas.two_node_canary_executor import CanaryExecutionResult

_MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_MAX_RECORDS = 4096
_UNIQUE_EVIDENCE_RECORDS = {
    "canary_plan",
    "fit_summary",
    "runtime_validation_claim",
    "canary_execution_receipt",
}


class CanaryHandoffError(RuntimeError):
    """A durable canary result or its evidence failed verification."""


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _open_parent(path: Path) -> int:
    if (
        not path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(component in {"", ".", ".."} for component in path.parts[1:])
    ):
        raise CanaryHandoffError("canary handoff paths must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parent.parts[1:]:
            following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_bounded(path: Path, limit: int) -> bytes:
    parent = _open_parent(path)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > limit:
            raise CanaryHandoffError("canary handoff input must be a bounded regular file")
        digest = bytearray()
        while len(digest) <= limit:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - len(digest)))
            if not chunk:
                break
            digest.extend(chunk)
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
        if (
            len(digest) != before.st_size
            or len(digest) > limit
            or identity_before != identity_after
        ):
            raise CanaryHandoffError("canary handoff input changed during read")
        return bytes(digest)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _evidence_records(encoded: bytes) -> dict[str, dict[str, Any]]:
    lines = encoded.splitlines()
    if not lines or len(lines) > _MAX_RECORDS:
        raise CanaryHandoffError("canary evidence record count is invalid")
    records: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CanaryHandoffError("canary evidence JSONL is invalid") from exc
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or not isinstance(record.get("record_type"), str)
            or not isinstance(record.get("payload"), dict)
        ):
            raise CanaryHandoffError("canary evidence record schema is invalid")
        record_type = record["record_type"]
        if record_type in records and record_type in _UNIQUE_EVIDENCE_RECORDS:
            raise CanaryHandoffError("canary evidence record types must be unique")
        if record_type in _UNIQUE_EVIDENCE_RECORDS:
            records[record_type] = record["payload"]
    return records


class RuntimeCanaryHandoff(BaseModel):
    """Immutable result that can authorize later candidate evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    handoff_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    plan: CanaryPlan
    execution: CanaryExecutionResult
    evidence_path: str = Field(pattern=r"^/")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_size_bytes: int = Field(gt=0)
    validated_for_evaluation: bool

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"handoff_sha256"})

    @model_validator(mode="after")
    def _bind_execution(self) -> RuntimeCanaryHandoff:
        plan_sha = self.plan.canonical_sha256()
        receipt = self.execution.receipt
        summary = self.execution.summary
        if receipt.plan_sha256 != plan_sha or summary.plan_sha256 != plan_sha:
            raise ValueError("canary execution is bound to a different plan")
        if summary.candidate != self.plan.candidate:
            raise ValueError("canary summary candidate differs from the plan")
        completed = tuple(receipt.completed_step_ids)
        expected_steps = tuple(step.step_id for step in self.plan.steps)
        valid = (
            receipt.runtime_claim_validated
            and receipt.evidence_kind is EvidenceKind.MEASURED
            and summary.evidence_kind is EvidenceKind.MEASURED
            and summary.both_nodes_measured
            and summary.fitted
            and completed == expected_steps
            and receipt.stop_reason is StopReason.COMPLETED
            and summary.stop_reason is StopReason.COMPLETED
        )
        if self.validated_for_evaluation != valid:
            raise ValueError("validated_for_evaluation differs from measured canary evidence")
        expected = _canonical_digest(self.identity_payload())
        if self.handoff_sha256 is not None and self.handoff_sha256 != expected:
            raise ValueError("canary handoff digest differs from canonical content")
        object.__setattr__(self, "handoff_sha256", expected)
        return self


def _verify_evidence(
    plan: CanaryPlan,
    execution: CanaryExecutionResult,
    *,
    evidence_path: Path,
) -> tuple[str, int]:
    encoded = _read_bounded(evidence_path, _MAX_EVIDENCE_BYTES)
    records = _evidence_records(encoded)
    required = {
        "canary_plan": plan.model_dump(mode="json"),
        "fit_summary": execution.summary.model_dump(mode="json"),
        "canary_execution_receipt": execution.receipt.model_dump(mode="json"),
    }
    for name, expected in required.items():
        if records.get(name) != expected:
            raise CanaryHandoffError(f"canary evidence is missing or mismatches {name}")
    claim = records.get("runtime_validation_claim")
    if (
        not isinstance(claim, dict)
        or claim.get("validated") is not execution.receipt.runtime_claim_validated
        or claim.get("reason") != execution.receipt.runtime_claim_reason
    ):
        raise CanaryHandoffError("runtime claim evidence differs from the receipt")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _exclusive_publish(path: Path, encoded: bytes) -> None:
    if len(encoded) > _MAX_RESULT_BYTES:
        raise CanaryHandoffError("canary handoff result exceeds its size bound")
    parent = _open_parent(path)
    temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = -1
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        published = True
        os.fsync(parent)
        os.unlink(temporary, dir_fd=parent)
        os.fsync(parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent)
        if published:
            try:
                os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
            except OSError:
                pass
        raise
    finally:
        os.close(parent)


def publish_runtime_canary_handoff(
    output_path: Path,
    *,
    plan: CanaryPlan,
    execution: CanaryExecutionResult,
    evidence_path: Path,
) -> RuntimeCanaryHandoff:
    if output_path == evidence_path:
        raise CanaryHandoffError("canary result and evidence paths must differ")
    evidence_sha256, evidence_size = _verify_evidence(plan, execution, evidence_path=evidence_path)
    receipt = execution.receipt
    summary = execution.summary
    completed = tuple(receipt.completed_step_ids)
    expected_steps = tuple(step.step_id for step in plan.steps)
    validated = (
        receipt.runtime_claim_validated
        and receipt.evidence_kind is EvidenceKind.MEASURED
        and summary.evidence_kind is EvidenceKind.MEASURED
        and summary.both_nodes_measured
        and summary.fitted
        and completed == expected_steps
        and receipt.stop_reason is StopReason.COMPLETED
        and summary.stop_reason is StopReason.COMPLETED
    )
    handoff = RuntimeCanaryHandoff(
        plan=plan,
        execution=execution,
        evidence_path=str(evidence_path),
        evidence_sha256=evidence_sha256,
        evidence_size_bytes=evidence_size,
        validated_for_evaluation=validated,
    )
    encoded = (handoff.model_dump_json(indent=2) + "\n").encode()
    _exclusive_publish(output_path, encoded)
    persisted = load_verified_runtime_canary_handoff(output_path, expected_plan=plan)
    if persisted != handoff:
        raise CanaryHandoffError("published canary handoff differs from verified content")
    return handoff


def load_verified_runtime_canary_handoff(
    path: Path,
    *,
    expected_plan: CanaryPlan | None = None,
    require_evaluation_ready: bool = False,
) -> RuntimeCanaryHandoff:
    encoded = _read_bounded(path, _MAX_RESULT_BYTES)
    try:
        handoff = RuntimeCanaryHandoff.model_validate_json(encoded)
    except ValueError as exc:
        raise CanaryHandoffError("canary handoff schema or digest is invalid") from exc
    if expected_plan is not None and handoff.plan != expected_plan:
        raise CanaryHandoffError("canary handoff differs from the expected plan")
    evidence_path = Path(handoff.evidence_path)
    digest, size = _verify_evidence(handoff.plan, handoff.execution, evidence_path=evidence_path)
    if digest != handoff.evidence_sha256 or size != handoff.evidence_size_bytes:
        raise CanaryHandoffError("canary evidence bytes drifted from the handoff")
    if require_evaluation_ready and not handoff.validated_for_evaluation:
        raise CanaryHandoffError("canary did not produce an evaluation-ready runtime claim")
    return handoff


__all__ = [
    "CanaryHandoffError",
    "RuntimeCanaryHandoff",
    "load_verified_runtime_canary_handoff",
    "publish_runtime_canary_handoff",
]
