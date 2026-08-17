"""Bounded measured KLD/CKA over validated llama capture artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_atlas.evaluation.cka import centered_linear_cka
from model_atlas.evaluation.contracts import (
    EvidenceKind,
    MetricEvidence,
    TokenKLDResult,
    TokenKLDRow,
)
from model_atlas.evaluation.kld import build_domain_report
from model_atlas.evaluation.llamacpp_capture import (
    CAPTURE_ADAPTER_VERSION,
    AlignmentRow,
    CaptureArtifact,
    CaptureManifest,
    CapturePair,
    validate_capture_pair,
)

CAPTURE_METRICS_VERSION = "atlas-capture-metrics-v1"
_MAX_ALIGNMENT_BYTES = 64 * 1024 * 1024
_MAX_METRIC_ROWS = 4096
_MAX_DOMAINS = 128
_MAX_VOCAB = 262_144
_MAX_CKA_ROWS = 128
_MAX_CKA_CELLS = 1_000_000
_IDENTITY_KLD_TOL = 1e-12
_IDENTITY_CKA_TOL = 1e-12
_CHUNK = 4 * 1024 * 1024


class CaptureMetricError(RuntimeError):
    """Capture metrics cannot be produced without weakening evidence."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class LayerCKAResult(_Frozen):
    layer: int = Field(ge=0)
    observation_count: int = Field(ge=2)
    score: float = Field(ge=-1.0, le=1.0)
    evidence: MetricEvidence


class CaptureMetricReport(_Frozen):
    schema_version: Literal[1] = 1
    report_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    pair: CapturePair
    temperature: float = Field(gt=0.0)
    kld: TokenKLDResult
    layer_cka: tuple[LayerCKAResult, ...]
    identity_control_passed: bool | None

    @model_validator(mode="after")
    def _identity(self) -> CaptureMetricReport:
        payload = self.model_dump(mode="json", exclude={"report_id"})
        digest = _canonical_sha256(payload)
        if self.report_id is not None and self.report_id != digest:
            raise ValueError("capture metric report_id does not match content")
        object.__setattr__(self, "report_id", digest)
        return self


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact(manifest: CaptureManifest, name: str) -> CaptureArtifact:
    matches = [artifact for artifact in manifest.artifacts if artifact.name == name]
    if len(matches) != 1:
        raise CaptureMetricError(f"capture manifest has no unique {name} artifact")
    return matches[0]


def _open_verified(root: Path, artifact: CaptureArtifact) -> int:
    is_fd_anchor = (
        len(root.parts) == 6
        and root.parts[:4] == ("/", "proc", "self", "fd")
        and root.parts[4].isdecimal()
    )
    if is_fd_anchor:
        parent_fd = os.dup(int(root.parts[4]))
        try:
            root_fd = os.open(
                root.name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)
    else:
        if not root.is_absolute() or root.is_symlink() or root != root.resolve(strict=True):
            raise CaptureMetricError("capture root must be canonical and symlink-free")
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(
            artifact.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
    finally:
        os.close(root_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size != artifact.size_bytes:
        os.close(descriptor)
        raise CaptureMetricError(f"{artifact.name} size/type differs from manifest")
    return descriptor


def _verify_descriptor(
    descriptor: int, artifact: CaptureArtifact
) -> tuple[int, int, int, int, int]:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    total = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, _CHUNK):
        digest.update(chunk)
        total += len(chunk)
    after = os.fstat(descriptor)

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        total != artifact.size_bytes
        or digest.hexdigest() != artifact.sha256
        or identity(before) != identity(after)
    ):
        raise CaptureMetricError(f"{artifact.name} bytes differ from capture manifest")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return identity(after)


def _read_alignment(root: Path, manifest: CaptureManifest) -> list[AlignmentRow]:
    artifact = _artifact(manifest, "alignment.jsonl")
    if artifact.size_bytes > _MAX_ALIGNMENT_BYTES:
        raise CaptureMetricError("capture alignment exceeds bounded read limit")
    descriptor = _open_verified(root, artifact)
    try:
        _verify_descriptor(descriptor, artifact)
        encoded = os.read(descriptor, artifact.size_bytes + 1)
    finally:
        os.close(descriptor)
    if len(encoded) != artifact.size_bytes:
        raise CaptureMetricError("capture alignment short read")
    if hashlib.sha256(encoded).hexdigest() != artifact.sha256:
        raise CaptureMetricError("capture alignment changed during read")
    lines = encoded.splitlines()
    if len(lines) > _MAX_METRIC_ROWS:
        raise CaptureMetricError("capture row count exceeds the metric safety limit")
    try:
        rows = [AlignmentRow.model_validate_json(line) for line in lines]
    except ValueError as exc:
        raise CaptureMetricError("capture alignment schema is invalid") from exc
    if len(rows) != manifest.row_count:
        raise CaptureMetricError("capture alignment count differs from manifest")
    if len({row.domain for row in rows}) > _MAX_DOMAINS:
        raise CaptureMetricError("capture domain count exceeds the metric safety limit")
    return rows


def _read_f32_row(descriptor: int, width: int) -> list[float]:
    size = width * 4
    encoded = bytearray()
    while len(encoded) < size:
        chunk = os.read(descriptor, size - len(encoded))
        if not chunk:
            raise CaptureMetricError("capture FP32 tensor ended mid-row")
        encoded.extend(chunk)
    values = list(struct.unpack(f"<{width}f", encoded))
    if any(not math.isfinite(value) for value in values):
        raise CaptureMetricError("capture FP32 row contains non-finite values")
    return values


def full_vocab_kld_row(
    reference: list[float], candidate: list[float], temperature: float = 1.0
) -> float:
    """Stable exact ``KL(reference || candidate)`` for one full-vocabulary row."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CaptureMetricError("temperature must be finite and positive")
    if len(reference) != len(candidate) or len(reference) < 2:
        raise CaptureMetricError("capture logits row shapes differ")
    ref_scaled = [value / temperature for value in reference]
    cand_scaled = [value / temperature for value in candidate]
    ref_max = max(ref_scaled)
    cand_max = max(cand_scaled)
    ref_log_z = ref_max + math.log(sum(math.exp(value - ref_max) for value in ref_scaled))
    cand_log_z = cand_max + math.log(sum(math.exp(value - cand_max) for value in cand_scaled))
    result = sum(
        math.exp(ref_value - ref_log_z) * ((ref_value - ref_log_z) - (cand_value - cand_log_z))
        for ref_value, cand_value in zip(ref_scaled, cand_scaled, strict=True)
    )
    if not math.isfinite(result) or result < -_IDENTITY_KLD_TOL:
        raise CaptureMetricError("capture KLD row is invalid")
    return max(0.0, result)


def _input_evidence_digest(
    pair: CapturePair,
    reference: CaptureManifest,
    candidate: CaptureManifest,
    temperature: float,
    cka_rows: int,
) -> str:
    return _canonical_sha256(
        {
            "pair": pair.model_dump(mode="json"),
            "reference": reference.capture_id,
            "candidate": candidate.capture_id,
            "temperature": temperature,
            "cka_rows": cka_rows,
            "producer": CAPTURE_METRICS_VERSION,
        }
    )


def evaluate_capture_pair(
    *,
    reference_root: Path,
    reference: CaptureManifest,
    candidate_root: Path,
    candidate: CaptureManifest,
    temperature: float = 1.0,
    cka_rows: int = 128,
) -> CaptureMetricReport:
    """Stream exact full-vocabulary KLD and bounded exact linear CKA."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CaptureMetricError("temperature must be finite and positive")
    if cka_rows < 2 or cka_rows > _MAX_CKA_ROWS:
        raise CaptureMetricError("cka_rows must be in [2, 128]")
    pair = validate_capture_pair(reference, candidate)
    if reference.row_count > _MAX_METRIC_ROWS or candidate.row_count > _MAX_METRIC_ROWS:
        raise CaptureMetricError("capture row count exceeds the metric safety limit")
    if reference.vocab_size > _MAX_VOCAB or candidate.vocab_size > _MAX_VOCAB:
        raise CaptureMetricError("capture vocabulary exceeds the metric safety limit")
    reference_alignment = _read_alignment(reference_root, reference)
    candidate_alignment = _read_alignment(candidate_root, candidate)
    if reference_alignment != candidate_alignment:
        raise CaptureMetricError("capture alignment rows differ")
    evidence_digest = _input_evidence_digest(pair, reference, candidate, temperature, cka_rows)

    ref_logits_artifact = _artifact(reference, "logits.f32")
    cand_logits_artifact = _artifact(candidate, "logits.f32")
    ref_logits = _open_verified(reference_root, ref_logits_artifact)
    cand_logits = _open_verified(candidate_root, cand_logits_artifact)
    rows: list[TokenKLDRow] = []
    try:
        ref_logits_identity = _verify_descriptor(ref_logits, ref_logits_artifact)
        cand_logits_identity = _verify_descriptor(cand_logits, cand_logits_artifact)
        for alignment in reference_alignment:
            value = full_vocab_kld_row(
                _read_f32_row(ref_logits, reference.vocab_size),
                _read_f32_row(cand_logits, candidate.vocab_size),
                temperature,
            )
            rows.append(
                TokenKLDRow(
                    sample_id=alignment.sample_id,
                    position=alignment.input_position,
                    token_id=alignment.target_token_id,
                    kld=value,
                    domain=alignment.domain,
                )
            )
        if _verify_descriptor(ref_logits, ref_logits_artifact) != ref_logits_identity:
            raise CaptureMetricError("reference logits identity changed during evaluation")
        if _verify_descriptor(cand_logits, cand_logits_artifact) != cand_logits_identity:
            raise CaptureMetricError("candidate logits identity changed during evaluation")
    finally:
        os.close(ref_logits)
        os.close(cand_logits)
    report = build_domain_report(rows)
    mean_kld = report.overall.token_weighted_mean
    kld = TokenKLDResult(
        sample_ids=list(dict.fromkeys(row.sample_id for row in rows)),
        rows=rows,
        report=report,
        evidence=MetricEvidence(
            kind=EvidenceKind.MEASURED,
            value=mean_kld,
            artifact_digest=evidence_digest,
            producer=CAPTURE_METRICS_VERSION,
            producer_version=CAPTURE_ADAPTER_VERSION,
        ),
    )

    observations = min(reference.row_count, cka_rows)
    if observations < 2:
        raise CaptureMetricError("CKA requires at least two captured observations")
    if observations * max(reference.hidden_size, candidate.hidden_size) > _MAX_CKA_CELLS:
        raise CaptureMetricError("CKA observation cells exceed the memory safety limit")
    layer_results: list[LayerCKAResult] = []
    for layer in pair.layers:
        name = f"layer-{layer:03d}.f32"
        ref_artifact = _artifact(reference, name)
        cand_artifact = _artifact(candidate, name)
        ref_fd = _open_verified(reference_root, ref_artifact)
        cand_fd = _open_verified(candidate_root, cand_artifact)
        try:
            ref_identity = _verify_descriptor(ref_fd, ref_artifact)
            cand_identity = _verify_descriptor(cand_fd, cand_artifact)
            ref_matrix = [_read_f32_row(ref_fd, reference.hidden_size) for _ in range(observations)]
            cand_matrix = [
                _read_f32_row(cand_fd, candidate.hidden_size) for _ in range(observations)
            ]
            if _verify_descriptor(ref_fd, ref_artifact) != ref_identity:
                raise CaptureMetricError(f"reference layer {layer} changed during evaluation")
            if _verify_descriptor(cand_fd, cand_artifact) != cand_identity:
                raise CaptureMetricError(f"candidate layer {layer} changed during evaluation")
        finally:
            os.close(ref_fd)
            os.close(cand_fd)
        measured = centered_linear_cka(ref_matrix, cand_matrix)
        if not measured.valid or measured.score is None:
            raise CaptureMetricError(f"layer {layer} CKA is blocked: {measured.reason}")
        layer_results.append(
            LayerCKAResult(
                layer=layer,
                observation_count=observations,
                score=measured.score,
                evidence=MetricEvidence(
                    kind=EvidenceKind.MEASURED,
                    value=measured.score,
                    artifact_digest=evidence_digest,
                    producer=CAPTURE_METRICS_VERSION,
                    producer_version=CAPTURE_ADAPTER_VERSION,
                ),
            )
        )

    identity_passed: bool | None = None
    if pair.identity_control:
        exact_artifacts = ref_logits_artifact.sha256 == cand_logits_artifact.sha256 and all(
            _artifact(reference, f"layer-{layer:03d}.f32").sha256
            == _artifact(candidate, f"layer-{layer:03d}.f32").sha256
            for layer in pair.layers
        )
        identity_passed = (
            exact_artifacts
            and max((row.kld for row in rows), default=0.0) <= _IDENTITY_KLD_TOL
            and all(abs(result.score - 1.0) <= _IDENTITY_CKA_TOL for result in layer_results)
        )
        if not identity_passed:
            raise CaptureMetricError("identity-control KLD/CKA did not match exact expectations")
    return CaptureMetricReport(
        pair=pair,
        temperature=temperature,
        kld=kld,
        layer_cka=tuple(layer_results),
        identity_control_passed=identity_passed,
    )


__all__ = [
    "CAPTURE_METRICS_VERSION",
    "CaptureMetricError",
    "LayerCKAResult",
    "CaptureMetricReport",
    "full_vocab_kld_row",
    "evaluate_capture_pair",
]
