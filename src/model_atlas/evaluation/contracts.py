"""Strict, versioned evaluation and reproducibility contracts.

Backend-independent contracts for held-out evaluation, metric evidence
typing, token/domain KLD, layer/router divergence, and reproducible metric
runs. Schemas use ``extra="forbid"`` so unknown fields are rejected, reject
NaN and both infinities for every numeric metric, match the existing Atlas
evidence conventions (measured kinds never implied), and pinned identities
are derived only from content hashes — never from timestamps.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Exactly the schema/report/manifest versions this contract supports. Anything
# else (including future "999") is rejected at parse time.
_SUPPORTED_SCHEMA_VERSION = 1
_SUPPORTED_REPORT_VERSION = 1
_SUPPORTED_MANIFEST_VERSION = 1

# SHA-256 hex digests: exactly 64 lowercase hex characters.
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _validate_version(v: int, supported: int, label: str) -> int:
    if v != supported:
        raise ValueError(
            f"{label} must be exactly {supported} (got {v}); unsupported version"
        )
    return v


class _StrictModel(BaseModel):
    """Shared strict base: no unknown fields, no NaN/inf on any numeric field."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class EvidenceKind(StrEnum):
    """How a metric value was produced. Measured is never implied by inference."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    PREDICTED = "predicted"
    INFERRED = "inferred"


class MetricEvidence(_StrictModel):
    """Typed evidence for a metric. MEASURED requires an immutable artifact digest.

    Any measured-looking metric requires a nonempty artifact digest, producer,
    and producer version. Estimated/predicted/inferred kinds are allowed to
    omit the digest because they do not assert direct measurement.
    """

    kind: EvidenceKind
    value: float
    # Immutable artifact digest + provenance. REQUIRED nonempty when
    # kind == MEASURED.
    artifact_digest: str | None = None
    producer: str | None = None
    producer_version: str | None = None

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("metric value must be finite")
        return v

    @field_validator("artifact_digest", "producer", "producer_version")
    @classmethod
    def _nonempty_when_set(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError("evidence provenance fields must be nonempty")
        return v

    @model_validator(mode="after")
    def _measured_requires_artifact(self) -> MetricEvidence:
        if self.kind == EvidenceKind.MEASURED:
            for name, val in (
                ("artifact_digest", self.artifact_digest),
                ("producer", self.producer),
                ("producer_version", self.producer_version),
            ):
                if not val:
                    raise ValueError(
                        f"MEASURED evidence requires nonempty {name}"
                    )
        return self


class CorpusSlice(_StrictModel):
    """Pinned held-out corpus slice used by an evaluation run."""

    manifest_hash: str = Field(pattern=_SHA256_PATTERN)
    held_out_partition: str = Field(min_length=1)
    ordered_sample_id_hash: str = Field(pattern=_SHA256_PATTERN)
    tokenizer_hash: str = Field(pattern=_SHA256_PATTERN)
    template_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    n_samples: int = Field(gt=0, description="must be a nonempty held-out corpus")


class EvaluationIdentity(_StrictModel):
    """Stable identity pinning teacher and candidate model."""

    teacher_id: str = Field(min_length=1)
    teacher_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidate_id: str = Field(min_length=1)
    candidate_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    teacher_relative: bool = True


@dataclass(frozen=True)
class SampleAlignment:
    """Ordered per-sample alignment identity for one batch member.

    ``sample_id`` is the batch-bound identity; ``positions`` are the valid
    token positions within the sample's sequence (ascending, unique), and
    ``token_ids`` are the vocabulary token ids at those positions. Token
    positions reset per sample. Used for both teacher and candidate so that
    mismatched alignment is rejected before any math.
    """

    sample_id: str
    positions: tuple[int, ...]
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.token_ids):
            raise ValueError(
                "positions and token_ids must have equal length per sample"
            )
        if len(self.positions) != len(set(self.positions)):
            raise ValueError("positions must be unique within a sample")
        if list(self.positions) != sorted(self.positions):
            raise ValueError("positions must be sorted ascending within a sample")
        if any(p < 0 for p in self.positions):
            raise ValueError("positions must be non-negative")
        if any(t < 0 for t in self.token_ids):
            raise ValueError("token_ids must be non-negative")


def _alignment_key(a: SampleAlignment) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (a.sample_id, a.positions, a.token_ids)


class TokenKLDRow(_StrictModel):
    """Per-token KLD(teacher || candidate) with forced-position alignment.

    ``(sample_id, position)`` uniquely identify a row within a run. ``domain``
    is ``"unknown"`` when no explicit per-sample domain mapping was supplied,
    so unknown domains are always explicit.
    """

    sample_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    kld: float = Field(ge=0.0)
    masked: bool = False
    domain: str = Field(default="unknown", min_length=1)


class DomainKLDAggregate(_StrictModel):
    """Token-weighted KLD aggregate over one domain."""

    domain: str = Field(min_length=1)  # "unknown" is explicit when no domain is known
    n_tokens: int = Field(ge=0)
    token_weighted_mean: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    p99: float = Field(ge=0.0)
    max: float = Field(ge=0.0)


class DomainKLDReport(_StrictModel):
    """Token-weighted aggregates per domain plus a token-weighted overall."""

    overall: DomainKLDAggregate
    by_domain: list[DomainKLDAggregate] = Field(default_factory=list)


class TokenKLDResult(_StrictModel):
    """Complete token-KLD result family: alignment, rows, and aggregates.

    Rows carry strict ``(sample_id, position)`` identity, and the whole result
    is bound to immutable MetricEvidence (artifact digest + producer +
    producer version) so a measured-looking result cannot exist unprovenanced.
    """

    sample_ids: list[str] = Field(min_length=1)
    rows: list[TokenKLDRow] = Field(default_factory=list)
    report: DomainKLDReport
    evidence: MetricEvidence

    @model_validator(mode="after")
    def _validate_identity(self) -> TokenKLDResult:
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("sample_ids must be unique")
        seen: set[tuple[str, int]] = set()
        for row in self.rows:
            key = (row.sample_id, row.position)
            if key in seen:
                raise ValueError(
                    f"duplicate TokenKLDRow identity (sample_id, position)={key!r}"
                )
            seen.add(key)
            if row.sample_id not in set(self.sample_ids):
                raise ValueError(
                    f"row sample_id {row.sample_id!r} not in sample_ids"
                )
        return self


class LayerDivergence(_StrictModel):
    """Divergence of one layer's weights/activations between models."""

    layer_index: int = Field(ge=0)
    layer_type: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float = Field(ge=0.0)
    evidence: MetricEvidence


class RouterDivergenceRecord(_StrictModel):
    """One router's divergence between teacher and candidate routing."""

    layer_index: int = Field(ge=0)
    expert_count: int = Field(gt=0)
    matched_tokens: int = Field(ge=0)
    route_agreement: float = Field(ge=0.0, le=1.0)
    kl_divergence: float = Field(ge=0.0)


class RouterDivergenceSummary(_StrictModel):
    """Full router divergence summary across layers, bound to evidence."""

    records: list[RouterDivergenceRecord] = Field(default_factory=list)
    matched_tokens_total: int = Field(ge=0)
    overall_agreement: float = Field(ge=0.0, le=1.0)
    evidence: MetricEvidence


class ReproducibilityManifest(_StrictModel):
    """Pins every input that determines a metric run's output.

    Timestamps are intentionally absent from identity fields so that a rerun
    of the identical configuration yields byte-identical canonical identity.
    """

    manifest_version: int = Field(default=1)

    @field_validator("manifest_version")
    @classmethod
    def _manifest_version_exact(cls, v: int) -> int:
        return _validate_version(v, _SUPPORTED_MANIFEST_VERSION, "manifest_version")

    source_manifest_hash: str = Field(pattern=_SHA256_PATTERN)
    candidate_manifest_hash: str = Field(pattern=_SHA256_PATTERN)
    corpus_hash: str = Field(pattern=_SHA256_PATTERN)
    tokenizer_hash: str = Field(pattern=_SHA256_PATTERN)
    template_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    config_hash: str = Field(pattern=_SHA256_PATTERN)
    harness_revision: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    seed: int
    dtype: str = Field(min_length=1)
    device: str = Field(min_length=1)
    topology: str = Field(min_length=1)
    argv: list[str] = Field(default_factory=list)
    input_hash: str = Field(pattern=_SHA256_PATTERN)
    output_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class EvaluationReport(_StrictModel):
    """A complete, reproducible evaluation report."""

    schema_version: int = Field(default=1)
    report_version: int = Field(default=1)

    @field_validator("schema_version")
    @classmethod
    def _schema_version_exact(cls, v: int) -> int:
        return _validate_version(v, _SUPPORTED_SCHEMA_VERSION, "schema_version")

    @field_validator("report_version")
    @classmethod
    def _report_version_exact(cls, v: int) -> int:
        return _validate_version(v, _SUPPORTED_REPORT_VERSION, "report_version")

    identity: EvaluationIdentity
    corpus: CorpusSlice
    reproducibility: ReproducibilityManifest
    kld: TokenKLDResult | None = None
    layer_divergence: list[LayerDivergence] = Field(default_factory=list)
    router_summary: RouterDivergenceSummary | None = None


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity_digest(model: BaseModel) -> str:
    """Canonical content digest of a contract model.

    Deterministic, sorted-key JSON of the model's fields; timestamps are never
    part of any identity model, so a rerun of identical content hashes to the
    same digest.
    """
    return _sha256_hex(
        json.dumps(
            model.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def canonical_evaluation_identity(report: EvaluationReport) -> str:
    """Stable identity of an evaluation: identity + corpus + reproducibility
    pins only — never the metric results or timestamps."""
    payload = {
        "identity": report.identity.model_dump(),
        "corpus": report.corpus.model_dump(),
        "reproducibility": report.reproducibility.model_dump(),
    }
    return _sha256_hex(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


__all__ = [
    "_SUPPORTED_SCHEMA_VERSION",
    "_SUPPORTED_REPORT_VERSION",
    "_SUPPORTED_MANIFEST_VERSION",
    "EvidenceKind",
    "MetricEvidence",
    "CorpusSlice",
    "EvaluationIdentity",
    "SampleAlignment",
    "TokenKLDRow",
    "DomainKLDAggregate",
    "DomainKLDReport",
    "TokenKLDResult",
    "LayerDivergence",
    "RouterDivergenceRecord",
    "RouterDivergenceSummary",
    "ReproducibilityManifest",
    "EvaluationReport",
    "identity_digest",
    "canonical_evaluation_identity",
]
