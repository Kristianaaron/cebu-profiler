"""Strict, versioned evaluation and reproducibility contracts.

Backend-independent contracts for held-out evaluation, metric evidence
typing, token/domain KLD, layer/router divergence, and reproducible metric
runs. Schemas use ``extra="forbid"`` so unknown fields are rejected, match the
existing Atlas evidence conventions (measured kinds never implied), and
pinned identities are derived only from content hashes — never from
timestamps.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class EvidenceKind(StrEnum):
    """How a metric value was produced. Measured is never implied by inference."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    PREDICTED = "predicted"
    INFERRED = "inferred"


# Kinds that do NOT assert direct measurement and thus need no artifact digest.
_NON_MEASURED_KINDS = frozenset(
    {EvidenceKind.ESTIMATED, EvidenceKind.PREDICTED, EvidenceKind.INFERRED}
)


class MetricEvidence(BaseModel):
    """Typed evidence for a metric. MEASURED requires an immutable artifact digest."""

    model_config = ConfigDict(extra="forbid")

    kind: EvidenceKind
    value: float
    # Immutable artifact digest + provenance. REQUIRED when kind == MEASURED.
    artifact_digest: str | None = None
    producer: str | None = None
    producer_version: str | None = None

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        # pydantic v2 rejects inf/nan for float fields by default when
        # `allow_inf_nan=False`; keep an explicit guard for clarity.
        import math

        if not math.isfinite(v):
            raise ValueError("metric value must be finite")
        return v

    @model_validator(mode="after")
    def _measured_requires_artifact(self) -> MetricEvidence:
        if self.kind == EvidenceKind.MEASURED:
            if not self.artifact_digest:
                raise ValueError(
                    "MEASURED evidence requires artifact_digest, producer, "
                    "producer_version"
                )
            if not self.producer:
                raise ValueError("MEASURED evidence requires producer")
            if not self.producer_version:
                raise ValueError("MEASURED evidence requires producer_version")
        return self


class CorpusSlice(BaseModel):
    """Pinned held-out corpus slice used by an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    manifest_hash: str
    held_out_partition: str
    ordered_sample_id_hash: str
    tokenizer_hash: str
    template_hash: str | None = None
    n_samples: int = Field(gt=0, description="must be a nonempty held-out corpus")


class EvaluationIdentity(BaseModel):
    """Stable identity pinning teacher and candidate model."""

    model_config = ConfigDict(extra="forbid")

    teacher_id: str
    teacher_hash: str | None = None
    candidate_id: str
    candidate_hash: str | None = None
    teacher_relative: bool = True


class TokenKLDRow(BaseModel):
    """Per-token KLD(teacher || candidate) with forced-position alignment.

    ``(sample_id, position)`` uniquely identify a row within a run. ``domain``
    is ``"unknown"`` when no explicit per-token domain mapping was supplied, so
    unknown domains are always explicit.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    kld: float = Field(ge=0.0)
    masked: bool = False
    domain: str = "unknown"


class DomainKLDAggregate(BaseModel):
    """Token-weighted KLD aggregate over one domain."""

    model_config = ConfigDict(extra="forbid")

    domain: str  # "unknown" is explicit when no domain is known
    n_tokens: int = Field(ge=0)
    token_weighted_mean: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    p99: float = Field(ge=0.0)
    max: float = Field(ge=0.0)


class DomainKLDReport(BaseModel):
    """Token-weighted aggregates per domain plus a token-weighted overall."""

    model_config = ConfigDict(extra="forbid")

    overall: DomainKLDAggregate
    by_domain: list[DomainKLDAggregate] = Field(default_factory=list)


class LayerDivergence(BaseModel):
    """Divergence of one layer's weights/activations between models."""

    model_config = ConfigDict(extra="forbid")

    layer_index: int = Field(ge=0)
    layer_type: str
    metric: str
    value: float = Field(ge=0.0)
    evidence: MetricEvidence


class RouterDivergenceRecord(BaseModel):
    """One router's divergence between teacher and candidate routing."""

    model_config = ConfigDict(extra="forbid")

    layer_index: int = Field(ge=0)
    expert_count: int = Field(gt=0)
    matched_tokens: int = Field(ge=0)
    route_agreement: float = Field(ge=0.0, le=1.0)
    kl_divergence: float = Field(ge=0.0)


class RouterDivergenceSummary(BaseModel):
    """Full router divergence summary across layers."""

    model_config = ConfigDict(extra="forbid")

    records: list[RouterDivergenceRecord] = Field(default_factory=list)
    matched_tokens_total: int = Field(ge=0)
    overall_agreement: float = Field(ge=0.0, le=1.0)


class ReproducibilityManifest(BaseModel):
    """Pins every input that determines a metric run's output.

    Timestamps are intentionally absent from identity fields so that a rerun
    of the identical configuration yields byte-identical canonical identity.
    """

    model_config = ConfigDict(extra="forbid")

    manifest_version: int = 1
    source_manifest_hash: str
    candidate_manifest_hash: str
    corpus_hash: str
    tokenizer_hash: str
    template_hash: str | None = None
    config_hash: str
    harness_revision: str
    adapter_version: str
    backend_version: str
    seed: int
    dtype: str
    device: str
    topology: str
    argv: list[str] = Field(default_factory=list)
    input_hash: str
    output_hash: str | None = None


class EvaluationReport(BaseModel):
    """A complete, reproducible evaluation report."""

    model_config = ConfigDict(extra="forbid")

    report_version: int = 1
    identity: EvaluationIdentity
    corpus: CorpusSlice
    reproducibility: ReproducibilityManifest
    kld: DomainKLDReport | None = None
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
    "EvidenceKind",
    "MetricEvidence",
    "CorpusSlice",
    "EvaluationIdentity",
    "TokenKLDRow",
    "DomainKLDAggregate",
    "DomainKLDReport",
    "LayerDivergence",
    "RouterDivergenceRecord",
    "RouterDivergenceSummary",
    "ReproducibilityManifest",
    "EvaluationReport",
    "identity_digest",
    "canonical_evaluation_identity",
]
