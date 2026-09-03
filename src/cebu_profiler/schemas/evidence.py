"""Evidence grades, uncertainty, and negative controls (v2 §20, §31:4/20).

Every interpretation must carry an evidence grade, a production kind
(measured vs estimated/predicted/inferred), and uncertainty. A causal- or
behavioural-grade claim may never be backed by an inference — that enforces the
"correlation is not causation" and "predictions are not measured results" rules.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class EvidenceLevel(StrEnum):
    """Analysis depth (v2): Basic = correlation, not causation."""

    BASIC_SALIENCY = "basic_saliency"
    ENHANCED_PROFILER = "enhanced_profiler"
    CAUSAL_PROFILER = "causal_profiler"


class EvidenceKind(StrEnum):
    """How a claim was produced; measured is never implied by inference."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    PREDICTED = "predicted"
    INFERRED = "inferred"
    CAUSALLY_TESTED = "causally_tested"


class EvidenceGrade(StrEnum):
    OBSERVED_ASSOCIATION = "observed_association"
    SEED_STABLE_ASSOCIATION = "seed_stable_association"
    HELD_OUT_ASSOCIATION = "held_out_association"
    LOCAL_INTERVENTION_EFFECT = "local_intervention_effect"
    DOWNSTREAM_CAUSAL_EFFECT = "downstream_causal_effect"
    HELD_OUT_BEHAVIOURAL_EFFECT = "held_out_behavioural_effect"


class CausalValidation(StrEnum):
    NOT_RUN = "not_run"
    CORRELATED = "correlated"
    LOCALLY_CAUSAL = "locally_causal"
    DOWNSTREAM_CAUSAL = "downstream_causal"
    HELD_OUT_BEHAVIOURAL = "held_out_behavioural"


class NegativeControlKind(StrEnum):
    RANDOM_EXPERT_MASK = "random_expert_mask"
    RANDOM_QUANTIZATION = "random_quantization"
    FREQUENCY_MATCHED = "frequency_matched"
    LABEL_SHUFFLED = "label_shuffled"
    PROMPT_PARAPHRASE = "prompt_paraphrase"
    ALTERNATE_CORPORA = "alternate_corpora"
    ROUTE_PRESERVING_QUANT = "route_preserving_quantization"
    ROUTE_CHANGING_QUANT = "route_changing_quantization"
    RANDOM_COALITION = "random_coalition"


class NegativeControlRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: NegativeControlKind
    control_id: str | None = None
    passed: bool | None = None
    note: str | None = None


# Grades that assert a causal / behavioural effect (require direct testing).
_CAUSAL_GRADES = frozenset(
    {
        EvidenceGrade.LOCAL_INTERVENTION_EFFECT,
        EvidenceGrade.DOWNSTREAM_CAUSAL_EFFECT,
        EvidenceGrade.HELD_OUT_BEHAVIOURAL_EFFECT,
    }
)
# Kinds that are NOT direct measurement / causal testing.
_NON_MEASURED_KINDS = frozenset(
    {EvidenceKind.ESTIMATED, EvidenceKind.PREDICTED, EvidenceKind.INFERRED}
)


class Uncertainty(BaseModel):
    """Supporting evidence / confidence for a claim."""

    model_config = ConfigDict(extra="forbid")

    sample_count: int | None = Field(default=None, ge=0)
    token_count: int | None = Field(default=None, ge=0)
    task_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    label_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    seed_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    corpus_resample_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_interval: tuple[float, float] | None = None
    known_confounds: list[str] = Field(default_factory=list)
    negative_controls: list[NegativeControlRecord] = Field(default_factory=list)
    causal_validation_status: CausalValidation = CausalValidation.NOT_RUN


class EvidenceClaim(BaseModel):
    """One interpretation tied to evidence. Never present an inference as causal."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    grade: EvidenceGrade
    kind: EvidenceKind
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)

    @field_validator("kind")
    @classmethod
    def _causal_grade_requires_direct_kind(
        cls, v: EvidenceKind, info: ValidationInfo
    ) -> EvidenceKind:
        grade = info.data.get("grade")
        if grade in _CAUSAL_GRADES and v in _NON_MEASURED_KINDS:
            raise ValueError(
                f"causal/behavioural grade {grade!s} cannot be backed by {v!s}; "
                "require measured or causally_tested"
            )
        return v


def is_direct_kind(kind: EvidenceKind) -> bool:
    """A claim is directly measured/causal-tested, not estimated/predicted/inferred."""
    return kind not in _NON_MEASURED_KINDS
