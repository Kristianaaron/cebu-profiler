"""Atlas trace schema (v2 §8, §11) with typed per-family payloads.

A single trace record links one observation to its identity (task, sample,
suite, labels, stage, token, mode, success), to the source model/layer/expert,
and to a typed family payload. Full representation tensors are never embedded —
they are stored under the artifact store and referenced here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_atlas.schemas.evidence import EvidenceClaim
from model_atlas.schemas.ontology import (
    CapabilityLabel,
    DataPartition,
    GenerationMode,
    InterventionType,
    SuccessState,
    TraceFamily,
    TrajectoryStage,
)


class RepresentationStorage(StrEnum):
    """Storage granularity for representation samples (v2 §11)."""

    FULL = "full"
    FP16 = "fp16"
    FP8 = "fp8"
    RANDOM_PROJECTION = "random_projection"
    PCA = "pca"
    STATISTICS_ONLY = "statistics_only"
    PRINCIPAL_DIRECTIONS = "principal_directions"
    SPARSE_FEATURE_ACTIVATIONS = "sparse_feature_activations"


class RoutedSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["routing"] = "routing"
    selected_expert_ids: list[int]
    router_logits: list[float]
    router_probabilities: list[float]
    top_k_order: list[int] = Field(default_factory=list)
    routing_margin: float | None = None
    routing_entropy: float | None = Field(default=None, ge=0.0)


class Contribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["contribution"] = "contribution"
    expert_output_norm: float | None = None
    router_weighted_output: float | None = None
    combined_moe_output: float | None = None
    post_norm_effect: float | None = None
    residual_delta: float | None = None
    downstream_logit_effect: float | None = None


class Representation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["representation"] = "representation"
    storage: RepresentationStorage = RepresentationStorage.STATISTICS_ONLY
    layer_input: str | None = None
    post_attention: str | None = None
    pre_moe: str | None = None
    post_moe: str | None = None
    post_norm: str | None = None
    post_residual: str | None = None
    kda_summary: str | None = None
    mla_summary: str | None = None
    final_hidden: str | None = None
    selected_logits: str | None = None


class Intervention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["intervention"] = "intervention"
    intervention_type: InterventionType
    reference_output: float | None = None
    intervention_output: float | None = None
    hidden_state_delta: float | None = None
    post_norm_delta: float | None = None
    logit_kl: float | None = Field(default=None, ge=0.0)
    task_score_delta: float | None = None
    runtime_delta: float | None = None
    memory_delta: float | None = None


TracePayload = Annotated[
    RoutedSelection | Contribution | Representation | Intervention,
    Field(discriminator="kind"),
]


class AtlasTrace(BaseModel):
    """One trace record with full identity + typed payload (v2 §8)."""

    model_config = ConfigDict(extra="forbid")

    atlas_run_id: str
    trace_schema_version: str = "1.0"
    family: TraceFamily

    task_id: str | None = None
    sample_id: str | None = None
    suite_id: str | None = None
    data_partition: DataPartition = DataPartition.ATLAS_CALIBRATION
    capability_labels: list[CapabilityLabel] = Field(default_factory=list)
    trajectory_stage: TrajectoryStage | None = None
    token_index: int | None = Field(default=None, ge=0)
    token_range: tuple[int, int] | None = None

    generation_mode: GenerationMode = GenerationMode.TEACHER_FORCED
    success_state: SuccessState = SuccessState.UNKNOWN

    source_model_id: str
    checkpoint_revision: str | None = None
    layer_index: int | None = Field(default=None, ge=0)
    source_expert_id: int | None = Field(default=None, ge=0)

    payload: TracePayload
    evidence: EvidenceClaim | None = None

    @model_validator(mode="after")
    def _family_matches_payload(self) -> AtlasTrace:
        if self.payload.kind != self.family.value:
            raise ValueError(f"family {self.family!s} != payload kind {self.payload.kind}")
        return self

    @model_validator(mode="after")
    def _intervention_needs_layer(self) -> AtlasTrace:
        if self.payload.kind == "intervention" and self.layer_index is None:
            raise ValueError("intervention trace requires layer_index")
        return self
