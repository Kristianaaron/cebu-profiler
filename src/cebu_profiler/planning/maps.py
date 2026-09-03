"""Typed planning maps and derivative candidates (v2 §25).

Each map is a versioned, source-identity-preserving record. Source expert IDs
are never overwritten; predictions are separate from measured results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class KeepEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_model_id: str
    layer_index: int
    source_expert_id: int
    keep: bool
    reason: str  # saliency | protected_coalition | path_preserved | budget


class KeepMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_model_id: str
    checkpoint_revision: str | None = None
    entries: list[KeepEntry] = Field(default_factory=list)

    def kept(self, layer: int) -> list[int]:
        return sorted(e.source_expert_id for e in self.entries if e.layer_index == layer and e.keep)

    def kept_count(self) -> int:
        return sum(1 for e in self.entries if e.keep)


class PrecisionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    precision: str
    bits: float
    reconstruction_error: float


class PrecisionMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[PrecisionEntry] = Field(default_factory=list)


class ResidencyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    location: str  # node_a | node_b | nvme_a | nvme_b

    @property
    def is_resident(self) -> bool:
        return self.location in {"node_a", "node_b"}


class ResidencyMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[ResidencyEntry] = Field(default_factory=list)


class CoalitionProtectionMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protections: list[tuple[int, tuple[int, ...]]] = Field(
        default_factory=list
    )  # (layer, coalition)


class PathPreservationMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protected_paths: list[tuple[tuple[int, ...], ...]] = Field(default_factory=list)


class SubstituteEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    candidates: list[int]
    confidence: float  # candidate, not validated substitutability


class SubstituteMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[SubstituteEntry] = Field(default_factory=list)


class ChannelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    channel_id: int
    importance: float  # measured channel-importance proxy (uniqueness/TENP)
    keep: bool


class ChannelMap(BaseModel):
    """§25 channel map: which channels each expert retains (v2 §18 grain)."""

    model_config = ConfigDict(extra="forbid")

    entries: list[ChannelEntry] = Field(default_factory=list)


class TileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    tile_index: int
    channel_start: int
    importance: float
    keep: bool


class TileMap(BaseModel):
    """§25 tile map: block granularity over the expert's channel dim."""

    model_config = ConfigDict(extra="forbid")

    entries: list[TileEntry] = Field(default_factory=list)


class NodeOwnershipEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_key: str
    role: str
    layer_index: int | None = None
    source_expert_id: int | None = None
    node: str  # node_a | node_b | nvme_a | nvme_b | replicated


class NodeOwnershipMap(BaseModel):
    """§25 node-ownership map: which physical node each tensor lives on."""

    model_config = ConfigDict(extra="forbid")

    entries: list[NodeOwnershipEntry] = Field(default_factory=list)


class OverflowPackEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    tier: str  # nvme_a | nvme_b (stored but not resident)
    reason: str


class OverflowPackMap(BaseModel):
    """§25 overflow-pack map: experts stored on the NVMe tier (non-resident)."""

    model_config = ConfigDict(extra="forbid")

    entries: list[OverflowPackEntry] = Field(default_factory=list)


class RouterRepairEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    old_index: int
    new_index: int | None  # None => dropped
    action: str  # keep | drop
    route_bias: bool  # correction bias must move in lockstep (v2 §31:18)


class RouterRepairMap(BaseModel):
    """§25 router-repair map: reindex router slots exactly with renumbering."""

    model_config = ConfigDict(extra="forbid")

    entries: list[RouterRepairEntry] = Field(default_factory=list)


class ResidualRepairEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    component: str  # residual_bias | expert_output | routing_bias
    severity: float
    target: str  # what to repair (e.g. "bias", "distill")


class ResidualRepairMap(BaseModel):
    """§25 residual-repair map: residuals needing bias/behaviour repair."""

    model_config = ConfigDict(extra="forbid")

    entries: list[ResidualRepairEntry] = Field(default_factory=list)


class DistillationTargetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_index: int
    source_expert_id: int
    target_type: str  # expert | lane | layer
    priority: float


class DistillationTargetMap(BaseModel):
    """§25 distillation-target map: high-value components kept for KD/student."""

    model_config = ConfigDict(extra="forbid")

    entries: list[DistillationTargetEntry] = Field(default_factory=list)


class CandidatePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    keep: KeepMap
    precision: PrecisionMap
    residency: ResidencyMap
    coalition_protection: CoalitionProtectionMap
    path_preservation: PathPreservationMap
    substitutes: SubstituteMap
    kept_per_layer: dict[int, int] = Field(default_factory=dict)
    resident_bytes_a: float = 0.0
    resident_bytes_b: float = 0.0
    stored_bytes: float = 0.0
    active_bytes_per_token: float = 0.0
    protected_coalitions_kept: int = 0
    protected_paths_kept: int = 0
    fitted: bool = False  # within node budgets
