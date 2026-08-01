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
