"""Tensor ownership: which tensor lives where, with source identity preserved.

Ownership is the backbone of the Atlas's "full picture of experts." Every real
tensor maps to exactly one role (invariant: no unclassified tensors) and one
physical location. Source expert IDs are never conflated with candidate/local
slot IDs or physical locations.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_atlas.schemas.architecture import DTYPE_BYTES, DType, TensorRole


class PhysicalLocation(StrEnum):
    NODE_A = "node_a"
    NODE_B = "node_b"
    NVME_A = "nvme_a"
    NVME_B = "nvme_b"
    REPLICATED = "replicated"


class PlacementPolicy(StrEnum):
    """How tensors are assigned to physical locations for planning."""

    EXPERT_PARALLEL = "expert_parallel"  # routed experts split across A/B
    REPLICATE_SHARED_ON_A = "replicate_shared_on_a"  # + non-expert on node A


class TensorOwnership(BaseModel):
    """One tensor's identity and home. Immutable record."""

    model_config = ConfigDict(extra="forbid")

    key: str  # canonical, globally unique tensor name
    role: TensorRole
    dtype: DType
    numel: int
    layer_index: int | None = None  # None for global tensors (embedding, lm_head)
    expert_index: int | None = None  # None unless a routed/shared expert slot
    location: PhysicalLocation = PhysicalLocation.NODE_A
    copied_from_source: bool = True  # True = byte-copied source; False = synthetic

    @property
    def bytes(self) -> float:
        return self.numel * DTYPE_BYTES[self.dtype]


class OwnershipManifest(BaseModel):
    """The result of a census: every tensor accounted for, none unclassified."""

    model_config = ConfigDict(extra="forbid")

    architecture: str
    records: list[TensorOwnership] = Field(default_factory=list)
    status: str  # "measured" | "synthetic" | "needs_source_measurement"

    # ---- validation (invariants) ----

    @model_validator(mode="after")
    def _check_keys_unique(self) -> OwnershipManifest:
        keys = [r.key for r in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate tensor keys in ownership manifest")
        return self

    # ---- derived facts ----

    def total_bytes(self) -> float:
        return sum(r.bytes for r in self.records)

    def bytes_by_node(self) -> dict[PhysicalLocation, float]:
        acc: dict[PhysicalLocation, float] = {loc: 0.0 for loc in PhysicalLocation}
        for r in self.records:
            acc[r.location] += r.bytes
        return acc

    def bytes_by_role(self) -> dict[TensorRole, float]:
        acc: dict[TensorRole, float] = {}
        for r in self.records:
            acc[r.role] = acc.get(r.role, 0.0) + r.bytes
        return acc
