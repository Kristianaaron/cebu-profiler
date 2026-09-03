"""Byte-accurate memory planner with per-node go/no-go.

Separates, per the invariants:
  - stored bytes (all tensor bytes regardless of location);
  - resident bytes per node (Node A/B own tensors + replicated tensors;
    NVMe tier is stored but NOT resident);
  - active expert bytes per token (an estimate of the MoE read per token).

A plan is `unsafe` when either node's resident bytes plus the runtime reserve
exceed that node's budget. Predictions are separate from measured results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from model_atlas.census.tensor_ownership import OwnershipManifest, PhysicalLocation
from model_atlas.schemas.architecture import DTYPE_BYTES, ArchitectureSpec, TensorRole

GIB = 1024**3


class PlanAssessment(BaseModel):
    node_a_resident_bytes: float
    node_b_resident_bytes: float
    stored_bytes: float
    active_expert_bytes_per_token: float
    safe: bool
    failures: list[str] = Field(default_factory=list)

    def node_a_resident_gib(self) -> float:
        return self.node_a_resident_bytes / GIB

    def node_b_resident_gib(self) -> float:
        return self.node_b_resident_bytes / GIB


def resident_bytes_by_node(
    manifest: OwnershipManifest,
) -> dict[PhysicalLocation, float]:
    """Resident bytes on each node: own tensors + replicated (counted on both)."""
    loc = manifest.bytes_by_node()
    return {
        PhysicalLocation.NODE_A: loc[PhysicalLocation.NODE_A] + loc[PhysicalLocation.REPLICATED],
        PhysicalLocation.NODE_B: loc[PhysicalLocation.NODE_B] + loc[PhysicalLocation.REPLICATED],
    }


def active_expert_bytes_per_token(arch: ArchitectureSpec) -> float:
    """Estimated expert bytes read per generated token (routing-driven)."""
    if TensorRole.EXPERTS not in arch.tensor_params:
        return 0.0
    per_expert = arch.tensor_params[TensorRole.EXPERTS] * DTYPE_BYTES[arch.moe.expert_dtype]
    return arch.num_text_layers * arch.moe.top_k * per_expert


def assess(
    arch: ArchitectureSpec,
    manifest: OwnershipManifest,
    *,
    budget_a_gb: float,
    budget_b_gb: float,
    runtime_reserve_gb: float = 30.0,
) -> PlanAssessment:
    """Assess a placement against per-node resident budgets (go/no-go)."""
    if arch.needs_source_measurement:
        return PlanAssessment(
            node_a_resident_bytes=0.0,
            node_b_resident_bytes=0.0,
            stored_bytes=0.0,
            active_expert_bytes_per_token=0.0,
            safe=False,
            failures=["architecture has no measured tensor sizes; cannot plan"],
        )

    by_node = resident_bytes_by_node(manifest)
    a, b = by_node[PhysicalLocation.NODE_A], by_node[PhysicalLocation.NODE_B]
    budget_a = budget_a_gb * GIB
    budget_b = budget_b_gb * GIB
    reserve = runtime_reserve_gb * GIB

    failures: list[str] = []
    if a + reserve > budget_a:
        failures.append(
            f"node A resident {a / GIB:.2f} GiB + reserve {runtime_reserve_gb:.1f} GiB "
            f"> budget {budget_a_gb:.1f} GiB"
        )
    if b + reserve > budget_b:
        failures.append(
            f"node B resident {b / GIB:.2f} GiB + reserve {runtime_reserve_gb:.1f} GiB "
            f"> budget {budget_b_gb:.1f} GiB"
        )

    return PlanAssessment(
        node_a_resident_bytes=a,
        node_b_resident_bytes=b,
        stored_bytes=manifest.total_bytes(),
        active_expert_bytes_per_token=active_expert_bytes_per_token(arch),
        safe=not failures,
        failures=failures,
    )
