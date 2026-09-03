"""Structural fallback planner (v3 %11 / blueprint §3.2, TENP/FlexMoE).

Only after a no-pruning Pareto frontier exists, generate 5/10/15/20% selective
intra-expert reductions and re-run allocation/evaluation. Preserve all routing
destinations where possible, and block destructive actions on
INSUFFICIENT_EVIDENCE regions (evidence gate). Returns a set of folding-fallback
candidate budgets — never a default-prune result.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.schemas.coverage import CapacityCoverage, EvidenceGate
from model_atlas.schemas.evidence import EvidenceKind


class StructuralFallbackPlan(BaseModel):
    """One selective intra-expert reduction plan (5-20%)."""

    model_config = ConfigDict(extra="forbid")

    reduction_percent: float = Field(ge=0.0, le=100.0)
    retained_channels: dict[tuple[int, int], int] = Field(default_factory=dict)
    blocked_capacity: list[str] = Field(default_factory=list)  # INSUFFICIENT_EVIDENCE ids
    preserved_routing_destinations: bool = True
    evidence_kind: EvidenceKind = EvidenceKind.PREDICTED


def structural_fallback_plans(
    widths: dict[tuple[int, int], int],
    *,
    coverage: dict[str, CapacityCoverage],
    gate: EvidenceGate | None = None,
    reductions: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
) -> list[StructuralFallbackPlan]:
    """Generate nested fallback reduction plans with evidence gating.

    ``widths`` maps (layer, expert) -> current full width. For each reduction
    level, planned width = round(full * (1 - r)). Capacity units whose coverage
    is INSUFFICIENT_EVIDENCE are never reduced (blocked) unless overridden.
    """
    gate = gate or EvidenceGate()
    plans: list[StructuralFallbackPlan] = []
    for r in reductions:
        retained: dict[tuple[int, int], int] = {}
        blocked: list[str] = []
        for (layer, e), full in widths.items():
            cap = coverage.get(f"{layer}:{e}")
            can, _why = gate.allow(cap) if cap else (True, "")
            if cap and not can:
                blocked.append(cap.capacity_id)
                retained[(layer, e)] = full  # never reduce under-observed
                continue
            new_w = int(round(full * (1.0 - r)))
            retained[(layer, e)] = max(1, new_w)
        plans.append(
            StructuralFallbackPlan(
                reduction_percent=round(r * 100.0, 1),
                retained_channels=retained,
                blocked_capacity=sorted(blocked),
                preserved_routing_destinations=True,
            )
        )
    return plans
