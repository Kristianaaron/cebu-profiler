"""V3 Pareto engine: frontier, knee region, neighbor deltas (blueprint %7, %10.1).

Dominance postulates a candidate dominates another when no worse in every
active objective and strictly better in at least one. We expose the nondominated
frontier, dominated points are kept for provenance, the knee is a *scored
region* (not a single magical point), and selecting a candidate shows the cost
of moving one step toward fidelity and one toward compactness (marginal
quality/GB, tok/s/GB, context/GB).

Measured and predicted points are never conflated: predictions carry an
explicit badge and can never be marked deployable.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.schemas.evidence import EvidenceKind


class ParetoObjective(StrEnum):
    """Active dimensions of the frontier (v3 %10 objectives)."""

    QUALITY = "quality"  # maximize
    DECODE_TPS = "decode_tps"  # maximize
    PREFILL_TPS = "prefill_tps"  # maximize
    CONTEXT = "context"  # maximize (safe context)
    MTP_ACCEPTANCE = "mtp_acceptance"  # maximize
    RESIDENT_GIB = "resident_gib"  # minimize
    PEAK_RUNTIME_GIB = "peak_runtime_gib"  # minimize
    COMM_BYTES_PER_TOKEN = "comm_bytes_per_token"  # minimize


class FrontierPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    values: dict[str, float] = Field(default_factory=dict)  # objective -> value
    frontier: bool = False
    dominated_by: list[str] = Field(default_factory=list)
    knee_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED


class NeighborDelta(BaseModel):
    """Cost of moving one step to an adjacent frontier neighbor."""

    model_config = ConfigDict(extra="forbid")

    direction: str  # fidelity | compact | context
    candidate_id: str
    dquality: float = 0.0
    dresident_gib: float = 0.0
    ddecode_tps: float = 0.0
    dcontext: float = 0.0  # context tokens delta
    quality_per_gib: float = 0.0  # dquality / dGB (sign-aware)
    tok_per_gib: float = 0.0


class ParetoAnalysis(BaseModel):
    """The computed frontier + knee region + neighbor deltas."""

    model_config = ConfigDict(extra="forbid")

    objectives: list[str] = Field(default_factory=list)
    points: list[FrontierPoint] = Field(default_factory=list)
    frontier_ids: list[str] = Field(default_factory=list)
    knee_region: list[str] = Field(default_factory=list)
    neighbor_deltas: dict[str, list[NeighborDelta]] = Field(default_factory=dict)


_OBJECTIVE_DIRECTIONS: dict[str, str] = {
    ParetoObjective.QUALITY.value: "max",
    ParetoObjective.DECODE_TPS.value: "max",
    ParetoObjective.PREFILL_TPS.value: "max",
    ParetoObjective.CONTEXT.value: "max",
    ParetoObjective.MTP_ACCEPTANCE.value: "max",
    ParetoObjective.RESIDENT_GIB.value: "min",
    ParetoObjective.PEAK_RUNTIME_GIB.value: "min",
    ParetoObjective.COMM_BYTES_PER_TOKEN.value: "min",
}


def _dominates(a: dict[str, float], b: dict[str, float], objectives: list[str]) -> bool:
    any_strict = False
    for o in objectives:
        direction = _OBJECTIVE_DIRECTIONS.get(o, "max")
        av, bv = a.get(o, 0.0), b.get(o, 0.0)
        if direction == "max":
            if av < bv:
                return False
            if av > bv:
                any_strict = True
        else:
            if av > bv:
                return False
            if av < bv:
                any_strict = True
    return any_strict


def restrict_frontier(
    points: list[FrontierPoint],
    objectives: list[str] | None = None,
) -> ParetoAnalysis:
    """Compute nondominated frontier over the active objectives."""
    obs = objectives or [
        ParetoObjective.QUALITY.value,
        ParetoObjective.RESIDENT_GIB.value,
    ]
    # missing objective values default to "worse" so partial candidates never
    # sneak onto the frontier without measured evidence on all active axes.
    for p in points:
        for o in obs:
            p.values.setdefault(o, 0.0 if _OBJECTIVE_DIRECTIONS.get(o, "max") == "max" else 1e18)

    dominated_by: dict[str, list[str]] = {p.candidate_id: [] for p in points}
    for a in points:
        for b in points:
            if a.candidate_id == b.candidate_id:
                continue
            if _dominates(a.values, b.values, obs):
                dominated_by[b.candidate_id].append(a.candidate_id)

    frontier_ids = sorted(p.candidate_id for p in points if not dominated_by[p.candidate_id])
    for p in points:
        p.frontier = p.candidate_id in frontier_ids
        p.dominated_by = dominated_by[p.candidate_id]

    # neighborhood along the frontier for neighbor deltas (compact vs fidelity):
    # sort frontier by resident_gib ascending.
    _resident = ParetoObjective.RESIDENT_GIB.value
    frontier = sorted((p for p in points if p.frontier), key=lambda p: p.values[_resident])
    neighbor_deltas: dict[str, list[NeighborDelta]] = {p.candidate_id: [] for p in points}
    for i, p in enumerate(frontier):
        q = p.values.get(ParetoObjective.QUALITY.value, 0.0)
        r = p.values.get(ParetoObjective.RESIDENT_GIB.value, 0.0)
        d = p.values.get(ParetoObjective.DECODE_TPS.value, 0.0)
        c = p.values.get(ParetoObjective.CONTEXT.value, 0.0)
        if i > 0:
            n = frontier[i - 1]
            dq = q - n.values.get(ParetoObjective.QUALITY.value, 0.0)
            dr = r - n.values.get(ParetoObjective.RESIDENT_GIB.value, 0.0)
            neighbor_deltas[p.candidate_id].append(
                NeighborDelta(
                    direction="fidelity",
                    candidate_id=n.candidate_id,
                    dquality=round(dq, 5),
                    dresident_gib=round(dr, 3),
                    ddecode_tps=round(d - n.values.get(ParetoObjective.DECODE_TPS.value, 0.0), 3),
                    dcontext=round(c - n.values.get(ParetoObjective.CONTEXT.value, 0.0), 0),
                    quality_per_gib=round(dq / dr, 5) if dr != 0 else 0.0,
                )
            )
        if i < len(frontier) - 1:
            n = frontier[i + 1]
            dq = n.values.get(ParetoObjective.QUALITY.value, 0.0) - q
            dr = n.values.get(ParetoObjective.RESIDENT_GIB.value, 0.0) - r
            neighbor_deltas[p.candidate_id].append(
                NeighborDelta(
                    direction="compact",
                    candidate_id=n.candidate_id,
                    dquality=round(dq, 5),
                    dresident_gib=round(dr, 3),
                    ddecode_tps=round(n.values.get(ParetoObjective.DECODE_TPS.value, 0.0) - d, 3),
                    dcontext=round(n.values.get(ParetoObjective.CONTEXT.value, 0.0) - c, 0),
                    quality_per_gib=round(dq / dr, 5) if dr != 0 else 0.0,
                )
            )

    knee_region = _knee_region(points, frontier)
    for p in points:
        if p.candidate_id in knee_region:
            p.knee_score = 0.8
    return ParetoAnalysis(
        objectives=obs,
        points=points,
        frontier_ids=frontier_ids,
        knee_region=knee_region,
        neighbor_deltas=neighbor_deltas,
    )


def _knee_region(
    points: list[FrontierPoint],
    frontier: list[FrontierPoint],
) -> list[str]:
    """Mark the frontier region where quality-per-GB slope changes most sharply.

    Computes the marginal quality cost per GB across adjacent frontier points
    and flags the segment with the largest slope change as the knee region
    (a band, not a single point).
    """
    if len(frontier) < 3:
        return [p.candidate_id for p in frontier]
    segs: list[tuple[float, int]] = []
    for i in range(1, len(frontier)):
        a, b = frontier[i - 1], frontier[i]
        dq = a.values.get(ParetoObjective.QUALITY.value, 0.0) - b.values.get(
            ParetoObjective.QUALITY.value, 0.0
        )
        dr = abs(
            a.values.get(ParetoObjective.RESIDENT_GIB.value, 0.0)
            - b.values.get(ParetoObjective.RESIDENT_GIB.value, 0.0)
        )
        slope = dq / dr if dr else 0.0  # quality lost per GB saved
        segs.append((slope, i))
    if not segs:
        return [p.candidate_id for p in frontier]
    # knee = segment with the highest quality-per-GB (steepest quality cliff)
    _, knee_idx = max(segs, key=lambda x: x[0])
    lo = max(0, knee_idx - 1)
    hi = min(len(frontier), knee_idx + 2)
    return [p.candidate_id for p in frontier[lo:hi]]
