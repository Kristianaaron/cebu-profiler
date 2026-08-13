"""Evidence-coverage typing: INSUFFICIENT_EVIDENCE gate (v3 %6.5.2 / §3E).

"Absence of activation is not evidence of irrelevance." Every destructive
planning decision must be backed by above-threshold calibration coverage for
the capacity it targets. This module provides:

- `CoverageStatus` (GOOD/FAIR/LOW/INSUFFICIENT_EVIDENCE) backed by explicit,
  stored thresholds — never hard-coded visual labels;
- `CapacityCoverage` per capacity unit (expert / channel / semantic region) with
  meaningful-observation counts, token counts, and per-label/per-stage coverage;
- `EvidenceGate` that blocks automatic aggressive pruning/precision reduction on
  INSUFFICIENT_EVIDENCE capacity unless an explicit override records a reason.

It also distinguishes **rare-specialist** capacity (under-observed but
causally/spectrally unique) from **expendable/redundant** capacity so planners
never equate low frequency with low importance.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CoverageStatus(StrEnum):
    GOOD = "good"
    FAIR = "fair"
    LOW = "low"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CoverageThresholds(BaseModel):
    """Explicit evidence thresholds; stored with the experiment, not hard-coded."""

    model_config = ConfigDict(extra="forbid")

    good_min_observations: int = 256
    fair_min_observations: int = 64
    low_min_observations: int = 8  # below this = INSUFFICIENT_EVIDENCE
    rare_specialist_max_observations: int = 64  # "rare" for specialist detection


class CapacityCoverage(BaseModel):
    """Calibration coverage for one capacity unit (expert/channel/region)."""

    model_config = ConfigDict(extra="forbid")

    capacity_id: str  # e.g. "L0:E3" or "L2:E5:C7" or "cluster:cuda-kernel"
    meaningful_observations: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)
    label_observations: dict[str, int] = Field(default_factory=dict)
    stage_observations: dict[str, int] = Field(default_factory=dict)
    activation_frequency: float | None = Field(default=None, ge=0.0, le=1.0)
    routing_entropy: float | None = Field(default=None, ge=0.0)
    semantic_uniqueness: float | None = Field(default=None, ge=0.0, le=1.0)
    spectral_uniqueness: float | None = Field(default=None, ge=0.0, le=1.0)
    rarity_flagged: bool = False  # observed rarely but unique -> possible specialist

    def status(self, thr: CoverageThresholds) -> CoverageStatus:
        n = self.meaningful_observations
        if n >= thr.good_min_observations:
            return CoverageStatus.GOOD
        if n >= thr.fair_min_observations:
            return CoverageStatus.FAIR
        if n >= thr.low_min_observations:
            return CoverageStatus.LOW
        return CoverageStatus.INSUFFICIENT_EVIDENCE


class CoverageReport(BaseModel):
    """Whole-model coverage report against stored thresholds."""

    model_config = ConfigDict(extra="forbid")

    model: str
    thresholds: CoverageThresholds = Field(default_factory=CoverageThresholds)
    capacity: dict[str, CapacityCoverage] = Field(default_factory=dict)

    def insufficient(self) -> list[CapacityCoverage]:
        def _is_insufficient(c: CapacityCoverage) -> bool:
            return c.status(self.thresholds) is CoverageStatus.INSUFFICIENT_EVIDENCE

        return sorted(
            (c for c in self.capacity.values() if _is_insufficient(c)),
            key=lambda c: c.meaningful_observations,
        )

    def rare_specialists(self) -> list[CapacityCoverage]:
        """Rare but (spectral/semantic) unique capacity — protect, do not prune."""
        return sorted(
            (
                c
                for c in self.capacity.values()
                if c.meaningful_observations <= self.thresholds.rare_specialist_max_observations
                and ((c.spectral_uniqueness or 0.0) >= 0.9 or (c.semantic_uniqueness or 0.0) >= 0.9)
            ),
            key=lambda c: c.meaningful_observations,
        )


class ExplicitOverride(BaseModel):
    """User/experiment override of an evidence gate, recorded for audit."""

    model_config = ConfigDict(extra="forbid")

    capacity_id: str
    reason: str
    override_kind: str = "prune"  # prune | precision_reduction | none
    reference: str = ""  # experiment id / ticket


class EvidenceGate(BaseModel):
    """Blocks destructive actions on INSUFFICIENT_EVIDENCE capacity."""

    model_config = ConfigDict(extra="forbid")

    thresholds: CoverageThresholds = Field(default_factory=CoverageThresholds)
    overrides: dict[str, ExplicitOverride] = Field(default_factory=dict)

    def allow(self, coverage: CapacityCoverage, *, kind: str = "prune") -> tuple[bool, str]:
        """Return (allowed, reason). Never auto-prunes an under-observed region."""
        if coverage.capacity_id in self.overrides:
            ov = self.overrides[coverage.capacity_id]
            if ov.override_kind == kind:
                return True, f"explicit override: {ov.reason}"
        st = coverage.status(self.thresholds)
        if st is CoverageStatus.INSUFFICIENT_EVIDENCE:
            return (
                False,
                "insufficient_evidence: absence of activation is not evidence of "
                "irrelevance; override required for aggressive change",
            )
        if st is CoverageStatus.LOW:
            return True, "low evidence: allow but flag for conservative action"
        return True, f"{st.value} coverage"
