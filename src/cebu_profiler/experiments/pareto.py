"""Quality-size Pareto sweep (blueprint §17 Stage 2).

Sweeps the retained-channel budget and, per level, reports the uniform and
heterogeneous retention/fidelity at the same budget. The frontier answers
Atlas's founding question: does measured heterogeneous allocation dominate a
fixed-width control at equal parameter size?
"""

from __future__ import annotations

from dataclasses import dataclass

from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE
from model_atlas.experiments.controls import budget_for, matched_budget_compare


@dataclass
class ParetoPoint:
    retain_fraction: float
    budget: int
    uniform_retention: float
    hetero_retention: float
    uniform_kl: float
    hetero_kl: float
    uniform_topk: float
    hetero_topk: float

    def as_row(self) -> dict[str, object]:
        return {
            "keep_fraction": round(self.retain_fraction, 4),
            "budget": self.budget,
            "uniform_retention": round(self.uniform_retention, 4),
            "hetero_retention": round(self.hetero_retention, 4),
            "uniform_logit_kl": round(self.uniform_kl, 5),
            "hetero_logit_kl": round(self.hetero_kl, 5),
            "uniform_topk_agreement": round(self.uniform_topk, 4),
            "hetero_topk_agreement": round(self.hetero_topk, 4),
        }


def pareto_sweep(
    model: MiniMoE,
    calibration: list[CalibrationSample],
    heldout: list[CalibrationSample],
    fractions: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> list[ParetoPoint]:
    """Sweep retention levels, comparing uniform vs heterogeneous at each budget."""
    points: list[ParetoPoint] = []
    for frac in fractions:
        b = budget_for(model, frac)
        out = matched_budget_compare(model, calibration, heldout, b)
        points.append(
            ParetoPoint(
                retain_fraction=out.retain_fraction,
                budget=out.budget,
                uniform_retention=out.uniform.retention,
                hetero_retention=out.hetero.retention,
                uniform_kl=out.uniform.mean_logit_kl,
                hetero_kl=out.hetero.mean_logit_kl,
                uniform_topk=out.uniform.topk_agreement,
                hetero_topk=out.hetero.topk_agreement,
            )
        )
    return points
