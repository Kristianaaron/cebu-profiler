"""Held-out evaluation of a derivative vs its source (v2 §14, §26).

Measures capability retention on data the derivative did NOT see during
planning/calibration, split by capability label, and derives router-repair
targets from the biggest held-out drops. Predictions/regressions are separate
from measured results.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from model_atlas.atlas.counterfactual import final_utility
from model_atlas.atlas.reap import CalibrationSample, SaliencyAccumulator
from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.planning.maps import CandidatePlan


@dataclass
class LabelRetention:
    label: str
    n_samples: int
    source_utility: float
    derivative_utility: float
    retention: float  # derivative/source (may exceed 1 if derivative "helps")


@dataclass
class HeldOutReport:
    n_samples: int
    overall_source: float
    overall_derivative: float
    overall_retention: float
    per_label: list[LabelRetention]
    worst_label_drop: float  # max over labels of max(0, 1 - retention)

    def as_rows(self) -> list[dict[str, object]]:
        return [
            {
                "label": r.label,
                "sample_count": r.n_samples,
                "source_utility": r.source_utility,
                "derivative_utility": r.derivative_utility,
                "retention": r.retention,
            }
            for r in self.per_label
        ]


def _utility_by_label(
    model: MiniMoE, samples: list[CalibrationSample]
) -> tuple[float, dict[str, tuple[float, int]]]:
    total = 0.0
    by_label: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for s in samples:
        u = final_utility(forward(model, s.tokens).logits)
        total += u
        for label in s.labels:
            cur = by_label[label.value]
            by_label[label.value] = (cur[0] + u, cur[1] + 1)
    return total / len(samples) if samples else 0.0, dict(by_label)


def evaluate_heldout(
    source: MiniMoE, derivative: MiniMoE, samples: list[CalibrationSample]
) -> HeldOutReport:
    """Compare source vs derivative utility on (ideally held-out) samples."""
    src_total, src_by_label = _utility_by_label(source, samples)
    deriv_total, deriv_by_label = _utility_by_label(derivative, samples)

    per_label: list[LabelRetention] = []
    merged = set(src_by_label) | set(deriv_by_label)
    for label in sorted(merged):
        su, sn = src_by_label.get(label, (0.0, 0))
        du, dn = deriv_by_label.get(label, (0.0, 0))
        retention = du / su if su > 0 else 1.0
        per_label.append(
            LabelRetention(
                label=label,
                n_samples=sn,
                source_utility=su,
                derivative_utility=du,
                retention=retention,
            )
        )
    overall_retention = deriv_total / src_total if src_total > 0 else 1.0
    worst = max((max(0.0, 1.0 - r.retention) for r in per_label), default=0.0)
    return HeldOutReport(
        n_samples=len(samples),
        overall_source=src_total,
        overall_derivative=deriv_total,
        overall_retention=overall_retention,
        per_label=per_label,
        worst_label_drop=worst,
    )


def router_repair_targets(
    plan: CandidatePlan,
    heldout_saliency: SaliencyAccumulator,
    *,
    saliency_threshold: float = 0.0,
    source: MiniMoE | None = None,
) -> list[tuple[int, int]]:
    """(layer, source_expert) pairs dropped by the plan yet still salient on
    held-out data — implicated for router/keep repair (v2 §26 repair targets)."""
    n_layers = source.arch.num_text_layers if source else 2
    n_exp = source.n_exp if source else 8
    targets: list[tuple[int, int]] = []
    for layer in range(n_layers):
        kept = set(plan.keep.kept(layer)) if plan.keep.entries else set()
        for e in range(n_exp):
            if e in kept:
                continue
            if heldout_saliency.total_value(layer, e) > saliency_threshold:
                targets.append((layer, e))
    return targets
