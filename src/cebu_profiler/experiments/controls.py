"""Matched-budget controls: Atlas heterogeneous vs uniform width (blueprint §17, Milestone E).

At an equal total retained-channel budget ``B`` over all (layer, expert) slots:

- **uniform** keeps the same width for every expert (``B // (n_layers*n_exp)``,
  remainder spread deterministically), each expert holding its top-`w`
  measured-importance channels.
- **heterogeneous** keeps the global top-``B`` measured-importance channels
  (automatically a nested per-expert prefix, FlexMoE-style), so more-important
  experts keep more width.

Importance comes from the real Atlas TENP scorer (forward-activation x projected
output-norm), so the heterogeneous arm is anchored in the measured pipeline
rather than a hand-picked mask. Quality/fidelity are measured on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_atlas.atlas.collector import ChannelStatsAccumulator
from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.executor.structural import build_clone
from model_atlas.experiments.fidelity import FidelityReport, measure_fidelity
from model_atlas.scoring.tenp import tenp_rank


@dataclass
class ExperimentOutcome:
    """Result of one matched-budget uniform vs heterogeneous comparison."""

    budget: int
    total_channels: int
    uniform: FidelityReport
    hetero: FidelityReport

    @property
    def retain_fraction(self) -> float:
        return self.budget / self.total_channels if self.total_channels else 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "keep_fraction": round(self.retain_fraction, 4),
            "uniform_retention": round(self.uniform.retention, 4),
            "hetero_retention": round(self.hetero.retention, 4),
            "uniform_logit_kl": round(self.uniform.mean_logit_kl, 5),
            "hetero_logit_kl": round(self.hetero.mean_logit_kl, 5),
            "delta_retention": round(self.hetero.retention - self.uniform.retention, 4),
        }


def budget_for(model: MiniMoE, retain_fraction: float) -> int:
    """Total retained-channel budget for an overall retention fraction on ``model``."""
    total = len(model.layers) * model.n_exp * model.mid
    b = round(total * retain_fraction)
    return min(max(b, 1), total)


def channel_importance(
    model: MiniMoE, samples: list[CalibrationSample]
) -> dict[tuple[int, int, int], float]:
    """Per-(layer, expert, channel) TENP importance from a forward pass."""
    acc = ChannelStatsAccumulator()
    for s in samples:
        forward(model, s.tokens, channel_stats=acc)
    return tenp_rank(model, acc)


def _ranked(model: MiniMoE, importance: dict[tuple[int, int, int], float]) -> list[list[int]]:
    """Per (layer,e) channel rankings (desc by importance), nested-prefix order.
    Slots indexed ``layer * n_exp + e``; missing channels treated as importance 0.
    """
    mid = model.mid
    per: list[list[tuple[int, float]]] = [
        [(c, importance.get((layer, e, c), 0.0)) for c in range(mid)]
        for layer in range(len(model.layers))
        for e in range(model.n_exp)
    ]
    ranked: list[list[int]] = []
    for lst in per:
        lst.sort(key=lambda x: (-x[1], x[0]))  # desc importance, tie by channel
        ranked.append([c for c, _ in lst])
    return ranked


def uniform_clone(
    model: MiniMoE, importance: dict[tuple[int, int, int], float], budget: int
) -> MiniMoE:
    """Pruine to equal per-expert widths totalling ``budget`` channels."""
    ranked = _ranked(model, importance)
    n_slots = len(model.layers) * model.n_exp
    w = budget // n_slots
    r = budget - w * n_slots
    orders: dict[tuple[int, int], list[int]] = {}
    for slot in range(n_slots):
        layer, e = slot // model.n_exp, slot % model.n_exp
        count = min(model.mid, w + (1 if slot < r else 0))
        orders[(layer, e)] = ranked[slot][:count]
    return build_clone(model, orders)


def hetero_clone(
    model: MiniMoE, importance: dict[tuple[int, int, int], float], budget: int
) -> MiniMoE:
    """Pruine keeping the global top-``budget`` channels (nested per-expert prefix)."""
    ranked = _ranked(model, importance)
    n_slots = len(model.layers) * model.n_exp
    # select global top-budget channel keys
    all_chan = sorted(
        ((layer, e, c, importance.get((layer, e, c), 0.0)) for layer, e, c in _all_keys(model)),
        key=lambda x: -x[3],
    )
    selected = {(layer, e, c) for layer, e, c, _ in all_chan[:budget]}
    orders: dict[tuple[int, int], list[int]] = {}
    for slot in range(n_slots):
        layer, e = slot // model.n_exp, slot % model.n_exp
        kept = [c for c in ranked[slot] if (layer, e, c) in selected]
        orders[(layer, e)] = kept
    return build_clone(model, orders)


def _all_keys(model: MiniMoE) -> list[tuple[int, int, int]]:
    return [
        (layer, e, c)
        for layer in range(len(model.layers))
        for e in range(model.n_exp)
        for c in range(model.mid)
    ]


def control_c_clone(
    model: MiniMoE,
    importance: dict[tuple[int, int, int], float],
    budget: int,
    layer_power: float = 1.0,
) -> MiniMoE:
    """Blueprint §17 Control C: TENP-only, depth-aware (trapezoidal) allocation.

    Per-layer channel budgets grow with depth (``(layer+1)**layer_power``), split
    equally across the layer's experts, and each expert keeps its top-TENP
    channels within its share. No causal/semantic input, isolating the value of
    pure depth-aware TENP allocation."""
    ranked = _ranked(model, importance)
    n_layers = len(model.layers)
    n_exp = model.n_exp
    depth = [(i + 1) ** layer_power for i in range(n_layers)]
    total_w = sum(depth) * n_exp
    budget = min(max(budget, n_layers * n_exp), n_layers * n_exp * model.mid)
    orders: dict[tuple[int, int], list[int]] = {}
    allocated = 0
    for layer in range(n_layers):
        share = int(budget * (depth[layer] * n_exp) / total_w)
        base = share // n_exp
        rem = share % n_exp
        for e in range(n_exp):
            slot = layer * n_exp + e
            count = base + (1 if slot % n_exp < rem else 0)
            # never fully prune an expert; cap at full width
            count = min(model.mid, max(1, count))
            if allocated + count > budget:
                count = max(1, budget - allocated)
            orders[(layer, e)] = ranked[slot][:count]
            allocated += count
    return build_clone(model, orders)


def compare_controls(
    model: MiniMoE,
    calibration: list[CalibrationSample],
    heldout: list[CalibrationSample],
    budget: int,
    layer_power: float = 1.0,
) -> dict[str, FidelityReport]:
    """Uniform (Control B), TENP-depth-aware (Control C), Atlas heterogeneous
    (Experiment D) at an equal budget."""
    importance = channel_importance(model, calibration)
    return {
        "uniform": measure_fidelity(model, uniform_clone(model, importance, budget), heldout),
        "control_c": measure_fidelity(
            model, control_c_clone(model, importance, budget, layer_power), heldout
        ),
        "hetero": measure_fidelity(model, hetero_clone(model, importance, budget), heldout),
    }


def matched_budget_compare(
    model: MiniMoE,
    calibration: list[CalibrationSample],
    heldout: list[CalibrationSample],
    budget: int,
) -> ExperimentOutcome:
    """Compare uniform vs heterogeneous clones at an equal channel budget."""
    importance = channel_importance(model, calibration)
    uniform = uniform_clone(model, importance, budget)
    hetero = hetero_clone(model, importance, budget)
    total = len(model.layers) * model.n_exp * model.mid
    return ExperimentOutcome(
        budget=budget,
        total_channels=total,
        uniform=measure_fidelity(model, uniform, heldout),
        hetero=measure_fidelity(model, hetero, heldout),
    )
