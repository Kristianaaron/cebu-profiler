"""Coalitions and multi-component causal tracing (v2 §10.7, §15, §17).

Coalition discovery via expert coactivation over real routing traces.

Multi-component causal tracing measures the joint effect of removing a set of
experts from routing (ablation on the frozen model), distinguishing
catastrophic cascades (A alone small, B alone small, A+B catastrophic) from
redundant/independent components:

    effect(set) = baseline_utility - ablated_utility
    synergy(A,B) = effect(A,B) - effect(A) - effect(B)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from cebu_profiler.profiler.counterfactual import final_utility
from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, forward


def _mean_utility(
    model: MiniMoE,
    samples: list[CalibrationSample],
    *,
    excluded: dict[int, frozenset[int]] | None = None,
) -> float:
    """Mean final-peak utility over samples, optionally with a frozen ablation."""
    u = 0.0
    n = 0
    for s in samples:
        u += final_utility(forward(model, s.tokens, excluded=excluded).logits)
        n += 1
    return u / n if n else 0.0


def single_effect(
    model: MiniMoE, samples: list[CalibrationSample], layer: int, removed: set[int]
) -> float:
    """Effect of removing `removed` experts at `layer`: baseline - ablated (mean)."""
    base = _mean_utility(model, samples)
    abl = _mean_utility(model, samples, excluded={layer: frozenset(removed)})
    return base - abl


# --------------------------------------------------------------------------- #
# Coactivation / coalition map (§10.7)
# --------------------------------------------------------------------------- #


@dataclass
class CoalitionMap:
    layer: int
    pair_counts: dict[tuple[int, int], int]  # (a, b) sorted, a < b -> co-routed count

    def candidate_coalitions(self, min_coactivity: int) -> list[tuple[int, int]]:
        return sorted(
            (pair for pair, c in self.pair_counts.items() if c >= min_coactivity),
            key=lambda p: -self.pair_counts[p],
        )


def coactivation_map(
    model: MiniMoE, samples: list[CalibrationSample], layer: int, top_k: int | None = None
) -> CoalitionMap:
    """Count how often pairs of experts are routed together at a layer."""
    counts: dict[tuple[int, int], int] = {}
    for s in samples:
        r = forward(model, s.tokens, top_k=top_k)
        trace = r.traces[layer]
        for ids in trace.topk_ids:
            for a, b in itertools.combinations(sorted(set(ids)), 2):
                key = (a, b)
                counts[key] = counts.get(key, 0) + 1
    return CoalitionMap(layer=layer, pair_counts=counts)


# --------------------------------------------------------------------------- #
# Multi-component causal tracing (§17)
# --------------------------------------------------------------------------- #


@dataclass
class PairAnalysis:
    layer: int
    a: int
    b: int
    effect_a: float
    effect_b: float
    effect_ab: float
    # positive ==> jointly more harmful than the sum (catastrophic cascade)
    synergy_ab: float = field(init=False)
    redundant: bool = field(init=False)
    catastrophic: bool = field(init=False)

    def __post_init__(self) -> None:
        self.synergy_ab = self.effect_ab - (self.effect_a + self.effect_b)
        self.redundant = self.synergy_ab < -1e-9
        self.catastrophic = (
            self.synergy_ab > 1e-9 and self.effect_ab > max(self.effect_a, self.effect_b) + 1e-9
        )


def pairwise_causal(
    model: MiniMoE, samples: list[CalibrationSample], layer: int, a: int, b: int
) -> PairAnalysis:
    """Individual, joint, and synergy effects of removing experts {a} and {b}."""
    base = _mean_utility(model, samples)
    e_a = base - _mean_utility(model, samples, excluded={layer: frozenset({a})})
    e_b = base - _mean_utility(model, samples, excluded={layer: frozenset({b})})
    e_ab = base - _mean_utility(model, samples, excluded={layer: frozenset({a, b})})
    return PairAnalysis(layer, a, b, e_a, e_b, e_ab)


def synergic_pairs(
    model: MiniMoE, samples: list[CalibrationSample], layer: int, max_experts: int | None = None
) -> list[PairAnalysis]:
    """All pairs, ordered by descending synergy (biggest cascade risk first)."""
    n_exp = max_experts or model.n_exp
    analyses = [
        pairwise_causal(model, samples, layer, a, b)
        for a, b in itertools.combinations(range(n_exp), 2)
    ]
    analyses.sort(key=lambda x: x.synergy_ab, reverse=True)
    return analyses


# --------------------------------------------------------------------------- #
# Minimum destructive set (§17)
# --------------------------------------------------------------------------- #


def minimum_destructive_set(
    model: MiniMoE,
    samples: list[CalibrationSample],
    layer: int,
    *,
    damage_threshold: float,
    max_size: int = 3,
) -> tuple[frozenset[int], float] | None:
    """Smallest removed-set whose effect crosses `damage_threshold`.

    Returns (set, effect) or None if no set up to `max_size` crosses it,
    searching size-1 first (cheapest falsification order) and always leaving at
    least top_k experts available for a valid route.
    """
    base = _mean_utility(model, samples)
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(range(model.n_exp), size):
            dropped = frozenset(subset)
            if model.n_exp - len(dropped) < model.arch.moe.top_k:
                continue  # would leave too few experts to route
            ablated = _mean_utility(model, samples, excluded={layer: dropped})
            effect = base - ablated
            if effect >= damage_threshold:
                return dropped, effect
    return None
