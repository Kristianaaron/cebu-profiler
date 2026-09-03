"""Counterfactual routing and route regret (v2 §13).

Tests the hypothesis that the frozen router selected a suboptimal route. For a
target (layer, token), we sample alternative equal-compute top-k expert sets,
force each through the unchanged model, and measure the downstream effect vs
the original route's outcome.

    route regret = best sampled alternative utility - original route utility

A positive regret means some alternative equal-compute route would have been
better — the router was locally suboptimal for that token (a §2 negative
finding the platform should record, not hide).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations

from cebu_profiler.profiler.runtime import MiniMoE, forward


def _softmax(x: list[float]) -> list[float]:
    m = max(x)
    e = [v - m for v in x]  # stable logits
    from math import exp

    ex = [exp(v) for v in e]
    s = sum(ex)
    return [v / s for v in ex]


def final_utility(logits: list[float]) -> float:
    """Peak softmax probability of the final distribution = 'decisiveness'."""
    return max(_softmax(logits))


def logit_kl(p_logits: list[float], q_logits: list[float]) -> float:
    """KL(softmax(p) || softmax(q)) over the full vocabulary."""
    from math import log

    p = _softmax(p_logits)
    q = _softmax(q_logits)
    return sum(pi * (log(pi) - log(qi)) for pi, qi in zip(p, q, strict=True) if pi > 0.0)


@dataclass
class AlternativeRoute:
    route: list[int]  # sorted forced expert ids
    utility: float
    logit_kl: float  # vs original route outcome
    local_output_delta: float  # |output_norm_forced - output_norm_original| at (layer, token)


@dataclass
class RouteRegretResult:
    layer: int
    token_index: int
    original_route: list[int]
    original_utility: float
    alternatives: list[AlternativeRoute]
    route_regret: float
    fragile: bool
    best_route: list[int] = field(init=False)
    best_utility: float = field(init=False)

    def __post_init__(self) -> None:
        best = max(self.alternatives, key=lambda a: a.utility)
        self.best_route = best.route
        self.best_utility = best.utility


def sample_topk_subsets(
    n_exp: int, k: int, exclude: frozenset[int], n: int, seed: int
) -> list[list[int]]:
    """Sample `n` distinct k-subsets of experts, none equal to `exclude`."""
    rng = random.Random(seed)
    all_subsets = [sorted(c) for c in combinations(range(n_exp), k) if frozenset(c) != exclude]
    rng.shuffle(all_subsets)
    return all_subsets[:n]


def counterfactual_scan(
    model: MiniMoE,
    tokens: list[int],
    *,
    layer: int,
    token_index: int,
    n_alternatives: int = 8,
    seed: int = 0,
    k_override: int | None = None,
) -> RouteRegretResult:
    """Sample alternative equal-compute routes for one (layer, token) and compute regret."""
    base = forward(model, tokens)
    original_route = sorted(base.traces[layer].topk_ids[token_index])
    original_utility = final_utility(base.logits)
    original_output_norm = base.traces[layer].output_norm[token_index]

    k = k_override or len(original_route)
    subsets = sample_topk_subsets(model.n_exp, k, frozenset(original_route), n_alternatives, seed)

    alternatives: list[AlternativeRoute] = []
    for subset in subsets:
        override = {(layer, token_index): subset}
        result = forward(model, tokens, route_override=override)
        utility = final_utility(result.logits)
        kl = logit_kl(base.logits, result.logits)
        local_delta = abs(result.traces[layer].output_norm[token_index] - original_output_norm)
        alternatives.append(
            AlternativeRoute(
                route=subset, utility=utility, logit_kl=kl, local_output_delta=local_delta
            )
        )

    best = max(alternatives, key=lambda a: a.utility)
    regret = best.utility - original_utility
    # fragile if the best alternative materially changes the outcome (large delta)
    fragile = best.local_output_delta > 1e-6 and regret > 1e-9

    return RouteRegretResult(
        layer=layer,
        token_index=token_index,
        original_route=original_route,
        original_utility=original_utility,
        alternatives=alternatives,
        route_regret=regret,
        fragile=fragile,
    )
