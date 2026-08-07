"""Held-out quality / fidelity measurement (blueprint §18.1).

Reuses the established ``final_utility`` (peak softmax "decisiveness") as the
quality scalar and ``logit_kl`` as the fidelity penalty, averaged over a
held-out sample set. Retention is mutated/source utility, mirroring the F12
held-out evaluation so results are comparable across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_atlas.atlas.counterfactual import final_utility, logit_kl
from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE, forward


@dataclass
class FidelityReport:
    """Measured quality/fidelity of a mutated model vs its source."""

    n_samples: int
    utility_source: float
    utility_mutated: float
    retention: float  # mutated/source utility (may exceed 1)
    mean_logit_kl: float  # mean KL(softmax src || softmax mut), >= 0
    topk_agreement: float  # fraction of samples where top-1 logit argmax agrees
    mean_hidden_drift: float  # mean relative L2 drift of the final hidden state


def _drift(src_hidden: list[list[float]], mut_hidden: list[list[float]]) -> float:
    """Mean relative L2 drift of per-token hidden states: ||d|| / (||s||+eps)."""
    from math import sqrt

    acc = 0.0
    n = 0
    for s, m in zip(src_hidden, mut_hidden, strict=True):
        if not s:
            continue
        sn = sqrt(sum(v * v for v in s)) or 1e-12
        diff = sum((a - b) ** 2 for a, b in zip(s, m, strict=True))
        acc += sqrt(diff) / sn
        n += 1
    return acc / n if n else 0.0


def _softmax(x: list[float]) -> list[float]:
    from math import exp

    m = max(x)
    e = [exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def measure_fidelity(
    source: MiniMoE, mutated: MiniMoE, samples: list[CalibrationSample]
) -> FidelityReport:
    """Compare ``source`` vs ``mutated`` output behavior on ``samples``."""
    if not samples:
        return FidelityReport(0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    src_u = 0.0
    mut_u = 0.0
    kl = 0.0
    drift = 0.0
    agree = 0
    for s in samples:
        src_r = forward(source, s.tokens)
        mut_r = forward(mutated, s.tokens)
        src_logits, mut_logits = src_r.logits, mut_r.logits
        src_u += final_utility(src_logits)
        mut_u += final_utility(mut_logits)
        kl += logit_kl(src_logits, mut_logits)
        drift += _drift(src_r.final_hidden_states, mut_r.final_hidden_states)
        if _softmax(src_logits).index(max(_softmax(src_logits))) == _softmax(mut_logits).index(
            max(_softmax(mut_logits))
        ):
            agree += 1
    n = len(samples)
    return FidelityReport(
        n_samples=n,
        utility_source=src_u / n,
        utility_mutated=mut_u / n,
        retention=mut_u / src_u if src_u > 0 else 1.0,
        mean_logit_kl=kl / n,
        topk_agreement=agree / n,
        mean_hidden_drift=drift / n,
    )
