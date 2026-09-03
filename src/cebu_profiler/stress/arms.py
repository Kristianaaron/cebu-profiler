"""Causal prune-arm stress matrix (blueprint §17 / AGENTS.md invariants 5+12).

Pattern after the published five-arm REAP stress protocol (alesha-pro/atlas
GLM-5.3 bundle `pruning` block): remove a fraction of experts by score at
multiple sizes, with a random-removal control and a high-score-removal control,
and score each arm by output damage — not by the proxy that chose the victims.
Re-implemented on Cebu's frozen-model intervention API (runtime.forward's
`excluded`), with the identity arm required by invariant 5.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from cebu_profiler.profiler.reap import SaliencyAccumulator
from cebu_profiler.profiler.runtime import MiniMoE, forward

DEFAULT_FRACTIONS = (0.02, 0.05, 0.10)


@dataclass
class ArmResult:
    """One arm of the matrix: which experts were removed and what it cost."""

    arm: str  # low_reap | high_reap | random | identity
    fraction: float
    removed: list[tuple[int, int]] = field(default_factory=list)
    removed_slot_fraction: float = 0.0
    # Damage metrics over the probe tokens:
    mean_logit_kl: float = 0.0
    output_cosine_mean: float = 1.0
    sequence_exact: float = 1.0  # fraction of probes with identical argmax
    n_probes: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "fraction": self.fraction,
            "removed_slot_fraction": self.removed_slot_fraction,
            "n_removed": len(self.removed),
            "mean_logit_kl": self.mean_logit_kl,
            "output_cosine_mean": self.output_cosine_mean,
            "sequence_exact": self.sequence_exact,
            "n_probes": self.n_probes,
        }


def _expert_scores(acc: SaliencyAccumulator) -> dict[tuple[int, int], float]:
    """Mean REAP saliency per (layer, expert), ascending = prune-first order."""
    out: dict[tuple[int, int], float] = {}
    layers = {k[0] for k in acc._sum}
    for layer in layers:
        # discover experts for this layer from the accumulator keys
        experts = {k[1] for k in acc._sum if k[0] == layer}
        for e in experts:
            out[(layer, e)] = acc.total_value(layer, e)
    return out


def _prune_order(
    scores: dict[tuple[int, int], float],
    arm: str,
    fraction: float,
    rng: random.Random,
) -> list[tuple[int, int]]:
    slots = sorted(scores)
    n_remove = int(len(slots) * fraction)
    if arm == "random":
        picked = slots[:]
        rng.shuffle(picked)
        return picked[:n_remove]
    reverse = arm == "high_reap"
    ordered = sorted(slots, key=lambda s: scores[s], reverse=reverse)
    return ordered[:n_remove]


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    nu = sum(a * a for a in u) ** 0.5
    nv = sum(b * b for b in v) ** 0.5
    if nu == 0.0 or nv == 0.0:
        return 1.0 if nu == nv else 0.0
    return float(dot / (nu * nv))


def _logit_kl(p: list[float], q: list[float]) -> float:
    from math import log

    m = max(p) or 1.0
    e_p = [math.exp(x - m) for x in p]
    e_q = [math.exp(x - m) for x in q]
    z_p, z_q = sum(e_p), sum(e_q)
    kl = 0.0
    for a, b in zip(e_p, e_q, strict=True):
        pi = a / z_p
        qi = b / z_q
        if pi > 0.0:
            kl += pi * (log(pi) - log(qi))
    return kl


def run_prune_arms(
    model: MiniMoE,
    acc: SaliencyAccumulator,
    probe_token_sets: list[list[int]],
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> list[ArmResult]:
    """Run the stress matrix and return one ArmResult per (arm, fraction).

    Arms: low_reap (prune lowest scores), high_reap (control: prune highest),
    random (control: prune uniformly), identity (no-op, invariant 5).
    The identity arm must reproduce the baseline exactly or the whole matrix
    is invalid evidence.
    """
    scores = _expert_scores(acc)
    rng = random.Random(seed)
    base_logits: list[list[float]] = []
    base_hidden: list[list[float]] = []
    for tokens in probe_token_sets:
        res = forward(model, tokens)
        base_logits.append(res.logits)
        base_hidden.append(res.final_hidden)

    arms: list[ArmResult] = [
        ArmResult(arm="identity", fraction=0.0, n_probes=len(probe_token_sets))
    ]
    # identity arm: verify the no-op reproduces the baseline bit-for-bit
    for i, tokens in enumerate(probe_token_sets):
        res = forward(model, tokens)
        if res.logits != base_logits[i]:
            raise AssertionError("identity arm drifted from baseline — invalid evidence")
    arms[0].sequence_exact = 1.0
    arms[0].output_cosine_mean = 1.0

    for fraction in fractions:
        for arm in ("low_reap", "random", "high_reap"):
            removed = _prune_order(scores, arm, fraction, rng)
            excluded: dict[int, frozenset[int]] = {}
            for layer, e in removed:
                excluded.setdefault(layer, frozenset()).union({e})
                excluded[layer] = excluded[layer] | {e}
            kls: list[float] = []
            cosines: list[float] = []
            exact = 0
            for i, tokens in enumerate(probe_token_sets):
                res = forward(model, tokens, excluded=excluded)
                kls.append(_logit_kl(base_logits[i], res.logits))
                cosines.append(_cosine(base_hidden[i], res.final_hidden))
                amax_base = base_logits[i].index(max(base_logits[i], key=float))
                amax_res = res.logits.index(max(res.logits, key=float))
                if amax_res == amax_base:
                    exact += 1
            n = len(probe_token_sets)
            arms.append(
                ArmResult(
                    arm=arm,
                    fraction=fraction,
                    removed=removed,
                    removed_slot_fraction=len(removed) / len(scores) if scores else 0.0,
                    mean_logit_kl=sum(kls) / n if n else 0.0,
                    output_cosine_mean=sum(cosines) / n if n else 1.0,
                    sequence_exact=exact / n if n else 1.0,
                    n_probes=n,
                )
            )
    return arms


def matrix_payload(
    arms: list[ArmResult], *, limitations: dict[str, str] | None = None
) -> dict[str, Any]:
    """JSON bundle for the run directory (evidence-typed, limitations explicit)."""
    return {
        "arms": [a.payload() for a in arms],
        "limits": dict(
            limitations
            or {
                "method": "frozen-model route ablation (router renormalized over survivors), "
                "not a physically pruned checkpoint"
            }
        ),
    }
