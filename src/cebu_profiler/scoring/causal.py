"""Causal / boundary scorer (blueprint §7-Module D).

Surfaces rare-but-critical channels by measuring the *worst-case single-token*
expert-output perturbation of dropping a channel: `causal(c) = ||down[:,c]|| *
peak|z_c|`. Because it uses the peak activation (outlier statistics) rather than
the mean, it ranks differently from TENP — a low-frequency but high-impact
channel is captured even when its average contribution is small.

`triage` then divides each expert's channels into DEFINITE_KEEP / DEFINITE_PRUNE
/ UNCERTAIN_BOUNDARY by importance, resolving the boundary with the causal view
(blueprint §7-D: spend causal compute on the uncertain boundary).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from model_atlas.atlas.collector import ChannelStatsAccumulator
from model_atlas.scoring.base import (
    AtlasScorer,
    ChannelScore,
    ScoreNeed,
    ScorerRequirements,
    ScoreTable,
)

if TYPE_CHECKING:
    from model_atlas.atlas.runtime import MiniMoE


class Boundary(StrEnum):
    DEFINITE_KEEP = "definite_keep"
    UNCERTAIN_BOUNDARY = "uncertain_boundary"
    DEFINITE_PRUNE = "definite_prune"


def causal_scores(
    model: MiniMoE, stats: ChannelStatsAccumulator
) -> dict[tuple[int, int, int], float]:
    """Per-(layer, expert, channel) worst-case output-perturbation causal score."""
    final = {(s.layer, s.expert, s.channel): s for s in stats.finalize()}
    out: dict[tuple[int, int, int], float] = {}
    for layer, layer_w in enumerate(model.layers):
        for e, exp in enumerate(layer_w.experts):
            down = exp["down"]
            for c in range(len(down[0])):
                s = final.get((layer, e, c))
                if s is None:
                    continue
                col_norm = (sum(row[c] * row[c] for row in down)) ** 0.5
                out[(layer, e, c)] = col_norm * s.peak
    return out


def triage(
    importance: dict[tuple[int, int, int], float],
    causal: dict[tuple[int, int, int], float],
    keep_share: float,
) -> dict[tuple[int, int, int], Boundary]:
    """Triage channels within each expert into keep / boundary / prune.

    `keep_share` is the target retained fraction of channels per expert. High
    importance -> definite keep; low importance -> definite prune; the rest form
    the uncertain boundary, resolved toward keep when the causal view ranks them
    above the expert's median causal score.
    """
    from collections import defaultdict

    by_expert: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for key in importance:
        by_expert[(key[0], key[1])].append(key)

    result: dict[tuple[int, int, int], Boundary] = {}
    for (_, _), keys in by_expert.items():
        keys_sorted = sorted(keys, key=lambda k: importance[k], reverse=True)
        n = len(keys_sorted)
        keep_n = max(1, round(n * keep_share))
        prune_n = max(0, n - round(n * (1 - keep_share)) )
        # band around the boundary
        band = max(1, n // 4)
        for idx, key in enumerate(keys_sorted):
            if idx < max(1, keep_n - band):
                result[key] = Boundary.DEFINITE_KEEP
            elif idx >= prune_n + band:
                result[key] = Boundary.DEFINITE_PRUNE
            else:
                result[key] = Boundary.UNCERTAIN_BOUNDARY

        # resolve boundary with the causal view: above-expert-median causal -> keep
        med = _median([causal.get(k, 0.0) for k in keys])
        for key, status in result.items():
            if status == Boundary.UNCERTAIN_BOUNDARY and causal.get(key, 0.0) >= med:
                result[key] = Boundary.DEFINITE_KEEP
    return result


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


class CausalScorer(AtlasScorer):
    name = "causal"
    version = "1.0"

    def __init__(self, model: MiniMoE, stats: ChannelStatsAccumulator) -> None:
        self._model = model
        self._scores = causal_scores(model, stats)

    def requirements(self) -> ScorerRequirements:
        return ScorerRequirements(
            frozenset({ScoreNeed.FORWARD_ACTIVATIONS, ScoreNeed.RAW_EXPERT_TENSORS})
        )

    def finalize(self) -> ScoreTable:
        rows = [
            ChannelScore(layer=layer, expert=e, channel=c, causal=v)
            for (layer, e, c), v in sorted(self._scores.items())
        ]
        return ScoreTable(
            model=self._model.arch.name, scorer_versions={self.name: self.version}, rows=rows
        )
