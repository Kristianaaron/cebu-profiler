"""Grouped Taylor / FlexMoE scorer (blueprint §7-Module C, §9.1).

A full graded first-order score needs `(theta * dL/dtheta)^2` summed over the
coupled structural group `Theta_j = {gate[j,:], up[j,:], down[:,j]}` — which
requires real gradients against a higher-precision / autograd-capable parent.

This module provides the *grouped structural unit* scoring interface and a
documented **surrogate** implementation (first-order magnitude × activation),
validated on the synthetic MiniMoE. It is marked `authoritative=False`; the
authoritative gradient result is explicitly pending a higher-precision parent
(blueprint §5.2, §9.1) and must not be treated as the final pruning map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cebu_profiler.scoring.base import (
    ChannelScore,
    ProfilerScorer,
    ScoreNeed,
    ScorerRequirements,
    ScoreTable,
)

if TYPE_CHECKING:
    from cebu_profiler.profiler.collector import ChannelStatsAccumulator
    from cebu_profiler.profiler.runtime import MiniMoE


def score_grouped_surrogate(
    model: MiniMoE, stats: ChannelStatsAccumulator
) -> tuple[dict[tuple[int, int, int], float], bool]:
    """Surrogate grouped-importance per channel.

    score(c) = (mean|z_c|)^2 * (||gate[j,:]||^2 + ||up[j,:]||^2 + ||down[:,j]||^2),
    a first-order magnitude proxy. Returns (scores, authoritative=False).
    """
    final = {(s.layer, s.expert, s.channel): s for s in stats.finalize()}
    out: dict[tuple[int, int, int], float] = {}
    for layer, layer_w in enumerate(model.layers):
        for e, exp in enumerate(layer_w.experts):
            gate = exp["gate"]
            up = exp["up"]
            down = exp["down"]
            for c in range(len(gate)):
                s = final.get((layer, e, c))
                if s is None:
                    continue
                w2 = (
                    sum(gate[c][q] ** 2 for q in range(len(gate[c])))
                    + sum(up[c][q] ** 2 for q in range(len(up[c])))
                    + sum(down[q][c] ** 2 for q in range(len(down)))
                )
                out[(layer, e, c)] = (s.mean_abs**2) * w2
    return out, False


class GroupedTaylorScorer(ProfilerScorer):
    """Grouped structural-unit scorer.

    Surrogate by default (no autograd on the synthetic runtime). When a
    higher-precision, autograd-capable parent is provided this can be replaced
    with the authoritative `(theta*dL/dtheta)^2` ranking.
    """

    name = "taylor_grouped"
    version = "1.0"
    authoritative = False  # surrogate until graded against a BF16 parent

    def __init__(self, model: MiniMoE, stats: ChannelStatsAccumulator) -> None:
        self._model = model
        self._scores, self._authoritative = score_grouped_surrogate(model, stats)

    def requirements(self) -> ScorerRequirements:
        needs = {ScoreNeed.RAW_EXPERT_TENSORS, ScoreNeed.FORWARD_ACTIVATIONS}
        if not self.authoritative:
            needs.add(ScoreNeed.GRADIENTS)  # authoritative path is pending
        return ScorerRequirements(
            frozenset(needs),
            note="surrogate until (theta*dL/dtheta)^2 graded against a BF16 parent",
        )

    def finalize(self) -> ScoreTable:
        rows = [
            ChannelScore(layer=layer, expert=e, channel=c, taylor=v)
            for (layer, e, c), v in sorted(self._scores.items())
        ]
        return ScoreTable(
            model=self._model.arch.name,
            scorer_versions={self.name: self.version},
            rows=rows,
        )
