"""TENP-style structural-importance scorer (blueprint §7-Module B, §3.1).

Ranks expert FFN channels by their projected contribution to the expert output:
`score(c) = mean|z_c| * ||down[:,c]||`, where `z_c = gate_c * up_c` is the
intermediate activation and `down[:,c]` the output-projection column. This is a
cheap, forward-pass-only ranking (needs activations + raw expert tensors), so it
runs immediately on the NVFP4 checkpoint (blueprint §9.1).
"""

from __future__ import annotations

import math
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


def _col_norm(down: list[list[float]], c: int) -> float:
    return math.sqrt(sum(row[c] * row[c] for row in down))


def tenp_rank(
    model: MiniMoE,
    stats: ChannelStatsAccumulator,
    expert_weight: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int, int], float]:
    """Per-(layer, expert, channel) TENP importance from measured statistics."""
    weights = expert_weight or {}
    final = {(s.layer, s.expert, s.channel): s for s in stats.finalize()}
    out: dict[tuple[int, int, int], float] = {}
    for layer, layer_w in enumerate(model.layers):
        for e, exp in enumerate(layer_w.experts):
            down = exp["down"]
            w = weights.get((layer, e), 1.0)
            for c in range(len(down[0])):
                # structured channel: activation magnitude x output-projection col
                s = final.get((layer, e, c))
                if s is None:
                    continue
                out[(layer, e, c)] = s.mean_abs * _col_norm(down, c) * w
    return out


class TenpScorer(AtlasScorer):
    name = "tenp"
    version = "1.0"

    def __init__(
        self,
        model: MiniMoE,
        stats: ChannelStatsAccumulator | None = None,
    ) -> None:
        self._model = model
        self._scores: dict[tuple[int, int, int], float] = {}
        if stats is not None:
            self.observe(stats)

    def requirements(self) -> ScorerRequirements:
        return ScorerRequirements(
            frozenset({ScoreNeed.FORWARD_ACTIVATIONS, ScoreNeed.RAW_EXPERT_TENSORS})
        )

    def observe(
        self,
        stats: ChannelStatsAccumulator,
        expert_weight: dict[tuple[int, int], float] | None = None,
    ) -> None:
        self._scores.update(tenp_rank(self._model, stats, expert_weight))

    def finalize(self) -> ScoreTable:
        rows = [
            ChannelScore(layer=layer, expert=e, channel=c, tenp=v)
            for (layer, e, c), v in sorted(self._scores.items())
        ]
        return ScoreTable(
            model=self._model.arch.name,
            scorer_versions={self.name: self.version},
            rows=rows,
        )
