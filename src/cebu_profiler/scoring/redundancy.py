"""Redundancy / substitutability and KEEP_VALUE (blueprint §8.3).

Measures channel *uniqueness* from the structural expert tensors: two FFN
channels are redundant when their output projections ``down[:,c]`` point along
the same direction (they produce near-identical expert-output contributions).
``uniqueness(c) = 1 - mean_{c'≠c} |cos(down[:,c], down[:,c'])|`` — high value =
distinct, low value = redundant.

KEEP_VALUE (blueprint §8.3) fuses the independent views:
    KEEP_VALUE = importance * causal_effect * uniqueness * stability
so a locally strong but redundant channel does not rank for retention.
"""

from __future__ import annotations

import math

from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.scoring.base import (
    ChannelScore,
    ProfilerScorer,
    ScoreNeed,
    ScorerRequirements,
    ScoreTable,
)


def _col_cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (na * nb)


def channel_uniqueness(model: MiniMoE) -> dict[tuple[int, int, int], float]:
    """Per-channel uniqueness from down-projection direction overlap."""
    uniq: dict[tuple[int, int, int], float] = {}
    for layer, layer_w in enumerate(model.layers):
        for e, exp in enumerate(layer_w.experts):
            down = exp["down"]  # [hidden, mid]
            mid = len(down[0]) if down else 0
            cols = [[row[c] for row in down] for c in range(mid)]
            for c in range(mid):
                if mid <= 1:
                    uniq[(layer, e, c)] = 1.0
                    continue
                overlap = sum(abs(_col_cos(cols[c], cols[d])) for d in range(mid) if d != c)
                uniq[(layer, e, c)] = 1.0 - overlap / (mid - 1)
    return uniq


def expert_uniqueness(model: MiniMoE) -> dict[tuple[int, int], float]:
    """Per-expert uniqueness = mean channel uniqueness."""
    per_channel = channel_uniqueness(model)
    acc: dict[tuple[int, int], list[float]] = {}
    for (layer, e, _c), v in per_channel.items():
        acc.setdefault((layer, e), []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def channel_kvalue(
    importance: dict[tuple[int, int, int], float],
    uniqueness: dict[tuple[int, int, int], float],
    causal: dict[tuple[int, int, int], float] | None = None,
    stability: dict[tuple[int, int, int], float] | None = None,
) -> dict[tuple[int, int, int], float]:
    """KEEP_VALUE = importance * causal_effect * uniqueness * stability (§8.3).

    Missing causal/stability views default to 1.0 (no penalty when unmeasured).
    """
    k = {}
    keys = set(importance) | set(uniqueness)
    for key in keys:
        imp = importance.get(key, 0.0)
        u = uniqueness.get(key, 1.0)
        c = causal.get(key, 1.0) if causal else 1.0
        s = stability.get(key, 1.0) if stability else 1.0
        k[key] = imp * c * u * s
    return k


class RedundancyScorer(ProfilerScorer):
    name = "redundancy"
    version = "1.0"

    def __init__(
        self,
        model: MiniMoE,
        tenp: dict[tuple[int, int, int], float] | None = None,
    ) -> None:
        self._model = model
        self._uniqueness = channel_uniqueness(model)
        self._kvalue = channel_kvalue(tenp or {}, self._uniqueness)

    def requirements(self) -> ScorerRequirements:
        return ScorerRequirements(frozenset({ScoreNeed.RAW_EXPERT_TENSORS}))

    def finalize(self) -> ScoreTable:
        rows = [
            ChannelScore(
                layer=layer,
                expert=e,
                channel=c,
                uniqueness=round(self._uniqueness[(layer, e, c)], 5),
                kvalue=round(self._kvalue.get((layer, e, c), 0.0), 5),
            )
            for (layer, e, c) in sorted(self._uniqueness)
        ]
        return ScoreTable(
            model=self._model.arch.name,
            scorer_versions={self.name: self.version},
            rows=rows,
        )
