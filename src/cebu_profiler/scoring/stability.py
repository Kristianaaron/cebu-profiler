"""Stability/confidence aggregator (blueprint §7-Module E).

Combines several independent runs (e.g. per calibration split / seed / context)
of per-channel importance into a single rank-stable, confidence-carrying view,
so the planner heavily penalizes low-confidence pruning decisions even when the
average importance is low.
"""

from __future__ import annotations

from cebu_profiler.scoring.base import (
    ChannelScore,
    ProfilerScorer,
    ScorerRequirements,
    ScoreTable,
)

_ScoreMap = dict[tuple[int, int, int], float]


def _rank(items: list[float]) -> dict[float, float]:
    """Average ranks for a list of scores (ties share the mean rank)."""
    order = sorted(range(len(items)), key=lambda i: items[i], reverse=True)
    ranks: dict[float, float] = {}
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and items[order[j + 1]] == items[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[items[order[k]]] = avg
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation between two equal-length score lists."""
    n = len(a)
    if n < 2:
        return 1.0
    ra = _rank(a)
    rb = _rank(b)
    d2 = sum((ra[x] - rb[y]) ** 2 for x, y in zip(a, b, strict=True))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


class StabilityAggregator(ProfilerScorer):
    name = "stability"
    version = "1.0"

    def __init__(self, runs: list[_ScoreMap]) -> None:
        self._runs = runs

    def requirements(self) -> ScorerRequirements:
        # consumes already-computed importance maps; itself needs no raw tensors
        return ScorerRequirements(frozenset())

    def finalize(self) -> ScoreTable:
        model = "synthetic"
        rows = self.aggregate()  # list of ChannelScore
        return ScoreTable(model=model, scorer_versions={self.name: self.version}, rows=rows)

    def aggregate(self) -> list[ChannelScore]:
        if not self._runs:
            return []
        # all channels that appear in any run
        channels = {key for run in self._runs for key in run}
        out: list[ChannelScore] = []
        # group by expert to get within-expert rankings/stability
        by_expert: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for layer, e, c in channels:
            by_expert.setdefault((layer, e), []).append((layer, e, c))

        for (layer, e), chans in by_expert.items():
            chans_sorted = sorted(chans, key=lambda k: (k[2],))
            for key in chans_sorted:
                vals = [run.get(key, 0.0) for run in self._runs]
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                stability = mean / (mean + std) if mean + std > 0 else (1.0 if mean > 0 else 0.0)

                # within-expert rank correlation across runs vs the mean ranking
                ranks_per_run = []
                for run in self._runs:
                    v_list = [run.get(k, 0.0) for k in chans_sorted]
                    m_list = [
                        sum(r.get(k, 0.0) for r in self._runs) / len(self._runs)
                        for k in chans_sorted
                    ]
                    ranks_per_run.append(_spearman(v_list, m_list))
                rank_stability = sum(ranks_per_run) / len(ranks_per_run)
                confidence = max(0.0, min(1.0, stability * rank_stability))

                out.append(
                    ChannelScore(
                        layer=layer,
                        expert=e,
                        channel=key[2],
                        stability=stability,
                        rank_stability=rank_stability,
                        confidence=confidence,
                    )
                )
        out.sort(key=lambda r: (r.layer, r.expert, r.channel))
        return out
