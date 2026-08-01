"""Streamed REAP saliency + success/failure/recovery contrasts (v2 §9/§12).

Accumulates per (layer, expert, capability-label, trajectory-stage):

    base REAP score  = mean over calibration tokens of  router_prob(e) * norm(expert_out(e))

and separately per success-state so we can contrast e.g. `success − failure`
and `recovery − unrecovered`, and find which experts participate in recovery
versus which appear only in one group. All values are `measured` (real math).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.schemas.ontology import (
    CapabilityLabel,
    SuccessState,
    TrajectoryStage,
)

_SUC = str
_LBL = str
_STG = str


@dataclass
class CalibrationSample:
    tokens: list[int]
    labels: list[CapabilityLabel]
    stage: TrajectoryStage
    success_state: SuccessState = SuccessState.UNKNOWN


def _iter_events(
    model: MiniMoE, sample: CalibrationSample, top_k: int | None
) -> Iterator[tuple[int, int, float, bool]]:
    """Yield (layer, expert, router-weighted score, routed) per calibration token."""
    n_exp = model.n_exp
    result = forward(model, sample.tokens, top_k=top_k)
    for trace in result.traces:
        for t in range(len(sample.tokens)):
            sel = set(trace.topk_ids[t])
            for e in range(n_exp):
                yield trace.layer, e, trace.router_weighted[t][e], e in sel


@dataclass
class SaliencyAccumulator:
    # (layer, expert, label, stage) -> accumulated score / count / routing freq
    _sum: dict[tuple[int, int, str, str], float] = field(default_factory=lambda: defaultdict(float))
    _count: dict[tuple[int, int, str, str], int] = field(default_factory=lambda: defaultdict(int))
    _freq: dict[tuple[int, int, str, str], int] = field(default_factory=lambda: defaultdict(int))

    def add(
        self,
        layer: int,
        expert: int,
        score: float,
        label: CapabilityLabel,
        stage: TrajectoryStage,
        *,
        routed: bool,
    ) -> None:
        key = (layer, expert, label.value, stage.value)
        self._sum[key] += score
        self._count[key] += 1
        if routed:
            self._freq[key] += 1

    def mean(
        self, layer: int, expert: int, label: CapabilityLabel, stage: TrajectoryStage
    ) -> float:
        key = (layer, expert, label.value, stage.value)
        c = self._count[key]
        return self._sum[key] / c if c else 0.0

    def frequency(
        self, layer: int, expert: int, label: CapabilityLabel, stage: TrajectoryStage
    ) -> float:
        key = (layer, expert, label.value, stage.value)
        c = self._count[key]
        return self._freq[key] / c if c else 0.0

    def total_value(self, layer: int, expert: int) -> float:
        """Mean measured saliency of an expert across all cells (kept experts)."""
        tot = 0.0
        n = 0
        for (l2, e2, _lab, _stg), s in self._sum.items():
            if l2 == layer and e2 == expert:
                tot += s
                n += self._count[(l2, e2, _lab, _stg)]
        return tot / n if n else 0.0

    def rank(
        self,
        label: CapabilityLabel,
        stage: TrajectoryStage | None = None,
        topk: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Top (layer, expert, mean-score) pairs; unique when stage is None."""
        lab = label.value
        if stage is not None:
            stg = stage.value
            rows = [
                (
                    layer,
                    expert,
                    self._sum[(layer, expert, lab, stg)] / self._count[(layer, expert, lab, stg)],
                )
                for (layer, expert, lbl, stg2), _ in self._sum.items()
                if lbl == lab and stg2 == stg
            ]
            rows.sort(key=lambda r: r[2], reverse=True)
            return rows[:topk]

        agg_sum: dict[tuple[int, int], float] = {}
        agg_count: dict[tuple[int, int], int] = {}
        for (layer, expert, lbl, _stg), s in self._sum.items():
            if lbl != lab:
                continue
            key = (layer, expert)
            agg_sum[key] = agg_sum.get(key, 0.0) + s
            agg_count[key] = agg_count.get(key, 0) + self._count[(layer, expert, lbl, _stg)]
        rows = [
            (layer, expert, agg_sum[(layer, expert)] / agg_count[(layer, expert)])
            for layer, expert in agg_sum
        ]
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:topk]


@dataclass
class ContrastAccumulator:
    # (layer, expert, label, stage, success) -> sum / count / routed
    _sum: dict[tuple[int, int, _LBL, _STG, _SUC], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    _count: dict[tuple[int, int, _LBL, _STG, _SUC], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _routed: dict[tuple[int, int, _LBL, _STG, _SUC], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def add(
        self,
        layer: int,
        expert: int,
        score: float,
        label: CapabilityLabel,
        stage: TrajectoryStage,
        success: SuccessState,
        *,
        routed: bool,
    ) -> None:
        key = (layer, expert, label.value, stage.value, success.value)
        self._sum[key] += score
        self._count[key] += 1
        if routed:
            self._routed[key] += 1

    def saliency(
        self,
        layer: int,
        expert: int,
        label: CapabilityLabel,
        stage: TrajectoryStage,
        success: SuccessState,
    ) -> float:
        key = (layer, expert, label.value, stage.value, success.value)
        c = self._count[key]
        return self._sum[key] / c if c else 0.0

    def contrast(
        self,
        label: CapabilityLabel,
        pos: SuccessState,
        neg: SuccessState,
        stage: TrajectoryStage | None = None,
        topk: int = 20,
    ) -> list[tuple[int, int, float]]:
        """Ranked (layer, expert, pos_saliency − neg_saliency) pairs."""
        pairs = self._folded_pairs(label, stage)
        rows: list[tuple[int, int, float]] = []
        for layer, expert in self._pairs_for(label, stage):
            cell = pairs.get((layer, expert), {})
            delta = cell.get(pos.value, 0.0) - cell.get(neg.value, 0.0)
            rows.append((layer, expert, delta))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:topk]

    def participates(
        self,
        label: CapabilityLabel,
        states: set[SuccessState],
        stage: TrajectoryStage | None = None,
    ) -> list[tuple[int, int]]:
        """(layer, expert) pairs routed at least once in any of `states`."""
        stgs = {stage.value} if stage else self._stages()
        out: set[tuple[int, int]] = set()
        for (layer, expert, lbl, stg, suc), c in self._routed.items():
            if lbl == label.value and stg in stgs and SuccessState(suc) in states and c > 0:
                out.add((layer, expert))
        return sorted(out)

    # -- internal helpers --

    def _stages(self) -> set[str]:
        return {stg for (_, _, _, stg, _) in self._sum}

    def _pairs_for(
        self, label: CapabilityLabel, stage: TrajectoryStage | None
    ) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for (layer, expert, lbl, stg, _suc), _v in self._sum.items():
            if lbl == label.value and (stage is None or stg == stage.value):
                pairs.add((layer, expert))
        return pairs

    def _folded_pairs(
        self, label: CapabilityLabel, stage: TrajectoryStage | None
    ) -> dict[tuple[int, int], dict[_SUC, float]]:
        """(layer, expert) -> success -> mean saliency, aggregated across stages."""
        acc: dict[tuple[int, int], dict[_SUC, tuple[float, int]]] = {}
        for (layer, expert, lbl, stg, suc), _s in self._sum.items():
            if lbl != label.value or (stage is not None and stg != stage.value):
                continue
            cur = acc.setdefault((layer, expert), {})
            cur[suc] = (
                cur.get(suc, (0.0, 0))[0] + self._sum[(layer, expert, lbl, stg, suc)],
                cur.get(suc, (0.0, 0))[1] + self._count[(layer, expert, lbl, stg, suc)],
            )
        return {
            key: {s: ssum / cnt for s, (ssum, cnt) in inner.items()} for key, inner in acc.items()
        }


def run_calibration(
    model: MiniMoE, samples: list[CalibrationSample], top_k: int | None = None
) -> SaliencyAccumulator:
    """Run the model over a calibration corpus and accumulate REAP saliency."""
    acc = SaliencyAccumulator()
    for sample in samples:
        for layer, expert, score, routed in _iter_events(model, sample, top_k):
            for label in sample.labels:
                acc.add(layer, expert, score, label, sample.stage, routed=routed)
    return acc


def run_contrast(
    model: MiniMoE, samples: list[CalibrationSample], top_k: int | None = None
) -> ContrastAccumulator:
    """Accumulate saliency split by success-state for contrast analysis."""
    acc = ContrastAccumulator()
    for sample in samples:
        for layer, expert, score, routed in _iter_events(model, sample, top_k):
            for label in sample.labels:
                acc.add(
                    layer, expert, score, label, sample.stage, sample.success_state, routed=routed
                )
    return acc


def make_synthetic_corpus(
    *,
    n_samples: int,
    seq_len: int,
    vocab: int,
    seed: int = 0,
) -> tuple[list[CalibrationSample], list[CapabilityLabel], list[TrajectoryStage]]:
    """Deterministic calibration corpus cycling through labels, stages, success states."""
    labels = list(CapabilityLabel)
    stages = list(TrajectoryStage)
    successes = list(SuccessState)
    rng = random.Random(seed)
    samples: list[CalibrationSample] = []
    for i in range(n_samples):
        tokens = [rng.randrange(vocab) for _ in range(seq_len)]
        samples.append(
            CalibrationSample(
                tokens=tokens,
                labels=[labels[i % len(labels)]],
                stage=stages[i % len(stages)],
                success_state=successes[i % len(successes)],
            )
        )
    return samples, labels, stages
