"""Streamed REAP saliency over real routing/contribution traces (v2 §9 / §10).

Accumulates per (layer, expert, capability-label, trajectory-stage):

    base REAP score  = mean over calibration tokens of  router_prob(e) * norm(expert_out(e))

where both the router probability and the expert output norm are computed from
real activations by the mini-MoE forward pass. Also tracks routing frequency.
All values here are `measured` (real math), never fabricated.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.schemas.ontology import CapabilityLabel, TrajectoryStage


@dataclass
class CalibrationSample:
    tokens: list[int]
    labels: list[CapabilityLabel]
    stage: TrajectoryStage


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

    def rank(
        self,
        label: CapabilityLabel,
        stage: TrajectoryStage | None = None,
        topk: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Top (layer, expert, mean-score) pairs for a label.

        With `stage` given, only that stage is included. Without it, scores are
        aggregated across all stages so each (layer, expert) appears once.
        """
        lab = label.value
        rows: list[tuple[int, int, float]] = []
        if stage is not None:
            stg_key = stage.value
            for (layer, expert, lbl, stg), _s in self._sum.items():
                if lbl == lab and stg == stg_key:
                    rows.append(
                        (
                            layer,
                            expert,
                            self._sum[(layer, expert, lbl, stg)]
                            / self._count[(layer, expert, lbl, stg)],
                        )
                    )
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


def run_calibration(
    model: MiniMoE, samples: list[CalibrationSample], top_k: int | None = None
) -> SaliencyAccumulator:
    """Run the model over a calibration corpus and accumulate REAP saliency."""
    acc = SaliencyAccumulator()
    n_exp = model.n_exp
    for sample in samples:
        result = forward(model, sample.tokens, top_k=top_k)
        for trace in result.traces:
            for t in range(len(sample.tokens)):
                sel = set(trace.topk_ids[t])
                for e in range(n_exp):
                    score = trace.router_weighted[t][e]  # p_e * norm(expert_out_e)
                    for label in sample.labels:
                        acc.add(trace.layer, e, score, label, sample.stage, routed=(e in sel))
    return acc


def make_synthetic_corpus(
    *,
    n_samples: int,
    seq_len: int,
    vocab: int,
    seed: int = 0,
) -> tuple[list[CalibrationSample], list[CapabilityLabel], list[TrajectoryStage]]:
    """Deterministic calibration corpus cycling through all labels and stages."""
    labels = list(CapabilityLabel)
    stages = list(TrajectoryStage)
    rng = random.Random(seed)
    samples: list[CalibrationSample] = []
    for i in range(n_samples):
        tokens = [rng.randrange(vocab) for _ in range(seq_len)]
        samples.append(
            CalibrationSample(
                tokens=tokens,
                labels=[labels[i % len(labels)]],
                stage=stages[i % len(stages)],
            )
        )
    return samples, labels, stages
