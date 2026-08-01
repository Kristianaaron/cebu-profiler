"""Cross-layer route pathways and neuron/channel sensitivity (v2 §14, §18).

Cross-layer pathways: compress each token's route into a signature — for every
layer, the sorted set of experts routed. Then measure path frequency, per-
success-state rates, capability-specific paths, and the layers where success
and failure routes diverge.

Channel sensitivity (§18): for an expert's real weight matrices, zero each
channel (matrix row) and measure the resulting change in the expert's output
direction — a magnitude- and activation-conditioned sensitivity proxy.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass

from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.schemas.ontology import SuccessState

PathSignature = tuple[tuple[int, ...], ...]  # per-layer sorted selected expert ids


def route_path(
    model: MiniMoE, tokens: list[int], top_k: int | None = None, token_index: int = 0
) -> PathSignature:
    """Per-token route signature across all layers (v2 §14)."""
    result = forward(model, tokens, top_k=top_k)
    sig: list[tuple[int, ...]] = []
    for trace in result.traces:
        sig.append(tuple(sorted(trace.topk_ids[token_index])))
    return tuple(sig)


@dataclass
class PathRecord:
    signature: PathSignature
    count: int
    success_rate: float  # fraction of occurrences in SUCCESS state
    labels: list[str]  # labels that appeared with this path (top few)


@dataclass
class PathStats:
    records: list[PathRecord]
    top_frequent: int
    n_layers: int
    n_experts: int

    def most_frequent(self, topk: int = 10) -> list[PathRecord]:
        return sorted(self.records, key=lambda r: -r.count)[:topk]

    def success_rate(self, index: int) -> float:
        return self.records[index].success_rate


def path_stats(
    model: MiniMoE,
    samples: list[CalibrationSample],
    *,
    top_k: int | None = None,
    token_index: int = 0,
) -> PathStats:
    """Aggregate route paths across a corpus with success-rate + label linkage."""
    counts: dict[PathSignature, int] = defaultdict(int)
    success: dict[PathSignature, int] = defaultdict(int)
    labels: dict[PathSignature, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        sig = route_path(model, s.tokens, top_k=top_k, token_index=token_index)
        counts[sig] += 1
        if s.success_state == SuccessState.SUCCESS:
            success[sig] += 1
        for label in s.labels:
            labels[sig][label.value] += 1

    records = [
        PathRecord(
            signature=sig,
            count=c,
            success_rate=(success[sig] / c),
            labels=sorted(labels[sig], key=lambda v: labels[sig][v], reverse=True)[:4],
        )
        for sig, c in counts.items()
    ]
    n_layers = model.arch.num_text_layers
    return PathStats(records=records, top_frequent=0, n_layers=n_layers, n_experts=model.n_exp)


def capability_paths(
    model: MiniMoE,
    samples: list[CalibrationSample],
    *,
    label: str,
    top_k: int | None = None,
    topk: int = 5,
    token_index: int = 0,
) -> list[PathSignature]:
    """Most-frequent path signatures seen for a capability label (v2 §14)."""
    counts: dict[PathSignature, int] = defaultdict(int)
    for s in samples:
        if any(labx.value == label for labx in s.labels):
            sig = route_path(model, s.tokens, top_k=top_k, token_index=token_index)
            counts[sig] += 1
    ranked = sorted(counts, key=lambda sig: counts[sig], reverse=True)
    return ranked[:topk]


@dataclass
class DivergenceResult:
    layer: int
    success_experts: set[int]
    failure_experts: set[int]
    jaccard: float
    distinct_failure_only: int
    distinct_success_only: int


def success_failure_divergence(
    model: MiniMoE,
    samples: list[CalibrationSample],
    *,
    layer: int,
    top_k: int | None = None,
    token_index: int = 0,
) -> DivergenceResult:
    """How different success vs failure routes are at a layer (v2 §12/§14)."""
    success_experts: set[int] = set()
    failure_experts: set[int] = set()
    for s in samples:
        if s.success_state not in (SuccessState.SUCCESS, SuccessState.FAILURE):
            continue
        result = forward(model, s.tokens, top_k=top_k)
        ids = set(result.traces[layer].topk_ids[token_index])
        if s.success_state == SuccessState.SUCCESS:
            success_experts |= ids
        else:
            failure_experts |= ids
    union = success_experts | failure_experts
    inter = success_experts & failure_experts
    jaccard = len(inter) / len(union) if union else 0.0
    return DivergenceResult(
        layer=layer,
        success_experts=success_experts,
        failure_experts=failure_experts,
        jaccard=jaccard,
        distinct_failure_only=len(failure_experts - success_experts),
        distinct_success_only=len(success_experts - failure_experts),
    )


# --------------------------------------------------------------------------- #
# Neuron / channel sensitivity (§18)
# --------------------------------------------------------------------------- #


@dataclass
class ChannelSensitivity:
    layer: int
    expert: int
    n_channels: int
    sensitivity: list[float]  # per-channel relative output delta when zeroed
    top_channels: list[tuple[int, float]]  # (channel, sensitivity) desc

    def top_channel(self) -> tuple[int, float]:
        return self.top_channels[0]


def channel_sensitivity(
    model: MiniMoE,
    tokens: list[int],
    *,
    layer: int,
    expert: int,
    token_index: int = 0,
) -> ChannelSensitivity:
    """Zero each 'down' output channel and measure the expert-output change.

    The expert's down-projection (down: [hidden, mid]) maps latent->output; each
    of its `hidden` rows is a channel. Zeroing row c and rerunning on the same
    frozen input gives that channel's contribution to the output vector.
    """
    k = model.arch.moe.top_k
    override = {(layer, token_index): [expert] * k}
    base = forward(model, tokens, route_override=override)
    base_dir = list(base.traces[layer].combined[token_index])
    down = model.layers[layer].experts[expert]["down"]
    n_channels = len(down)  # hidden dim (rows of down)
    sensitivities: list[float] = []
    for c in range(n_channels):
        m2 = copy.deepcopy(model)
        # zero channel c in the expert's down projection
        m2.layers[layer].experts[expert]["down"][c] = [0.0] * len(down[c])
        res = forward(m2, tokens, route_override=override)
        qdir = list(res.traces[layer].combined[token_index])
        rel = _rel_l2(base_dir, qdir)
        sensitivities.append(rel)
    top = sorted(enumerate(sensitivities), key=lambda x: -x[1])
    return ChannelSensitivity(
        layer=layer,
        expert=expert,
        n_channels=n_channels,
        sensitivity=sensitivities,
        top_channels=top,
    )


def _rel_l2(a: list[float], b: list[float]) -> float:
    sse = sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
    norm = sum(x * x for x in a)
    return math.sqrt(sse / norm) if norm > 0 else 0.0
