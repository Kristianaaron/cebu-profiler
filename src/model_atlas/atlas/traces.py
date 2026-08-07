"""Measured trace-record builder for blueprint §10 (Router/Expert/Channel)."""

from __future__ import annotations

from model_atlas.atlas.collector import ChannelStatsAccumulator
from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.schemas.trace_records import (
    ChannelAggregate,
    ExpertAggregate,
    RouterRecord,
    TraceRecords,
)
from model_atlas.scoring.tenp import tenp_rank


def trace_records(
    model: MiniMoE, samples: list[CalibrationSample], top_k: int | None = None
) -> TraceRecords:
    """Compose RouterRecord / ExpertAggregate / ChannelAggregate from measurements."""
    router: list[RouterRecord] = []
    acc = ChannelStatsAccumulator()
    for s_i, s in enumerate(samples):
        result = forward(model, s.tokens, top_k=top_k, channel_stats=acc)
        for trace in result.traces:
            for tok in range(len(s.tokens)):
                router.append(
                    RouterRecord(
                        sample_id=s_i,
                        token_index=s.tokens[tok],
                        layer_id=trace.layer,
                        selected_experts=list(trace.topk_ids[tok]),
                        gate_weights=[round(w, 5) for w in trace.topk_probs[tok]],
                        routing_entropy=round(trace.entropy[tok], 5),
                    )
                )

    # per (layer, expert) aggregates from the same forward traces
    exp_sums: dict[tuple[int, int], list[float]] = {}
    exp_count: dict[tuple[int, int], int] = {}
    for s in samples:
        result = forward(model, s.tokens, top_k=top_k)
        for trace in result.traces:
            n_tokens = len(s.tokens)
            for t in range(n_tokens):
                input_norm = trace.input_norm[t]
                for e in range(model.n_exp):
                    key = (trace.layer, e)
                    exp_sums.setdefault(key, [0.0, 0.0, 0.0])
                    exp_sums[key][0] += trace.router_weighted[t][e]
                    exp_sums[key][1] += trace.expert_norm[t][e]
                    if input_norm:
                        exp_sums[key][2] += trace.expert_norm[t][e] / input_norm
                    exp_count[key] = exp_count.get(key, 0) + 1
    experts: list[ExpertAggregate] = [
        ExpertAggregate(
            layer_id=layer,
            expert_id=e,
            activation_count=exp_count[key],
            gate_weight_mean=round(vals[0] / exp_count[key], 5),
            output_norm_mean=round(vals[1] / exp_count[key], 5),
            direction_change_mean=round(vals[2] / exp_count[key], 5),
        )
        for key, vals in sorted(exp_sums.items())
        for layer, e in [key]
    ]

    tenp = tenp_rank(model, acc)
    channels: list[ChannelAggregate] = [
        ChannelAggregate(
            layer_id=s.layer,
            expert_id=s.expert,
            channel_id=s.channel,
            activation_rms=round(s.rms, 5),
            activation_frequency=round(s.frequency, 5),
            tenp_score=round(tenp.get((s.layer, s.expert, s.channel), 0.0), 5),
        )
        for s in acc.finalize()
    ]
    return TraceRecords(router=router, experts=experts, channels=channels)
