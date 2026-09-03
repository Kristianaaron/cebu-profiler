"""Elastic NVMe-overflow simulation (v2 §14 / §5-B).

A resident core plus an overflow tier. Over real routing traces, count cold
misses when a routed expert lives on the overflow tier and is not loaded, and
decide whether overflow helps: if the token-level cold-miss rate exceeds the
threshold, overflow is DISABLED (fall back to a broader resident profile) —
v2 §14 stop condition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.serving.runtime import per_expert_bytes


@dataclass
class ElasticPolicy:
    resident: set[tuple[int, int]] = field(default_factory=set)  # (layer, slot)
    overflow: set[tuple[int, int]] = field(default_factory=set)


def build_resident_policy(derivative: MiniMoE, resident_fraction: float = 0.6) -> ElasticPolicy:
    """Split each layer's kept experts into resident core + overflow tier by slot
    order (slots are value-ordered by the planner)."""
    resident: set[tuple[int, int]] = set()
    overflow: set[tuple[int, int]] = set()
    for layer in range(derivative.arch.num_text_layers):
        n = len(derivative.layers[layer].experts)
        n_res = max(1, int(n * resident_fraction))
        for slot in range(n):
            (resident if slot < n_res else overflow).add((layer, slot))
    return ElasticPolicy(resident=resident, overflow=overflow)


@dataclass
class MissResult:
    resident_bytes: float
    overflow_bytes: float
    total_routed: int
    cold_misses: int
    miss_rate: float
    overflow_disabled: bool  # stop condition triggered

    def decision(self) -> str:
        return "overflow_disabled" if self.overflow_disabled else "overflow_kept"


def simulate_overflow(
    derivative: MiniMoE,
    tokens: list[int],
    policy: ElasticPolicy,
    *,
    miss_threshold: float = 0.1,
) -> MissResult:
    """Count cold overflow misses over a real forward's routed experts + decide."""
    result = forward(derivative, tokens)
    routed_experts: Counter[tuple[int, int]] = Counter()
    for trace in result.traces:
        for ids in trace.topk_ids:
            for slot in ids:
                routed_experts[(trace.layer, slot)] += 1

    total_routed = sum(routed_experts.values())
    cold = sum(c for (layer, slot), c in routed_experts.items() if (layer, slot) in policy.overflow)
    miss_rate = cold / total_routed if total_routed else 0.0
    b = per_expert_bytes(derivative)

    return MissResult(
        resident_bytes=len(policy.resident) * b,
        overflow_bytes=len(policy.overflow) * b,
        total_routed=total_routed,
        cold_misses=cold,
        miss_rate=miss_rate,
        overflow_disabled=miss_rate > miss_threshold,
    )
