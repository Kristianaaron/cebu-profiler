"""Channel / neuron saliency math for profiler-driven width pruning.

Profiled per-neuron importance is aggregated up to the exporter's atomic
NVFP4 16-channel group granularity (see ``model_atlas.loader.GROUP_VALUES``).
All functions are pure and deterministic so the selection decision can be
proved with synthetic data before any real model is touched.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

GROUP_VALUES = 16


class SaliencyError(ValueError):
    """Raised when saliency evidence is malformed / non-finite."""


def neuron_saliency_from_router_norm(
    router_gate: float,
    expert_output_norms: Sequence[float],
) -> list[float]:
    """Elementwise REAP-style S = router_gate * norm(expert_output) per neuron.

    ``expert_output_norms`` must be finite and non-empty. This is the
    channel-level analogue of the per-expert REAP score.
    """
    if not isfinite(router_gate):
        raise SaliencyError("router_gate must be finite")
    if not expert_output_norms:
        raise SaliencyError("expert_output_norms must be non-empty")
    out: list[float] = []
    for value in expert_output_norms:
        if not isfinite(value):
            raise SaliencyError("expert_output_norm must be finite")
        out.append(router_gate * value)
    return out


def aggregate_group_saliency(
    neuron_saliency: Sequence[float],
    group: int = GROUP_VALUES,
) -> list[float]:
    """Aggregate per-neuron saliency into per-group saliency (mean of its members).

    ``len(neuron_saliency)`` must be a positive multiple of ``group``. The
    returned list is the mean saliency of each contiguous 16-channel group,
    preserving channel order so group index ``g`` maps to channels
    ``[g*group, (g+1)*group)`` exactly as the loader slices.
    """
    if group <= 0:
        raise SaliencyError("group must be positive")
    if not neuron_saliency:
        raise SaliencyError("neuron_saliency must be non-empty")
    if len(neuron_saliency) % group != 0:
        raise SaliencyError(
            f"neuron count {len(neuron_saliency)} is not a multiple of group {group}"
        )
    out: list[float] = []
    for start in range(0, len(neuron_saliency), group):
        block = neuron_saliency[start : start + group]
        for value in block:
            if not isfinite(value):
                raise SaliencyError("neuron_saliency must be finite")
        out.append(sum(block) / group)
    return out


def expert_saliency(
    group_saliency: Sequence[float],
) -> float:
    """Whole-expert fallback score = mean of its group saliencies.

    This is the higher-level ranking signal (decision C). It never removes an
    expert; it only tells us which experts tolerate the most aggressive width
    reduction and which must stay widest.
    """
    if not group_saliency:
        raise SaliencyError("group_saliency must be non-empty")
    return sum(group_saliency) / len(group_saliency)


__all__ = [
    "GROUP_VALUES",
    "SaliencyError",
    "aggregate_group_saliency",
    "expert_saliency",
    "neuron_saliency_from_router_norm",
]
