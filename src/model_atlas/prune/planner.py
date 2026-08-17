"""Byte-budget -> retention-width planner for uniform NVFP4 width pruning.

The canary/GLM path uses a uniform retention width per expert (the exporter
requires a single aligned ``width`` per sparse layer). Given how much of the
source footprint is expert FFN versus the protected backbone, solve the width
that keeps a derivative under a resident-byte budget. This is a deterministic
*estimate* (the recipe marks evidence PREDICTED, never MEASURED); the KLD gate
is what actually decides quality after the derivative is built.
"""
from __future__ import annotations

from model_atlas.prune.channel_saliency import GROUP_VALUES


class PlannerError(ValueError):
    """Raised when the budget/geometry cannot yield a valid width."""


def plan_uniform_width(
    *,
    expert_source_gib: float,
    protected_gib: float,
    target_gib: float,
    full: int,
    min_width: int = GROUP_VALUES,
    group: int = GROUP_VALUES,
) -> int:
    """Return the aligned retention ``width`` that fits ``target_gib``.

    Model: resident footprint ~= protected_gib + expert_source_gib*(width/full).
    We solve for ``width`` and clamp + 16-align it. ``protected_gib`` is the
    non-prunable backbone/embedding/KV estimate that width pruning leaves
    untouched. Fails closed on an impossible target.
    """
    if expert_source_gib <= 0.0:
        raise PlannerError("expert_source_gib must be positive")
    if target_gib <= protected_gib:
        raise PlannerError(
            f"target_gib {target_gib} leaves no room for experts under "
            f"protected_gib {protected_gib}"
        )
    if full <= 0 or full % group != 0:
        raise PlannerError(f"full {full} must be a positive multiple of group {group}")
    if min_width <= 0 or min_width % group != 0:
        raise PlannerError(f"min_width {min_width} must be a positive multiple of group")
    if min_width > full:
        raise PlannerError("min_width cannot exceed full")

    budget = target_gib - protected_gib
    frac = budget / expert_source_gib
    width = frac * full
    if width <= 0.0:
        raise PlannerError("budget admits no expert retention")
    width = int(width)
    width -= width % group  # 16-align down
    width = max(width, min_width)
    width = min(width, full)
    return width


__all__ = ["PlannerError", "plan_uniform_width"]
