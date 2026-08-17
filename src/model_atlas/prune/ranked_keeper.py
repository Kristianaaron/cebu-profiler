"""Saliency-ranked keep-map selection for NVFP4 width pruning.

Produces the per-(layer, expert) channel keep-map consumed by
``model_atlas.loader.materialize_uniform_width(..., keep_channels=...)``.

The exporter is atomic at 16-channel groups and requires, for every
``(sparse_layer, expert)`` target, an ascending union of ``width`` channels
made of aligned 16-channel groups with complete coverage. This module
selects the *highest-saliency* groups to reach ``width`` per expert, and the
result is deterministic (ties broken by ascending group index).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite

from model_atlas.prune.channel_saliency import GROUP_VALUES, SaliencyError


class KeepMapError(ValueError):
    """Raised when the saliency/geometry cannot yield a valid keep-map."""


def _validate_geometry(width: int, full: int, group: int) -> int:
    if width <= 0 or full <= 0:
        raise KeepMapError("width and full must be positive")
    if width > full:
        raise KeepMapError("width cannot exceed full channel count")
    if full % group != 0:
        raise KeepMapError(f"full {full} must be a multiple of group {group}")
    if width % group != 0:
        raise KeepMapError(f"width {width} must be a multiple of group {group}")
    return full // group


def _build_expert_keep(
    group_saliency: Sequence[float],
    width: int,
    full: int,
    group: int,
) -> list[int]:
    n_groups = _validate_geometry(width, full, group)
    if len(group_saliency) != n_groups:
        raise KeepMapError(
            f"group saliency length {len(group_saliency)} != groups {n_groups}"
        )
    for value in group_saliency:
        if not isfinite(value):
            raise SaliencyError("group_saliency must be finite")
    n_keep = width // group
    # rank groups by descending saliency, ties by ascending group index
    ranked = sorted(
        range(n_groups), key=lambda g: (-group_saliency[g], g)
    )
    keep_groups = sorted(ranked[:n_keep])
    channels: list[int] = []
    for g in keep_groups:
        channels.extend(range(g * group, (g + 1) * group))
    return channels


def select_keep_map(
    expert_group_saliency: Mapping[tuple[int, int], Sequence[float]],
    *,
    width: int,
    full: int,
    sparse_layers: Iterable[int],
    n_exp: int,
    group: int = GROUP_VALUES,
) -> dict[tuple[int, int], list[int]]:
    """Return ``{(layer, expert): [kept_channels_ascending]}`` for all targets.

    Every ``(layer, expert)`` for ``layer in sparse_layers`` and
    ``expert in range(n_exp)`` must be present in ``expert_group_saliency``,
    and each value must have exactly ``full // group`` finite saliencies.
    Raises if coverage is incomplete (fail closed — no silent fallback).
    """
    keep_map: dict[tuple[int, int], list[int]] = {}
    for li in sparse_layers:
        for e in range(n_exp):
            key = (int(li), int(e))
            if key not in expert_group_saliency:
                raise KeepMapError(
                    f"keep selection missing saliency for {key} — complete "
                    "coverage required"
                )
            keep_map[key] = _build_expert_keep(
                expert_group_saliency[key], width, full, group
            )
    return keep_map


__all__ = ["KeepMapError", "select_keep_map"]
