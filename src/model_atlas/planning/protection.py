"""Coalition/cross-path driven channel protection (blueprint §8.2, §12).

A locally modest channel can still be load-bearing because it participates in a
repeated downstream coalition/path (L29:E18 -> L36:E204 -> ...). Before pruning,
experts that appear in persistent co-routed coalitions are marked protected and
forced to keep their full channel set — the planner widens their budget rather
than cutting them (blueprint §8.2 preserves important trajectories).
"""

from __future__ import annotations

from model_atlas.atlas.coalition import coactivation_map
from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE


def coalition_protected_experts(
    model: MiniMoE,
    samples: list[CalibrationSample],
    top_k: int | None = None,
    min_coactivity: int = 1,
) -> set[tuple[int, int]]:
    """(layer, expert) pairs routed together persistently (coactivity "frequent").

    ``min_coactivity`` is the minimum number of co-routings before a pair counts
    as a persistent coalition; higher = more conservative protection.
    """
    protected: set[tuple[int, int]] = set()
    for layer in range(len(model.layers)):
        cmap = coactivation_map(model, samples, layer, top_k=top_k)
        for a, b in cmap.candidate_coalitions(min_coactivity=min_coactivity):
            protected.add((layer, a))
            protected.add((layer, b))
    return protected


def full_channel_protection(
    model: MiniMoE, experts: set[tuple[int, int]]
) -> dict[tuple[int, int], set[int]]:
    """Expand protected (layer, expert) pairs to keep-all-channels dicts.

    The compression planner consumes ``{(layer, expert): set(channels)}`` and
    guarantees every protected channel survives, so a protected expert is never
    intra-expert pruned — it only changes width via its own measured scores if
    all its channels remain.
    """
    return {(layer, e): set(range(model.mid)) for (layer, e) in experts}
