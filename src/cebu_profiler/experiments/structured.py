"""Synthetic MoE with injected channel-importance structure (blueprint §17).

The stock ``k3-mini`` synthetic model draws i.i.d. Gaussian weights, so every
channel is roughly equally important and heterogeneous planning cannot
distinguish itself from a uniform control. For a meaningful differential-cost
experiment (Milestone E) this builds a deterministic variant where a subset of
experts carry a few strongly-important channels and the rest are weak, so the
measured TENP scorer has real signal to act on.
"""

from __future__ import annotations

from model_atlas.atlas.runtime import MiniMoE, build_mini_moe
from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.architecture import ArchitectureSpec


def build_structured_model(
    seed: int = 0,
    *,
    n_strong: int = 2,
    strong_scale: float = 8.0,
    channels: int = 4,
    arch: ArchitectureSpec | None = None,
) -> MiniMoE:
    """Mini-MoE where experts ``0..n_strong-1`` have ``channels`` scaled-up
    output-projection columns (high projected importance); other experts stay
    uniform. Importance is injected on the ``down`` columns only, so TENP
    (``mean_abs * ||down[:,c]||``) ranks them high without inflating the
    gate/up pre-activations (which would overflow the pure-Python silu)."""
    arch = arch or get_registry().get("k3-mini")
    model = build_mini_moe(arch, seed=seed)
    mid = model.mid
    n_strong = min(n_strong, model.n_exp)
    per_expert = min(channels, mid)
    for layer in range(len(model.layers)):
        layer_w = model.layers[layer]
        for e in range(n_strong):
            down = layer_w.experts[e]["down"]
            for c in range(per_expert):
                for j in range(model.hidden):
                    down[j][c] *= strong_scale
    return model
