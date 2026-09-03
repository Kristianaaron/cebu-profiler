"""Kimi K3 architecture adapter.

Holds the *structural* layout facts for the Kimi K3 parent checkpoint, as cited
in the project blueprint §4. Exact per-tensor shapes and hashes must come from a
real checkpoint census; this module provides the layout contract + drift-failing
checks. Figures are from the Kimi K3 model card / technical report; they describe
layout, not exact per-tensor shapes, which is why `tensor_params` is
intentionally empty and vocab size is unknown.
"""

from __future__ import annotations

from model_atlas.schemas.architecture import (
    ArchitectureSpec,
    DType,
    LayerKind,
    MoELayout,
)

K3_NAME = "k3"


def k3_spec() -> ArchitectureSpec:
    """Kimi K3 layout (structural facts only — tensor sizes require measurement)."""
    return ArchitectureSpec(
        name=K3_NAME,
        total_params=2_800_000_000_000,  # ~2.8T (reported)
        active_params=104_000_000_000,  # ~104B active (reported)
        checkpoint_bytes=1_560_000_000_000,  # ~1.56 TB (reported)
        num_text_layers=93,
        layers_by_kind={LayerKind.KDA: 69, LayerKind.MLA: 24},
        moe=MoELayout(
            num_routed_experts=896,
            top_k=16,
            num_shared_experts=2,
            latent_dim=3584,  # Stable LatentMoE working space
            hidden_dim=7168,  # residual width
            expert_dtype=DType.MXFP4,  # routed experts are source MXFP4
            dense_dtype=DType.BF16,  # non-expert tensors primarily BF16
        ),
        hidden_dim=7168,
        vocabulary_size=None,  # requires source measurement
        tensor_params={},
    )
