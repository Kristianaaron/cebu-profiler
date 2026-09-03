"""A synthetic miniature MoE shaped like Kimi K3, small enough for fast,
deterministic unit tests. Every tensor count is chosen by us (synthetic) — the
full K3 counts must come from the real checkpoint, not from here.
"""

from __future__ import annotations

from model_atlas.schemas.architecture import (
    ArchitectureSpec,
    DType,
    LayerKind,
    MoELayout,
    TensorRole,
)

MINI_MOE_NAME = "k3-mini"

# Synthetic but explicit per-role numel (ONE unit occurrence). These are invented
# values for deterministic testing, documented as such.
_MIN_TENSOR_PARAMS: dict[TensorRole, int] = {
    TensorRole.ROUTER: 8 * 64,  # experts x latent_dim -> router logits
    TensorRole.ROUTER_BIAS: 8,
    TensorRole.EXPERTS: 8192,  # one routed expert
    TensorRole.SHARED_EXPERT: 2048,
    TensorRole.LATENT_PROJ: 128 * 64,  # hidden x latent projections
    TensorRole.ATTENTION: 4096,
    TensorRole.MLA_STATE: 2048,
    TensorRole.KDA_DECAY: 1024,
    TensorRole.NORM: 128,
    TensorRole.EMBEDDING: 1000 * 128,  # vocab x hidden
    TensorRole.LM_HEAD: 1000 * 128,
}


def mini_moe_spec() -> ArchitectureSpec:
    """Return the synthetic miniature K3-shaped MoE architecture."""
    return ArchitectureSpec(
        name=MINI_MOE_NAME,
        num_text_layers=2,
        layers_by_kind={LayerKind.KDA: 1, LayerKind.MLA: 1},
        moe=MoELayout(
            num_routed_experts=8,
            top_k=2,
            num_shared_experts=1,
            latent_dim=64,
            hidden_dim=128,
            expert_dtype=DType.MXFP4,
            dense_dtype=DType.BF16,
        ),
        hidden_dim=128,
        vocabulary_size=1000,
        tensor_params=_MIN_TENSOR_PARAMS,
    )
