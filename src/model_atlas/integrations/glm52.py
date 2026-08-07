"""GLM-5.2 architecture adapter (blueprint §4, §19-P3).

Holds the *structural* layout facts for zai-org/GLM-5.2 / nvidia/GLM-5.2-NVFP4
(from the published config.json, as cited in the blueprint §4). Exact per-tensor
shapes and hashes must come from a real checkpoint census (needs the download);
this module provides the layout contract + drift-failing checks.

Routed-expert geometry (blueprint §4):
    hidden_size 6144, 78 layers, 3 dense + 75 sparse MoE,
    256 routed experts, 1 shared expert, top-8, moe_intermediate 2048,
    router fp32, silu.
"""

from __future__ import annotations

from model_atlas.schemas.architecture import (
    ArchitectureSpec,
    DType,
    LayerKind,
    MoELayout,
)

GLM52_NAME = "glm-5.2"
NUM_LAYERS = 78
NUM_DENSE_LAYERS = 3
NUM_SPARSE_LAYERS = 75  # layers 3..77
HIDDEN = 6144
NUM_ROUTED = 256
NUM_SHARED = 1
TOP_K = 8
MOE_INTERMEDIATE = 2048
PER_EXPERT_PARAMS = 3 * HIDDEN * MOE_INTERMEDIATE  # gate + up + down


def glm52_layout_params() -> dict[str, int]:
    """Structural parameter math used by the drift checks."""
    return {
        "per_expert_params": PER_EXPERT_PARAMS,
        "routed_expert_params": NUM_SPARSE_LAYERS * NUM_ROUTED * PER_EXPERT_PARAMS,
    }


def glm52_spec() -> ArchitectureSpec:
    """Structural GLM-5.2 spec (real tensor sizes require checkpoint census)."""
    # Layers 0-2 dense; layers 3-77 are routed-MoE sparse layers.
    layers_by_kind = {LayerKind.DENSE: NUM_DENSE_LAYERS, LayerKind.MOE: NUM_SPARSE_LAYERS}
    return ArchitectureSpec(
        name=GLM52_NAME,
        num_text_layers=NUM_LAYERS,
        layers_by_kind=layers_by_kind,
        moe=MoELayout(
            num_routed_experts=NUM_ROUTED,
            top_k=TOP_K,
            num_shared_experts=NUM_SHARED,
            latent_dim=MOE_INTERMEDIATE,  # moe_intermediate_size working space
            hidden_dim=HIDDEN,
            expert_dtype=DType.MXFP4,  # current source checkpoint is NVFP4-quantized
            dense_dtype=DType.BF16,
        ),
        hidden_dim=HIDDEN,
        vocabulary_size=None,  # requires measurement
        tensor_params={},  # structural only; real sizes from census
    )
