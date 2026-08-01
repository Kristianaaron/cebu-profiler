"""Model-agnostic architecture specification.

Describes the structural layout of a transformer/MoE model enough to enumerate
its tensors, account bytes, and plan placements without depending on any one
checkpoint. Real per-tensor shapes come from the actual checkpoint census when
available; the spec carries deterministic values for synthetic fixtures and
leaves source-derived sizes to measurement otherwise.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DType(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    MXFP4 = "mxfp4"


DTYPE_BYTES: dict[DType, float] = {
    DType.FP32: 4.0,
    DType.BF16: 2.0,
    DType.FP16: 2.0,
    DType.INT8: 1.0,
    DType.INT4: 0.5,
    DType.MXFP4: 0.5,
}


class LayerKind(StrEnum):
    KDA = "kda"
    MLA = "mla"
    DENSE = "dense"


class TensorRole(StrEnum):
    """Role of a tensor within the model. Every tensor must map to exactly one."""

    ROUTER = "router"
    ROUTER_BIAS = "router_bias"
    EXPERTS = "experts"  # a routed expert bank (one record per expert)
    SHARED_EXPERT = "shared_expert"
    LATENT_PROJ = "latent_proj"  # Stable LatentMoE input/output projections
    ATTENTION = "attention"
    MLA_STATE = "mla_state"
    KDA_DECAY = "kda_decay"
    NORM = "norm"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    VISION = "vision"


class MoELayout(BaseModel):
    """Per-layer mixture-of-experts geometry."""

    model_config = ConfigDict(extra="forbid")

    num_routed_experts: int
    top_k: int
    num_shared_experts: int = 0
    latent_dim: int  # Stable LatentMoE working space
    hidden_dim: int  # residual width in/out of the layer
    expert_dtype: DType = DType.MXFP4
    dense_dtype: DType = DType.BF16


def _numel_positive(v: int) -> int:
    if v <= 0:
        raise ValueError(f"expected positive value, got {v}")
    return v


class ArchitectureSpec(BaseModel):
    """Structural descriptor of a model family.

    `tensor_params` maps a TensorRole to the numel of ONE unit occurrence:
    for `EXPERTS`, numel of a single routed expert; for the other roles, numel
    of a single per-layer tensor (for per-layer roles) or the single global
    tensor (for `EMBEDDING` / `LM_HEAD`). It is REQUIRED for synthetic fixtures
    that need deterministic accounting, and left unset for real checkpoints
    whose sizes must be measured from the source.

    Numbers that are not known without measuring the real checkpoint (e.g. K3's
    vocabulary size, per-expert shapes) are `None`, never invented.
    """

    model_config = ConfigDict(extra="forbid")

    spec_version: str = "1.0"
    name: str
    total_params: int | None = None
    active_params: int | None = None
    checkpoint_bytes: int | None = None
    num_text_layers: int
    layers_by_kind: dict[LayerKind, int] = Field(default_factory=dict)
    moe: MoELayout
    hidden_dim: int
    vocabulary_size: int | None = None
    tensor_params: dict[TensorRole, int] = Field(default_factory=dict)

    @property
    def needs_source_measurement(self) -> bool:
        return not self.tensor_params
