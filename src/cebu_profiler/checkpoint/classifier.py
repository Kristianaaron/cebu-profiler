"""Model-agnostic tensor classification by role / layer / expert.

Heuristics over tensor names. Model-agnostic: patterns are convention-based.
Unmatched tensors get `role=None` + `unclassified=True` so the structural graph
can fail closed rather than silently drop coverage (no-unclassified invariant).

Verified against real-world naming conventions:
- GLM / DeepSeek MoE routers are ``mlp.gate[.e_score_correction_bias]`` — the
  bare ``gate`` segment (not ``gate_proj``) is the router.
- Vision towers (``model.visual.*``) sub-name their tensors like the text stack
  (``attn``, ``norm``, ``gate_proj``), so the vision check must run first.
- NVFP4 checkpoints carry ``weight_scale_inv`` / ``weight_shape`` companions;
  these keep their owner's role but are flagged ``is_quant_metadata``.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from cebu_profiler.schemas.architecture import TensorRole

_LAYER_RE = re.compile(r"layers\.(\d+)\b", re.IGNORECASE)
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\b", re.IGNORECASE)
# `gate` as a standalone path segment = MoE router (GLM/DeepSeek naming).
# `gate_proj` (FFN weight) must NOT match: the segment must end at `gate`.
_ROUTER_GATE_RE = re.compile(r"(?:^|\.)(?:router|gate)(?:\.|$)", re.IGNORECASE)
# Standalone `norm` segment or an explicit *layernorm suffix. Checked before
# the attention branch so `post_attention_layernorm` lands on NORM, not
# ATTENTION (the substring "attention" alone would misfile it).
_NORM_RE = re.compile(r"(?:^|\.)norm(?:\.|$)|layernorm", re.IGNORECASE)

_QUANT_META_MARKERS = ("weight_scale", "weight_shape", "scale_inv")


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: TensorRole | None  # None => could not classify
    unclassified: bool
    layer_index: int | None = None
    expert_index: int | None = None
    is_quant_metadata: bool = False


def _parse_layer(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _parse_expert(name: str) -> int | None:
    m = _EXPERT_RE.search(name)
    return int(m.group(1)) if m else None


def is_unclassified(c: Classification) -> bool:
    return c.unclassified


def classify_tensor(name: str) -> Classification:
    """Classify a tensor name into a role (+ layer/expert when present)."""
    low = name.lower()
    expert_index = _parse_expert(name)
    layer_index = _parse_layer(name)
    is_quant_meta = any(marker in low for marker in _QUANT_META_MARKERS)

    role: TensorRole | None
    if "visual" in low:
        # Vision tower (ViT blocks, merger, patch embed, downsample). Its
        # sub-names imitate the text stack, so this must win over every
        # text-stack pattern below.
        role = TensorRole.VISION
    elif "shared" in low or "share" in low:
        role = TensorRole.SHARED_EXPERT
    elif "router" in low or (expert_index is None and _ROUTER_GATE_RE.search(low)):
        is_bias = "bias" in low or "correction" in low
        role = TensorRole.ROUTER_BIAS if is_bias else TensorRole.ROUTER
    elif expert_index is not None or "expert" in low:
        role = TensorRole.EXPERTS
    elif "embed" in low:
        role = TensorRole.EMBEDDING
        layer_index = None
    elif "lm_head" in low or (low.endswith("output") and "norm" not in low) or "eh_proj" in low:
        role = TensorRole.LM_HEAD
        layer_index = None
    elif _NORM_RE.search(low):
        role = TensorRole.NORM
    elif "latent" in low or "gate_proj" in low or "up_proj" in low or "down_proj" in low:
        role = TensorRole.LATENT_PROJ
    elif "kda" in low:
        role = TensorRole.KDA_DECAY
    elif "mla" in low:
        role = TensorRole.MLA_STATE
    elif "attn" in low or "attention" in low:
        role = TensorRole.ATTENTION
    elif "norm" in low or "layernorm" in low:
        role = TensorRole.NORM
    else:
        role = None

    return Classification(
        name=name,
        role=role,
        unclassified=role is None,
        layer_index=layer_index,
        expert_index=expert_index,
        is_quant_metadata=is_quant_meta,
    )
