"""Model-agnostic tensor classification by role / layer / expert.

Heuristics over tensor names. Model-agnostic: patterns are convention-based.
Unmatched tensors get `role=None` + `unclassified=True` so the structural graph
can fail closed rather than silently drop coverage (no-unclassified invariant).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from model_atlas.schemas.architecture import TensorRole

_LAYER_RE = re.compile(r"layers\.(\d+)\b", re.IGNORECASE)
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\b", re.IGNORECASE)


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

    role: TensorRole | None
    if "shared" in low or "share" in low:
        role = TensorRole.SHARED_EXPERT
    elif "router" in low:
        role = TensorRole.ROUTER
    elif expert_index is not None or "expert" in low:
        role = TensorRole.EXPERTS
    elif "embed" in low:
        role = TensorRole.EMBEDDING
        layer_index = None
    elif "lm_head" in low or (low.endswith("output") and "norm" not in low) or "eh_proj" in low:
        role = TensorRole.LM_HEAD
        layer_index = None
    elif (
        "latent" in low
        or "gate_proj" in low
        or "up_proj" in low
        or "down_proj" in low
        or "gate" in low
    ):
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
    )
