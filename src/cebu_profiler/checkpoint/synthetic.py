"""Builder for a synthetic on-disk mini-K3 checkpoint (test fixture).

Writes a real (tiny) Safetensors checkpoint shaped like the miniature K3, so
the census → classifier → structural-graph path can be tested deterministically
without the 1.56 TB source. Optional: include one unclassified tensor name to
exercise the coverage-failure path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cebu_profiler.checkpoint.safetensors import write_safetensors

_NAMES = [
    "model.embed_tokens",  # EMBEDDING
    "model.lm_head",  # LM_HEAD
    "model.layers.0.input_layernorm.weight",  # NORM l0
    "model.layers.0.router.weight",  # ROUTER l0
    "model.layers.0.self_attn.q_proj.weight",  # ATTENTION l0
    "model.layers.0.mla_state.weight",  # MLA l0
    "model.layers.0.kda_decay.weight",  # KDA l0
    "model.layers.0.latent_proj.weight",  # LATENT l0
    "model.layers.0.experts.0.gate_proj.weight",  # EXPERT l0 e0
    "model.layers.0.experts.1.gate_proj.weight",  # EXPERT l0 e1
    "model.layers.0.moe_shared_expert.weight",  # SHARED l0
    "model.layers.1.input_layernorm.weight",  # NORM l1
    "model.layers.1.router.weight",  # ROUTER l1
    "model.layers.1.self_attn.q_proj.weight",  # ATTENTION l1
    "model.layers.1.mla_state.weight",  # MLA l1
    "model.layers.1.kda_decay.weight",  # KDA l1
    "model.layers.1.latent_proj.weight",  # LATENT l1
    "model.layers.1.experts.0.gate_proj.weight",  # EXPERT l1 e0
    "model.layers.1.experts.1.gate_proj.weight",  # EXPERT l1 e1
    "model.layers.1.moe_shared_expert.weight",  # SHARED l1
]

# A name the classifier does not recognize (for the coverage-failure test).
UNCLASSIFIED_NAME = "mystery.weights.quant_stuff"

# name -> (dtype, shape) ; bytes filled deterministically.
_LAYOUT: dict[str, tuple[str, list[int]]] = {
    "model.embed_tokens": ("F16", [1000, 128]),
    "model.lm_head": ("F16", [1000, 128]),
    "model.layers.0.input_layernorm.weight": ("F32", [128]),
    "model.layers.0.router.weight": ("F16", [8, 64]),
    "model.layers.0.self_attn.q_proj.weight": ("F16", [128, 128]),
    "model.layers.0.mla_state.weight": ("F16", [128, 128]),
    "model.layers.0.kda_decay.weight": ("F16", [64]),
    "model.layers.0.latent_proj.weight": ("F16", [128, 64]),
    "model.layers.0.experts.0.gate_proj.weight": ("F16", [8, 64]),
    "model.layers.0.experts.1.gate_proj.weight": ("F16", [8, 64]),
    "model.layers.0.moe_shared_expert.weight": ("F16", [64, 64]),
    "model.layers.1.input_layernorm.weight": ("F32", [128]),
    "model.layers.1.router.weight": ("F16", [8, 64]),
    "model.layers.1.self_attn.q_proj.weight": ("F16", [128, 128]),
    "model.layers.1.mla_state.weight": ("F16", [128, 128]),
    "model.layers.1.kda_decay.weight": ("F16", [64]),
    "model.layers.1.latent_proj.weight": ("F16", [128, 64]),
    "model.layers.1.experts.0.gate_proj.weight": ("F16", [8, 64]),
    "model.layers.1.experts.1.gate_proj.weight": ("F16", [8, 64]),
    "model.layers.1.moe_shared_expert.weight": ("F16", [64, 64]),
}

_CONFIG = {
    "model_type": "k3-mini",
    "num_hidden_layers": 2,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "hidden_size": 128,
    "latent_dim": 64,
}


def _bytes_for(dtype: str, shape: list[int]) -> bytes:
    """Deterministic filler bytes sized for dtype."""
    import math

    width = {"F16": 2, "F32": 4, "I8": 1, "BF16": 2}[dtype]
    n = math.prod(shape) if shape else 0
    return bytes(i % 256 for i in range(n * width))


def make_synthetic_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    include_unclassified: bool = False,
) -> str:
    """Write a synthetic mini-K3 checkpoint dir; return its path."""
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_CONFIG))

    tensors: dict[str, dict[str, object]] = {}
    names = list(_NAMES)
    if include_unclassified:
        names.append(UNCLASSIFIED_NAME)
        _LAYOUT[UNCLASSIFIED_NAME] = ("F16", [16, 16])
    for name in names:
        dtype, shape = _LAYOUT[name]
        tensors[name] = {"dtype": dtype, "shape": shape, "bytes": _bytes_for(dtype, shape)}

    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    return str(root)
