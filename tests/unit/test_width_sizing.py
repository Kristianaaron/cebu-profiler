import json
import struct
from pathlib import Path

import pytest

from model_atlas.checkpoint.safetensors import write_safetensors
from model_atlas.prune.width_sizing import SizingError, size_checkpoint_for_width


def _tiny(hidden: int = 32, full: int = 16) -> Path:
    import tempfile

    root = Path(tempfile.mkdtemp())
    config = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 1,
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": hidden,
        "moe_intermediate_size": full,
        "vocab_size": 8,
        "quantization_config": {"quant_algo": "NVFP4"},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, dict[str, object]] = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [8, hidden],
                                      "bytes": b"\x01\x02" * (8 * hidden)},
        "lm_head.weight": {"dtype": "BF16", "shape": [8, hidden],
                           "bytes": b"\x03\x04" * (8 * hidden)},
        "model.layers.0.input_layernorm.weight": {
            "dtype": "F32", "shape": [hidden],
            "bytes": struct.pack("<f", 1.0) * hidden},
        "model.layers.0.mlp.gate.weight": {"dtype": "BF16", "shape": [1, hidden],
                                           "bytes": b"\x05\x06" * hidden},
    }
    prefix = "model.layers.0.mlp.experts.0"
    tensors[f"{prefix}.gate_proj.weight"] = {
        # NVFP4-style expert FFN: U8 packed + group scale
        "dtype": "U8", "shape": [full, hidden // 2],
        "bytes": bytes(range(256)) * (full * hidden // 2 // 256),
    }
    tensors[f"{prefix}.gate_proj.weight_scale"] = {
        "dtype": "F8_E4M3", "shape": [full, hidden // 16],
        "bytes": b"\x7f" * (full * (hidden // 16)),
    }
    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    return root


def test_sizing_reads_full_width_and_classes_expert() -> None:
    root = _tiny(full=16, hidden=32)
    sizing = size_checkpoint_for_width(root)
    assert sizing.full_width == 16
    assert sizing.shards_scanned == 1
    # expert = gate_proj.weight (16*16 U8=256) + weight_scale (16*2 F8=32)
    assert sizing.expert_bytes == 256 + 32
    assert sizing.expert_gib > 0.0
    # protected = embed + lm_head + layernorm + router gate (all non-expert)
    assert sizing.protected_bytes > 0
    assert sizing.total_gib == pytest.approx(sizing.expert_gib + sizing.protected_gib)


def test_sizing_full_shared_across_experts() -> None:
    # two experts each with FFN -> expert bytes scale, full unchanged
    import tempfile

    root = Path(tempfile.mkdtemp())
    hidden, full = 32, 16
    (root / "config.json").write_text(json.dumps({"moe_intermediate_size": full}),
                                      encoding="utf-8")
    tensors: dict[str, dict[str, object]] = {}
    for e in range(2):
        p = f"model.layers.0.mlp.experts.{e}"
        tensors[f"{p}.up_proj.weight"] = {"dtype": "U8", "shape": [full, hidden // 2],
                                          "bytes": bytes(range(2)) * (full * hidden // 2)}
    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    sizing = size_checkpoint_for_width(root)
    assert sizing.expert_bytes == 2 * (full * (hidden // 2))


def test_sizing_fails_closed_on_missing_config() -> None:
    import tempfile

    root = Path(tempfile.mkdtemp())
    with pytest.raises(SizingError, match="config"):
        size_checkpoint_for_width(root)


def test_sizing_fails_closed_on_no_experts() -> None:
    import tempfile

    root = Path(tempfile.mkdtemp())
    (root / "config.json").write_text(json.dumps({"moe_intermediate_size": 16}),
                                      encoding="utf-8")
    # only a protected tensor, no expert FFN
    write_safetensors(root / "model-00001-of-00001.safetensors", {
        "model.layers.0.input_layernorm.weight": {"dtype": "F32", "shape": [32],
                                                  "bytes": struct.pack("<f", 1.0) * 32},
    })
    with pytest.raises(SizingError, match="expert"):
        size_checkpoint_for_width(root)
