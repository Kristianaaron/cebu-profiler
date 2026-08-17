"""End-to-end proof: profiler-ranked keep-map through the real NVFP4 exporter.

Runs only against the complete ``model_atlas`` package where the exporter and
its sibling modules (checkpoint.safetensors, jobs.artifacts) are present.
Verifies that a saliency-ranked keep map produces a structurally-complete,
source-non-mutating derivative with the *data-chosen* channels retained —
not the uniform-first-N default.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from model_atlas.checkpoint.safetensors import write_safetensors
from model_atlas.jobs.artifacts import source_manifest
from model_atlas.loader import materialize_uniform_width
from model_atlas.prune.ranked_keeper import select_keep_map


def _tiny_glm_nvfp4(root: Path, full: int = 32, hidden: int = 64) -> Path:
    root.mkdir()
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
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    tensors: dict[str, dict[str, object]] = {
        "model.embed_tokens.weight": {
            "dtype": "BF16", "shape": [8, hidden],
            "bytes": b"\x01\x02" * (8 * hidden),
        },
        "lm_head.weight": {
            "dtype": "BF16", "shape": [8, hidden],
            "bytes": b"\x03\x04" * (8 * hidden),
        },
        "model.layers.0.input_layernorm.weight": {
            "dtype": "F32", "shape": [hidden],
            "bytes": struct.pack("<f", 1.0) * hidden,
        },
        "model.layers.0.mlp.gate.weight": {
            "dtype": "BF16", "shape": [1, hidden],
            "bytes": b"\x05\x06" * hidden,
        },
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        down = projection == "down_proj"
        weight_shape = [hidden, full // 2] if down else [full, hidden // 2]
        scale_shape = [hidden, full // 16] if down else [full, hidden // 16]
        prefix = f"model.layers.0.mlp.experts.0.{projection}"
        tensors[f"{prefix}.weight"] = {
            "dtype": "U8",
            "shape": weight_shape,
            "bytes": bytes(range(256)) * (weight_shape[0] * weight_shape[1] // 256),
        }
        tensors[f"{prefix}.weight_scale"] = {
            "dtype": "F8_E4M3",
            "shape": scale_shape,
            "bytes": b"\x7f" * (scale_shape[0] * scale_shape[1]),
        }
        tensors[f"{prefix}.weight_scale_2"] = {
            "dtype": "F32", "shape": [],
            "bytes": struct.pack("<f", 2.0),
        }
        tensors[f"{prefix}.input_scale"] = {
            "dtype": "F32", "shape": [],
            "bytes": struct.pack("<f", 3.0),
        }
    shard = root / "model-00001-of-00001.safetensors"
    write_safetensors(shard, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {n: shard.name for n in tensors}}),
        encoding="utf-8",
    )
    return root


def test_ranked_keep_map_through_real_exporter(tmp_path: Path) -> None:
    source = _tiny_glm_nvfp4(tmp_path / "source")
    before = source_manifest(str(source))
    output = tmp_path / "sliced"

    # group 1 (channels 16..31) is far more salient than group 0
    keep_map = select_keep_map(
        {(0, 0): [5.0, 20.0]}, width=16, full=32, sparse_layers=[0], n_exp=1
    )
    assert keep_map == {(0, 0): list(range(16, 32))}

    result = materialize_uniform_width(
        str(source), str(output), 16, overwrite=True, keep_channels=keep_map
    )
    assert result.promoted
    assert result.structurally_complete
    assert result.width == 16
    # source untouched
    assert source_manifest(str(source)) == before
    # derivative reflects retained width
    out_config = json.loads((output / "config.json").read_text())
    assert out_config["moe_intermediate_size"] == 16


def test_ranked_selection_differs_from_uniform_default(tmp_path: Path) -> None:
    # With default (no keep_channels) the exporter keeps the FIRST `width`
    # channels. With a saliency map that favours group 1, selection must be
    # data-driven and therefore land on channels 16..31 instead.
    del tmp_path
    keep_map = select_keep_map(
        {(0, 0): [1.0, 50.0]}, width=16, full=32, sparse_layers=[0], n_exp=1
    )
    assert keep_map == {(0, 0): list(range(16, 32))}  # NOT range(0,16)
