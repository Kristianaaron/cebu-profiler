import json
import struct
from pathlib import Path

import pytest

from model_atlas.analysis.nvfp4_quant_risk import (
    _percentile_ranks,
    profile_nvfp4_quant_risk,
)
from model_atlas.checkpoint.safetensors import write_safetensors


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "quantization_config": {"quant_algo": "NVFP4"},
            }
        )
    )
    tensors: dict[str, dict[str, object]] = {}
    for layer in range(2):
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                tensors[f"{prefix}.weight"] = {
                    "dtype": "U8",
                    "shape": [4, 16],
                    "bytes": bytes([0x76 if layer else 0x11]) * 64,
                }
                tensors[f"{prefix}.weight_scale"] = {
                    "dtype": "F8_E4M3",
                    "shape": [4, 2],
                    "bytes": bytes([0x38 + layer]) * 8,
                }
                tensors[f"{prefix}.weight_scale_2"] = {
                    "dtype": "F32",
                    "shape": [],
                    "bytes": struct.pack("<f", 2.0 if layer else 0.25),
                }
    # An MTP/mixed tensor without ModelOpt scale sidecars is ignored.
    tensors["model.layers.2.mlp.experts.0.gate_proj.weight"] = {
        "dtype": "BF16",
        "shape": [4, 32],
        "bytes": b"\x00\x00" * 128,
    }
    shard = root / "model.safetensors"
    write_safetensors(shard, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}})
    )
    return root


def test_profile_is_bounded_deterministic_and_layer_projection_granular(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    first = profile_nvfp4_quant_risk(
        checkpoint, sample_experts=2, sample_rows=2, sensitive_fraction=0.34
    )
    second = profile_nvfp4_quant_risk(
        checkpoint, sample_experts=2, sample_rows=2, sensitive_fraction=0.34
    )

    assert first == second
    assert len(first.rows) == 6
    assert all(row.layer in (0, 1) for row in first.rows)
    assert sum(row.retained_type == "NVFP4" for row in first.rows) == 3
    assert {row.layer for row in first.rows if row.retained_type == "NVFP4"} == {1}
    assert first.tensor_type_lines[-1] == r"blk\..*\.ffn_(gate|up|down)_exps\.weight=Q1_0"
    assert first.tensor_type_sha256
    assert first.evidence_kind == "estimated"
    assert "no activation/Hessian/KLD claim" in first.note


def test_profile_rejects_unbounded_sampling(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="sample_experts"):
        profile_nvfp4_quant_risk(checkpoint, sample_experts=17)
    with pytest.raises(ValueError, match="sample_rows"):
        profile_nvfp4_quant_risk(checkpoint, sample_rows=5)


def test_percentile_ties_receive_the_same_midrank() -> None:
    assert _percentile_ranks([1.0, 1.0, 1.0]) == [0.5, 0.5, 0.5]
    assert _percentile_ranks([1.0, 2.0, 2.0, 3.0]) == [0.0, 0.5, 0.5, 1.0]
