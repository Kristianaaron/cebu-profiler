"""Tests for config-driven ArchitectureSpec derivation (registry/config_driven)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cebu_profiler.checkpoint.source_manifest import load_manifest
from cebu_profiler.checkpoint.synthetic import make_synthetic_checkpoint
from cebu_profiler.registry.config_driven import (
    SpecDerivationError,
    spec_from_checkpoint_dir,
    spec_from_config,
    verify_spec_against_manifest,
)

# Shape mirrors the released GLM-5.x family config (text_config wrapper,
# first_k_dense_replace, compressed-tensors nvfp4-pack-quantized).
GLM5NEXT_STYLE = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "text_config": {
        "model_type": "glm5_next_text",
        "hidden_size": 6144,
        "num_hidden_layers": 45,
        "first_k_dense_replace": 3,
        "n_routed_experts": 288,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "vocab_size": 154820,
    },
    "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
    },
}


def test_glm5next_style_derivation():
    spec = spec_from_config(GLM5NEXT_STYLE)
    assert spec.name == "glm5_next"
    assert spec.num_text_layers == 45
    assert spec.layers_by_kind == {"dense": 3, "moe": 42}
    assert spec.moe.num_routed_experts == 288
    assert spec.moe.top_k == 8
    assert spec.moe.num_shared_experts == 1
    assert spec.vocabulary_size == 154820
    assert spec.needs_source_measurement  # real sizes still from census


def test_flat_config_no_wrapper():
    flat = {
        "model_type": "some_moe",
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "first_k_dense_replace": 1,
        "n_routed_experts": 16,
        "num_experts_per_tok": 2,
    }
    spec = spec_from_config(flat)
    assert spec.layers_by_kind == {"dense": 1, "moe": 5}
    assert spec.moe.num_routed_experts == 16


def test_missing_required_field_fails_closed():
    bad = {"model_type": "x", "hidden_size": 128}  # no layers/experts
    with pytest.raises(SpecDerivationError):
        spec_from_config(bad)


def test_measured_experts_win_over_config(tmp_path: Path):
    cfg_dir = Path(make_synthetic_checkpoint(tmp_path / "ckpt"))
    manifest = load_manifest(str(cfg_dir))
    spec = spec_from_config(
        dict(
            GLM5NEXT_STYLE["text_config"],
            hidden_size=128,
            num_hidden_layers=2,
            first_k_dense_replace=0,
            n_routed_experts=999,
        ),  # deliberately wrong
        manifest=manifest,
    )
    # measured from tensor names: experts.0/experts.1 -> 2
    assert spec.moe.num_routed_experts == 2


def test_verify_against_manifest_consistent(tmp_path: Path):
    cfg_dir = Path(make_synthetic_checkpoint(tmp_path / "ckpt"))
    manifest = load_manifest(str(cfg_dir))
    spec = spec_from_config(
        dict(
            GLM5NEXT_STYLE["text_config"],
            hidden_size=128,
            num_hidden_layers=2,
            first_k_dense_replace=0,
            n_routed_experts=2,
        ),
        manifest=manifest,
    )
    notes = verify_spec_against_manifest(spec, manifest)
    assert notes == []


def test_verify_flags_layer_drift(tmp_path: Path):
    cfg_dir = Path(make_synthetic_checkpoint(tmp_path / "ckpt"))
    manifest = load_manifest(str(cfg_dir))
    spec = spec_from_config(
        dict(
            GLM5NEXT_STYLE["text_config"],
            hidden_size=128,
            num_hidden_layers=1,
            first_k_dense_replace=0,
            n_routed_experts=2,
        ),
        manifest=manifest,
    )
    notes = verify_spec_against_manifest(spec, manifest)
    assert any("layer drift" in n for n in notes)


def test_spec_from_checkpoint_dir(tmp_path: Path):
    d = tmp_path / "real"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(GLM5NEXT_STYLE))
    spec = spec_from_checkpoint_dir(str(d))
    assert spec.moe.num_routed_experts == 288


def test_missing_config_fails(tmp_path: Path):
    with pytest.raises(SpecDerivationError):
        spec_from_checkpoint_dir(str(tmp_path))


def test_dense_model_fallback():
    dense = {
        "model_type": "denseish",
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "vocab_size": 1000,
    }
    spec = spec_from_config(dense)
    assert spec.layers_by_kind == {"dense": 4}
    assert spec.moe.num_routed_experts >= 1
