"""Phase 3: real GLM-5.2 config facts + bounded streamed routing trace tests."""

import struct

import pytest

from model_atlas.checkpoint.safetensors import write_safetensors
from model_atlas.glm52trace import (
    GlmRoutingTrace,
    _topk_rank,
    classify_mlp,
    load_glm52_facts,
    stream_routing_trace,
)


def _make_router_fixture(tmp_path):
    """A tiny checkpoint with a real-looking GLM sparse-layer router."""
    root = tmp_path / "glm"
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 4,
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "vocab_size": 100,
        "num_nextn_predict_layers": 1,
        "kv_lora_rank": 8,
        "q_lora_rank": 4,
        "v_head_dim": 4,
        "quantization_config": {
            "quant_algo": "NVFP4",
            "kv_cache_scheme": {"type": "float", "dynamic": False, "num_bits": 8},
            "config_groups": {"group_0": {"weights": {"group_size": 16}}},
        },
    }
    import json

    (root / "config.json").write_text(json.dumps(cfg))
    # 8 experts x 16 hidden BF16 gate, plus F32 correction bias
    n_exp, hidden = 8, 16
    gate = struct.pack("<H", 0x3F80) * (n_exp * hidden)  # all 1.0
    bias = struct.pack("<f", 0.1) * n_exp
    write_safetensors(
        root / "model-00001-of-00001.safetensors",
        {
            "model.layers.3.mlp.gate.weight": {
                "dtype": "BF16",
                "shape": [n_exp, hidden],
                "bytes": gate,
            },
            "model.layers.3.mlp.gate.e_score_correction_bias": {
                "dtype": "F32",
                "shape": [n_exp],
                "bytes": bias,
            },
        },
    )
    return str(root)


@pytest.mark.integration
def test_load_glm52_facts(tmp_path):
    f = load_glm52_facts(_make_router_fixture(tmp_path))
    assert f.n_layers == 4
    assert f.n_dense_layers == 3
    assert f.n_sparse_layers == 1
    assert f.n_routed_experts == 8
    assert f.top_k == 2
    assert f.quant_algo == "NVFP4"
    assert f.group_size == 16
    assert f.num_mtp_layers == 1
    assert f.model_type == "glm_moe_dsa"


@pytest.mark.integration
def test_classify_mlp():
    assert classify_mlp(0, None) == "dense"
    assert classify_mlp(2, None) == "dense"
    assert classify_mlp(3, None) == "sparse"
    assert classify_mlp(5, ["dense", "sparse", "sparse", "sparse", "sparse", "sparse"]) == "sparse"


@pytest.mark.integration
def test_topk_rank_deterministic():
    sel, probs = _topk_rank([0.5, 1.0, 0.2, 0.9], 2)
    assert sel == [1, 3]
    assert abs(sum(probs) - 1.0) < 1e-9


@pytest.mark.integration
def test_stream_routing_trace_fixture(tmp_path):
    t = stream_routing_trace(_make_router_fixture(tmp_path), layer=3, n_hidden_rows=4)
    assert isinstance(t, GlmRoutingTrace)
    assert t.n_experts == 8
    assert t.top_k == 2
    assert len(t.records) == 4
    # real top-k picks the same experts each row (router all-ones + constant bias)
    assert all(len(r.selected_experts) == 2 for r in t.records)
    assert t.frequency  # at least one expert observed


@pytest.mark.integration
def test_routing_frequencies_correct(tmp_path):
    t = stream_routing_trace(_make_router_fixture(tmp_path), layer=3, n_hidden_rows=4)
    total_routed = sum(t.frequency.values())
    assert total_routed == 4 * 2  # rows x top_k
