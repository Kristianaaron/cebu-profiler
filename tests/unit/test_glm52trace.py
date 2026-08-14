"""Phase 3 review-fix: GLM-5.2 facts + REAL_ROUTER_SYNTHETIC_INPUT_PROBE tests.

Covers review findings: the routing trace is relabelled (real router + synthetic
input -> PREDICTED, never measured corpus evidence); entropy is non-negative;
coactivation counts each unordered distinct expert combination exactly once;
`sorce_gate_bias` typo removed (`gate_bias_values`); config adapter drops the
incompatible `layer_types` instead of DSA->sparse / full_attention replacements.
"""

import struct

import pytest

from model_atlas.checkpoint.safetensors import write_safetensors
from model_atlas.glm52trace import (
    PROBE_INPUT_LABEL,
    GlmRoutingTrace,
    _sum_p_log,
    _topk_rank,
    classify_mlp,
    load_glm52_facts,
    normalized_glm52_config,
    stream_routing_trace,
)
from model_atlas.schemas.evidence import EvidenceKind


def _make_router_fixture(tmp_path):
    """A tiny checkpoint with a real-looking GLM sparse-layer router."""
    root = tmp_path / "glm"
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 4,
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "layer_types": ["deepseek_sparse_attention"] * 4,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "vocab_size": 100,
        "num_nextn_predict_layers": 1,
        "quantization_config": {
            "quant_algo": "NVFP4",
            "kv_cache_scheme": {"type": "float", "dynamic": False, "num_bits": 8},
            "config_groups": {"group_0": {"weights": {"group_size": 16}}},
        },
    }
    import json

    (root / "config.json").write_text(json.dumps(cfg))
    n_exp, hidden = 8, 16
    gate = struct.pack("<H", 0x3F80) * (n_exp * hidden)
    bias = struct.pack("<f", 0.1) * n_exp
    write_safetensors(
        root / "model-00001-of-00001.safetensors",
        {
            "model.layers.3.mlp.gate.weight": {
                "dtype": "BF16", "shape": [n_exp, hidden], "bytes": gate,
            },
            "model.layers.3.mlp.gate.e_score_correction_bias": {
                "dtype": "F32", "shape": [n_exp], "bytes": bias,
            },
        },
    )
    return str(root)


@pytest.mark.integration
def test_normalized_config_drops_incompatible_layer_types(tmp_path):
    cfg = normalized_glm52_config(_make_router_fixture(tmp_path))
    assert "layer_types" not in cfg  # incompatible key dropped, not overwritten
    assert cfg["mlp_layer_types"] == ["dense", "dense", "dense", "sparse"]


@pytest.mark.integration
def test_sum_p_log_is_entropy_and_non_negative():
    # uniform over 2 -> log(2) ~ 0.693 positive
    h = _sum_p_log([0.5, 0.5])
    assert h > 0
    assert abs(h - 0.693147) < 0.002
    assert h >= 0  # never negative after the fix


@pytest.mark.integration
def test_topk_rank_deterministic():
    sel, probs = _topk_rank([0.5, 1.0, 0.2, 0.9], 2)
    assert sel == [1, 3]
    assert abs(sum(probs) - 1.0) < 1e-9


@pytest.mark.integration
def test_stream_probe_relabelled_predicted(tmp_path):
    t = stream_routing_trace(_make_router_fixture(tmp_path), layer=3, n_hidden_rows=4)
    assert isinstance(t, GlmRoutingTrace)
    assert t.input_label == PROBE_INPUT_LABEL
    assert t.evidence_kind is EvidenceKind.PREDICTED  # synthetic input, not measured
    assert "REAL_ROUTER_SYNTHETIC_INPUT_PROBE" in t.provenance
    assert "NOT measured" in t.provenance
    # typo gone
    assert not hasattr(t, "sorce_gate_bias")
    assert t.gate_bias_values  # real correction-bias values carried
    # entropy non-negative on every record
    assert all(r.entropy >= 0 for r in t.records)


@pytest.mark.integration
def test_coactivation_unordered_distinct_once(tmp_path):
    # force a known selection to check counting: the all-1.0 router + bias gives
    # a deterministic order; coactivation must count each unordered pair once per
    # token, exactly (symmetric pair counted once).
    t = stream_routing_trace(_make_router_fixture(tmp_path), layer=3, n_hidden_rows=3)
    topk = t.top_k
    # each token contributes exactly C(topk,2) distinct unordered combos
    expected_events = 3 * (topk * (topk - 1)) // 2
    total = sum(t.coactivation.values())
    assert total == expected_events


@pytest.mark.integration
def test_fixture_facts(tmp_path):
    f = load_glm52_facts(_make_router_fixture(tmp_path))
    assert f.n_sparse_layers == 1
    assert f.quant_algo == "NVFP4"
    assert f.group_size == 16
    assert classify_mlp(3, None) == "sparse"
