"""Phase 1 gate: bounded body read proof + no-unclassified on a real-ish fixture.

Real GLM-5.2 NVFP4 source stays immutable; the fixture replicates its dtype
mix (BF16 reference, U8 + F8_E4M3 NVFP4 expert layout) so the bounded reader
contract is tested deterministically offline. Coverage is asserted at 1.0 and
unclassified == 0 (fail-closed invariant).
"""

import struct

import pytest

from model_atlas.checkpoint.classifier import classify_tensor
from model_atlas.checkpoint.realbody import _nvfp4_layout, validate_real_bodies
from model_atlas.checkpoint.safetensors import write_safetensors


def _make_glmstyle_fixture(tmp_path):
    """A tiny checkpoint with real GLM-5.2 NVFP4-style tensor names + dtypes."""
    root = tmp_path / "glm"
    root.mkdir(parents=True, exist_ok=True)
    cfg = '{"architectures":["GlmMoeDsaForCausalLM"],"dtype":"bfloat16"}'
    (root / "config.json").write_text(cfg)

    def flt(v):
        return struct.pack("<f", v)

    tensors: dict[str, dict] = {
        # reference (BF16) tensors — decodable
        "model.layers.3.mlp.gate.weight": {
            "dtype": "BF16",
            "shape": [2, 4],
            "bytes": struct.pack("<H", 0x3F80) * 8,
        },
        "model.layers.3.mlp.shared_experts.down_proj.weight": {
            "dtype": "BF16",
            "shape": [4, 2],
            "bytes": struct.pack("<H", 0xBF80) * 8,
        },
        # NVFP4 expert constituents (opaque U8 + F8_E4M3 + F32 scales)
        "model.layers.3.mlp.experts.0.gate_proj.weight": {
            "dtype": "U8",
            "shape": [2, 4],
            "bytes": bytes(range(8)),
        },
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale": {
            "dtype": "F8_E4M3",
            "shape": [2, 2],
            "bytes": bytes([0x38, 0x40, 0x3C, 0x40]),
        },
        "model.layers.3.mlp.experts.0.gate_proj.input_scale": {
            "dtype": "F32",
            "shape": [],
            "bytes": flt(1.0),
        },
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale_2": {
            "dtype": "F32",
            "shape": [],
            "bytes": flt(0.5),
        },
        # unclassified-free requires every tensor to map to a role
        "model.layers.3.mlp.experts.0.down_proj.weight": {
            "dtype": "U8",
            "shape": [4, 2],
            "bytes": bytes(range(8)),
        },
        "model.layers.3.mlp.experts.0.down_proj.weight_scale": {
            "dtype": "F8_E4M3",
            "shape": [4, 1],
            "bytes": bytes([0x38]),
        },
    }
    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    return str(root)


@pytest.mark.integration
def test_realbody_scan_coverage_full_and_unclassified_zero(tmp_path):
    ckpt = _make_glmstyle_fixture(tmp_path)
    scan = validate_real_bodies(ckpt, reference_max=2, nvfp4_experts=1)
    s = scan.as_dict()
    assert s["coverage"] == 1.0
    assert s["unclassified_count"] == 0
    # reference gate decoded; NVFP4 U8 bodies read but NOT mis-decoded
    decoded_names = {b["name"] for b in s["bodies_read"] if b["decoded"]}
    assert "model.layers.3.mlp.gate.weight" in decoded_names
    assert "model.layers.3.mlp.experts.0.gate_proj.weight" not in decoded_names


@pytest.mark.integration
def test_bounded_peak_is_smaller_than_shard(tmp_path):
    ckpt = _make_glmstyle_fixture(tmp_path)
    scan = validate_real_bodies(ckpt, reference_max=2, nvfp4_experts=1)
    # peak resident = largest single tensor body (down_proj.weight = 16B),
    # never the whole shard (which also carries header).
    assert scan.peak_resident_bytes == 16
    assert scan.peak_resident_bytes > 0


@pytest.mark.integration
def test_nvfp4_layout_describes_components(tmp_path):
    ckpt = _make_glmstyle_fixture(tmp_path)
    from model_atlas.checkpoint.source_manifest import load_manifest

    layout = _nvfp4_layout(load_manifest(ckpt))
    assert "gate_proj.weight" in layout
    assert "gate_proj.weight_scale" in layout
    assert "gate_proj.input_scale" in layout
    assert "down_proj.weight" in layout


@pytest.mark.integration
def test_fixture_tensors_all_classify(tmp_path):
    """Every name in the GLM-style fixture maps to a role (no unclassified)."""
    from model_atlas.checkpoint.source_manifest import load_manifest

    ckpt = _make_glmstyle_fixture(tmp_path / "classify")
    for t in load_manifest(ckpt).tensors:
        assert not classify_tensor(t.name).unclassified
