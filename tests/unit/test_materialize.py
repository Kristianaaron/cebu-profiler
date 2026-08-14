"""Phase 4: derivative materializer tests (crash journal, hashes, immutability,
fail-closed coverage)."""

import json

import pytest

from model_atlas.checkpoint.source_manifest import load_manifest
from model_atlas.materialize import _sliced_nvfp4, materialize_expert_bank


def _make_nvfp4_fixture(tmp_path):
    """A tiny GLM-style NVFP4 layer fixture (one sparse layer, few experts)."""
    import struct

    from model_atlas.checkpoint.safetensors import write_safetensors

    root = tmp_path / "glm"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"glm_moe_dsa","num_hidden_layers":4}')
    gate = struct.pack("<H", 0x3F80) * (2 * 8)  # BF16 router [8,8]
    # expert 0 gate_proj: weight U8[2,4], scale F8[2,1], scale2 F32[2], input_scale F32[]
    w = bytes(range(8))
    scale = bytes([0x38, 0x40])
    scale2 = struct.pack("<ff", 1.0, 0.5)
    inp = struct.pack("<f", 1.0)
    # down_proj: weight U8[8,2], scale F8[8,1] -- channel axis is column
    dw = bytes(range(16))
    dscale = bytes([0x38] * 8)
    tensors = {
        "model.layers.3.mlp.gate.weight": {"dtype": "BF16", "shape": [2, 8], "bytes": gate},
        "model.layers.3.mlp.experts.0.gate_proj.weight": {
            "dtype": "U8", "shape": [2, 4], "bytes": w,
        },
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale": {
            "dtype": "F8_E4M3", "shape": [2, 1], "bytes": scale,
        },
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale_2": {
            "dtype": "F32", "shape": [2], "bytes": scale2,
        },
        "model.layers.3.mlp.experts.0.gate_proj.input_scale": {
            "dtype": "F32", "shape": [], "bytes": inp,
        },
        "model.layers.3.mlp.experts.0.up_proj.weight": {
            "dtype": "U8", "shape": [2, 4], "bytes": w,
        },
        "model.layers.3.mlp.experts.0.up_proj.weight_scale": {
            "dtype": "F8_E4M3", "shape": [2, 1], "bytes": scale,
        },
        "model.layers.3.mlp.experts.0.up_proj.weight_scale_2": {
            "dtype": "F32", "shape": [2], "bytes": scale2,
        },
        "model.layers.3.mlp.experts.0.up_proj.input_scale": {
            "dtype": "F32", "shape": [], "bytes": inp,
        },
        "model.layers.3.mlp.experts.0.down_proj.weight": {
            "dtype": "U8", "shape": [8, 2], "bytes": dw,
        },
        "model.layers.3.mlp.experts.0.down_proj.weight_scale": {
            "dtype": "F8_E4M3", "shape": [8, 1], "bytes": dscale,
        },
        "model.layers.3.mlp.experts.0.down_proj.weight_scale_2": {
            "dtype": "F32", "shape": [8], "bytes": scale2 * 4,
        },
        "model.layers.3.mlp.experts.0.down_proj.input_scale": {
            "dtype": "F32", "shape": [], "bytes": inp,
        },
    }
    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    return str(root)


@pytest.mark.integration
def test_sliced_nvfp4_gate_up_rows(tmp_path):
    # gate: keep channels [0,1]; weight 2 rows -> both kept
    w = bytes(range(8))  # 2 rows x 4 cols
    s = bytes([0x38, 0x40])
    s2 = bytes([1, 0, 0, 0, 0, 0, 0, 64])
    nw, ns, n2 = _sliced_nvfp4(
        w, s, s2, w_rows=2, w_cols=4, group_size=4, keep_channels=[0, 1], is_down=False
    )
    assert len(nw) == 8  # 2 rows x 4 cols
    assert len(ns) == 2  # 2 scale rows
    assert len(n2) == 8  # 2 scale2 floats


@pytest.mark.integration
def test_sliced_nvfp4_down_columns(tmp_path):
    # down weight [8,2], keep channel 0 -> keep only column 0 of each row
    dw = bytes(range(16))  # 8 rows x 2 cols
    dscale = bytes([0x30] * 8)  # 8 rows x 1 scale col
    nw, ns, _ = _sliced_nvfp4(
        dw, dscale, b"", w_rows=8, w_cols=2, group_size=2, keep_channels=[0], is_down=True
    )
    assert len(nw) == 8  # 8 rows x 1 col
    assert len(ns) == 8  # 8 scale rows


@pytest.mark.integration
def test_materialize_promotes_and_hashes(tmp_path):
    src = _make_nvfp4_fixture(tmp_path)
    out = tmp_path / "out"
    res = materialize_expert_bank(
        src, str(out), corner_layer=3, keep_channels=[0, 1], num_experts=1, group_size=4
    )
    assert res.validated is True
    assert res.promoted is True
    assert res.coverage == 1.0
    # manifest with per-shard hashes + source immutability flag
    manifest = json.loads((out / "derivative_manifest.json").read_text())
    assert manifest["source_immutable"] is True
    assert manifest["shard_hashes"]
    for h in manifest["shard_hashes"].values():
        assert len(h) == 64  # sha256
    # journal recorded open/slice/validate/promote
    steps = [e.step for e in res.journal]
    assert "open" in steps and "slice" in steps and "validate" in steps and "promote" in steps


@pytest.mark.integration
def test_materialize_source_untouched(tmp_path):
    src = _make_nvfp4_fixture(tmp_path)
    manifest_before = load_manifest(src)
    b4 = {t.name: t.byte_size for t in manifest_before.tensors}
    out = tmp_path / "out2"
    materialize_expert_bank(src, str(out), corner_layer=3, keep_channels=[0], num_experts=1)
    manifest_after = load_manifest(src)
    after = {t.name: t.byte_size for t in manifest_after.tensors}
    assert b4 == after  # source immutability: never rewritten
    assert str(out) != src


@pytest.mark.integration
def test_materialize_fail_closed_on_empty_keep(tmp_path):
    src = _make_nvfp4_fixture(tmp_path)
    out = tmp_path / "out3"
    with pytest.raises(ValueError):
        materialize_expert_bank(src, str(out), corner_layer=3, keep_channels=[], num_experts=1)
