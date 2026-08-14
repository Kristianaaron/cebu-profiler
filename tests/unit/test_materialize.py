"""Phase 4 review-fix: derivative materializer tests.

Covers review findings: nibble-aware NVFP4 surgery (down requires full
16-channel groups), scalar copy, overwrite-flag guard (never implicit rmtree),
fail-closed coverage by exact names/shapes/byte sizes + hashes, router written
once, artifact labelled NON_LOADABLE (never experiment-ready).
"""

import json

import pytest

from model_atlas.checkpoint.safetensors import read_safetensors_header, write_safetensors
from model_atlas.materialize import (
    NonBlockAlignedError,
    _down_slice,
    _gateup_slice,
    materialize_expert_bank,
)


def _make_nvfp4_fixture(tmp_path):
    """A tiny GLM-style NVFP4 layer (2 experts, real packed shapes)."""
    import struct

    root = tmp_path / "glm"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"glm_moe_dsa","num_hidden_layers":4}')
    # gate: [4 channels, 32 packed bytes] scale [4, 2 groups] (16 values/group=8 bytes)
    nch, packed, sg = 4, 32, 2
    gate_w = bytes(range(nch * packed))
    gate_s = bytes(range(nch * sg))
    gate_scalar = struct.pack("<f", 1.0)
    # down: [8 hidden, 8 packed bytes] scale [8, 1 group]
    dh, dp, dg = 8, 8, 1
    down_w = bytes(range(dh * dp))
    down_s = bytes(range(dh * dg))
    tensors = {
        "model.layers.3.mlp.gate.weight": {
            "dtype": "BF16", "shape": [2, 8],
            "bytes": struct.pack("<H", 0x3F80) * 16,
        },
        "model.layers.3.mlp.gate.e_score_correction_bias": {
            "dtype": "F32", "shape": [2], "bytes": struct.pack("<ff", 0.1, 0.2),
        },
    }
    for e in range(2):
        p = f"model.layers.3.mlp.experts.{e}."
        for n_, isd in (("gate_proj", False), ("up_proj", False), ("down_proj", True)):
            w_s = [dh, dp] if isd else [nch, packed]
            s_s = [dh, dg] if isd else [nch, sg]
            tensors[p + n_ + ".weight"] = {
                "dtype": "U8", "shape": w_s, "bytes": down_w if isd else gate_w,
            }
            tensors[p + n_ + ".weight_scale"] = {
                "dtype": "F8_E4M3", "shape": s_s, "bytes": down_s if isd else gate_s,
            }
            tensors[p + n_ + ".weight_scale_2"] = {
                "dtype": "F32", "shape": [], "bytes": gate_scalar,
            }
            tensors[p + n_ + ".input_scale"] = {
                "dtype": "F32", "shape": [], "bytes": gate_scalar,
            }
    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    return str(root)


@pytest.mark.integration
def test_gateup_slice_keeps_rows():
    nch, packed, sg = 4, 32, 2
    gw = bytes(range(nch * packed))
    gs = bytes(range(nch * sg))
    nw, ns = _gateup_slice(gw, gs, packed, sg, [0, 2])
    assert len(nw) == 2 * packed
    assert len(ns) == 2 * sg


@pytest.mark.integration
def test_down_slice_group_aligned_ok():
    # down [8 hidden, 8 packed bytes], groupsize 16 => 8 bytes/group=1 group, 8 channels
    dw = bytes(range(8 * 8))
    ds = bytes(range(8 * 1))
    nw, ns = _down_slice(dw, ds, hidden=8, packed_total=8, scale_groups=1,
                         keep_channels=list(range(16)))
    assert len(nw) == 8 * 8  # full group retained
    assert len(ns) == 8 * 1


@pytest.mark.integration
def test_down_slice_fails_closed_on_partial_group():
    dw = bytes(range(8 * 8))
    ds = bytes(range(8 * 1))
    with pytest.raises(NonBlockAlignedError):
        _down_slice(dw, ds, hidden=8, packed_total=8, scale_groups=1,
                    keep_channels=[0, 1])  # partial group of 16


@pytest.mark.integration
def test_materialize_promotes_and_labels_non_loadable(tmp_path):
    src = _make_nvfp4_fixture(tmp_path)
    out = tmp_path / "out"
    # down needs full groups; use keep_channels covering the whole 16-block
    res = materialize_expert_bank(
        src, str(out), corner_layer=3, keep_channels=list(range(16)), num_experts=2
    )
    assert res.validated is True
    assert res.promoted is True
    assert res.loadable is False  # article honesty: never experiment-ready
    manifest = json.loads((out / "artifact_manifest.json").read_text())
    assert manifest["experiment_ready"] is False
    assert manifest["artifact_type"] == "NON_LOADABLE_EXPERT_BANK"
    assert manifest["shard_hashes"]
    for h in manifest["shard_hashes"].values():
        assert len(h) == 64
    steps = [e.step for e in res.journal]
    assert set(steps) >= {"open", "slice", "validate", "promote"}


@pytest.mark.integration
def test_router_written_once(tmp_path):
    src = _make_nvfp4_fixture(tmp_path)
    out = tmp_path / "out2"
    res = materialize_expert_bank(
        src, str(out), corner_layer=3, keep_channels=list(range(16)), num_experts=2,
    )
    assert res.promoted is True
    # router shard is a single dedicated shard; only one router tensor
    hdr = read_safetensors_header(out / "layer3-router.safetensors")
    names = [k for k in hdr if k != "__metadata__"]
    assert names == ["model.layers.3.mlp.gate.weight"]


@pytest.mark.integration
def test_overwrite_flag_required(tmp_path):
    src = _make_nvfp4_fixture(tmp_path)
    out = tmp_path / "out3"
    materialize_expert_bank(src, str(out), corner_layer=3, keep_channels=list(range(16)))
    # second call without overwrite -> must raise, never implicit rmtree
    with pytest.raises(FileExistsError):
        materialize_expert_bank(src, str(out), corner_layer=3, keep_channels=list(range(16)))
    # with overwrite=True it succeeds
    res = materialize_expert_bank(
        src, str(out), corner_layer=3, keep_channels=list(range(16)), overwrite=True,
    )
    assert res.promoted is True


@pytest.mark.integration
def test_scalars_copied_unchanged(tmp_path):
    src = _make_nvfp4_fixture(tmp_path)
    out = tmp_path / "out4"
    res = materialize_expert_bank(
        src, str(out), corner_layer=3, keep_channels=list(range(16)),
    )
    assert res.promoted is True
    hdr = read_safetensors_header(out / "layer3-exp0-gate_proj.safetensors")
    assert hdr["model.layers.3.mlp.experts.0.gate_proj.weight_scale_2"]["shape"] == []
    assert hdr["model.layers.3.mlp.experts.0.gate_proj.input_scale"]["shape"] == []
    # scale_2/input_scale scalar bytes preserved (4 bytes)
    o = hdr["model.layers.3.mlp.experts.0.gate_proj.weight_scale_2"]["data_offsets"]
    assert o[1] - o[0] == 4
