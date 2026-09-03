"""Tests for the deep census (measured SQNR + distribution stats)."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from cebu_profiler.census.deep import (
    _fp8_e4m3_roundtrip,
    _quant_int4_group128,
    _quant_int8_perchannel,
    deep_scan_manifest,
    deep_scan_tensor,
)


def _write_safetensors(path: Path, tensors: dict[str, tuple[list[int], list[float]]]) -> None:
    """Minimal safetensors writer for F32 tensors (header + data, 8-byte align)."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (shape, vals) in tensors.items():
        data = struct.pack(f"<{len(vals)}f", *vals)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    hb = json.dumps(header).encode()
    pad = (-len(hb)) % 8
    hb += b" " * pad
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def _tensor_bytes(path: Path, name: str) -> tuple[int, int]:
    import json as _json

    with path.open("rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = _json.loads(fh.read(hlen))
    start, end = header[name]["data_offsets"]
    return 8 + hlen + start, end - start


def test_fp8_roundtrip_is_close():
    for x in (0.0, 1.0, -2.5, 0.125, 100.0, 448.0):
        q = _fp8_e4m3_roundtrip(x)
        assert math.isclose(q, x, rel_tol=0.07, abs_tol=0.01)
    # out-of-range clamps
    assert _fp8_e4m3_roundtrip(1000.0) == 448.0
    assert _fp8_e4m3_roundtrip(1e-9) == 0.0


def test_int8_perchannel_exact_when_values_fit():
    rows, cols = 2, 4
    vals = [127.0, -127.0, 63.5, -63.5, 10.0, -10.0, 5.0, -5.0]
    chunks = [vals]
    sqnr = _quant_int8_perchannel(chunks, rows, cols)
    # representable values -> quantization error ~0 -> huge (but finite-ish) SQNR
    assert sqnr > 40.0


def test_int4_group128_losses_are_larger_than_int8():
    vals = [((i % 17) - 8) * 0.37 for i in range(256)]
    sqnr8 = _quant_int8_perchannel([vals], 2, 128)
    sqnr4 = _quant_int4_group128([vals], 2, 128)
    assert sqnr4 < sqnr8


def test_deep_scan_tensor_reports_measured_fields(tmp_path: Path):
    p = tmp_path / "m.safetensors"
    shape = [4, 8]
    vals = [((r * 8 + c) % 13 - 6) * 0.21 for r in range(4) for c in range(8)]
    _write_safetensors(p, {"model.layers.0.mlp.experts.0.gate_proj.weight": (shape, vals)})
    off, size = _tensor_bytes(p, "model.layers.0.mlp.experts.0.gate_proj.weight")
    rep = deep_scan_tensor(
        p,
        name="model.layers.0.mlp.experts.0.gate_proj.weight",
        dtype="F32",
        shape=shape,
        offset=off,
        byte_size=size,
        shard="m.safetensors",
        with_spectrum=True,
    )
    assert rep is not None
    assert rep.mean is not None and rep.std is not None and rep.absmax is not None
    assert rep.sqnr_int8_perchannel is not None
    assert rep.sqnr_int4_group128 is not None
    assert rep.sqnr_int4_group128 < rep.sqnr_int8_perchannel
    assert rep.sqnr_fp8_e4m3 is not None
    assert rep.sv_leading and rep.sv_leading[0] > 0
    assert rep.stable_rank is not None and rep.stable_rank >= 1.0
    payload = rep.payload()
    assert payload["evidence"] == "measured"
    assert isinstance(json.dumps(payload), str)


def test_1d_tensor_reports_none_sqnr_not_zero(tmp_path: Path):
    p = tmp_path / "m.safetensors"
    vals = [0.5, -0.25, 0.75, 1.5]
    _write_safetensors(p, {"model.layers.0.input_layernorm.weight": ([4], vals)})
    off, size = _tensor_bytes(p, "model.layers.0.input_layernorm.weight")
    rep = deep_scan_tensor(
        p,
        name="model.layers.0.input_layernorm.weight",
        dtype="F32",
        shape=[4],
        offset=off,
        byte_size=size,
        shard="m.safetensors",
    )
    assert rep is not None
    assert rep.sqnr_int8_perchannel is None
    assert any("not applicable" in note for note in rep.notes)


def test_deep_scan_manifest_end_to_end(tmp_path: Path):
    p = tmp_path / "shard.safetensors"
    w = [((i % 11) - 5) * 0.13 for i in range(2 * 16)]
    b = [0.01, -0.02, 0.03, 0.04]
    _write_safetensors(
        p,
        {
            "model.layers.0.mlp.experts.0.up_proj.weight": ([2, 16], w),
            "model.layers.0.input_layernorm.weight": ([4], b),
        },
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
    from cebu_profiler.checkpoint.source_manifest import load_manifest

    manifest = load_manifest(str(tmp_path))
    bundle = deep_scan_manifest(str(tmp_path), manifest=manifest)
    assert bundle["evidence"] == "measured"
    assert bundle["tensors_scanned"] == 1  # only_2d default skips the 1-D norm
    names = {r["name"] for r in bundle["reports"]}
    assert names == {"model.layers.0.mlp.experts.0.up_proj.weight"}
    rep = bundle["reports"][0]
    assert {"sqnr_int8_perchannel", "sqnr_int4_group128", "sqnr_fp8_e4m3"} <= set(rep)
    assert isinstance(json.dumps(bundle), str)


def test_unsupported_dtype_is_skipped_with_reason(tmp_path: Path):
    from cebu_profiler.checkpoint.source_manifest import CheckpointManifest, TensorEntry

    manifest = CheckpointManifest(
        checkpoint_dir=str(tmp_path),
        tensors=[
            TensorEntry(
                name="x",
                dtype="U8",
                shape=[2, 2],
                numel=4,
                byte_size=4,
                shard="s.safetensors",
                offset_start=0,
                offset_end=4,
            )
        ],
    )
    (tmp_path / "s.safetensors").write_bytes(b"\0" * 64)
    bundle = deep_scan_manifest(str(tmp_path), manifest=manifest)
    assert bundle["tensors_scanned"] == 0
    assert bundle["tensors_skipped"] == [{"name": "x", "reason": "unsupported dtype U8"}]
