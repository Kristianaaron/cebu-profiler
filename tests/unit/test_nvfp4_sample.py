from __future__ import annotations

from pathlib import Path

import pytest

from model_atlas.backend.nvfp4_sample import (
    decode_e4m3fn,
    decode_nvfp4_rows,
    probe_nvfp4_tensor,
)


def test_e4m3fn_known_values_and_special() -> None:
    assert decode_e4m3fn(0x00) == 0.0
    assert decode_e4m3fn(0x01) == 2**-9
    assert decode_e4m3fn(0x38) == 1.0
    assert decode_e4m3fn(0x7E) == 448.0
    assert decode_e4m3fn(0xB8) == -1.0
    assert decode_e4m3fn(0x7F) != decode_e4m3fn(0x7F)  # NaN


def test_decode_uses_low_nibble_first_and_block_scale() -> None:
    packed = bytes([0x21] * 8 + [0xAB] * 8)
    scales = bytes([0x38, 0x40])  # 1.0 then 2.0
    decoded = decode_nvfp4_rows(
        packed,
        scales,
        0.5,
        rows=1,
        packed_columns=16,
        block_size=16,
    )
    assert decoded[:4] == (0.25, 0.5, 0.25, 0.5)
    assert decoded[16:20] == (-1.5, -1.0, -1.5, -1.0)


def test_decode_is_strictly_bounded() -> None:
    with pytest.raises(ValueError, match="rows must be"):
        decode_nvfp4_rows(b"", b"", 1.0, rows=9, packed_columns=8)
    with pytest.raises(ValueError, match="scale byte count"):
        decode_nvfp4_rows(bytes(8), b"", 1.0, rows=1, packed_columns=8)


def test_real_glm_nvfp4_sample_is_stable_when_mounted() -> None:
    source = Path("/media/glm52/models/nvidia/GLM-5.2-NVFP4")
    if not source.exists():
        pytest.skip("real GLM-5.2 NVFP4 checkpoint is not mounted")
    report = probe_nvfp4_tensor(
        source,
        "model.layers.10.mlp.experts.0.gate_proj.weight",
        row_start=0,
        rows=1,
    )
    assert report.bytes_read == 3460
    assert report.decoded_shape == (1, 6144)
    assert report.raw_sha256 == "9dfd6bb2c7687c3cc7381e3166b16ab8a5a7c830ea9964c2b8e87468bb7e2cfc"
    assert report.finite is True
    assert report.producer == "modelopt@0.46.0.dev65+g977d34dc3"
