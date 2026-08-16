"""Bounded, dependency-free decode probe for ModelOpt NVFP4 safetensors.

This module is deliberately a probe, not a quantization backend.  It reads at
most eight rows from one packed NVFP4 tensor and emits immutable evidence that
the mounted checkpoint layout can be decoded without materializing a shard.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_atlas.checkpoint.safetensors import read_safetensors_header

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_MAX_ROWS = 8


@dataclass(frozen=True)
class NVFP4SampleReport:
    schema_version: int
    source: str
    producer: str
    tensor: str
    row_start: int
    rows: int
    decoded_shape: tuple[int, int]
    block_size: int
    bytes_read: int
    raw_sha256: str
    decoded_f32_sha256: str
    finite: bool
    minimum: float
    maximum: float
    l2_norm: float
    scale2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decode_e4m3fn(raw: int) -> float:
    """Decode one IEEE-like E4M3FN byte, including subnormals."""
    if not 0 <= raw <= 255:
        raise ValueError("E4M3FN byte must be in [0, 255]")
    sign = -1.0 if raw & 0x80 else 1.0
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    if exponent == 0:
        return sign * math.ldexp(float(mantissa), -9)
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    return sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def decode_nvfp4_rows(
    packed: bytes,
    scales_e4m3: bytes,
    scale2: float,
    *,
    rows: int,
    packed_columns: int,
    block_size: int = 16,
) -> tuple[float, ...]:
    """Decode packed low-nibble-first E2M1 values with per-block scales."""
    if rows <= 0 or rows > _MAX_ROWS:
        raise ValueError(f"rows must be in [1, {_MAX_ROWS}]")
    if packed_columns <= 0 or block_size <= 0 or block_size % 2:
        raise ValueError("packed_columns and an even block_size must be positive")
    expected_weight = rows * packed_columns
    scale_columns, remainder = divmod(packed_columns * 2, block_size)
    if remainder:
        raise ValueError("decoded columns must be divisible by block_size")
    if len(packed) != expected_weight:
        raise ValueError("packed row byte count does not match shape")
    if len(scales_e4m3) != rows * scale_columns:
        raise ValueError("scale byte count does not match block geometry")
    values: list[float] = []
    for row in range(rows):
        weight_base = row * packed_columns
        scale_base = row * scale_columns
        for column in range(packed_columns):
            raw = packed[weight_base + column]
            block = (column * 2) // block_size
            scale = decode_e4m3fn(scales_e4m3[scale_base + block]) * scale2
            values.append(_E2M1[raw & 0x0F] * scale)
            values.append(_E2M1[raw >> 4] * scale)
    return tuple(values)


def probe_nvfp4_tensor(
    checkpoint_dir: str | Path,
    tensor: str,
    *,
    row_start: int = 0,
    rows: int = 1,
) -> NVFP4SampleReport:
    """Read and decode a bounded row range from a real ModelOpt NVFP4 tensor."""
    if not tensor.endswith(".weight"):
        raise ValueError("tensor must name an NVFP4 .weight tensor")
    if row_start < 0 or rows <= 0 or rows > _MAX_ROWS:
        raise ValueError(f"row_start must be nonnegative and rows in [1, {_MAX_ROWS}]")
    root = Path(checkpoint_dir).resolve()
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight_map")
    scale_name = tensor.removesuffix(".weight") + ".weight_scale"
    scale2_name = tensor.removesuffix(".weight") + ".weight_scale_2"
    names = (tensor, scale_name, scale2_name)
    try:
        shards = {str(weight_map[name]) for name in names}
    except KeyError as exc:
        raise ValueError(f"missing NVFP4 constituent {exc.args[0]}") from exc
    if len(shards) != 1:
        raise ValueError("NVFP4 constituents must reside in one shard")
    shard = root / shards.pop()
    header = read_safetensors_header(shard)
    weight = _tensor_spec(header, tensor, "U8", 2)
    scale = _tensor_spec(header, scale_name, "F8_E4M3", 2)
    scale2_spec = _tensor_spec(header, scale2_name, "F32", 0)
    weight_rows, packed_columns = (int(v) for v in weight["shape"])
    scale_rows, scale_columns = (int(v) for v in scale["shape"])
    if weight_rows != scale_rows or packed_columns * 2 != scale_columns * 16:
        raise ValueError("NVFP4 weight/scale geometry mismatch")
    if row_start + rows > weight_rows:
        raise ValueError("requested row range exceeds tensor shape")
    header_size = _header_size(shard)
    packed = _read_rows(shard, header_size, weight, row_start, rows, packed_columns)
    scale_bytes = _read_rows(shard, header_size, scale, row_start, rows, scale_columns)
    scale2_bytes = _read_exact_range(shard, header_size, scale2_spec)
    if len(scale2_bytes) != 4:
        raise ValueError("weight_scale_2 must be an F32 scalar")
    scale2 = struct.unpack("<f", scale2_bytes)[0]
    decoded = decode_nvfp4_rows(
        packed,
        scale_bytes,
        scale2,
        rows=rows,
        packed_columns=packed_columns,
    )
    decoded_bytes = b"".join(struct.pack("<f", value) for value in decoded)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    producer = ((config.get("quantization_config") or {}).get("producer") or {})
    producer_text = f"{producer.get('name', 'unknown')}@{producer.get('version', 'unknown')}"
    return NVFP4SampleReport(
        schema_version=1,
        source=str(root),
        producer=producer_text,
        tensor=tensor,
        row_start=row_start,
        rows=rows,
        decoded_shape=(rows, packed_columns * 2),
        block_size=16,
        bytes_read=len(packed) + len(scale_bytes) + len(scale2_bytes),
        raw_sha256=hashlib.sha256(packed + scale_bytes + scale2_bytes).hexdigest(),
        decoded_f32_sha256=hashlib.sha256(decoded_bytes).hexdigest(),
        finite=all(math.isfinite(value) for value in decoded),
        minimum=min(decoded),
        maximum=max(decoded),
        l2_norm=math.sqrt(math.fsum(value * value for value in decoded)),
        scale2=scale2,
    )


def write_sample_report(report: NVFP4SampleReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _tensor_spec(
    header: dict[str, Any], name: str, dtype: str, dimensions: int
) -> dict[str, Any]:
    spec = header.get(name)
    if not isinstance(spec, dict):
        raise ValueError(f"missing tensor header for {name}")
    if spec.get("dtype") != dtype or len(spec.get("shape") or []) != dimensions:
        raise ValueError(f"unexpected dtype/shape for {name}")
    offsets = spec.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise ValueError(f"invalid data offsets for {name}")
    return spec


def _header_size(path: Path) -> int:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError("truncated safetensors header length")
    return 8 + int(struct.unpack("<Q", raw)[0])


def _read_rows(
    path: Path,
    data_base: int,
    spec: dict[str, Any],
    row_start: int,
    rows: int,
    row_bytes: int,
) -> bytes:
    start, end = (int(value) for value in spec["data_offsets"])
    if end - start != int(spec["shape"][0]) * row_bytes:
        raise ValueError("tensor byte length does not match row geometry")
    offset = data_base + start + row_start * row_bytes
    size = rows * row_bytes
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError("short safetensors row read")
    return data


def _read_exact_range(path: Path, data_base: int, spec: dict[str, Any]) -> bytes:
    start, end = (int(value) for value in spec["data_offsets"])
    size = end - start
    with path.open("rb") as handle:
        handle.seek(data_base + start)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError("short safetensors scalar read")
    return data
