"""Bounded, indexed Safetensors tensor-body access (Phase 1 substrate).

Dependency-free (no numpy/torch): opens a shard with ``mmap`` and reads only the
exact byte range of one tensor or one of its nested members, `never materializing
the whole shard`. Decodes the float formats the real GLM-5.2 checkpoint carries
(BF16 / FP16 / FP32 / F32 scalars) with pure-Python bit math. Provides identity
read→write→compare proof and records peak resident estimate.

This is the primitive the streaming layer/expert trace and the derivative
materializer will build on; it does not mutate source.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from model_atlas.checkpoint.source_manifest import CheckpointManifest, TensorEntry

_BF16 = struct.Struct("<H")
_F16 = struct.Struct("<H")
_F32 = struct.Struct("<f")
_I8 = struct.Struct("<b")

_DTYPE_ITEMSIZE: dict[str, int] = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}


class UnsupportedDtypeError(ValueError):
    pass


def _bf16_to_float(bits: int) -> float:
    """Convert an IEEE 754 bfloat16 bit pattern to a Python float."""
    f32_bits = bits << 16
    return float(struct.unpack("<f", struct.pack("<I", f32_bits))[0])


def _f16_to_float(bits: int) -> float:
    """Convert an IEEE 754 half-precision bit pattern to a Python float."""
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0x1F:
        return float("inf") if frac == 0 else float("nan")
    if exp == 0:
        value: float = frac * (2 ** (-14 - 10))
    else:
        value = (1 + frac / 1024) * (2 ** (exp - 15))
    return -value if sign else value


def decode_values(data: bytes, dtype: str, shape: list[int]) -> list[float]:
    """Decode a raw byte buffer into a flat list of float values.

    Only the float dtype the GLM-5.2 NVFP4 checkpoint carries in its non-quant
    (reference / BF16-tier) tensors is supported here; NVFP4 weights need their
    token + scale layout handled separately (see AMDQuantNote / real body
    validation module).
    """
    dtype_upper = dtype.upper()
    n = len(data)
    if dtype_upper == "BF16":
        out = [0.0] * (n // 2)
        for i in range(0, n, 2):
            out[i // 2] = _bf16_to_float(_BF16.unpack_from(data, i)[0])
        return out
    if dtype_upper == "F16":
        out = [0.0] * (n // 2)
        for i in range(0, n, 2):
            out[i // 2] = _f16_to_float(_F16.unpack_from(data, i)[0])
        return out
    if dtype_upper == "F32":
        return [float(v) for v in struct.unpack(f"<{n // 4}f", data)]
    if dtype_upper in {"I8", "U8"}:
        return [float(_I8.unpack_from(data, i)[0]) for i in range(n)]
    wanted = _DTYPE_ITEMSIZE.get(dtype_upper)
    if wanted is None or (n % wanted) != 0:
        raise UnsupportedDtypeError(
            f"unsupported dtype {dtype!r} with {n} bytes (itemsize {wanted})"
        )
    if dtype_upper == "I16":
        return [float(v) for v in struct.unpack_from(f"<{n//2}h", data, 0)]
    if dtype_upper == "I32":
        return [float(v) for v in struct.unpack_from(f"<{n//4}i", data, 0)]
    if dtype_upper == "I64":
        return [float(v) for v in struct.unpack_from(f"<{n//8}q", data, 0)]
    if dtype_upper == "U8":
        return [float(b) for b in data]
    raise UnsupportedDtypeError(f"unsupported dtype {dtype!r}")

    _ = shape  # shape is carried on the entry; decode is flat


@dataclass
class BodyRead:
    """One bounded tensor-body read."""

    name: str
    dtype: str
    shape: list[int]
    byte_size: int
    values: list[float]
    source_shard: str
    offset_start: int

    @property
    def numel(self) -> int:
        return len(self.values)


class BoundedShardReader:
    """mmap-backed reader that fetches only the requested tensor bodies."""

    def __init__(self, shard_path: str | os.PathLike[str]) -> None:
        self._path = Path(shard_path)
        self._file = open(self._path, "rb")  # noqa: SIM115 — must stay open for mmap lifetime
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._size = self._mm.size()
        (self._header_len,) = struct.unpack("<Q", self._mm[:8])
        self._data_base = 8 + self._header_len
        self.peak_bytes = 0
        self.reads = 0

    def __enter__(self) -> BoundedShardReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._file.close()

    def read_body(self, entry: TensorEntry) -> BodyRead:
        """Read exactly one tensor body's bytes from this shard."""
        start = entry.offset_start
        end = entry.offset_end
        if (start < 0 or end > self._size - self._data_base or start >= end):
            raise ValueError(
                f"tensor {entry.name}: byte range [{start},{end}) out of data section "
                f"(size {self._size - self._data_base})"
            )
        data = self._mm[self._data_base + start : self._data_base + end]
        self.peak_bytes = max(self.peak_bytes, len(data))
        self.reads += 1
        # Decode only reference float dtypes (BF16/F16/F32). NVFP4-produced
        # tensors carry an opaque 4-bit token layout that needs the modelopt
        # scale scheme; they are validated separately, never mis-decoded here.
        if entry.dtype.upper() in {"BF16", "F16", "F32"}:
            try:
                vals = decode_values(data, entry.dtype, entry.shape)
            except UnsupportedDtypeError:
                vals = []
        else:
            vals = []
        return BodyRead(
            name=entry.name,
            dtype=entry.dtype,
            shape=list(entry.shape),
            byte_size=len(data),
            values=vals,
            source_shard=self._path.name,
            offset_start=start,
        )


@dataclass
class StreamingStats:
    shards_opened: int = 0
    tensors_read: int = 0
    bytes_read: int = 0
    peak_bytes: int = 0
    _open: dict[str, BoundedShardReader] = field(default_factory=dict, repr=False)


class CheckpointStream:
    """Open a checkpoint and stream tensors by name / by (layer, expert, suffix)
    without materializing the full model. Tracks peak resident bytes.

    Readers stay open (mmap) per shard; each `read` fetches only the requested
    byte range. Never mutates the source.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        from model_atlas.checkpoint.source_manifest import load_manifest

        self._dir = Path(checkpoint_dir)
        self.manifest: CheckpointManifest = load_manifest(checkpoint_dir)
        self._by_name: dict[str, TensorEntry] = {t.name: t for t in self.manifest.tensors}
        self._readers: dict[str, BoundedShardReader] = {}
        self.stats = StreamingStats(shards_opened=0)

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers.clear()

    def __enter__(self) -> CheckpointStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _reader(self, shard: str) -> BoundedShardReader:
        if shard not in self._readers:
            self._readers[shard] = BoundedShardReader(self._dir / shard)
            self.stats.shards_opened += 1
        return self._readers[shard]

    def get(self, name: str) -> BodyRead | None:
        entry = self._by_name.get(name)
        if entry is None:
            return None
        br = self._reader(entry.shard).read_body(entry)
        self.stats.peak_bytes = max(self.stats.peak_bytes, br.byte_size)
        self.stats.tensors_read += 1
        self.stats.bytes_read += br.byte_size
        return br

    def tensors(self, predicate: Callable[[TensorEntry], bool]) -> list[BodyRead]:
        """Stream every tensor matching `predicate(entry)`, one at a time."""
        out: list[BodyRead] = []
        for entry in self.manifest.tensors:
            if predicate(entry):
                br = self._reader(entry.shard).read_body(entry)
                out.append(br)
                self.stats.tensors_read += 1
                self.stats.bytes_read += br.byte_size
        return out


def identity_copy(manifest: CheckpointManifest, names: list[str] | None = None) -> dict[str, bytes]:
    """Read selected tensor bodies and return {name: raw_bytes} — a bounded,
    identity-preserving read used for copy/upstream-equivalence tests."""
    out: dict[str, bytes] = {}
    by_shard: dict[str, list[TensorEntry]] = {}
    for t in manifest.tensors:
        if names and t.name not in names:
            continue
        by_shard.setdefault(t.shard, []).append(t)
    for shard, entries in by_shard.items():
        with open(Path(manifest.checkpoint_dir) / shard, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            base = 8 + header_len
            for entry in entries:
                start = base + entry.offset_start
                end = base + entry.offset_end
                f.seek(start)
                out[entry.name] = f.read(end - start)
    return out

