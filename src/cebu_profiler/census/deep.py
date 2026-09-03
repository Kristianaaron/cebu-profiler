"""Deep census: measured per-tensor quantization damage and distribution stats.

Where a checkpoint's tensors live in memory, pure-Python (stdlib-only, no
torch/numpy) and chunked so a multi-hundred-GB shard never loads wholesale.

For each 2-D tensor we MEASURE (not estimate) the quantization error of three
schemes — INT8 per-channel, INT4 group-128, FP8 e4m3 — as SQNR in dB,
10·log10(||W||²/||W−Ŵ||²), plus distribution facts (mean/std/absmax, outlier
ratios, log2 histogram) and, optionally, a few leading singular values by
one-pass power iteration.

This extends the header-only census (`checkpoint.source_manifest`, which stays
the default for huge checkpoints) with an opt-in evidence upgrade, in the spirit
of measured-not-projected weight autopsies such as alesha-pro/atlas's
weight_atlas scan (SQNR/spectrum/histogram per tensor), re-implemented here on
Cebu's manifest model and from raw safetensors bytes.

Evidence typing (invariant 12): every field returned here is `measured` for
the tensors actually scanned; tensors skipped by --limit are simply absent.
Metrics that cannot apply (1-D biases, norms) are reported as None, never
fabricated as zeros.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# safetensors dtype -> (bytes per element, struct char). We decode exactly the
# dtypes the deep census supports; anything else is skipped with a reason.
# (Byte order is composed at unpack time as f"<{n}{char}".)
_DTYPE_FMT: dict[str, tuple[int, str | None]] = {
    "F32": (4, "f"),
    "F16": (2, "e"),
    "BF16": (2, None),  # handled specially: bf16 = top 16 bits of f32
    "F64": (8, "d"),
}

_SQNR_REF: dict[str, str] = {
    "int8_perchannel": "INT8 symmetric, per-row scale",
    "int4_group128": "INT4 symmetric, group size 128 along the row",
    "fp8_e4m3": "IEEE-style FP8 e4m3 round-trip",
}


@dataclass
class DeepTensorReport:
    name: str
    shape: list[int]
    dtype: str
    numel: int
    shard: str
    # distribution facts (measured)
    mean: float | None = None
    std: float | None = None
    absmax: float | None = None
    outlier_3s: float | None = None  # fraction of elements > 3σ
    # per-scheme SQNR in dB (2-D tensors only; None = not applicable)
    sqnr_int8_perchannel: float | None = None
    sqnr_int4_group128: float | None = None
    sqnr_fp8_e4m3: float | None = None
    # optional spectrum (one-pass power iteration, when requested)
    sv_leading: list[float] = field(default_factory=list)
    stable_rank: float | None = None
    notes: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "numel": self.numel,
            "shard": self.shard,
            "evidence": "measured",
        }
        for k in (
            "mean",
            "std",
            "absmax",
            "outlier_3s",
            "sqnr_int8_perchannel",
            "sqnr_int4_group128",
            "sqnr_fp8_e4m3",
            "stable_rank",
        ):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        if self.sv_leading:
            out["sv_leading"] = self.sv_leading
        if self.notes:
            out["notes"] = self.notes
        return out


def _iter_shard_values(
    path: Path, dtype: str, offset: int, byte_size: int, max_elements: int | None
) -> Iterator[list[float]]:
    """Yield chunks of decoded floats from one tensor's data region."""
    spec = _DTYPE_FMT.get(dtype)
    if spec is None:
        return
    nbytes_per, fmt = spec
    with path.open("rb") as fh:
        fh.seek(offset)
        remaining = byte_size
        # chunk: ~1M elements at a time (bounded RAM regardless of tensor size)
        chunk_elems = 1_000_000
        if max_elements is not None:
            remaining = min(remaining, max_elements * nbytes_per)
        while remaining > 0:
            take = min(chunk_elems * nbytes_per, remaining)
            buf = fh.read(take)
            if not buf:
                break
            if dtype == "BF16":
                # bf16: 2-byte mantissa-truncated float; widen to f32 by
                # appending zero bytes then decoding as f32 pairs.
                words = struct.unpack(f"<{len(buf) // 2}H", buf)
                yield [struct.unpack("<f", struct.pack("<I", w << 16))[0] for w in words]
            else:
                n = len(buf) // nbytes_per
                yield list(struct.unpack(f"<{n}{fmt}", buf[: n * nbytes_per]))
            remaining -= len(buf)


def _quant_int8_perchannel(chunks: list[list[float]], rows: int, cols: int) -> float:
    """Symmetric INT8 with a per-row scale; SQNR in dB over the whole tensor."""
    # accumulate per-row sum of squares of W and of (W - Q) in one pass
    num = 0.0  # ||W||^2
    den = 0.0  # ||W - Q||^2
    flat = [x for chunk in chunks for x in chunk]
    for r in range(rows):
        row = flat[r * cols : (r + 1) * cols]
        amax = max((abs(x) for x in row), default=0.0)
        scale = amax / 127.0 if amax > 0 else 1.0
        for x in row:
            q = round(x / scale)
            q = max(-127, min(127, q))
            num += x * x
            d = x - q * scale
            den += d * d
    return 10.0 * math.log10(num / den) if den > 0 and num > 0 else math.inf


def _quant_int4_group128(chunks: list[list[float]], rows: int, cols: int) -> float:
    """Symmetric INT4 with group size 128 along each row; SQNR in dB."""
    g = 128
    num = 0.0
    den = 0.0
    flat = [x for chunk in chunks for x in chunk]
    for r in range(rows):
        row = flat[r * cols : (r + 1) * cols]
        for start in range(0, len(row), g):
            grp = row[start : start + g]
            amax = max((abs(x) for x in grp), default=0.0)
            scale = amax / 7.0 if amax > 0 else 1.0
            for x in grp:
                q = round(x / scale)
                q = max(-7, min(7, q))
                num += x * x
                d = x - q * scale
                den += d * d
    return 10.0 * math.log10(num / den) if den > 0 and num > 0 else math.inf


def _fp8_e4m3_roundtrip(x: float) -> float:
    """Quantize one float to FP8 e4m3 precision (4 exp bits, 3 mantissa bits)."""
    if x == 0.0 or not math.isfinite(x):
        return x
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x)
    # e4m3: max ~448, mantissa steps of 2^-3 relative
    if ax > 448.0:
        return sign * 448.0
    if ax < 2.0**-6 / 2:  # below min subnormal/2 -> 0
        return 0.0
    exp = math.floor(math.log2(ax))
    # clamp exponent to e4m3 range (-6..8)
    exp = max(-6, min(8, exp))
    mant = ax / (2.0**exp)  # in [1, 2)
    mant_q = round(mant * 8.0) / 8.0  # 3 mantissa bits
    if mant_q >= 2.0:
        mant_q = 1.0
        exp = min(8, exp + 1)
    return sign * mant_q * (2.0**exp)


def _quant_fp8(chunks: list[list[float]]) -> float:
    num = 0.0
    den = 0.0
    for chunk in chunks:
        for x in chunk:
            q = _fp8_e4m3_roundtrip(x)
            num += x * x
            d = x - q
            den += d * d
    return 10.0 * math.log10(num / den) if den > 0 and num > 0 else math.inf


def _power_iteration_sv(chunks: list[list[float]], rows: int, cols: int, k: int = 3) -> list[float]:
    """A few leading singular values by power iteration on A^T A (approximate)."""
    flat = [x for chunk in chunks for x in chunk]
    svs: list[float] = []
    # work on a deflated copy in-place via subtraction of rank-1 approximations
    a = flat[:]
    for _ in range(k):
        v = [0.0] * cols
        # init v deterministically from column 0
        for r in range(rows):
            aij = a[r * cols]
            if aij != 0.0:
                for c in range(cols):
                    v[c] += aij * a[r * cols + c]
        nv = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / nv for x in v]
        # u = A v
        u = [0.0] * rows
        for r in range(rows):
            s = 0.0
            for c in range(cols):
                s += a[r * cols + c] * v[c]
            u[r] = s
        nu = math.sqrt(sum(x * x for x in u))
        if nu <= 0.0:
            break
        svs.append(nu)
        # deflate: A <- A - sigma * u v^T
        for r in range(rows):
            if u[r] == 0.0:
                continue
            coef = nu * u[r] / (nu * nu)
            for c in range(cols):
                a[r * cols + c] -= coef * u[r] * v[c]
    return svs


def deep_scan_tensor(
    shard_path: Path,
    *,
    name: str,
    dtype: str,
    shape: list[int],
    offset: int,
    byte_size: int,
    shard: str,
    with_spectrum: bool = False,
    max_elements: int | None = 4_000_000,
) -> DeepTensorReport | None:
    """Measure one tensor. Returns None when the dtype is unsupported."""
    if dtype not in _DTYPE_FMT:
        return None
    report = DeepTensorReport(
        name=name, shape=shape, dtype=dtype, numel=math.prod(shape), shard=shard
    )
    chunks = list(_iter_shard_values(shard_path, dtype, offset, byte_size, max_elements))
    vals = [x for chunk in chunks for x in chunk]
    n = len(vals)
    if n == 0:
        report.notes.append("no elements decoded (truncated shard?)")
        return report
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    absmax = max(abs(x) for x in vals)
    report.mean, report.std, report.absmax = mean, std, absmax
    if std > 0:
        report.outlier_3s = sum(1 for x in vals if abs(x - mean) > 3 * std) / n
    if len(shape) == 2:
        rows, cols = shape
        if cols == 0 or rows == 0:
            report.notes.append("degenerate 2-D shape")
            return report
        budget = max_elements if max_elements is not None else n
        if n <= budget:
            report.sqnr_int8_perchannel = _quant_int8_perchannel(chunks, rows, cols)
            report.sqnr_int4_group128 = _quant_int4_group128(chunks, rows, cols)
        report.sqnr_fp8_e4m3 = _quant_fp8(chunks)
        if with_spectrum and n <= budget:
            svs = _power_iteration_sv(chunks, rows, cols, k=3)
            report.sv_leading = svs
            if svs and svs[0] > 0:
                total_energy = sum(x * x for x in vals)
                # stable rank approximation: ||A||_F^2 / sigma_max^2
                report.stable_rank = total_energy / (svs[0] ** 2)
    else:
        report.notes.append("1-D: SQNR not applicable, reported as None")
    return report


def deep_scan_manifest(
    checkpoint_dir: str,
    *,
    manifest: Any = None,
    limit: int | None = None,
    only_2d: bool = True,
    with_spectrum: bool = False,
    max_elements: int | None = 4_000_000,
) -> dict[str, Any]:
    """Scan a checkpoint dir (or a provided manifest) and return the bundle.

    The default header-only census stays untouched; this is the opt-in deep
    pass. Output is JSON-serializable and evidence-typed.
    """
    from cebu_profiler.checkpoint.source_manifest import load_manifest

    if manifest is None:
        manifest = load_manifest(checkpoint_dir)
    root = Path(checkpoint_dir)
    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    scanned = 0
    for t in manifest.tensors:
        if limit is not None and scanned >= limit:
            break
        if only_2d and len(t.shape) != 2:
            continue
        rep = deep_scan_tensor(
            root / t.shard,
            name=t.name,
            dtype=t.dtype,
            shape=list(t.shape),
            offset=t.offset_start,
            byte_size=t.byte_size,
            shard=t.shard,
            with_spectrum=with_spectrum,
            max_elements=max_elements,
        )
        if rep is None:
            skipped.append({"name": t.name, "reason": f"unsupported dtype {t.dtype}"})
            continue
        reports.append(rep.payload())
        scanned += 1
    return {
        "schema": 1,
        "evidence": "measured",
        "checkpoint": str(checkpoint_dir),
        "tensors_scanned": scanned,
        "tensors_skipped": skipped,
        "sqnr_schemes": _SQNR_REF,
        "reports": reports,
    }


def write_deep_census(bundle: dict[str, Any], out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(bundle, indent=1))
