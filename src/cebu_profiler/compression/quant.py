"""Real quantization math for compression probes (v2 §21–§23).

Uniform signed-integer quantization (absmax symmetric) and float-mantissa
rounding (bf16/fp16-style). Both are honest arithmetic on real weights; we
never claim deployable inference for them (probe-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QuantMeta:
    format: str
    effective_bits: float
    stored_bytes: float  # weight bytes + small metadata/scale allowance
    numel: int
    scale_count: int
    sparsity: float = 0.0  # fraction of zeroed coefficients (structured pruning)


def uniform_int_quant(rows: list[list[float]], bits: int) -> tuple[list[list[float]], QuantMeta]:
    """Symmetric uniform integer quantization with one scale per tensor.

    Returns dequantized rows + metadata. `bits` = effective bit width (e.g.
    int8=8, int4=4, nvfp4≈4, fp8≈8). Per-tensor absmax scale keeps the probe
    deterministic and dependency-free.
    """
    numel = sum(len(r) for r in rows)
    amax = max((abs(v) for r in rows for v in r), default=0.0)
    levels = 2 ** max(1, bits - 1)  # symmetric, one side for zero
    step = amax / levels if amax > 0 else 0.0
    out: list[list[float]] = []
    n_zero = 0
    for r in rows:
        row: list[float] = []
        for v in r:
            if step == 0.0:
                q = 0
            else:
                q = max(-levels, min(levels - 1, round(v / step)))
                q = int(q)
            row.append(q * step)
            if q == 0:
                n_zero += 1
        out.append(row)
    # one scale per tensor (per expert weight matrix) — 32-bit scale + tiny header
    meta_bytes = numel * bits / 8 + 4 + 16
    return out, QuantMeta(
        format=f"int{bits}",
        effective_bits=float(bits),
        stored_bytes=meta_bytes,
        numel=numel,
        scale_count=1,
        sparsity=n_zero / numel if numel else 0.0,
    )


def float_mantissa_quant(
    rows: list[list[float]], bit_width: int, mantissa: int
) -> tuple[list[list[float]], QuantMeta]:
    """bf16/fp16-style rounding of the mantissa (keeps exponent)."""
    numel = sum(len(r) for r in rows)
    out: list[list[float]] = []
    for r in rows:
        row: list[float] = []
        for v in r:
            if v == 0.0:
                row.append(0.0)
                continue
            e = math.floor(math.log2(abs(v)))
            unit = 2 ** (e - mantissa)
            q = math.copysign(round(abs(v) / unit) * unit, v)
            row.append(q)
        out.append(row)
    meta_bytes = numel * bit_width / 8 + 16
    return out, QuantMeta(
        format=f"f{bit_width}",
        effective_bits=float(bit_width),
        stored_bytes=meta_bytes,
        numel=numel,
        scale_count=0,
    )


def rel_l2(a: list[list[float]], b: list[list[float]]) -> float:
    """Relative L2 reconstruction error ||a-b|| / ||a|| between two tensors."""
    sse = 0.0
    norm = 0.0
    for ra, rb in zip(a, b, strict=True):
        for va, vb in zip(ra, rb, strict=True):
            sse += (va - vb) ** 2
            norm += va * va
    return math.sqrt(sse / norm) if norm > 0 else 0.0
