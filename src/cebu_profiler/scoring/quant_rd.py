"""Sampled rate-distortion (R-D) quantization screens over a real checkpoint.

Blueprint §7 (sensitivity signals) + AGENTS.md invariant 12 (typed evidence):
every error figure here is `measured` on real BF16 tensor bodies, sampled in a
deterministic, reproducible way (seeded row selection, fixed sample geometry).
It never touches the GPU and never holds a whole 12+ GiB expert bank: it reads
bounded row-slices via ``TensorFetcher.read_slice``.

What is measured
----------------
For each sampled tensor (or packed expert bank), we compute reconstruction
error curves across candidate representations at several bpw levels:

- uniform symmetric integer quantization (per-output-channel scales) — a
  stand-in for int4/int8 kernel paths,
- bf16→fp16 rounding (as an fp8-proxy upper bound is added by the allocator as
  a fixed offset, not measured here).

Output: ``RDReport`` — per tensor-name, per bpw: relative L2 error, measured
rows/cols sampled, and the sample geometry, so the allocator (planning module)
can do a global GEMQ-style bit allocation with honest measured slopes.

Probe-only: these errors bound *weight reconstruction*, not end-to-end
behavior. Behavioral gates remain the separate phase defined in AGENTS.md.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any

from cebu_profiler.checkpoint.source_manifest import CheckpointManifest, TensorEntry
from cebu_profiler.checkpoint.tensor_io import TensorFetcher


class RDError(RuntimeError):
    """Rate-distortion screen failure."""


# Sample geometry: rows are selected deterministically; a row slice of a
# [out, in] matrix is contiguous in safetensors row-major storage, so each
# sampled row is one bounded range read.
MAX_SAMPLE_ROWS = 24
ROW_ELEMS_CAP = 8192  # cap cols per row read for very wide matrices


@dataclass
class TensorRD:
    """Measured R-D points for one tensor."""

    name: str
    role: str
    layer_index: int | None
    expert_index: int | None
    shape: list[int]
    bf16_bytes: int
    # bpw -> rel L2 error (measured on sampled rows)
    errors: dict[float, float] = field(default_factory=dict)
    sample_rows: int = 0
    sample_cols: int = 0


@dataclass
class RDReport:
    """All measured R-D data for a run. JSON-serializable via ``to_dict``."""

    checkpoint: str
    seed: int
    tensors: list[TensorRD] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "evidence": "measured",
            "tensors": [
                {
                    "name": t.name,
                    "role": t.role,
                    "layer": t.layer_index,
                    "expert": t.expert_index,
                    "shape": t.shape,
                    "bf16_bytes": t.bf16_bytes,
                    "errors": t.errors,
                    "sample_rows": t.sample_rows,
                    "sample_cols": t.sample_cols,
                }
                for t in self.tensors
            ],
        }


def _numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError as exc:  # pragma: no cover
        raise RDError("numpy is required for R-D screens (pip install 'cebu-profiler[profile]')") from exc


def _bf16_rows_to_f32(np: Any, buf: bytes, n_rows: int, row_elems: int) -> Any:
    u16 = np.frombuffer(buf, dtype="<u2")
    out = np.empty(u16.shape, dtype=np.float32)
    out.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return out.reshape(n_rows, row_elems)


def _row_slice_bytes(entry: TensorEntry, row: int, cols: int) -> tuple[int, int]:
    """(rel-offset, size) of one contiguous row block of a BF16 2-D tensor.

    rel-offset is relative to the tensor data start (read_slice adds the
    tensor's own offset_start); the row STRIDE comes from entry.shape[1].
    """
    stride_bytes = entry.shape[1] * 2
    start = row * stride_bytes
    return start, cols * 2


def _select_rows(num_rows: int, seed: int, k: int) -> list[int]:
    """Deterministic evenly-strided row selection (seed only breaks ties)."""
    if num_rows <= k:
        return list(range(num_rows))
    stride = num_rows / k
    rows = sorted({min(num_rows - 1, int(i * stride)) for i in range(k)})
    # seed-based jitter of at most one row, deterministic
    if seed:
        shift = seed % max(1, num_rows // max(1, len(rows)))
        rows = [min(num_rows - 1, r + (shift if i % 2 else 0)) for i, r in enumerate(rows)]
        rows = sorted(set(rows))
    return rows


def _uniform_int_qerr(np: Any, mat: Any, bits: float) -> float:
    """Per-output-channel symmetric uniform quantization rel-L2 on a 2-D matrix.

    One absmax scale per row (output channel) — the same granularity the
    kernel-lab EXL3/NVFP4 paths assume (per-channel scales, tile codes beyond).
    """
    amax = np.max(np.abs(mat), axis=1, keepdims=True)
    amax = np.where(amax > 0, amax, 1.0)
    # symmetric grid: levels = number of positive steps; bits=1.5 -> ~3 codes (-1,0,+1)
    levels = max(2.0 ** (bits - 1.0), 1.0)
    step = amax / levels
    q = np.round(mat / step) * step
    q = np.clip(q, -(levels) * step, levels * step)
    num = float(np.sum((mat - q) ** 2))
    den = float(np.sum(mat ** 2))
    return math.sqrt(num / den) if den > 0 else 0.0


def screen_tensor(
    fetcher: TensorFetcher,
    entry: TensorEntry,
    role: str,
    layer_index: int | None,
    expert_index: int | None,
    bpw_levels: tuple[float, ...],
    seed: int,
) -> TensorRD:
    """Measure R-D points for one BF16 tensor via bounded row sampling.

    Accepts 2-D matrices and 3-D packed expert banks ``[E, out, in]``; for
    3-D banks a deterministic subset of (expert, row) slices is sampled, each
    one contiguous in safetensors row-major storage.
    """
    np = _numpy()
    if entry.dtype != "BF16":
        raise RDError(f"R-D screen expects BF16 teacher tensors, got {entry.dtype} for {entry.name}")
    if len(entry.shape) == 3:
        n_exp, rows, cols = entry.shape
        sample_cols = min(cols, ROW_ELEMS_CAP)
        exp_sel = _select_rows(n_exp, seed, min(MAX_SAMPLE_ROWS, n_exp))
        row_sel = _select_rows(rows, seed + 1, max(1, MAX_SAMPLE_ROWS // len(exp_sel)))
        mats = []
        stride = rows * cols * 2
        for e in exp_sel:
            for r in row_sel:
                rel = e * stride + r * cols * 2
                buf = fetcher.read_slice(entry, rel, sample_cols * 2)
                mats.append(_bf16_rows_to_f32(np, buf, 1, sample_cols))
        mat = np.concatenate(mats, axis=0)
        rd = TensorRD(
            name=entry.name,
            role=role,
            layer_index=layer_index,
            expert_index=expert_index,
            shape=list(entry.shape),
            bf16_bytes=entry.byte_size,
            sample_rows=len(mats),
            sample_cols=sample_cols,
        )
        for bits in bpw_levels:
            rd.errors[float(bits)] = _uniform_int_qerr(np, mat, bits)
        return rd
    if len(entry.shape) != 2:
        raise RDError(f"R-D screen expects 2-D/3-D tensors, got shape {entry.shape} for {entry.name}")

    rows, cols = entry.shape
    sample_cols = min(cols, ROW_ELEMS_CAP)
    sel = _select_rows(rows, seed, MAX_SAMPLE_ROWS)
    mats = []
    for r in sel:
        off, size = _row_slice_bytes(entry, r, sample_cols)
        buf = fetcher.read_slice(entry, off, size)
        mats.append(_bf16_rows_to_f32(np, buf, 1, sample_cols))
    mat = np.concatenate(mats, axis=0)

    rd = TensorRD(
        name=entry.name,
        role=role,
        layer_index=layer_index,
        expert_index=expert_index,
        shape=list(entry.shape),
        bf16_bytes=entry.byte_size,
        sample_rows=len(sel),
        sample_cols=sample_cols,
    )
    for bits in bpw_levels:
        rd.errors[float(bits)] = _uniform_int_qerr(np, mat, bits)
    return rd


def screen_checkpoint(
    manifest: CheckpointManifest,
    fetcher: TensorFetcher,
    *,
    bpw_levels: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0),
    seed: int = 0,
    max_tensors: int = 400,
    min_byte_size: int = 1 << 20,
) -> RDReport:
    """R-D-screen the checkpoint's tensors (largest first, deterministic).

    Skips quant-metadata companions and non-2-D tensors. Bounded memory:
    at most MAX_SAMPLE_ROWS × ROW_ELEMS_CAP fp32 elements held at once.
    """
    from cebu_profiler.checkpoint.classifier import classify_tensor

    report = RDReport(checkpoint=str(manifest.checkpoint_dir), seed=seed)
    entries = [
        t
        for t in manifest.tensors
        if t.dtype == "BF16" and len(t.shape) in (2, 3) and t.byte_size >= min_byte_size
    ]
    entries.sort(key=lambda t: (-t.byte_size, t.name))
    screened = 0
    for entry in entries:
        if screened >= max_tensors:
            break
        c = classify_tensor(entry.name)
        if c.unclassified or c.is_quant_metadata or c.role is None:
            continue
        try:
            report.tensors.append(
                screen_tensor(
                    fetcher,
                    entry,
                    role=str(c.role),
                    layer_index=c.layer_index,
                    expert_index=c.expert_index,
                    bpw_levels=bpw_levels,
                    seed=seed,
                )
            )
            screened += 1
        except RDError:
            continue
    return report
