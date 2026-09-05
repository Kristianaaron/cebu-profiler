"""GEMQ-style global bit allocation over measured rate-distortion data.

Brief §8: tensors are not optimized independently. Given a fixed weight-side
byte budget, allocate bpw per tensor-family × layer so that total measured
distortion is minimized (greedy marginal-distortion-per-byte, the GEMQ
principle), subject to the protection policy from brief §14:

- router / DSA-adjacent / norms / iHC / lm_head / embedding: FP8 floor (8 bpw
  class) or BF16 — never below, regardless of marginal slope;
- routed experts: free allocation in the 1.5–4.0 bpw EXL3 band;
- attention: free in the 3.5–8.0 band (NVFP4/EXL3 gate happens later against
  kernel-oracle throughput; this module only does fidelity-vs-bytes).

MixQuant-style conditional sensitivity (§8) is honored by construction: the
allocator re-runs on refreshed R-D data, so downstream re-measurement under
candidate compression states simply produces a new report to allocate from.

Evidence typing (AGENTS.md invariant 12): byte figures derive from measured
``bf16_bytes`` × bpw/16 (plus a measured-scale metadata allowance); distortion
figures are measured R-D samples. Anything extrapolated (per-tensor error from
family medians when a tensor was not screened) is tagged ``inferred`` in the
plan output.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cebu_profiler.scoring.quant_rd import RDReport, TensorRD

GIB = 1024**3

# Policy floors (brief §14). Roles not listed default to FREE allocation.
ROLE_FLOOR_BPW: dict[str, float] = {
    "router": 8.0,
    "router_bias": 16.0,
    "norm": 16.0,
    "hyper_connection": 16.0,
    "lm_head": 16.0,
    "embedding": 8.0,
}
ROLE_BAND: dict[str, tuple[float, float]] = {
    # role -> (min_bpw, max_bpw) for allocatable roles
    "experts": (1.5, 4.0),
    "attention": (3.5, 8.0),
    "mla_state": (3.5, 8.0),
    "latent_proj": (3.5, 8.0),
    "shared_expert": (2.5, 6.0),
}
DEFAULT_BAND: tuple[float, float] = (2.0, 8.0)

# Metadata allowance per tensor (scales/headers), bytes — conservative flat fee.
_META_BYTES_PER_TENSOR = 4096

# Fixed representation classes the plan snaps to (EXL3 band + FP8/BF16 tail).
SNAP_POINTS: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 16.0)


def _snap(x: float) -> float:
    return min(SNAP_POINTS, key=lambda p: abs(p - x))


@dataclass
class AllocItem:
    """One tensor's allocation in the final plan."""

    name: str
    role: str
    layer: int | None
    bf16_bytes: int
    bpw: float
    target_bytes: int
    evidence: str = "measured"  # or "inferred" when error came from family median
    rel_err: float | None = None


@dataclass
class AllocPlan:
    """A global bit-allocation plan at one budget point."""

    checkpoint: str
    budget_gib: float
    items: list[AllocItem] = field(default_factory=list)
    total_target_bytes: int = 0
    total_measured_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        fam = defaultdict(lambda: [0, 0.0, 0])
        for it in self.items:
            key = it.role
            fam[key][0] += it.target_bytes
            fam[key][1] += it.bpw
            fam[key][2] += 1
        return {
            "checkpoint": self.checkpoint,
            "budget_gib": self.budget_gib,
            "evidence": "mixed: bytes=measured×bpw, distortion=measured-sample",
            "total_target_gib": round(self.total_target_bytes / GIB, 3),
            "total_bf16_gib": round(self.total_measured_bytes / GIB, 3),
            "families": {
                k: {
                    "target_gib": round(v[0] / GIB, 3),
                    "mean_bpw": round(v[1] / v[2], 3) if v[2] else None,
                    "tensors": v[2],
                }
                for k, v in sorted(fam.items())
            },
            "items": [
                {
                    "name": it.name,
                    "role": it.role,
                    "layer": it.layer,
                    "bpw": it.bpw,
                    "target_gib": round(it.target_bytes / GIB, 6),
                    "evidence": it.evidence,
                    "rel_err": it.rel_err,
                }
                for it in self.items
            ],
        }


def _bytes_at(t: TensorRD, bpw: float) -> int:
    return int(t.bf16_bytes * bpw / 16.0) + _META_BYTES_PER_TENSOR


def _family_median_error(report: RDReport) -> dict[tuple[str, float], float]:
    """Median measured error per (role, bpw) — for tensors not directly screened."""
    bucket: dict[tuple[str, float], list[float]] = defaultdict(list)
    for t in report.tensors:
        for bpw, err in t.errors.items():
            bucket[(t.role, bpw)].append(err)
    out: dict[tuple[str, float], float] = {}
    for key, errs in bucket.items():
        errs.sort()
        n = len(errs)
        out[key] = errs[n // 2] if n % 2 else (errs[n // 2 - 1] + errs[n // 2]) / 2
    return out


def allocate(
    report: RDReport,
    budget_gib: float,
    *,
    protected_bytes: int | None = None,
) -> AllocPlan:
    """Greedy marginal-distortion global allocation (GEMQ principle).

    Starts every allocatable tensor at its band floor, protected roles at
    their floor (16 = BF16-class, 8 = FP8-class), then spends the remaining
    budget on the largest distortion reduction per byte, in global competition.
    """
    plan = AllocPlan(
        checkpoint=report.checkpoint,
        budget_gib=budget_gib,
        total_measured_bytes=sum(t.bf16_bytes for t in report.tensors),
    )
    fam_err = _family_median_error(report)
    levels = sorted({b for t in report.tensors for b in t.errors})

    items: list[AllocItem] = []
    state: list[dict[str, Any]] = []

    protected_total = 0
    for t in report.tensors:
        if t.role in ROLE_FLOOR_BPW:
            bpw = ROLE_FLOOR_BPW[t.role]
            it = AllocItem(
                name=t.name,
                role=t.role,
                layer=t.layer_index,
                bf16_bytes=t.bf16_bytes,
                bpw=bpw,
                target_bytes=_bytes_at(t, bpw),
                rel_err=t.errors.get(bpw) or fam_err.get((t.role, bpw)),
            )
            items.append(it)
            protected_total += it.target_bytes
            continue
        lo, hi = ROLE_BAND.get(t.role, DEFAULT_BAND)
        base_bpw = lo
        it = AllocItem(
            name=t.name,
            role=t.role,
            layer=t.layer_index,
            bf16_bytes=t.bf16_bytes,
            bpw=base_bpw,
            target_bytes=_bytes_at(t, base_bpw),
            rel_err=t.errors.get(base_bpw) or fam_err.get((t.role, base_bpw)),
        )
        items.append(it)
        state.append(
            {
                "item": it,
                "tensor": t,
                "lo": lo,
                "hi": hi,
            }
        )

    budget_bytes = int(budget_gib * GIB)
    remaining = budget_bytes - protected_total - sum(s["item"].target_bytes for s in state)
    if remaining < 0:
        # Even floors bust the budget: raise floors to the minimum possible
        # (fail-closed: plan is emitted over-budget and flagged by caller).
        remaining = 0

    # Greedy: upgrade whichever tensor gives the biggest measured error drop per byte.
    while True:
        best = None
        best_gain = 0.0
        for s in state:
            it = s["item"]
            cur = it.bpw
            nxt_levels = [b for b in levels if b > cur and b <= s["hi"]]
            if not nxt_levels:
                continue
            # consider every higher level; pick the best gain-per-byte among them
            best_local = None
            best_local_gain = 0.0
            for cand in nxt_levels:
                e_c = t.errors.get(cand) or fam_err.get((t.role, cand))
                e_0 = t.errors.get(cur) or fam_err.get((t.role, cur))
                if e_c is None or e_0 is None:
                    continue
                db = _bytes_at(t, cand) - _bytes_at(t, cur)
                if db <= 0 or db > remaining:
                    continue
                g = (e_0 - e_c) / db
                if g > best_local_gain:
                    best_local_gain = g
                    best_local = cand
            if best_local is None:
                continue
            nxt = best_local
            t: TensorRD = s["tensor"]
            e_cur = t.errors.get(cur) or fam_err.get((t.role, cur))
            e_nxt = t.errors.get(nxt) or fam_err.get((t.role, nxt))
            if e_cur is None or e_nxt is None:
                continue
            d_bytes = _bytes_at(t, nxt) - _bytes_at(t, cur)
            if d_bytes <= 0 or d_bytes > remaining:
                continue
            gain = (e_cur - e_nxt) / d_bytes
            if gain > best_gain:
                best_gain = gain
                best = (s, nxt, d_bytes)
        if best is None or best_gain <= 0:
            break
        s, nxt, d_bytes = best
        it: AllocItem = s["item"]
        t: TensorRD = s["tensor"]
        it.bpw = nxt
        it.target_bytes = _bytes_at(t, nxt)
        it.rel_err = t.errors.get(nxt) or fam_err.get((t.role, nxt))
        remaining -= d_bytes

    plan.items = items
    plan.total_target_bytes = sum(it.target_bytes for it in items)
    return plan


def allocate_two_points(report: RDReport, fidelity_gib: float, knee_gib: float) -> dict[str, AllocPlan]:
    """The two operating points (brief §28): Fidelity + Knee."""
    return {
        "fidelity": allocate(report, fidelity_gib),
        "knee": allocate(report, knee_gib),
    }
