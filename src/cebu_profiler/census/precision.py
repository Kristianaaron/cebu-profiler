"""Per-tensor / per-role precision census from a checkpoint manifest.

Pure, header-only measurement (blueprint §"final per-tensor format choice" and
Priority 6/7): for every tensor we report the **achieved** bits-per-weight the
checkpoint actually stores it at, grouped by role. This shows where precision
headroom actually lives (e.g. BF16 attention vs ~4-bit routed experts) so a
mixed-precision / EXL3 allocation decision is grounded in measured bytes — never
guessed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from cebu_profiler.checkpoint.classifier import classify_tensor
from cebu_profiler.checkpoint.source_manifest import CheckpointManifest
from cebu_profiler.schemas.architecture import TensorRole


@dataclass
class RolePrecision:
    role: str
    tensor_count: int
    stored_bytes: int
    numel: int
    achieved_bpw: float | None  # stored bits per weight for this role
    dominant_dtype: str | None  # most common declared dtype

    @property
    def headroom(self) -> str:
        """Plain tier: how much room this role has to shrink."""
        if self.achieved_bpw is None:
            return "unknown"
        if self.achieved_bpw <= 4.5:
            return "already ~4-bit (little headroom)"
        if self.achieved_bpw <= 8.5:
            return "mid (BF16/NVFP4-scale)"
        return "heavy — best target for precision reduction / EXL3"


@dataclass
class PrecisionCensus:
    total_stored_bytes: int
    total_numel: int
    overall_bpw: float | None
    by_role: list[RolePrecision]

    def role(self, role: TensorRole) -> RolePrecision | None:
        for r in self.by_role:
            if r.role == role.value:
                return r
        return None


def _bpw(bytes_: int, numel: int) -> float | None:
    return (bytes_ * 8) / numel if numel > 0 else None


def census_precision(manifest: CheckpointManifest) -> PrecisionCensus:
    agg: dict[str, dict[str, int]] = {}
    dtypes: dict[str, Counter[str]] = {}
    total_bytes = 0
    total_numel = 0
    for t in manifest.tensors:
        total_bytes += t.byte_size
        total_numel += t.numel
        c = classify_tensor(t.name)
        role = c.role.value if c.role else "unclassified"
        a = agg.setdefault(role, {"bytes": 0, "numel": 0, "count": 0})
        a["bytes"] += t.byte_size
        a["numel"] += t.numel
        a["count"] += 1
        dtypes.setdefault(role, Counter())[t.dtype] += 1

    by_role: list[RolePrecision] = []
    for role, a in sorted(agg.items()):
        dom = dtypes[role].most_common(1)[0][0] if dtypes[role] else None
        by_role.append(
            RolePrecision(
                role=role,
                tensor_count=a["count"],
                stored_bytes=a["bytes"],
                numel=a["numel"],
                achieved_bpw=_bpw(a["bytes"], a["numel"]),
                dominant_dtype=dom,
            )
        )

    return PrecisionCensus(
        total_stored_bytes=total_bytes,
        total_numel=total_numel,
        overall_bpw=_bpw(total_bytes, total_numel),
        by_role=by_role,
    )
