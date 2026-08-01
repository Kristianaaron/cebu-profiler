"""Calibration/held-out leakage detection + promotion gate (v2 §7, §26).

Detects exact token-sequence overlap and near-duplicate (high token-set
Jaccard) between the calibration corpus (used to build/plan) and the evaluation
corpus. Final promotion reports are BLOCKED by default when leakage exists; an
explicit development-only override must be recorded to bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_atlas.atlas.reap import CalibrationSample


@dataclass
class LeakageResult:
    exact_overlap: list[int] = field(default_factory=list)  # held-out sample ids reused
    near_duplicates: list[tuple[int, int]] = field(default_factory=list)  # (calib, heldout)

    @property
    def detected(self) -> bool:
        return bool(self.exact_overlap or self.near_duplicates)


def _tokenset(s: CalibrationSample) -> set[int]:
    return set(s.tokens)


def detect_leakage(
    calibration: list[CalibrationSample],
    heldout: list[CalibrationSample],
    *,
    jaccard_threshold: float = 0.8,
) -> LeakageResult:
    """Find held-out samples that are exact or near-duplicates of calibration."""
    seen_exact: dict[tuple[int, ...], int] = {}
    for i, s in enumerate(calibration):
        seen_exact.setdefault(tuple(s.tokens), i)

    exact: set[int] = set()
    near: list[tuple[int, int]] = []
    cal_tokensets = [(i, _tokenset(s), len(s.tokens)) for i, s in enumerate(calibration)]

    for h, s in enumerate(heldout):
        if tuple(s.tokens) in seen_exact:
            exact.add(h)
        hs = _tokenset(s)
        # near duplicates: high Jaccard with some calibration sample
        for ci, cs, _n in cal_tokensets:
            inter = len(hs & cs)
            union = len(hs | cs)
            if union and inter / union >= jaccard_threshold:
                near.append((ci, h))
                break
    return LeakageResult(exact_overlap=sorted(exact), near_duplicates=near)


def promote_allowed(
    leakage: LeakageResult,
    *,
    allow_development_override: bool = False,
) -> bool:
    """A promotion report may be produced only when no leakage exists, or an
    EXPLICIT development-only override was recorded. Never silent."""
    if not leakage.detected:
        return True
    return allow_development_override
