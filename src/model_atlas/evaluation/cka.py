"""Pure centered linear CKA (centered kernel alignment) kernel.

No dependency on Torch; implemented with NumPy over matched
``[observations, features]`` activation matrices. Returns a typed ``CKA``
result and never a misleading numeric score when the inputs are degenerate
(fewer than two observations or zero variance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CKA:
    """Result of a CKA computation.

    ``valid`` is False when fewer than two observations or zero variance
    blocks alignment; in that case ``score`` is None (never a misleading
    number) and ``reason`` explains the blocker.
    """

    valid: bool
    score: float | None
    reason: str | None = None


def _center_rows(x: np.ndarray) -> np.ndarray:
    return np.asarray(x - x.mean(axis=0, keepdims=True), dtype=np.float64)


def _linear_hsic(x: np.ndarray, y: np.ndarray) -> float:
    """Linear HSIC = || (1/(n-1)) * Xc^T Yc ||_F^2 (unnormalised)."""
    n = x.shape[0]
    gram = np.asarray(_center_rows(x).T @ _center_rows(y), dtype=np.float64)
    return float(np.asarray(np.sum(gram**2), dtype=np.float64) / ((n - 1.0) ** 2))


def centered_linear_cka(
    x: np.ndarray, y: np.ndarray
) -> CKA:
    """Centered linear CKA between matched activations.

    ``x`` and ``y`` are ``[observations, features]`` arrays with equal
    observation counts. Returns ``CKA``:

    - fewer than two observations → ``valid=False``
    - zero variance in either input → ``valid=False``
    - otherwise ``score = HSIC(x, y) / sqrt(HSIC(x, x) * HSIC(y, y))``.

    A degenerate input never yields a misleading numeric score.
    """
    xa = np.asarray(x)
    ya = np.asarray(y)
    if xa.ndim != 2 or ya.ndim != 2:
        raise ValueError("CKA inputs must be 2D [observations, features]")
    if xa.shape[0] != ya.shape[0]:
        raise ValueError(
            f"observation mismatch: x has {xa.shape[0]}, y has {ya.shape[0]}"
        )
    n = xa.shape[0]
    if n < 2:
        return CKA(valid=False, score=None, reason="fewer than two observations")

    # Zero variance in either input blocks alignment.
    if float(np.var(xa)) < 1e-300 or float(np.var(ya)) < 1e-300:
        which = "x" if np.var(xa) < np.var(ya) else "y"
        return CKA(valid=False, score=None, reason=f"zero variance in {which}")

    hsic_xy = _linear_hsic(xa, ya)
    hsic_xx = _linear_hsic(xa, xa)
    hsic_yy = _linear_hsic(ya, ya)
    denom = float(np.sqrt(hsic_xx * hsic_yy))
    if denom <= 0.0 or not np.isfinite(denom):
        return CKA(valid=False, score=None, reason="degenerate HSIC denominator")
    score = hsic_xy / denom
    if not np.isfinite(score):
        return CKA(valid=False, score=None, reason="non-finite CKA score")
    return CKA(valid=True, score=float(np.clip(score, -1.0, 1.0)), reason=None)


__all__ = ["CKA", "centered_linear_cka"]
