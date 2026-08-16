"""Pure centered linear CKA (centered kernel alignment) kernel.

No dependency on Torch or NumPy; implemented with stdlib math over matched
``[observations, features]`` nested activation matrices. Returns a typed
``CKA`` result and never a misleading numeric score when the inputs are
degenerate (fewer than two observations or zero variance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_ZERO_VAR_TOL = 1e-300


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


def _to_matrix(name: str, x: object) -> list[list[float]]:
    """Validate x is a rectangular finite [observations][features] matrix."""
    if not isinstance(x, (list, tuple)):
        raise ValueError(f"{name} must be a nested sequence [observations, features]")
    outer = list(x)
    if not outer:
        raise ValueError(f"{name} must be a nonempty [observations, features] matrix")
    nfeat: int | None = None
    out: list[list[float]] = []
    for ri, row in enumerate(outer):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{name} row {ri} must be a sequence")
        vec = list(row)
        if nfeat is None:
            nfeat = len(vec)
        elif len(vec) != nfeat:
            raise ValueError(f"{name} must be rectangular (row {ri} width differs)")
        flat: list[float] = []
        for v in vec:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be numeric at row {ri}")
            fv = float(v)
            if not math.isfinite(fv):
                raise ValueError(f"{name} values must be finite")
            flat.append(fv)
        out.append(flat)
    return out


def _mean_cols(x: list[list[float]]) -> list[float]:
    n = len(x)
    nfeat = len(x[0])
    return [sum(row[j] for row in x) / n for j in range(nfeat)]


def _center_rows(x: list[list[float]]) -> list[list[float]]:
    means = _mean_cols(x)
    return [[v - m for v, m in zip(row, means, strict=True)] for row in x]


def _linear_hsic(x: list[list[float]], y: list[list[float]]) -> float:
    """Linear HSIC = || (1/(n-1)) * Xc^T Yc ||_F^2 (unnormalised)."""
    n = len(x)
    nfeat = len(x[0])
    gram = 0.0
    for i in range(nfeat):
        for j in range(nfeat):
            s = sum(x[k][i] * y[k][j] for k in range(n))
            gram += s * s
    return gram / ((n - 1.0) ** 2)


def centered_linear_cka(x: object, y: object) -> CKA:
    """Centered linear CKA between matched activations.

    ``x`` and ``y`` are ``[observations, features]`` nested matrices with
    equal observation counts. Returns ``CKA``:

    - fewer than two observations → ``valid=False``
    - zero variance in either input → ``valid=False``
    - otherwise ``score = HSIC(x, y) / sqrt(HSIC(x, x) * HSIC(y, y))``.

    A degenerate input never yields a misleading numeric score.
    """
    xa = _to_matrix("x", x)
    ya = _to_matrix("y", y)
    if len(xa) != len(ya):
        raise ValueError(
            f"observation mismatch: x has {len(xa)}, y has {len(ya)}"
        )
    n = len(xa)
    if n < 2:
        return CKA(valid=False, score=None, reason="fewer than two observations")

    xc = _center_rows(xa)
    yc = _center_rows(ya)
    var_x = sum(v * v for row in xc for v in row) / (n * len(xc[0]))
    var_y = sum(v * v for row in yc for v in row) / (n * len(yc[0]))
    if var_x < _ZERO_VAR_TOL or var_y < _ZERO_VAR_TOL:
        which = "x" if var_x <= var_y else "y"
        return CKA(valid=False, score=None, reason=f"zero variance in {which}")

    hsic_xy = _linear_hsic(xc, yc)
    hsic_xx = _linear_hsic(xc, xc)
    hsic_yy = _linear_hsic(yc, yc)
    denom = math.sqrt(hsic_xx * hsic_yy)
    if not (math.isfinite(denom) and denom > 0.0):
        return CKA(valid=False, score=None, reason="degenerate HSIC denominator")
    score = hsic_xy / denom
    if not math.isfinite(score):
        return CKA(valid=False, score=None, reason="non-finite CKA score")
    return CKA(valid=True, score=max(-1.0, min(1.0, score)), reason=None)


__all__ = ["CKA", "centered_linear_cka"]
