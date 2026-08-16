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
        if not vec:
            raise ValueError(f"{name} feature dimension must be >= 1")
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


def _observation_gram(x: list[list[float]]) -> list[list[float]]:
    """Return ``X X^T`` using symmetry.

    Linear CKA is commonly written with the feature covariance
    ``X^T Y``.  For LLM activations that formulation costs
    ``O(observations * x_features * y_features)`` and becomes unusable at
    hidden widths such as 6,144.  The equivalent observation-Gram identity

        ``||X^T Y||_F^2 = <X X^T, Y Y^T>_F``

    costs ``O(observations^2 * features)``.  Held-out capture rows are much
    fewer than model features, so this is the exact—not projected—formulation
    needed by the GLM canary.
    """

    n = len(x)
    gram = [[0.0] * n for _ in range(n)]
    for i, left in enumerate(x):
        for j in range(i, n):
            value = sum(a * b for a, b in zip(left, x[j], strict=True))
            gram[i][j] = value
            gram[j][i] = value
    return gram


def _gram_hsic(x_gram: list[list[float]], y_gram: list[list[float]], n: int) -> float:
    value = sum(
        x_value * y_value
        for x_row, y_row in zip(x_gram, y_gram, strict=True)
        for x_value, y_value in zip(x_row, y_row, strict=True)
    )
    return value / ((n - 1.0) ** 2)


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
        raise ValueError(f"observation mismatch: x has {len(xa)}, y has {len(ya)}")
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

    x_gram = _observation_gram(xc)
    y_gram = _observation_gram(yc)
    hsic_xy = _gram_hsic(x_gram, y_gram, n)
    hsic_xx = _gram_hsic(x_gram, x_gram, n)
    hsic_yy = _gram_hsic(y_gram, y_gram, n)
    denom = math.sqrt(hsic_xx * hsic_yy)
    if not (math.isfinite(denom) and denom > 0.0):
        return CKA(valid=False, score=None, reason="degenerate HSIC denominator")
    score = hsic_xy / denom
    if not math.isfinite(score):
        return CKA(valid=False, score=None, reason="non-finite CKA score")
    return CKA(valid=True, score=max(-1.0, min(1.0, score)), reason=None)


__all__ = ["CKA", "centered_linear_cka"]
