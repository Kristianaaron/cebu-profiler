"""Spectral quality analyzer (v3 %2 / blueprint §3.2).

AlphaQ-style calibration-independent signal: singular-value energy curves,
effective rank, heavy-tail statistics, and spectral uniqueness per expert.
Explicitly never the sole ground truth — it complements corpus evidence, never
replaces it (v3 "calibration-independent signal, never as sole ground truth").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.schemas.evidence import EvidenceKind

EPS = 1e-12


def _flatten(rows: list[list[float]]) -> list[float]:
    return [v for r in rows for v in r]


def _gram_energy(rows: list[list[float]]) -> list[float]:
    """Empirical singular-value-ish energy spectrum via the row Gram matrix.

    Returns sorted descending approximate singular values (power iteration over
    the Gram matrix for a small model is deterministic and closed-form enough):
    we estimate the spectrum of ``rows @ rows^T`` eigen decomposition with power
    iteration per singular mode (largest first with deflation).
    """
    n, m = len(rows), len(rows[0]) if rows else 0
    if n == 0 or m == 0:
        return []
    import random

    rng = random.Random(0)
    v = [rng.gauss(0.0, 1.0) for _ in range(n)]
    vn = (sum(x * x for x in v) ** 0.5) or 1.0
    v = [x / vn for x in v]
    g = [[sum(rows[i][k] * rows[j][k] for k in range(m)) for j in range(n)] for i in range(n)]

    vals: list[float] = []
    for _ in range(n):
        w = [0.0] * n
        for i in range(n):
            w[i] = sum(g[i][j] * v[j] for j in range(n))
        lam = sum(v[i] * w[i] for i in range(n))
        if lam <= EPS:
            break
        vals.append(lam**0.5)
        # deflate rank-1
        for i in range(n):
            for j in range(n):
                g[i][j] -= lam * v[i] * v[j]
        w2 = [0.0] * n
        for i in range(n):
            w2[i] = sum(g[i][j] * v[j] for j in range(n))
        nrm = (sum(x * x for x in w2) ** 0.5) or 1.0
        for i in range(n):
            v[i] = w2[i] / nrm
    return vals


class SpectralProfile(BaseModel):
    """One expert/tensor spectral quality result."""

    model_config = ConfigDict(extra="forbid")

    layer: int
    expert: int
    tensor: str  # gate | up | down
    effective_rank: float = Field(ge=0.0)
    energy_ratio_top: float = Field(ge=0.0, le=1.0)  # share of energy in top modes
    heavy_tail: float = Field(ge=0.0)  # spectral exponent proxy (clamped)
    spectral_uniqueness: float = Field(ge=0.0, le=1.0)
    evidence_kind: EvidenceKind = EvidenceKind.ESTIMATED


class SpectralAnalysis(BaseModel):
    """Versioned whole-model spectral evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    model: str
    rows: list[SpectralProfile] = Field(default_factory=list)
    note: str = "calibration-independent signal: never the sole ground truth"


def _effective_rank(vals: list[float]) -> float:
    total = sum(vals)
    if total <= 0:
        return 0.0
    norm = [v / total for v in vals]
    # participation ratio (inverse Simpson) as a continuous rank proxy
    return (sum(norm) ** 2) / (sum(p * p for p in norm)) if any(p > 0 for p in norm) else 0.0


def _energy_ratio_top(vals: list[float], top: int = 3) -> float:
    total = sum(vals)
    return (sum(vals[:top]) / total) if total > 0 else 0.0


def _heavy_tail(vals: list[float]) -> float:
    """Spectral exponent proxy: log-log slope of the tail (negative = heavy tail).

    We fit the least-squares slope of ``log(s) ~ b + a*log(rank)`` on the
    largest modes; return ``-a`` (higher = fatter tail / slower decay).
    """
    import math

    if len(vals) < 2:
        return 0.0
    xs, ys = [], []
    for i, s in enumerate(vals):
        if s <= 0:
            continue
        xs.append(math.log(i + 1))
        ys.append(math.log(s))
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den > 0 else 0.0
    return -slope


def analyze_spectral(
    model: MiniMoE,
    *,
    expert_mats: tuple[str, ...] = ("gate", "up", "down"),
    top_energy_modes: int = 3,
    heavy_tail_modes: int = 20,
) -> SpectralAnalysis:
    """Spectral evidence for every expert tensor (alpha-q style)."""
    rows: list[SpectralProfile] = []
    # cross-expert mean uniqueness: compare each expert's top spectrum to the
    # layer's mean spectrum (correlation of normalized spectrum)
    for li, lw in enumerate(model.layers):
        for ei, exp in enumerate(lw.experts):
            for tensor in expert_mats:
                vals = _gram_energy(exp[tensor])
                eff = _effective_rank(vals)
                ratio = _energy_ratio_top(vals, top_energy_modes)
                tail = _heavy_tail(vals[:heavy_tail_modes])
                # uniqueness: share of this expert's energy not shared with the
                # average of other experts' spectra (cosine of normalized spectra)
                uniqueness = 1.0
                norm_s = _sort_norm(vals)
                if norm_s:
                    others: list[list[float]] = []
                    for e2, exp2 in enumerate(lw.experts):
                        if e2 == ei:
                            continue
                        v2 = sorted(_gram_energy(exp2[tensor]), reverse=True)
                        if v2:
                            padded = v2[: len(norm_s)] + [0.0] * max(0, len(norm_s) - len(v2))
                            others.append(_sort_norm(padded))
                    if others:
                        avg = [sum(o[i] for o in others) / len(others) for i in range(len(norm_s))]
                        cos = sum(a * b for a, b in zip(norm_s, avg, strict=True))
                        uniqueness = max(0.0, min(1.0, 1.0 - cos))
                rows.append(
                    SpectralProfile(
                        layer=li,
                        expert=ei,
                        tensor=tensor,
                        effective_rank=round(eff, 6),
                        energy_ratio_top=round(ratio, 6),
                        heavy_tail=round(max(0.0, tail), 6),
                        spectral_uniqueness=round(uniqueness, 6),
                    )
                )
    return SpectralAnalysis(model=model.arch.name, rows=rows)


def _sort_norm(vals: list[float]) -> list[float]:
    s = sorted(vals, reverse=True)
    total = sum(s)
    return [v / total for v in s] if total > 0 else s
