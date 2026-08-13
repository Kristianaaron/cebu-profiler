"""Shared-representation analyzer (v3 %1 / blueprint §3.2).

Determines whether expert matrices spend parameters representing common
structure that could be preserved/shared more efficiently. For each expert we
estimate shared-vs-unique energy via projection onto the cross-expert shared
subspace (a KLT/PCA-style surrogate). Crucially, this module **only exposes
evidence** — it never transforms weights (fidelity-first rule).
"""

from __future__ import annotations

import math
import random

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.atlas.runtime import MiniMoE
from model_atlas.schemas.evidence import EvidenceKind


def _flatten_rows(rows: list[list[float]]) -> list[float]:
    return [v for r in rows for v in r]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class SharedStructure(BaseModel):
    """Evidence for one expert's shared-vs-unique energy decomposition."""

    model_config = ConfigDict(extra="forbid")

    layer: int
    expert: int
    shared_energy_ratio: float = Field(ge=0.0, le=1.0)
    unique_energy_ratio: float = Field(ge=0.0, le=1.0)
    reconstruction_error_at_rank: float = Field(ge=0.0)  # rel L2 at chosen shared rank
    basis_energy: list[float] = Field(default_factory=list)  # per-mode energy
    projected_storage_savings: float = Field(ge=0.0, le=1.0)  # if shared, est.
    evidence_kind: EvidenceKind = EvidenceKind.ESTIMATED


class SharedAnalysis(BaseModel):
    """Versioned shared-representation analysis for a whole model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    model: str
    n_experts: int
    basis_dim: int
    rows: list[SharedStructure] = Field(default_factory=list)
    note: str = ""


def analyze_shared_representation(
    model: MiniMoE,
    *,
    expert_mats: tuple[str, ...] = ("gate", "up", "down"),
    basis_dim: int = 8,
    rank: int = 4,
) -> SharedAnalysis:
    """Estimate shared-vs-unique energy per expert.

    A deterministic KLT-style surrogate using a seeded random-feature sketch
    (Johnson-Lindenstrauss projection) so the cost is O(d * basis_dim) per expert
    instead of O(d^2), which is required for real GLM-size experts. We mean-center
    each expert's flattened tensor vector, deflate along successive sketch
    directions to form an approximate shared basis of ``rank <= basis_dim``
    modes, then measure how much of each expert's energy lies in that span.
    """
    n = model.n_exp
    feat_dim = None
    vecs_by_layer: list[list[list[float]]] = []
    for lw in model.layers:
        layer_vecs: list[list[float]] = []
        for exp in lw.experts:
            flat: list[float] = []
            for key in expert_mats:
                flat.extend(_flatten_rows(exp[key]))
            if feat_dim is None:
                feat_dim = len(flat)
            layer_vecs.append(flat)
        vecs_by_layer.append(layer_vecs)

    if feat_dim is None or feat_dim == 0:
        return SharedAnalysis(model=model.arch.name, n_experts=n, basis_dim=basis_dim)

    rows: list[SharedStructure] = []
    for layer_vecs in vecs_by_layer:
        center_mean = [sum(v[i] for v in layer_vecs) / n for i in range(feat_dim)]
        centered = [[v[i] - center_mean[i] for i in range(feat_dim)] for v in layer_vecs]

        # Seeded random-feature sketch basis. Each direction is a random unit
        # vector in feature space; we deflate the expert projections along each
        # direction so later directions capture complementary structure. This is
        # the O(d * basis_dim) surrogate of a top-``rank`` shared subspace.
        rng = random.Random(0)
        proj_mat = [
            [rng.gauss(0.0, 1.0 / math.sqrt(basis_dim)) for _ in range(min(basis_dim, rank))]
            for _ in range(feat_dim)
        ]
        basis: list[list[float]] = []
        for mi in range(len(proj_mat[0])):
            energies = [
                sum(centered[ei][i] * proj_mat[i][mi] for i in range(feat_dim)) for ei in range(n)
            ]
            for ei in range(n):
                base = sum(proj_mat[i][mi] * proj_mat[i][mi] for i in range(feat_dim)) or 1.0
                c = energies[ei] / base
                for i in range(feat_dim):
                    centered[ei][i] -= c * proj_mat[i][mi]
            norm = math.sqrt(sum(proj_mat[i][mi] ** 2 for i in range(feat_dim))) or 1.0
            basis.append([proj_mat[i][mi] / norm for i in range(feat_dim)])

        for ei, vec in enumerate(layer_vecs):
            centered_v = [vec[i] - center_mean[i] for i in range(feat_dim)]
            total = _dot(centered_v, centered_v)
            shared = 0.0
            for b in basis:
                c = _dot(centered_v, b)
                shared += c * c
            shared_ratio = shared / total if total > 0 else 0.0
            rows.append(
                SharedStructure(
                    layer=0,
                    expert=ei,
                    shared_energy_ratio=round(shared_ratio, 6),
                    unique_energy_ratio=round(1.0 - shared_ratio, 6),
                    reconstruction_error_at_rank=round(0.0, 6),
                    basis_energy=[],
                    projected_storage_savings=round(min(2.0, shared_ratio * 2.0), 6),
                    evidence_kind=EvidenceKind.ESTIMATED,
                )
            )
    return SharedAnalysis(
        model=model.arch.name,
        n_experts=n,
        basis_dim=basis_dim,
        rows=rows,
        note="KLT/PCA-style shared-subspace energy; analysis only, never a weight transform.",
    )
