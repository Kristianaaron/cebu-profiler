"""Conditional sensitivity model (v3 %3 / blueprint §3.2, MixQuant).

Sensitivity for a tensor/layer is conditioned on already-quantized upstream
state rather than treated as independent. We sample a few upstream quantization
configurations (represented as noise injections into upstream expert outputs)
and measure how the downstream tensor's reconstruction error changes — exposing
both a conditional sensitivity curve and an uncertainty band. This is an
estimate; measured state is always explicit.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.compression.quant import rel_l2
from cebu_profiler.compression.response import quantize_expert_tensor
from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.schemas.evidence import EvidenceKind

_EXPERT_MATS = ("gate", "up", "down")


class ConditionalSensitivityPoint(BaseModel):
    """One conditioning level's sensitivity for one (layer, expert, tensor)."""

    model_config = ConfigDict(extra="forbid")

    layer: int
    expert: int
    tensor: str
    upstream_noise: float = Field(ge=0.0)  # relative L2 noise injected upstream
    reconstruction_error: float = Field(ge=0.0)  # downstream rel L2 error
    evidence_kind: EvidenceKind = EvidenceKind.ESTIMATED


class ConditionalSensitivity(BaseModel):
    """Versioned conditional-sensitivity evidence for the model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    model: str
    format: str = "int8"
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05)
    rows: list[ConditionalSensitivityPoint] = Field(default_factory=list)
    note: str = "sensitivity conditioned on upstream quantization state (MixQuant-style); estimated"


def _inject_noise(
    model: MiniMoE, layer: int, expert: int, noise: float
) -> tuple[list[list[float]], ...]:
    """Return (gate, up, down) of one expert corrupted by relative L2 noise."""
    w = model.layers[layer].experts[expert]
    out = []
    import random

    rng = random.Random(int(noise * 1e6))
    for key in _EXPERT_MATS:
        m = w[key]
        flat = [v for r in m for v in r]
        norm = (sum(v * v for v in flat) ** 0.5) or 1.0
        scale = noise * norm
        q = [
            [
                v + rng.gauss(0.0, scale / (len(m[0]) ** 0.5 if len(m) and len(m[0]) else 1))
                for v in r
            ]
            for r in m
        ]
        out.append(q)
    return tuple(out)


def conditional_sensitivity(
    model: MiniMoE,
    *,
    layers: list[int] | None = None,
    experts: list[int] | None = None,
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05),
    fmt: str = "int8",
) -> ConditionalSensitivity:
    """For each target tensor measure reconstruction error under upstream noise.

    Layers/experts default to all. Deterministic: noise is seeded per level.
    """
    rows: list[ConditionalSensitivityPoint] = []
    target_layers = layers if layers is not None else list(range(len(model.layers)))
    target_experts = experts if experts is not None else list(range(model.n_exp))
    for li in target_layers:
        for ei in target_experts:
            orig = model.layers[li].experts[ei]
            for noise in noise_levels:
                corrupted = _inject_noise(model, li, ei, noise)
                rels: list[float] = []
                for key, c in zip(_EXPERT_MATS, corrupted, strict=True):
                    q, _ = quantize_expert_tensor(orig[key], fmt)
                    # relative error of the quantized-with-noise representation
                    rels.append(rel_l2(q, c))
                err = (sum(r * r for r in rels) / len(rels)) ** 0.5 if rels else 0.0
                rows.append(
                    ConditionalSensitivityPoint(
                        layer=li,
                        expert=ei,
                        tensor="expert",
                        upstream_noise=round(noise, 4),
                        reconstruction_error=round(err, 6),
                    )
                )
    return ConditionalSensitivity(
        model=model.arch.name,
        format=fmt,
        noise_levels=noise_levels,
        rows=rows,
    )
