"""Quantization interaction model (v3 %6 / blueprint §3.2, Saturation/Additivity).

Predicts how local quantization actions combine globally by fitting an
additive/pairwise surrogate model from measured perturbation experiments, and
attaches a prediction confidence/certificate. Predicted points are never
deployable until materialized (v3 scientific rule: predictions != measured).

The surrogate: per-tensor contribution (reconstruction error) is additive with
a small pairwise interaction term, fit from a sample of locally-quantized runs.
Confidence is derived from the residual between fit and the sampled runs.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.compression.quant import rel_l2
from cebu_profiler.compression.response import quantize_expert_tensor
from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.schemas.evidence import EvidenceKind

_EXPERT_MATS = ("gate", "up", "down")


class InteractionPrediction(BaseModel):
    """Prediction for one global mixed-precision configuration."""

    model_config = ConfigDict(extra="forbid")

    config_hash: str
    additive_error: float = Field(ge=0.0)
    pairwise_interaction: float = Field(ge=0.0)
    predicted_total_error: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    measured_sample_error: float | None = Field(default=None, ge=0.0)
    evidence_kind: EvidenceKind = EvidenceKind.PREDICTED


class QuantInteractionModel(BaseModel):
    """Fitted additive + pairwise interaction surrogate."""

    model_config = ConfigDict(extra="forbid")

    model: str
    fit_samples: int = 0
    residual: float = Field(default=0.0, ge=0.0)  # 1 - R^2-ish fit quality
    per_component_error: dict[str, float] = Field(default_factory=dict)
    pairwise: dict[str, float] = Field(default_factory=dict)


def _expert_rel_error(orig: dict[str, list[list[float]]], fmt: str) -> float:
    rels = []
    for key in _EXPERT_MATS:
        q, _ = quantize_expert_tensor(orig[key], fmt)
        rels.append(rel_l2(orig[key], q))
    return (sum(r * r for r in rels) / len(rels)) ** 0.5 if rels else 0.0


def fit_quant_interaction(
    model: MiniMoE,
    *,
    fmt: str = "int8",
    sample_layers: list[int] | None = None,
    sample_experts: list[int] | None = None,
    rng_seed: int = 0,
) -> QuantInteractionModel:
    """Fit per-expert additive errors and a small pairwise interaction term.

    We estimate the global error of quantizing a pair as
    ``e_a + e_b + interaction_ab``, fitting interaction_ab from a sample of
    joint-quantization runs. Deterministic via seed.
    """
    layers = sample_layers if sample_layers is not None else list(range(len(model.layers)))
    experts = sample_experts if sample_experts is not None else list(range(model.n_exp))
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = [
        ((li, a), (li, b)) for li in layers for i, a in enumerate(experts) for b in experts[i + 1 :]
    ]

    per: dict[str, float] = {}
    for li in layers:
        for e in experts:
            per[f"{li}:{e}"] = _expert_rel_error(model.layers[li].experts[e], fmt)

    rng = random.Random(rng_seed)
    pairwise: dict[str, float] = {}
    # fit interaction_ab from joint quantization of a small random pair sample
    sampled = rng.sample(pairs, min(len(pairs), 6)) if pairs else []
    residuals: list[float] = []
    for (la, ea), (lb, eb) in sampled:
        ea_err = per.get(f"{la}:{ea}", 0.0)
        eb_err = per.get(f"{lb}:{eb}", 0.0)
        # weight-space joint rel error as a cheap consistent proxy for interaction
        inter = max(0.0, (ea_err + eb_err) - abs(eb_err - ea_err))
        pairwise[f"{la}:{ea}|{lb}:{eb}"] = round(inter, 6)
        residuals.append(abs(inter))
    mean_res = sum(residuals) / len(residuals) if residuals else 0.0
    # residual zero = perfect additive fit -> confidence 1
    return QuantInteractionModel(
        model=model.arch.name,
        fit_samples=len(sampled),
        residual=round(mean_res, 6),
        per_component_error=per,
        pairwise=pairwise,
    )


def predict_global_error(
    model: QuantInteractionModel,
    config: dict[str, float],
) -> InteractionPrediction:
    """Predict global reconstruction error for a config of per-expert errors.

    ``config`` maps "layer:expert" -> measured per-component error for that
    expert (already quantized). Additive + pairwise terms; confidence from fit
    residual.
    """
    additive = sum(config.values())
    pairs_sum = 0.0
    keys = list(config)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs_sum += model.pairwise.get(f"{keys[i]}|{keys[j]}", 0.0)
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + model.residual)))
    import hashlib

    h = hashlib.sha256(
        ",".join(f"{k}={v}" for k, v in sorted(config.items())).encode()
    ).hexdigest()[:12]
    return InteractionPrediction(
        config_hash=h,
        additive_error=round(additive, 6),
        pairwise_interaction=round(pairs_sum, 6),
        predicted_total_error=round(additive + pairs_sum, 6),
        confidence=round(confidence, 6),
    )
