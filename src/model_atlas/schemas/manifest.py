"""Versioned compression manifest + validator (blueprint §11, §12).

Atlas's primary output: a deterministic, self-describing plan of retained
channels / target widths per layer+expert, with confidence, protected reasons,
and a per-tensor quantization recommendation. Atlas never touches weights; the
structural executor consumes this manifest.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_weight_gib: float | None = Field(default=None, ge=0.0)
    deployment: str = "2x-dgx-spark-sm121"


class QuantRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = "exl3"
    bpw: float = 3.25


class ExpertScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenp: float | None = None
    taylor: float | None = None
    causal: float | None = None
    stability: float | None = None


class ExpertPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_width: int
    target_width: int
    keep_channels: list[int]  # retained channel indices (sorted, stable)
    reorder_permutation: list[int] = Field(default_factory=list)  # [] = keep order
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    scores: ExpertScores = Field(default_factory=ExpertScores)
    protected_reasons: list[str] = Field(default_factory=list)
    quant_recommendation: QuantRecommendation = Field(default_factory=QuantRecommendation)


class LayerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experts: dict[str, ExpertPlan] = Field(default_factory=dict)


class CompressionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: int = 1
    model: str
    source_checkpoint: str
    source_hash: str | None = None
    atlas_version: str = "0.0.0"
    calibration_suite: str = "glm52-compression-v1"
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    allowed_widths: list[int]
    layers: dict[str, LayerPlan] = Field(default_factory=dict)
    generated_by: str = "model-atlas"


class ManifestValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str] = Field(default_factory=list)


def validate_manifest(manifest: CompressionManifest) -> ManifestValidation:
    """Validate a manifest against the versioned contract (blueprint §11)."""
    errors: list[str] = []
    allowed = set(manifest.allowed_widths)
    cardinal = True
    for layer_idx, layer in manifest.layers.items():
        for exp_str, plan in layer.experts.items():
            if plan.target_width != len(plan.keep_channels):
                errors.append(
                    f"layer {layer_idx} expert {exp_str}: target_width "
                    f"{plan.target_width} != len(keep_channels) {len(plan.keep_channels)}"
                )
                cardinal = False
            if plan.original_width < plan.target_width:
                errors.append(
                    f"layer {layer_idx} expert {exp_str}: target_width exceeds original_width"
                )
            if plan.target_width not in allowed:
                errors.append(
                    f"layer {layer_idx} expert {exp_str}: target_width {plan.target_width} "
                    f"not in allowed_widths {sorted(allowed)}"
                )
            if len(plan.keep_channels) != len(set(plan.keep_channels)):
                errors.append(f"layer {layer_idx} expert {exp_str}: duplicate keep_channels")
            if any(c < 0 or c >= plan.original_width for c in plan.keep_channels):
                errors.append(
                    f"layer {layer_idx} expert {exp_str}: keep_channel out of range"
                )
            if plan.reorder_permutation and sorted(plan.reorder_permutation) != list(
                range(len(plan.reorder_permutation))
            ):
                errors.append(
                    f"layer {layer_idx} expert {exp_str}: invalid reorder_permutation"
                )
    _ = cardinal
    return ManifestValidation(ok=not errors, errors=errors)
