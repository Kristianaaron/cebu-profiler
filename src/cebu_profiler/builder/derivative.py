"""Derivative checkpoint builder (v2 §26).

Turns a CandidatePlan + source MiniMoE into a derivative: dropped experts are
removed and the retained ones renumbered to contiguous slots, the router is
remapped to those slots, optionally-lowered precision is written, and source
identities are preserved in a provenance map. Validates tensor coverage,
routing (derivative only routes among retained experts), and a miniature
inference run. Registering produces a derivative_checkpoint ModelAsset.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from cebu_profiler.compression.quant import uniform_int_quant
from cebu_profiler.planning.maps import CandidatePlan
from cebu_profiler.profiler.runtime import LayerWeights, MiniMoE, forward
from cebu_profiler.schemas.model_asset import AssetType, ModelAsset

# precisions we actually rewrite to integer tiers; others keep source bytes.
_APPLY_BITS = {"int4": 4, "int8": 8, "nvfp4": 4, "fp8": 8}


@dataclass
class RenumberedExpert:
    layer: int
    derivative_slot: int
    source_expert_id: int


@dataclass
class DerivativeValidation:
    tensor_coverage: bool
    routing_valid: bool
    inference_ok: bool
    mapping_exact: bool
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.tensor_coverage and self.routing_valid and self.inference_ok and self.mapping_exact
        )


@dataclass
class DerivativeResult:
    model: MiniMoE
    plan: CandidatePlan
    identity_map: list[RenumberedExpert]
    validation: DerivativeValidation


def _precision_bits(precision: str) -> int | None:
    return _APPLY_BITS.get(precision)


def _rewrite_expert(
    src_weights: dict[str, list[list[float]]], precision: str
) -> dict[str, list[list[float]]]:
    bits = _precision_bits(precision)
    if bits is None:
        return {k: [list(r) for r in v] for k, v in src_weights.items()}
    out: dict[str, list[list[float]]] = {}
    for k, v in src_weights.items():
        q, _ = uniform_int_quant(v, bits)
        out[k] = q
    return out


def build_derivative(model: MiniMoE, plan: CandidatePlan) -> DerivativeResult:
    """Construct a derivative MiniMoE from a plan (renumber + remap + precision)."""
    arch = model.arch
    n_layers = arch.num_text_layers
    kept_per_layer: list[list[int]] = []
    identity: list[RenumberedExpert] = []
    new_layers: list[LayerWeights] = []

    kept0 = plan.keep.kept(0) if n_layers else []
    for layer in range(n_layers):
        kept = plan.keep.kept(layer)
        kept_per_layer.append(kept)
        src_weights = model.layers[layer].experts
        src_router = model.layers[layer].router
        prec_by_src = {
            e.source_expert_id: e.precision
            for e in plan.precision.entries
            if e.layer_index == layer
        }
        new_experts: list[dict[str, list[list[float]]]] = []
        new_router: list[list[float]] = []
        for slot, src in enumerate(kept):
            new_experts.append(_rewrite_expert(src_weights[src], prec_by_src.get(src, "")))
            new_router.append(list(src_router[src]))
            identity.append(
                RenumberedExpert(layer=layer, derivative_slot=slot, source_expert_id=src)
            )
        new_layers.append(
            LayerWeights(
                ln_w=list(model.layers[layer].ln_w),
                router=new_router,
                experts=new_experts,
            )
        )

    derivative = MiniMoE(
        arch=arch,
        hidden=model.hidden,
        n_exp=len(kept0),
        mid=model.mid,
        embed=copy.deepcopy(model.embed),
        lm_head=copy.deepcopy(model.lm_head),
        layers=new_layers,
    )
    validation = validate_derivative(derivative, plan, identity, model)
    return DerivativeResult(
        model=derivative, plan=plan, identity_map=identity, validation=validation
    )


def validate_derivative(
    derivative: MiniMoE,
    plan: CandidatePlan,
    identity: list[RenumberedExpert],
    source: MiniMoE,
) -> DerivativeValidation:
    """Structural coverage, routing validity, identity exactness, mini inference."""
    issues: list[str] = []
    n_layers = derivative.arch.num_text_layers

    # structural coverage: every retained expert present (no missing/unclassified)
    coverage = True
    for layer in range(n_layers):
        n_exp_now = len(derivative.layers[layer].experts)
        if n_exp_now != len(plan.keep.kept(layer)):
            coverage = False
            issues.append(f"layer {layer}: expert count mismatch")

    # routing validity: forward routes only within [0, n_exp)
    routing_valid = True
    try:
        result = forward(derivative, [3, 7, 11], top_k=2)
        for trace in result.traces:
            for ids in trace.topk_ids:
                if any(i < 0 or i >= derivative.n_exp for i in ids):
                    routing_valid = False
                    issues.append("route index out of range")
                    break
        inference_ok = len(result.logits) == derivative.arch.vocabulary_size and all(
            v == v
            for v in result.logits  # finite (NaN check)
        )
        if not inference_ok:
            issues.append("inference produced non-finite/incomplete logits")
    except Exception as exc:  # noqa: BLE001
        routing_valid = False
        inference_ok = False
        issues.append(f"inference raised: {exc}")

    # exact renumbering: slot -> source mapping is a bijection onto the kept set
    mapping_exact = True
    for layer in range(n_layers):
        kept = sorted(plan.keep.kept(layer))
        slots = [r for r in identity if r.layer == layer]
        if [r.source_expert_id for r in slots] != kept:
            mapping_exact = False
            issues.append(f"layer {layer}: identity mapping does not match kept set")

    return DerivativeValidation(
        tensor_coverage=coverage,
        routing_valid=routing_valid,
        inference_ok=inference_ok,
        mapping_exact=mapping_exact,
        issues=issues,
    )


def register_derivative(
    result: DerivativeResult,
    *,
    display_name: str,
    source_asset: ModelAsset,
    model_family: str = "k3-mini",
    source_experiment_id: str | None = None,
    derivative_path: str | None = None,
) -> ModelAsset:
    """Register the built derivative as a derivative_checkpoint ModelAsset."""
    path = derivative_path or f"/models/derivatives/{result.plan.name}"
    return ModelAsset(
        model_asset_id=f"deriv-{result.plan.name}-{len(identity_slots(result))}",
        display_name=display_name,
        asset_type=AssetType.DERIVATIVE_CHECKPOINT,
        model_family=model_family,
        architecture=result.model.arch.name,
        checkpoint_path=path,
        endpoint=None,
        stored_size_bytes=int(result.plan.stored_bytes),
        estimated_resident_bytes=int(result.plan.resident_bytes_a + result.plan.resident_bytes_b),
        parent_model_id=source_asset.model_asset_id,
        source_experiment_id=source_experiment_id,
        metadata={
            "plan": result.plan.name,
            "validated": result.validation.ok,
            "kept_per_layer": result.plan.kept_per_layer,
            "identity_source_slots": {
                f"l{r.layer}e{r.derivative_slot}": r.source_expert_id for r in result.identity_map
            },
        },
    )


def identity_slots(result: DerivativeResult) -> list[RenumberedExpert]:
    return result.identity_map
