"""Structural executor (blueprint §12).

Consumes a compression manifest and produces a *pruned clone* of the MiniMoE by
applying the coupled structural surgery invariant: for each retained FFN channel
`j`, keep `gate[j,:]`, `up[j,:]`, and `down[:,j]` together, applying the exact
same order/permutation to all three (blueprint §12.1). Every routed expert is
retained — only its width changes; routing topology and expert IDs are unchanged.

Also provides the six executor tests from blueprint §12.2 (dry-run validation,
coupled slicing, permutation equivalence, reload, manifest replay, protected
channels) — see the matching test module.
"""

from __future__ import annotations

import copy

from cebu_profiler.profiler.runtime import LayerWeights, MiniMoE
from cebu_profiler.schemas.manifest import (
    CompressionManifest,
    ManifestValidation,
    validate_manifest,
)


def _sliced(
    weights: dict[str, list[list[float]]], order: list[int]
) -> dict[str, list[list[float]]]:
    """Coupled slice: gate/up rows and down columns share `order`."""
    gate = weights["gate"]
    up = weights["up"]
    down = weights["down"]
    down_cols = [[row[c] for c in order] for row in down]
    return {
        "gate": [gate[c] for c in order],
        "up": [up[c] for c in order],
        "down": down_cols,
    }


def build_clone(
    model: MiniMoE,
    orders: dict[tuple[int, int], list[int]],
    default_width: int | None = None,
) -> MiniMoE:
    """Clone `model` pruning each routed expert to the given channel `orders`.

    `default_width` (when set) keeps the first N channels of experts without an
    explicit order, used to produce a uniform-width control clone.
    """
    new_layers: list[LayerWeights] = []
    for layer, layer_w in enumerate(model.layers):
        experts: list[dict[str, list[list[float]]]] = []
        for e, exp in enumerate(layer_w.experts):
            order = orders.get((layer, e))
            if order is None and default_width is not None:
                order = list(range(default_width))
            experts.append(_sliced(exp, order) if order is not None else copy.deepcopy(exp))
        new_layers.append(LayerWeights(ln_w=layer_w.ln_w, router=layer_w.router, experts=experts))
    return MiniMoE(
        arch=model.arch,
        hidden=model.hidden,
        n_exp=model.n_exp,
        mid=model.mid,
        embed=model.embed,
        lm_head=model.lm_head,
        layers=new_layers,
    )


def orders_from_manifest(manifest: CompressionManifest) -> dict[tuple[int, int], list[int]]:
    """Extract the kept-channel order per expert from a manifest."""
    orders: dict[tuple[int, int], list[int]] = {}
    for layer_idx, layer in manifest.layers.items():
        for exp_str, plan in layer.experts.items():
            orders[(int(layer_idx), int(exp_str))] = list(plan.keep_channels)
    return orders


def apply_manifest(model: MiniMoE, manifest: CompressionManifest) -> MiniMoE:
    """Prune `model` according to a manifest (coupled, topology-preserving)."""
    return build_clone(model, orders_from_manifest(manifest))


def dry_run(model: MiniMoE, manifest: CompressionManifest) -> ManifestValidation:
    """Validator-mode: verify a manifest is executable against `model` shapes."""
    errors: list[str] = []
    if len(model.layers) != len(manifest.layers):
        errors.append(
            f"layer count mismatch: model {len(model.layers)} vs manifest {len(manifest.layers)}"
        )
    for layer, layer_w in enumerate(model.layers):
        lp = manifest.layers.get(str(layer))
        if lp is None:
            errors.append(f"layer {layer} missing from manifest")
            continue
        for e, _exp in enumerate(layer_w.experts):
            plan = lp.experts.get(str(e))
            if plan is None:
                errors.append(f"layer {layer} expert {e} missing from manifest")
                continue
            if plan.original_width != model.mid:
                errors.append(
                    f"layer {layer} expert {e}: "
                    f"original_width {plan.original_width} != mid {model.mid}"
                )
            for c in plan.keep_channels:
                if c < 0 or c >= model.mid:
                    errors.append(f"layer {layer} expert {e}: keep_channel {c} out of range")
    manifest_ok = validate_manifest(manifest)
    return ManifestValidation(ok=not errors and manifest_ok.ok, errors=errors + manifest_ok.errors)


def reorder_channels(
    model: MiniMoE, manifest: CompressionManifest, key: tuple[int, int]
) -> list[int]:
    """Return a non-trivial permutation of an expert's kept channels (same set)."""
    keep = list(orders_from_manifest(manifest)[key])
    if len(keep) < 2:
        return keep
    perm = list(reversed(keep))
    return perm if set(perm) == set(keep) else keep
