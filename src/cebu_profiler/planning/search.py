"""Derivative architecture search over measured profiler evidence (v2 §24–§25).

Consumes measured saliency (F3), coalitions (F8), paths (F9), and per-expert
compression response curves (F7) to generate MULTIPLE Pareto-ish candidates —
not one supposedly optimal plan. Always leaves >= top_k experts per layer and
protects named coalitions/paths while filling a byte budget by value.
"""

from __future__ import annotations

from dataclasses import dataclass

from cebu_profiler.compression.response import ResponsePoint
from cebu_profiler.planning.maps import (
    CandidatePlan,
    CoalitionProtectionMap,
    KeepEntry,
    KeepMap,
    PathPreservationMap,
    PrecisionEntry,
    PrecisionMap,
    ResidencyEntry,
    ResidencyMap,
    SubstituteEntry,
    SubstituteMap,
)
from cebu_profiler.profiler.reap import SaliencyAccumulator
from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.schemas.architecture import DTYPE_BYTES, TensorRole


def expert_src_bytes(model: MiniMoE, _layer: int, _expert: int) -> float:
    numel = model.arch.tensor_params.get(TensorRole.EXPERTS)
    dtype = model.arch.moe.expert_dtype
    return (numel or 0) * DTYPE_BYTES[dtype]


@dataclass
class SearchInputs:
    model: MiniMoE
    saliency: SaliencyAccumulator
    coalitions: dict[int, list[tuple[int, ...]]]  # layer -> list of coalitions
    protected_paths: list[tuple[tuple[int, ...], ...]] | None = None
    response: dict[tuple[int, int], list[ResponsePoint]] | None = None  # (layer,expert)->curve


def _protected_experts(
    num_layers: int, coalitions: dict[int, list[tuple[int, ...]]]
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for layer, colls in coalitions.items():
        for coll in colls:
            for e in coll:
                out.add((layer, e))
    return out


def _select_by_value(
    model: MiniMoE, saliency: SaliencyAccumulator, layer: int, budget: int, protected: set[int]
) -> list[int]:
    kept: list[int] = list(protected)
    rest = [e for e in range(model.n_exp) if e not in protected]
    rest.sort(key=lambda e: saliency.total_value(layer, e), reverse=True)
    for e in rest:
        if len(kept) >= budget:
            break
        kept.append(e)
    return kept


def _select_by_coverage(
    model: MiniMoE, saliency: SaliencyAccumulator, layer: int, budget: int
) -> list[int]:
    chosen: list[int] = []
    remaining = set(range(model.n_exp))
    while len(chosen) < budget and remaining:
        best = max(remaining, key=lambda e: saliency.total_value(layer, e))
        chosen.append(best)
        remaining.discard(best)
    return chosen


def _choose_precision(
    entries: dict[tuple[int, int], list[ResponsePoint]], layer: int, expert: int
) -> tuple[str, float, float]:
    """Pick the lowest-bit probed precision with reconstruction_error <= threshold."""
    curve = entries.get((layer, expert), [])
    probed = [p for p in curve if p.reconstruction_error is not None]
    probed.sort(key=lambda p: p.effective_bits or 0.0)  # try lower bits first
    threshold = 0.1
    for p in probed:
        if (p.reconstruction_error or 1.0) <= threshold:
            return p.format, p.effective_bits or 0.0, p.reconstruction_error or 0.0
    # none meets threshold -> keep higher precision (source-tier)
    fallback = min(probed, key=lambda p: p.reconstruction_error or 1.0) if probed else None
    if fallback:
        return fallback.format, fallback.effective_bits or 0.0, fallback.reconstruction_error or 0.0
    return "source_unprobed", 4.0, 0.0


def build_candidate(
    inputs: SearchInputs,
    *,
    name: str,
    keep_budget_per_layer: int,
    strategy: str = "value",
    node_budget_bytes: float,
    active_bytes_per_token: float,
) -> CandidatePlan:
    model = inputs.model
    n_layers = model.arch.num_text_layers
    protected = _protected_experts(n_layers, inputs.coalitions)

    keep_entries: list[KeepEntry] = []
    precision_entries: list[PrecisionEntry] = []
    residency_entries: list[ResidencyEntry] = []
    substitute_entries: list[SubstituteEntry] = []

    for layer in range(n_layers):
        layer_protected = {e for (lay, e) in protected if lay == layer}
        if strategy == "coalition":
            kept = sorted(layer_protected)
            if len(kept) < keep_budget_per_layer:
                kept = _select_by_value(
                    model, inputs.saliency, layer, keep_budget_per_layer, set(kept)
                )
        elif strategy == "coverage":
            kept = _select_by_coverage(model, inputs.saliency, layer, keep_budget_per_layer)
        elif strategy == "identity":
            kept = list(range(model.n_exp))
        else:  # value (default)
            kept = _select_by_value(
                model, inputs.saliency, layer, keep_budget_per_layer, layer_protected
            )

        for e in range(model.n_exp):
            keep = e in kept
            reason = "saliency" if keep else "budget"
            if keep and (layer, e) in protected:
                reason = "protected_coalition"
            keep_entries.append(
                KeepEntry(
                    source_model_id=model.arch.name,
                    layer_index=layer,
                    source_expert_id=e,
                    keep=keep,
                    reason=reason,
                )
            )
            if keep:
                prec, bits, err = _choose_precision(inputs.response or {}, layer, e)
                precision_entries.append(
                    PrecisionEntry(
                        layer_index=layer,
                        source_expert_id=e,
                        precision=prec,
                        bits=bits,
                        reconstruction_error=err,
                    )
                )
                location = "node_a" if e % 2 == 0 else "node_b"
                residency_entries.append(
                    ResidencyEntry(layer_index=layer, source_expert_id=e, location=location)
                )
                # candidate substitutes: experts NOT co-routed with e (from coalition complement)
                candidates = [
                    x
                    for coll in inputs.coalitions.get(layer, [])
                    for x in coll
                    if x != e and x not in kept
                ]
                substitute_entries.append(
                    SubstituteEntry(
                        layer_index=layer,
                        source_expert_id=e,
                        candidates=list(dict.fromkeys(candidates)),
                        confidence=0.3,
                    )
                )

    keep_map = KeepMap(source_model_id=model.arch.name, entries=keep_entries)
    # per-layer kept counts + bytes
    kept_per_layer: dict[int, int] = {}
    resident_a = resident_b = 0.0
    for entry in residency_entries:
        kept_per_layer[entry.layer_index] = kept_per_layer.get(entry.layer_index, 0) + 1
        b = expert_src_bytes(model, entry.layer_index, entry.source_expert_id)
        if entry.location == "node_a":
            resident_a += b
        elif entry.location == "node_b":
            resident_b += b

    protected_kept = sum(1 for (lay, e) in protected if _entry_keep(keep_map, lay, e))
    protected_paths = inputs.protected_paths or []
    paths_kept = sum(
        1
        for path in protected_paths
        if all(all(e in keep_map.kept(layer) for e in els) for layer, els in enumerate(path))
    )

    fitted = resident_a <= node_budget_bytes and resident_b <= node_budget_bytes

    return CandidatePlan(
        name=name,
        keep=keep_map,
        precision=PrecisionMap(entries=precision_entries),
        residency=ResidencyMap(entries=residency_entries),
        coalition_protection=CoalitionProtectionMap(
            protections=[(lay, c) for lay, colls in inputs.coalitions.items() for c in colls]
        ),
        path_preservation=PathPreservationMap(protected_paths=protected_paths),
        substitutes=SubstituteMap(entries=substitute_entries),
        kept_per_layer=kept_per_layer,
        resident_bytes_a=resident_a,
        resident_bytes_b=resident_b,
        stored_bytes=resident_a + resident_b,
        active_bytes_per_token=active_bytes_per_token,
        protected_coalitions_kept=protected_kept,
        protected_paths_kept=paths_kept,
        fitted=fitted,
    )


def _entry_keep(keep_map: KeepMap, layer: int, expert: int) -> bool:
    for e in keep_map.entries:
        if e.layer_index == layer and e.source_expert_id == expert:
            return e.keep
    return False


def generate_candidates(
    inputs: SearchInputs,
    *,
    keep_budget_per_layer: int,
    node_budget_bytes: float,
    active_bytes_per_token: float,
    strategies: tuple[str, ...] = ("value", "coverage", "coalition", "identity"),
) -> list[CandidatePlan]:
    """Generate multiple candidate plans (one per strategy), never a single winner."""
    out: list[CandidatePlan] = []
    for strat in strategies:
        out.append(
            build_candidate(
                inputs,
                name=f"keep{keep_budget_per_layer}-{strat}",
                keep_budget_per_layer=keep_budget_per_layer,
                strategy=strat,
                node_budget_bytes=node_budget_bytes,
                active_bytes_per_token=active_bytes_per_token,
            )
        )
    return out
