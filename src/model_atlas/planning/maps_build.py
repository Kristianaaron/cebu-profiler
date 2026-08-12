"""Builders for the §25 planning-artifact set (v2 §18, §32-adjacent maps).

Produces the seven maps that did not yet have a producer — channel, tile,
node-ownership, overflow-pack, router-repair, residual-repair,
distillation-target — from a synthetic mini-MoE plus its measured REAP
saliency. Channel/tile maps are grounded in the measured channel-uniqueness
§8.3 view; node-ownership from the census ownership manifest; the remaining
maps derive from measured saliency ranking. Components that depend on removal
impact keep an explicit ``estimated`` posture in their docstrings/risk — no
removal decision is presented as causally validated here (§31:20; real causal
evidence requires routing/failure traces on the target).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_atlas.atlas.reap import SaliencyAccumulator
from model_atlas.atlas.runtime import MiniMoE
from model_atlas.census.census import build_manifest
from model_atlas.planning.maps import (
    ChannelEntry,
    ChannelMap,
    DistillationTargetEntry,
    DistillationTargetMap,
    NodeOwnershipEntry,
    NodeOwnershipMap,
    OverflowPackEntry,
    OverflowPackMap,
    ResidualRepairEntry,
    ResidualRepairMap,
    RouterRepairEntry,
    RouterRepairMap,
    TileEntry,
    TileMap,
)
from model_atlas.scoring.redundancy import channel_uniqueness


@dataclass
class PlanningMapSet:
    """The seven §25 maps plus the measured inputs they were built from."""

    model: str
    channel: ChannelMap = field(default_factory=ChannelMap)
    tile: TileMap = field(default_factory=TileMap)
    node_ownership: NodeOwnershipMap = field(default_factory=NodeOwnershipMap)
    overflow_pack: OverflowPackMap = field(default_factory=OverflowPackMap)
    router_repair: RouterRepairMap = field(default_factory=RouterRepairMap)
    residual_repair: ResidualRepairMap = field(default_factory=ResidualRepairMap)
    distillation_target: DistillationTargetMap = field(default_factory=DistillationTargetMap)


def _expert_saliency(
    saliency: SaliencyAccumulator, layer: int, expert: int
) -> float:
    return saliency.total_value(layer, expert)


def _kept_experts(
    model: MiniMoE, saliency: SaliencyAccumulator, keep_frac: float
) -> dict[int, set[int]]:
    """Kept expert set per layer = the top-`keep_frac` by measured saliency."""
    kept: dict[int, set[int]] = {}
    for layer in range(len(model.layers)):
        order = sorted(
            range(model.n_exp), key=lambda e: -_expert_saliency(saliency, layer, e)
        )
        n = max(1, round(model.n_exp * keep_frac))
        kept[layer] = set(order[:n])
    return kept


def build_channel_map(
    model: MiniMoE, uniq: dict[tuple[int, int, int], float], keep_frac: float
) -> ChannelMap:
    entries: list[ChannelEntry] = []
    by_exp: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for (layer, e, c), u in uniq.items():
        by_exp.setdefault((layer, e), []).append((c, u))
    for (layer, e), channels in sorted(by_exp.items()):
        channels_sorted = sorted(channels, key=lambda x: -x[1])
        n = max(1, round(len(channels) * keep_frac))
        keep_ids = {c for c, _ in channels_sorted[:n]}
        for c, u in sorted(channels):
            entries.append(
                ChannelEntry(
                    layer_index=layer,
                    source_expert_id=e,
                    channel_id=c,
                    importance=round(u, 5),
                    keep=c in keep_ids,
                )
            )
    return ChannelMap(entries=entries)


def build_tile_map(
    model: MiniMoE, uniq: dict[tuple[int, int, int], float], keep_frac: float, tile_size: int
) -> TileMap:
    entries: list[TileEntry] = []
    by_exp: dict[tuple[int, int], dict[int, list[float]]] = {}
    for (layer, e, c), u in uniq.items():
        tile = c // tile_size
        by_exp.setdefault((layer, e), {}).setdefault(tile, []).append(u)
    for (layer, e), tiles in sorted(by_exp.items()):
        means = {t: sum(v) / len(v) for t, v in tiles.items()}
        n_keep = max(1, round(len(tiles) * keep_frac))
        keep_tiles = set(t for t, _ in sorted(means.items(), key=lambda x: -x[1])[:n_keep])
        for t, _vals in sorted(tiles.items()):
            entries.append(
                TileEntry(
                    layer_index=layer,
                    source_expert_id=e,
                    tile_index=t,
                    channel_start=t * tile_size,
                    importance=round(means[t], 5),
                    keep=t in keep_tiles,
                )
            )
    return TileMap(entries=entries)


def build_node_ownership_map(model: MiniMoE) -> NodeOwnershipMap:
    manifest = build_manifest(model.arch)
    entries = [
        NodeOwnershipEntry(
            tensor_key=r.key,
            role=r.role.value,
            layer_index=r.layer_index,
            source_expert_id=r.expert_index,
            node=r.location.value,
        )
        for r in manifest.records
    ]
    return NodeOwnershipMap(entries=entries)


def build_overflow_pack_map(
    model: MiniMoE, saliency: SaliencyAccumulator, resident_frac: float
) -> OverflowPackMap:
    entries: list[OverflowPackEntry] = []
    for layer in range(len(model.layers)):
        order = sorted(
            range(model.n_exp), key=lambda e: -_expert_saliency(saliency, layer, e)
        )
        resident_n = max(1, round(model.n_exp * resident_frac))
        for rank, e in enumerate(order):
            if rank >= resident_n:
                entries.append(
                    OverflowPackEntry(
                        layer_index=layer,
                        source_expert_id=e,
                        tier="nvme_a" if e % 2 == 0 else "nvme_b",
                        reason="below-resident saliency (measured)",
                    )
                )
    return OverflowPackMap(entries=entries)


def build_router_repair_map(
    model: MiniMoE, saliency: SaliencyAccumulator, keep_frac: float
) -> RouterRepairMap:
    kept = _kept_experts(model, saliency, keep_frac)
    entries: list[RouterRepairEntry] = []
    for layer in range(len(model.layers)):
        keepers = sorted(kept[layer])
        new_by_old = {old: new for new, old in enumerate(keepers)}
        for e in range(model.n_exp):
            entries.append(
                RouterRepairEntry(
                    layer_index=layer,
                    old_index=e,
                    new_index=new_by_old.get(e),
                    action="keep" if e in kept[layer] else "drop",
                    route_bias=True,  # §31:18 — router + correction bias stay in lockstep
                )
            )
    return RouterRepairMap(entries=entries)


def build_residual_repair_map(
    model: MiniMoE, saliency: SaliencyAccumulator, keep_frac: float
) -> ResidualRepairMap:
    kept = _kept_experts(model, saliency, keep_frac)
    entries: list[ResidualRepairEntry] = []
    for layer in range(len(model.layers)):
        for e in range(model.n_exp):
            if e in kept[layer]:
                continue
            s = _expert_saliency(saliency, layer, e)
            if s <= 0:
                continue
            entries.append(
                ResidualRepairEntry(
                    layer_index=layer,
                    source_expert_id=e,
                    component="expert_output",
                    severity=round(s, 5),
                    target="distill",
                )
            )
    return ResidualRepairMap(entries=entries)


def build_distillation_target_map(
    model: MiniMoE, saliency: SaliencyAccumulator, per_layer: int = 3
) -> DistillationTargetMap:
    entries: list[DistillationTargetEntry] = []
    for layer in range(len(model.layers)):
        order = sorted(
            range(model.n_exp), key=lambda e: -_expert_saliency(saliency, layer, e)
        )
        for e in order[:per_layer]:
            entries.append(
                DistillationTargetEntry(
                    layer_index=layer,
                    source_expert_id=e,
                    target_type="expert",
                    priority=round(_expert_saliency(saliency, layer, e), 5),
                )
            )
    return DistillationTargetMap(entries=entries)


def build_planning_maps(
    model: MiniMoE,
    saliency: SaliencyAccumulator,
    *,
    keep_frac: float = 0.8,
    resident_frac: float = 0.75,
    tile_size: int = 8,
) -> PlanningMapSet:
    """Build all seven §25 maps from a synthetic model + measured saliency."""
    uniq = channel_uniqueness(model)
    return PlanningMapSet(
        model=model.arch.name,
        channel=build_channel_map(model, uniq, keep_frac),
        tile=build_tile_map(model, uniq, keep_frac, tile_size),
        node_ownership=build_node_ownership_map(model),
        overflow_pack=build_overflow_pack_map(model, saliency, resident_frac),
        router_repair=build_router_repair_map(model, saliency, keep_frac),
        residual_repair=build_residual_repair_map(model, saliency, keep_frac),
        distillation_target=build_distillation_target_map(model, saliency),
    )
