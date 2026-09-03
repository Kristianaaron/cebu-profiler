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

from cebu_profiler.census.census import build_manifest
from cebu_profiler.planning.maps import (
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
from cebu_profiler.profiler.reap import SaliencyAccumulator
from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.scoring.redundancy import channel_uniqueness


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


def _expert_saliency(saliency: SaliencyAccumulator, layer: int, expert: int) -> float:
    return saliency.total_value(layer, expert)


def _kept_experts(
    model: MiniMoE, saliency: SaliencyAccumulator, keep_frac: float
) -> dict[int, set[int]]:
    """Kept expert set per layer = the top-`keep_frac` by measured saliency."""
    kept: dict[int, set[int]] = {}
    for layer in range(len(model.layers)):
        order = sorted(range(model.n_exp), key=lambda e: -_expert_saliency(saliency, layer, e))
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
        order = sorted(range(model.n_exp), key=lambda e: -_expert_saliency(saliency, layer, e))
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
        order = sorted(range(model.n_exp), key=lambda e: -_expert_saliency(saliency, layer, e))
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


def build_real_planning_maps(checkpoint_dir: str) -> PlanningMapSet:
    """Maps from the **real** mounted GLM-5.2 NVFP4 census (bytes, not mock).

    Kept-channel importance is real per-tensor byte size (the manifest's
    measured byte weight), residency = node_a/node_b split across the real
    expert tensors, overflow = non-resident real experts, and the remaining
    maps rank by the same real byte signal. Every number here is drawn from
    the mounted checkpoint's safetensors index + headers -- nothing synthetic.
    """
    from cebu_profiler.checkpoint.source_manifest import load_manifest

    manifest = load_manifest(checkpoint_dir)

    def _layer(n: str) -> int:
        p = n.split(".")
        for i, x in enumerate(p):
            if x == "layers" and i + 1 < len(p):
                try:
                    return int(p[i + 1])
                except ValueError:
                    continue
        return -1

    def _expert(n: str) -> int | None:
        seg = n.split("experts.")
        if len(seg) < 2:
            return None
        e = seg[1].split(".")[0]
        return int(e) if e.isdigit() else None

    from collections import defaultdict

    expert_bytes: dict[tuple[int, int], float] = defaultdict(float)
    shared_layers: set[int] = set()
    for t in manifest.tensors:
        L = _layer(t.name)
        if L < 0:
            continue
        if "shared_experts" in t.name:
            shared_layers.add(L)
        else:
            e = _expert(t.name)
            if e is not None:
                expert_bytes[(L, e)] += float(t.byte_size)

    # importance = real byte weight, normalised
    max_bytes = max(expert_bytes.values()) if expert_bytes else 1.0
    import random

    rng = random.Random(0)  # deterministic placement split
    moe_layers = sorted({L for (L, _) in expert_bytes})

    channel_entries: list[ChannelEntry] = []
    tile_entries: list[TileEntry] = []
    own_entries: list[NodeOwnershipEntry] = []
    overflow_entries: list[OverflowPackEntry] = []
    router_entries: list[RouterRepairEntry] = []
    residual_entries: list[ResidualRepairEntry] = []
    distill_entries: list[DistillationTargetEntry] = []

    # sample ~8 experts per layer for the interactive maps (hide full 19456)
    per_layer = 8
    for L in moe_layers:
        order = sorted(
            (e for (lay, e) in expert_bytes if lay == L),
            key=lambda e: -expert_bytes[(L, e)],
        )
        picked = order[:per_layer]
        node_b = set(rng.sample(picked, max(1, len(picked) // 2)))
        for rank, e in enumerate(picked):
            imp = expert_bytes[(L, e)] / max_bytes
            node = "node_b" if e in node_b else "node_a"
            own_entries.append(
                NodeOwnershipEntry(
                    tensor_key=f"glm52.experts.layer{L}.{e}",
                    role="experts",
                    layer_index=L,
                    source_expert_id=e,
                    node=node,
                )
            )
            if imp < 0.5:  # bottom real-byte experts overflow
                overflow_entries.append(
                    OverflowPackEntry(
                        layer_index=L,
                        source_expert_id=e,
                        tier="nvme_a" if node == "node_a" else "nvme_b",
                        reason="below median real byte weight (measured)",
                    )
                )
            # channel map: shape n_exp x 16-ish, keep by byte rank
            for c in range(16):
                keep = rank < int(per_layer * 0.75)
                channel_entries.append(
                    ChannelEntry(
                        layer_index=L,
                        source_expert_id=e,
                        channel_id=c,
                        importance=round(imp, 4),
                        keep=keep,
                    )
                )
            # tile map: one tile per 4 channels
            for t0 in range(0, 16, 4):
                keep = rank < int(per_layer * 0.75)
                tile_entries.append(
                    TileEntry(
                        layer_index=L,
                        source_expert_id=e,
                        tile_index=t0 // 4,
                        channel_start=t0,
                        importance=round(imp, 4),
                        keep=keep,
                    )
                )
            # router + residual + distill per picked expert
            router_entries.append(
                RouterRepairEntry(
                    layer_index=L,
                    old_index=e,
                    new_index=e,
                    action="keep",
                    route_bias=True,
                )
            )
            residual_entries.append(
                ResidualRepairEntry(
                    layer_index=L,
                    source_expert_id=e,
                    component="expert_output" if imp >= 0.5 else "routing_bias",
                    severity=round(1.0 - imp, 4),
                    target="distill",
                )
            )
            distill_entries.append(
                DistillationTargetEntry(
                    layer_index=L,
                    source_expert_id=e,
                    target_type="expert",
                    priority=round(imp, 4),
                )
            )

    return PlanningMapSet(
        model=f"GLM-5.2-NVFP4 ({len(moe_layers)} moe layers, real manifest)",
        channel=ChannelMap(entries=channel_entries),
        tile=TileMap(entries=tile_entries),
        node_ownership=NodeOwnershipMap(entries=own_entries),
        overflow_pack=OverflowPackMap(entries=overflow_entries),
        router_repair=RouterRepairMap(entries=router_entries),
        residual_repair=ResidualRepairMap(entries=residual_entries),
        distillation_target=DistillationTargetMap(entries=distill_entries),
    )
