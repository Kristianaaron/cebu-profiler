"""F21 tests: §25 planning-artifact maps (channel/tile/node-ownership/overflow/
router-repair/residual-repair/distillation-target).

Each builder must be deterministic and respect its map's contract: channel/tile
keep ratios match the requested fraction, router-repair reindexes exactly (keep
0..k-1 contiguous new indices, drop -> None, route_bias always on), ownership
reports lawful node homes, and overflow/residual/distillation only flag
saliency-grounded components.
"""

from model_atlas.atlas.reap import make_synthetic_corpus, run_calibration
from model_atlas.atlas.runtime import build_mini_moe
from model_atlas.planning.maps_build import build_planning_maps
from model_atlas.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def _maps(seed: int = 0, keep_frac: float = 0.8, resident_frac: float = 0.75):
    model = build_mini_moe(ARCH, seed=seed)
    corpus = make_synthetic_corpus(
        n_samples=24, seq_len=6, vocab=ARCH.vocabulary_size or 1000, seed=seed
    )[0]
    sal = run_calibration(model, corpus, top_k=2)
    return build_planning_maps(model, sal, keep_frac=keep_frac, resident_frac=resident_frac)


def test_all_maps_nonempty_and_deterministic():
    a = _maps(seed=1)
    b = _maps(seed=1)
    for name in (
        "channel",
        "tile",
        "node_ownership",
        "overflow_pack",
        "router_repair",
        "residual_repair",
        "distillation_target",
    ):
        ma = getattr(a, name)
        mb = getattr(b, name)
        assert ma.entries, name
        assert [e.model_dump() for e in ma.entries] == [e.model_dump() for e in mb.entries]


def test_channel_map_keeps_fraction_and_bounds():
    ms = _maps(keep_frac=0.8)
    by_exp = {}
    for e in ms.channel.entries:
        by_exp.setdefault((e.layer_index, e.source_expert_id), []).append(e)
        assert 0.0 <= e.importance <= 1.0
    for (_lay, _exp), chans in by_exp.items():
        kept = sum(1 for c in chans if c.keep)
        assert kept == round(len(chans) * 0.8)  # top-fraction per expert (rounded)


def test_tile_map_groups_and_keeps_blocks():
    ms = _maps(keep_frac=0.8)
    assert all(e.channel_start == e.tile_index * 8 for e in ms.tile.entries)
    # tiles are groups of 8 channels, importance within unit interval
    assert all(-1e-9 <= e.importance <= 1.0 + 1e-9 for e in ms.tile.entries)
    assert any(e.keep for e in ms.tile.entries)


def test_node_ownership_lawful_homes():
    ms = _maps()
    homes = {"node_a", "node_b", "nvme_a", "nvme_b", "replicated"}
    for e in ms.node_ownership.entries:
        assert e.node in homes
        assert e.role  # classified, non-empty
    # every routed expert slot appears somewhere
    expert_rows = [e for e in ms.node_ownership.entries if e.source_expert_id is not None]
    assert expert_rows


def test_router_repair_reindexes_contiguously():
    ms = _maps(keep_frac=0.8)
    per_layer = {}
    for e in ms.router_repair.entries:
        assert e.route_bias is True  # v2 §31:18 lockstep
        per_layer.setdefault(e.layer_index, []).append(e)
    for _lay, rows in per_layer.items():
        accepted = [r for r in rows if r.action == "keep"]
        dropped = [r for r in rows if r.action == "drop"]
        assert accepted or dropped  # every layer has at least one slot row
        assert len(accepted) + len(dropped) == ARCH.moe.num_routed_experts
        # keep rows get contiguous 0..k-1 new indices, dropped get None
        assert sorted(r.new_index for r in accepted) == list(range(len(accepted)))
        assert all(r.new_index is None for r in dropped)
        # old indices are a bijection per layer
        assert sorted(r.old_index for r in rows) == list(range(ARCH.moe.num_routed_experts))


def test_overflow_pack_only_nonresident_tiers():
    ms = _maps(resident_frac=0.75)
    assert all(e.tier in {"nvme_a", "nvme_b"} for e in ms.overflow_pack.entries)
    for e in ms.overflow_pack.entries:
        assert "saliency" in e.reason


def test_residual_and_distillation_grounded():
    ms = _maps()
    assert all(e.severity > 0 for e in ms.residual_repair.entries)
    perf_layer = {}
    for e in ms.distillation_target.entries:
        assert e.priority >= 0
        assert e.target_type == "expert"
        perf_layer.setdefault(e.layer_index, 0)
        perf_layer[e.layer_index] += 1
    assert max(perf_layer.values()) <= 3  # per_layer arg caps targets
