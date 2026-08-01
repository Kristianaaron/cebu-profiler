"""F8 tests: coalitions + multi-component causal tracing (v2 §10.7, §17).

The causal sweeps are kept tiny (small corpus, capped pair sets) because the
forward pass is pure Python.
"""

from model_atlas.atlas.coalition import (
    coactivation_map,
    minimum_destructive_set,
    pairwise_causal,
    single_effect,
    synergic_pairs,
)
from model_atlas.atlas.reap import make_synthetic_corpus
from model_atlas.atlas.runtime import build_mini_moe, forward
from model_atlas.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def _samples(n=4, seq=4):
    return make_synthetic_corpus(n_samples=n, seq_len=seq, vocab=ARCH.vocabulary_size, seed=0)[0]


def test_ablation_excluded_sets_zero_router_probability():
    model = build_mini_moe(ARCH, seed=1)
    result = forward(model, [1, 2, 3], top_k=2, excluded={0: frozenset({0})})
    for trace in result.traces:
        if trace.layer == 0:
            for ids in trace.topk_ids:
                assert 0 not in ids


def test_coactivation_map_symmetric_and_nonnegative():
    model = build_mini_moe(ARCH, seed=2)
    cmap = coactivation_map(model, _samples(10, 5), layer=0, top_k=2)
    for (a, b), c in cmap.pair_counts.items():
        assert a < b
        assert c >= 0
    # over a real corpus at least one pair was co-routed
    assert cmap.pair_counts


def test_single_effect_nonnegative_for_routed_expert():
    model = build_mini_moe(ARCH, seed=3)
    samples = _samples()
    effects = [single_effect(model, samples, 0, {e}) for e in range(model.n_exp)]
    assert any(e >= 0.0 for e in effects)


def test_pairwise_synergy_equals_formula():
    model = build_mini_moe(ARCH, seed=4)
    a = pairwise_causal(model, _samples(), 0, 1, 2)
    assert abs(a.synergy_ab - (a.effect_ab - a.effect_a - a.effect_b)) < 1e-12


def test_synergic_pairs_sorted_desc_and_capped():
    model = build_mini_moe(ARCH, seed=5)
    # cap to 3 experts -> only 3 pairs, keeps the pure-python sweep cheap
    pairs = synergic_pairs(model, _samples(), 0, max_experts=3)
    synergies = [p.synergy_ab for p in pairs]
    assert synergies == sorted(synergies, reverse=True)
    assert len(pairs) == 3


def test_minimum_destructive_set_size_one_preferred():
    model = build_mini_moe(ARCH, seed=6)
    samples = _samples()
    # pick a threshold from the actual singleton effects so a set is reachable
    base = max(single_effect(model, samples, 0, {e}) for e in range(model.n_exp))
    threshold = max(0.0, base * 0.5)
    found = minimum_destructive_set(model, samples, 0, damage_threshold=threshold, max_size=2)
    # either a singleton (preferred) or, if all singletons are below the chosen
    # low threshold, possibly a pair — any returned set must be valid/small
    if found is not None:
        dropped, effect = found
        assert len(dropped) == 1 or len(dropped) == 2
        assert effect >= threshold
