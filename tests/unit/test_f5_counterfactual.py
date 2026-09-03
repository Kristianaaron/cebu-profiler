"""F5 tests: counterfactual routing + route regret (v2 §13)."""

from cebu_profiler.profiler.counterfactual import (
    counterfactual_scan,
    final_utility,
    logit_kl,
    sample_topk_subsets,
)
from cebu_profiler.profiler.runtime import build_mini_moe, forward
from cebu_profiler.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def test_softmax_utility_in_01():
    model = build_mini_moe(ARCH, seed=1)
    result = forward(model, [1, 2, 3], top_k=2)
    u = final_utility(result.logits)
    assert 0.0 <= u <= 1.0


def test_kl_nonnegative_and_zero_for_same():
    model = build_mini_moe(ARCH, seed=1)
    result = forward(model, [1, 2, 3], top_k=2)
    assert abs(logit_kl(result.logits, result.logits)) < 1e-12
    other = forward(model, [9, 9, 9], top_k=2)
    assert logit_kl(result.logits, other.logits) >= -1e-12


def test_topk_subset_sampling_excludes_original():
    subsets = sample_topk_subsets(8, 2, frozenset({0, 1}), n=10, seed=0)
    assert len(subsets) == 10
    assert all(frozenset(s) != frozenset({0, 1}) for s in subsets)
    assert all(len(s) == 2 and len(set(s)) == 2 for s in subsets)


def test_counterfactual_scan_returns_regret():
    model = build_mini_moe(ARCH, seed=3)
    tokens = [5, 12, 77]
    res = counterfactual_scan(model, tokens, layer=0, token_index=0, n_alternatives=6, seed=0)
    assert len(res.alternatives) == 6
    assert len(res.original_route) == ARCH.moe.top_k
    # best utility >= original utility by construction (or equal if never beaten)
    assert res.best_utility >= res.original_utility - 1e-12
    assert res.route_regret == res.best_utility - res.original_utility


def test_forced_original_route_is_no_change():
    # forcing the original route back must reproduce the original utility (regret 0)
    model = build_mini_moe(ARCH, seed=4)
    tokens = [3, 4, 5]
    base = forward(model, tokens, top_k=2)
    orig = sorted(base.traces[1].topk_ids[0])
    forced = forward(model, tokens, top_k=2, route_override={(1, 0): orig})
    assert orig == sorted(forced.traces[1].topk_ids[0])
    assert abs(final_utility(forced.logits) - final_utility(base.logits)) < 1e-12


def test_counterfactual_deterministic_with_seed():
    model = build_mini_moe(ARCH, seed=5)
    a = counterfactual_scan(model, [1, 2, 3], layer=0, token_index=1, n_alternatives=5, seed=1)
    b = counterfactual_scan(model, [1, 2, 3], layer=0, token_index=1, n_alternatives=5, seed=1)
    assert a.alternatives == b.alternatives
    assert a.route_regret == b.route_regret
