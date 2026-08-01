"""F13 tests: two-node serving + elastic overflow (v2 §14, §5)."""

from model_atlas.atlas.runtime import build_mini_moe, forward
from model_atlas.registry.architectures import get_registry
from model_atlas.serving import (
    assign_nodes,
    build_resident_policy,
    cross_node_activation_bytes_per_token,
    fit,
    resident_bytes_by_node,
    run_distributed,
    simulate_overflow,
)

ARCH = get_registry().get("k3-mini")


def _deriv(seed=1):
    # identity derivative (keep all experts) as the serving subject
    return build_mini_moe(ARCH, seed=seed)


def test_assign_nodes_splits_equal_parity():
    dv = _deriv()
    assignment = assign_nodes(dv)
    assert assignment.node_a and assignment.node_b
    # every kept expert assigned exactly one node
    for layer in range(dv.arch.num_text_layers):
        for slot in range(len(dv.layers[layer].experts)):
            assert assignment.node_of(layer, slot) in {"node_a", "node_b"}


def test_resident_bytes_and_fit_go_no_go():
    dv = _deriv()
    assignment = assign_nodes(dv)
    a, b = resident_bytes_by_node(assignment, dv)
    assert a > 0 and b > 0
    tight = fit(dv, assignment, node_budget_bytes=1.0)  # tiny -> cannot fit
    assert tight.fitted is False
    loose = fit(dv, assignment, node_budget_bytes=max(a, b) * 10, runtime_reserve_bytes=0.0)
    assert loose.fitted is True


def test_cross_node_activation_estimate_positive():
    dv = _deriv()
    assert cross_node_activation_bytes_per_token(dv) > 0.0


def test_run_distributed_attributes_routes_to_nodes():
    dv = _deriv()
    assignment = assign_nodes(dv)
    run = run_distributed(dv, [1, 2, 3], assignment)
    assert run.node_a_routed > 0 and run.node_b_routed > 0
    total = run.node_a_routed + run.node_b_routed
    # equals total routed expert executions from a plain forward
    plain = sum(len(ids) for t in forward(dv, [1, 2, 3]).traces for ids in t.topk_ids)
    assert total == plain


def test_elastic_overflow_miss_counting_and_stop():
    dv = _deriv()
    policy = build_resident_policy(dv, resident_fraction=0.5)
    assert policy.resident and policy.overflow
    # permissive threshold -> overflow kept
    ml = simulate_overflow(dv, [1, 2, 3, 4, 5], policy, miss_threshold=1.0)
    # strict threshold -> overflow disabled (stop condition)
    ms = simulate_overflow(dv, [1, 2, 3, 4, 5], policy, miss_threshold=0.0)
    assert ml.overflow_disabled is False
    assert ms.overflow_disabled is True
    assert 0.0 <= ml.miss_rate <= 1.0
    assert ml.total_routed > 0


def test_elastic_all_resident_zero_misses():
    dv = _deriv()
    policy = build_resident_policy(dv, resident_fraction=1.0)
    ml = simulate_overflow(dv, [1, 2], policy, miss_threshold=0.0)
    assert ml.cold_misses == 0
    assert ml.overflow_disabled is False
