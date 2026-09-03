"""F10 tests: derivative architecture search + planning maps (v2 §24–§25)."""

from cebu_profiler.compression import expert_response_curve, get_backend_registry
from cebu_profiler.planning import (
    SearchInputs,
    build_candidate,
    expert_src_bytes,
    generate_candidates,
)
from cebu_profiler.profiler.reap import make_synthetic_corpus, run_calibration
from cebu_profiler.profiler.runtime import build_mini_moe
from cebu_profiler.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def _inputs(seed=1):
    model = build_mini_moe(ARCH, seed=seed)
    samples = make_synthetic_corpus(n_samples=12, seq_len=6, vocab=ARCH.vocabulary_size, seed=0)[0]
    sal = run_calibration(model, samples, top_k=2)
    reg = get_backend_registry()
    response = {}
    for layer in range(model.arch.num_text_layers):
        for e in range(model.n_exp):
            response[(layer, e)] = expert_response_curve(
                model, [1, 2, 3], layer=layer, expert=e, backends=reg, formats=["int4", "int8"]
            )
    coalitions = {0: [(0, 2, 4)], 1: [(1, 3)]}  # protected coalition per layer
    return SearchInputs(model=model, saliency=sal, coalitions=coalitions, response=response), model


def test_expert_value_measured_and_nonnegative():
    inputs, _ = _inputs()
    v = inputs.saliency.total_value(0, 0)
    assert v >= 0.0


def test_build_candidate_value_keeps_budget_and_protects():
    inputs, model = _inputs()
    per_expert = expert_src_bytes(model, 0, 0)
    assert per_expert > 0.0
    plan = build_candidate(
        inputs,
        name="value4",
        keep_budget_per_layer=4,
        strategy="value",
        node_budget_bytes=per_expert * 8 * 10,  # generous
        active_bytes_per_token=1000.0,
    )
    # protected coalition expert (layer0 e0) kept even at low value
    assert 0 in plan.keep.kept(0)
    assert all(len(plan.keep.kept(lay)) <= 4 for lay in range(model.arch.num_text_layers))
    # kept count == budget (>= top_k)
    assert all(len(plan.keep.kept(lay)) == 4 for lay in range(model.arch.num_text_layers))
    # residency assigns every kept expert a node
    assert plan.residency.entries
    assert all(e.location in {"node_a", "node_b"} for e in plan.residency.entries)


def test_identity_candidate_keeps_all():
    inputs, model = _inputs()
    plan = build_candidate(
        inputs,
        name="identity",
        keep_budget_per_layer=model.n_exp,
        strategy="identity",
        node_budget_bytes=1e12,
        active_bytes_per_token=0.0,
    )
    for lay in range(model.arch.num_text_layers):
        assert len(plan.keep.kept(lay)) == model.n_exp


def test_candidate_fitted_reflects_budget():
    inputs, model = _inputs()
    tight = build_candidate(
        inputs,
        name="tight",
        keep_budget_per_layer=4,
        strategy="value",
        node_budget_bytes=1.0,  # tiny -> cannot fit resident bytes
        active_bytes_per_token=0.0,
    )
    assert tight.fitted is False
    loose = build_candidate(
        inputs,
        name="loose",
        keep_budget_per_layer=4,
        strategy="value",
        node_budget_bytes=1e12,
        active_bytes_per_token=0.0,
    )
    assert loose.fitted is True


def test_generate_candidates_multiple_strategies():
    inputs, _ = _inputs()
    plans = generate_candidates(
        inputs,
        keep_budget_per_layer=4,
        node_budget_bytes=1e12,
        active_bytes_per_token=100.0,
        strategies=("value", "coverage", "coalition", "identity"),
    )
    assert len(plans) == 4
    # each plan is a distinct strategy => at least not all identical keep counts
    names = {p.name for p in plans}
    assert names == {
        "keep4-value",
        "keep4-coverage",
        "keep4-coalition",
        "keep4-identity",
    }


def test_precision_map_driven_by_response_curve():
    inputs, _ = _inputs()
    plan = build_candidate(
        inputs,
        name="value4",
        keep_budget_per_layer=4,
        strategy="value",
        node_budget_bytes=1e12,
        active_bytes_per_token=0.0,
    )
    # kept experts have a precision assignment with bits >= 4 (int4) and error recorded
    assert plan.precision.entries
    assert all(e.precision in {"int4", "int8"} for e in plan.precision.entries)
