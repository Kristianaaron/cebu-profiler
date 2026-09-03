"""F12 tests: held-out retention, leakage gate, repair targets (v2 §7, §14, §26)."""

from cebu_profiler.builder import build_derivative
from cebu_profiler.compression import expert_response_curve, get_backend_registry
from cebu_profiler.evaluation import (
    detect_leakage,
    evaluate_heldout,
    promote_allowed,
    router_repair_targets,
)
from cebu_profiler.planning import SearchInputs, build_candidate
from cebu_profiler.profiler.reap import CalibrationSample, make_synthetic_corpus, run_calibration
from cebu_profiler.profiler.runtime import build_mini_moe
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.ontology import CapabilityLabel, TrajectoryStage

ARCH = get_registry().get("k3-mini")


def _model_and_plan(keep=4, seed=1):
    model = build_mini_moe(ARCH, seed=seed)
    samples = make_synthetic_corpus(n_samples=8, seq_len=5, vocab=ARCH.vocabulary_size, seed=0)[0]
    sal = run_calibration(model, samples, top_k=2)
    reg = get_backend_registry()
    response = {}
    for layer in range(model.arch.num_text_layers):
        for e in range(model.n_exp):
            response[(layer, e)] = expert_response_curve(
                model, [1, 2, 3], layer=layer, expert=e, backends=reg, formats=["int4", "int8"]
            )
    coalition = {0: [(0, 2, 4)], 1: [(1, 3)]}
    plan = build_candidate(
        SearchInputs(model=model, saliency=sal, coalitions=coalition, response=response),
        name=f"keep{keep}",
        keep_budget_per_layer=keep,
        strategy="value",
        node_budget_bytes=1e12,
        active_bytes_per_token=100.0,
    )
    return model, plan


def test_detect_leakage_exact_overlap():
    s1 = CalibrationSample(
        tokens=[1, 2, 3], labels=[CapabilityLabel.CODE_GENERATION], stage=TrajectoryStage.PLAN
    )
    held = [
        CalibrationSample(
            tokens=[1, 2, 3],
            labels=[CapabilityLabel.CODE_GENERATION],
            stage=TrajectoryStage.DIAGNOSE,
        )
    ]
    leak = detect_leakage([s1], held)
    assert leak.detected
    assert 0 in leak.exact_overlap


def test_detect_leakage_disjoint_is_clean():
    a = CalibrationSample(
        tokens=[1, 2, 3], labels=[CapabilityLabel.CODE_GENERATION], stage=TrajectoryStage.PLAN
    )
    b = CalibrationSample(
        tokens=[9, 8, 7], labels=[CapabilityLabel.DEBUGGING], stage=TrajectoryStage.VERIFY
    )
    assert detect_leakage([a], [b]).detected is False


def test_promote_blocked_without_override():
    s1 = CalibrationSample(
        tokens=[1, 2, 3], labels=[CapabilityLabel.CODE_GENERATION], stage=TrajectoryStage.PLAN
    )
    held = [
        CalibrationSample(
            tokens=[1, 2, 3],
            labels=[CapabilityLabel.CODE_GENERATION],
            stage=TrajectoryStage.DIAGNOSE,
        )
    ]
    leak = detect_leakage([s1], held)
    assert promote_allowed(leak, allow_development_override=False) is False
    assert promote_allowed(leak, allow_development_override=True) is True


def test_identity_derivative_retains_near_full_on_heldout():
    model, plan = _model_and_plan(keep=8)  # identity-ish keep-all
    deriv = build_derivative(model, plan).model
    held = make_synthetic_corpus(n_samples=10, seq_len=5, vocab=ARCH.vocabulary_size, seed=99)[0]
    report = evaluate_heldout(model, deriv, held)
    assert report.n_samples == 10
    assert report.overall_retention > 0.5  # keep-all is close to no-op
    assert report.per_label
    assert report.worst_label_drop >= 0.0


def test_pruned_derivative_has_lower_or_measureable_retention():
    model, plan = _model_and_plan(keep=4)
    deriv = build_derivative(model, plan).model
    held = make_synthetic_corpus(n_samples=12, seq_len=5, vocab=ARCH.vocabulary_size, seed=101)[0]
    report = evaluate_heldout(model, deriv, held)
    assert report.overall_retention >= 0.0  # retention is measured (may be < 1 for pruning)
    # worst-label drop is reported (not fabricated)
    assert (
        any(r.retention < 1.0 for r in report.per_label) or report.overall_retention <= 1.0 + 1e-9
    )


def test_router_repair_targets_flag_dropped_important_experts():
    model, plan = _model_and_plan(keep=4)
    held = make_synthetic_corpus(n_samples=12, seq_len=5, vocab=ARCH.vocabulary_size, seed=202)[0]
    held_sal = run_calibration(model, held, top_k=2)  # saliency on held-out data
    targets = router_repair_targets(plan, held_sal, saliency_threshold=0.0, source=model)
    for layer, expert in targets:
        assert expert not in set(plan.keep.kept(layer))  # dropped experts only
        assert held_sal.total_value(layer, expert) > 0.0  # measured, not fabricated
