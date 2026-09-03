"""F22 tests: Cebu Profiler v3 fidelity-first analyzers + candidate graph.

Covers each new v3 analyzer (shared-representation, spectral, conditional
sensitivity, routing-consistency, global EXL3 bit budget, quant-interaction,
fixed-grid refiner, residual-correction, NVFP4 suitability, KV ledger,
structural fallback) plus the candidate-graph immutability rules. All run on
the deterministic synthetic MiniMoE — no checkpoint required.
"""

from cebu_profiler.analysis import (
    analyze_shared_representation,
    analyze_spectral,
    conditional_sensitivity,
    enumerate_global_bit_maps,
    fit_quant_interaction,
    nvfp4_suitability,
    routing_consistency,
)
from cebu_profiler.analysis.kv_memory import MemoryLedger, plan_kv_budget
from cebu_profiler.analysis.quant_interaction import predict_global_error
from cebu_profiler.analysis.refiner import refine_expert_tensors
from cebu_profiler.analysis.residual_correction import residual_correction_plan
from cebu_profiler.analysis.structural_fallback import structural_fallback_plans
from cebu_profiler.candidates import (
    CandidateGraph,
    CandidateGraphError,
    CandidateNode,
    CandidateStage,
)
from cebu_profiler.profiler.reap import make_synthetic_corpus
from cebu_profiler.profiler.runtime import build_mini_moe
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.coverage import (
    CapacityCoverage,
    CoverageThresholds,
    EvidenceGate,
)

ARCH = get_registry().get("k3-mini")


def _model(seed: int = 0):
    return build_mini_moe(ARCH, seed=seed)


def _corpus(model, n: int = 8, seed: int = 0):
    return make_synthetic_corpus(
        n_samples=n, seq_len=4, vocab=model.arch.vocabulary_size or 1000, seed=seed
    )[0]


def test_shared_representation_rows_and_bounds() -> None:
    model = _model()
    a = analyze_shared_representation(model)
    b = analyze_shared_representation(model)
    assert len(a.rows) == len(model.layers) * model.n_exp
    assert a.model_dump() == b.model_dump()  # deterministic
    for r in a.rows:
        assert 0.0 <= r.shared_energy_ratio <= 1.0
        assert abs(r.shared_energy_ratio + r.unique_energy_ratio - 1.0) < 1e-6


def test_spectral_rows_three_tensors() -> None:
    model = _model()
    sp = analyze_spectral(model, heavy_tail_modes=4)
    assert len(sp.rows) == len(model.layers) * model.n_exp * 3
    for r in sp.rows:
        assert r.effective_rank >= 0.0
        assert 0.0 <= r.energy_ratio_top <= 1.0
        assert 0.0 <= r.spectral_uniqueness <= 1.0


def test_conditional_sensitivity_increases_with_noise() -> None:
    model = _model()
    levels = (0.0, 0.04, 0.08)
    cs = conditional_sensitivity(model, layers=[0], experts=[0], noise_levels=levels)
    errs = {p.upstream_noise: p.reconstruction_error for p in cs.rows}
    assert errs[0.0] <= errs[0.04] <= errs[0.08]  # monotonic in corrupted upstream state


def test_routing_consistency_identity_passes() -> None:
    model = _model()
    corpus = _corpus(model, n=4)
    rep = routing_consistency(model, model, corpus)
    assert rep.passed
    assert rep.mean_topk_agreement == 1.0


def test_global_bit_maps_respect_budget_and_bpw_bounds() -> None:
    model = _model()
    maps = enumerate_global_bit_maps(model, budgets=(0.001, 0.002))
    for budget, bm in maps.items():
        assert bm.total_bytes <= budget * 1024**3
        for a in bm.assignments:
            assert 3.0 <= a.bpw <= 4.0
            assert a.memory_bytes >= 0.0


def test_quant_interaction_prediction_has_certificate() -> None:
    model = _model()
    qi = fit_quant_interaction(model, sample_layers=[0], sample_experts=[0, 1])
    assert len(qi.per_component_error) == 2
    pred = predict_global_error(qi, {"0:0": 0.1, "0:1": 0.2})
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.additive_error == 0.3
    assert pred.predicted_total_error >= pred.additive_error


def test_refiner_fixed_storage_no_format_change() -> None:
    import random as _r

    rng = _r.Random(1)
    rows = [[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(4)]
    results = refine_expert_tensors({"gate": rows, "up": rows, "down": rows}, bits=3)
    for res in results:
        assert res.bits == 3
        assert "no format change" in res.note


def test_residual_correction_recommends_option() -> None:
    model = _model()
    plan = residual_correction_plan(model, layer=0, expert=0)
    kinds = {o.kind for o in plan.options}
    assert kinds == {"+bpw", "residual_correction", "nvfp4_fp8"}
    assert plan.recommended in kinds


def test_nvfp4_suitability_rows_and_recovery_flag() -> None:
    model = _model()
    corpus = _corpus(model, n=4)
    rep = nvfp4_suitability(model, corpus, layers=[0], experts=[0])
    assert len(rep.rows) == 1
    row = rep.rows[0]
    assert row.accepted in (True, False)
    assert row.recovery_kind in ("none", "qad", "cka_qad")


def test_kv_ledger_recommends_with_headroom_and_context() -> None:
    led = MemoryLedger(
        rank="node_a",
        physical_bytes=128 * 1024**3,
        weights_bytes=100 * 1024**3,
        safety_reserve_bytes=5 * 1024**3,
    )
    r = plan_kv_budget(led, arch_hidden=128, n_layers=2, context_target_tokens=32000)
    assert led.free_bytes > 0
    assert r.recommended_format in {opt.format for opt in r.options}
    assert r.options[0].format == "fp8"


def test_structural_fallback_never_reduces_insufficient() -> None:
    model = _model()
    widths = {(li, e): model.mid for li in range(len(model.layers)) for e in range(model.n_exp)}
    low = CapacityCoverage(capacity_id="0:0", meaningful_observations=2)
    gate = EvidenceGate()
    plans = structural_fallback_plans(
        widths,
        coverage={"0:0": low},
        gate=gate,
        reductions=(0.05, 0.20),
    )
    for p in plans:
        # insufficient-evidence expert never reduced
        assert p.retained_channels[(0, 0)] == model.mid
        assert "0:0" in p.blocked_capacity
        assert p.preserved_routing_destinations


def test_candidate_graph_immutable_provenance() -> None:
    g = CandidateGraph(source_teacher_id="t")
    g.add(
        CandidateNode(
            candidate_id="t",
            name="teacher",
            stage=CandidateStage.P0_REFERENCE,
            predicted=False,
            deployed=True,
        )
    )
    g.add(
        CandidateNode(
            candidate_id="c1",
            parent_ids=["t"],
            name="EXL3",
            stage=CandidateStage.P4_EXL3,
            predicted=True,
        )
    )
    # derived from deployable parent as predicted is allowed, but a measured
    # (non-predicted) candidate cannot derive from a deployable parent
    try:
        g.add(
            CandidateNode(
                candidate_id="c2",
                parent_ids=["t"],
                name="measured",
                stage=CandidateStage.P8_MATERIALIZED_BENCHMARK,
                predicted=False,
            )
        )
        raise AssertionError("expected CandidateGraphError for measured-from-deployable")
    except CandidateGraphError:
        pass
    root = g.root("c1")
    assert root.candidate_id == "t"
    assert len(g.measured()) == 1  # only teacher is measured
    assert all(len(node.parent_ids) <= 1 for node in g.nodes.values())


def test_evidence_gate_blocks_insufficient() -> None:
    gate = EvidenceGate(thresholds=CoverageThresholds())
    low = CapacityCoverage(capacity_id="L0:E2", meaningful_observations=2)
    ok = gate.allow(low)
    assert not ok[0]
    assert "insufficient_evidence" in ok[1]
    good = CapacityCoverage(capacity_id="L0:E1", meaningful_observations=300)
    assert gate.allow(good)[0]
