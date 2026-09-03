"""F18 tests: blueprint phase-2 offline modules (§8.1/8.3/8.4, §10, Priority 4#5, §17 Control C)."""

import random

from cebu_profiler.experiments import compare_controls
from cebu_profiler.planning.optimizer import rate_distortion_manifest
from cebu_profiler.profiler.compress import run_compression_pipeline
from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, build_mini_moe
from cebu_profiler.profiler.traces import trace_records
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.manifest import validate_manifest
from cebu_profiler.schemas.ontology import CapabilityLabel, TrajectoryStage
from cebu_profiler.scoring.quant_sensitivity import (
    SensitivityReport,
    expert_quant_sensitivity,
    recommend_bpw,
    sensitivity_report,
)
from cebu_profiler.scoring.redundancy import (
    RedundancyScorer,
    channel_kvalue,
    channel_uniqueness,
    expert_uniqueness,
)
from cebu_profiler.scoring.semantic import SemanticScorer, expert_semantic_score, semantic_map

ARCH = get_registry().get("k3-mini")


def _samples_n(model: MiniMoE, seed: int = 0, n: int = 20) -> list[CalibrationSample]:
    rng = random.Random(seed)
    vocab = model.arch.vocabulary_size
    assert vocab is not None
    return [
        CalibrationSample(
            tokens=[rng.randrange(vocab) for _ in range(12)],
            labels=[CapabilityLabel.FACTUAL_KNOWLEDGE],
            stage=TrajectoryStage.UNDERSTAND,
        )
        for _ in range(n)
    ]


# --- §10 trace records -------------------------------------------------------


def test_trace_records_compose_all_families() -> None:
    model = build_mini_moe(ARCH, seed=1)
    recs = trace_records(model, _samples_n(model))
    assert recs.router  # per sample/token/layer
    assert recs.experts
    assert recs.channels
    r = recs.router[0]
    assert len(r.selected_experts) == len(r.gate_weights)
    assert r.routing_entropy >= 0.0
    assert recs.channels[0].activation_frequency >= 0.0


# --- §8.1 semantic -----------------------------------------------------------


def test_semantic_map_and_score() -> None:
    model = build_mini_moe(ARCH, seed=1)
    m = semantic_map(model, _samples_n(model))
    assert m.associations
    assert m.experts_all()
    sem = expert_semantic_score(model, _samples_n(model))
    assert sem and all(v >= 0.0 for v in sem.values())
    rows = SemanticScorer(model, _samples_n(model)).finalize().rows
    assert any(r.semantic is not None for r in rows)


# --- §8.3 redundancy / KEEP_VALUE -------------------------------------------


def test_uniqueness_bounds_and_kvalue() -> None:
    model = build_mini_moe(ARCH, seed=1)
    u = channel_uniqueness(model)
    assert u and all(0.0 <= v <= 1.0 for v in u.values())
    assert expert_uniqueness(model)  # per-expert mean
    imp = {(layer, e, c): 1.0 for (layer, e, c) in u}
    kv = channel_kvalue(imp, u)  # no causal/stability -> penalty-free
    assert kv == {(layer, e, c): u[(layer, e, c)] for (layer, e, c) in u}
    rows = RedundancyScorer(model, imp).finalize().rows
    assert any(r.uniqueness is not None and r.kvalue is not None for r in rows)


# --- §8.4 quantization sensitivity ------------------------------------------


def test_quant_sensitivity_recommends_levels() -> None:
    model = build_mini_moe(ARCH, seed=1)
    sens = expert_quant_sensitivity(model)
    assert sens and all(v >= 0.0 for v in sens.values())
    bpw = recommend_bpw(sens)
    levels = {4.0, 3.5, 3.25, 3.0}
    assert set(bpw.values()).issubset(levels)
    rep = sensitivity_report(model)
    assert isinstance(rep, SensitivityReport)
    assert set(rep.bpw.values()).issubset(levels)


# --- Priority 4 #5 rate-distortion optimizer --------------------------------


def test_rate_distortion_manifest_respects_budget() -> None:
    model = build_mini_moe(ARCH, seed=1)
    from cebu_profiler.experiments.controls import channel_importance

    imp = channel_importance(model, _samples_n(model))
    n_slots = len(model.layers) * model.n_exp
    total = n_slots * model.mid
    plan = rate_distortion_manifest(
        model, imp, budget_channels=int(0.5 * total), allowed_widths=[16, 12, 8, 4]
    )
    assert plan.kept_channels <= int(0.5 * total)
    assert plan.kept_channels >= n_slots
    assert validate_manifest(plan.manifest).ok
    assert plan.estimated_params > 0


# --- §17 Control C -----------------------------------------------------------


def test_control_c_compare_returns_all_arms() -> None:
    model = build_mini_moe(ARCH, seed=1)
    n_slots = len(model.layers) * model.n_exp
    reports = compare_controls(
        model,
        _samples_n(model, 0),
        _samples_n(model, 2, n=20),
        budget=int(0.6 * n_slots * model.mid),
    )
    assert set(reports) == {"uniform", "control_c", "hetero"}
    for rep in reports.values():
        assert rep.mean_hidden_drift >= 0.0


# --- pipeline integration ----------------------------------------------------


def test_pipeline_enriches_manifest_with_phase2_views() -> None:
    model = build_mini_moe(ARCH, seed=1)
    manifest, validation = run_compression_pipeline(model, _samples_n(model), n_stability_runs=3)
    assert validation.ok, validation.errors
    any_with = []
    for lp in manifest.layers.values():
        for plan in lp.experts.values():
            any_with.append(plan.scores.uniqueness is not None)
            assert plan.quant_recommendation.bpw in {4.0, 3.5, 3.25, 3.0}
    assert any(any_with)  # uniqueness propagated to at least one expert
