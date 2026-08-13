"""F23 tests: v3 system modules — Pareto engine, corpus-semantic bidirectional,
v3 pipeline orchestrator, and output-contract additions."""

import json

from model_atlas.analysis import build_corpus_semantic_map, project_corpus_delta
from model_atlas.atlas.output_layout import ATLAS_RUN_FILES, expected_run_files
from model_atlas.atlas.reap import make_synthetic_corpus
from model_atlas.atlas.runtime import build_mini_moe
from model_atlas.atlas.v3_pipeline import run_v3_pipeline, v3_run_to_jsonable
from model_atlas.experiments.pareto_v3 import FrontierPoint, restrict_frontier
from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.coverage import EvidenceGate
from model_atlas.schemas.evidence import EvidenceLevel

ARCH = get_registry().get("k3-mini")


def _model():
    return build_mini_moe(ARCH, seed=0)


def _corpus(model, n=6):
    return make_synthetic_corpus(
        n_samples=n, seq_len=4, vocab=model.arch.vocabulary_size or 1000, seed=0
    )[0]


def test_pareto_frontier_deterministic_and_dominance() -> None:
    pts = [
        FrontierPoint(candidate_id="A", values={"quality": 0.99, "resident_gib": 214.0}),
        FrontierPoint(candidate_id="B", values={"quality": 0.995, "resident_gib": 196.0}),
        FrontierPoint(candidate_id="C", values={"quality": 0.96, "resident_gib": 150.0}),
        FrontierPoint(candidate_id="D", values={"quality": 0.96, "resident_gib": 140.0}),
    ]
    r1 = restrict_frontier(pts)
    r2 = restrict_frontier(pts)
    assert r1.frontier_ids == r2.frontier_ids  # deterministic
    a_point = next(p for p in r1.points if p.candidate_id == "A")
    assert a_point.dominated_by  # A dominated by better candidates
    assert all(p.frontier for p in r1.points if p.candidate_id in r1.frontier_ids)
    # D and B dominate their worse copies; A alone is not on frontier
    assert "A" not in r1.frontier_ids


def test_knee_region_is_band_not_singleton() -> None:
    pts = [
        FrontierPoint(
            candidate_id=f"p{i}",
            values={"quality": 1.0 - i * 0.03, "resident_gib": 200 - i * 10},
        )
        for i in range(6)
    ]
    r = restrict_frontier(pts)
    assert len(r.frontier_ids) == 6  # all nondominated (strictly improving)
    # knee is a scored region, never a single unquestionable point
    assert len(r.knee_region) >= 2
    assert all(p.frontier for p in r.points)


def test_neighbor_deltas_exist_for_moves() -> None:
    pts = [
        FrontierPoint(candidate_id="A", values={"quality": 0.99, "resident_gib": 214.0}),
        FrontierPoint(candidate_id="B", values={"quality": 0.995, "resident_gib": 196.0}),
        FrontierPoint(candidate_id="C", values={"quality": 0.96, "resident_gib": 150.0}),
    ]
    r = restrict_frontier(pts)
    # every frontier point has neighbor deltas filled for fidelity/compact moves
    for cid in r.frontier_ids:
        assert r.neighbor_deltas[cid]  # non-empty for A/B (sorted frontier)
    for deltas in r.neighbor_deltas.values():
        for d in deltas:
            assert d.direction in ("fidelity", "compact")


def test_corpus_bidirectional_reports_and_delta_projection() -> None:
    model = _model()
    corpus = _corpus(model, n=9)
    rep = build_corpus_semantic_map(model, corpus, top_k=2, gate=EvidenceGate())
    assert len(rep.clusters) >= 2
    assert len(rep.cluster_expert_coverage) == len(rep.clusters) * len(model.layers) * model.n_exp
    # inverse link present
    some = rep.expert_activation[0]
    assert some.layer == 0 and some.expert == 0

    # project deltas onto clusters; regressions remain visible even mixed sign
    per_sample = {i: -0.1 if i < 3 else 0.05 for i in range(9)}
    rep2 = project_corpus_delta(rep, candidate_id="mk", per_sample_delta=per_sample)
    assert rep2.deltas
    # teacher-relative deltas may be negative (regression), bounded within [-1, 1]
    for d in rep2.deltas:
        assert -1.0 <= d.quality_delta <= 1.0


def test_v3_pipeline_runs_all_stages_and_serializes() -> None:
    model = _model()
    corpus = _corpus(model, n=4)
    r = run_v3_pipeline(model, corpus, seed=0)
    assert "corpus_semantic" in r.stages_run
    assert "spectral" in r.stages_run
    assert "shared_structure" in r.stages_run
    assert "global_bit_budget" in r.stages_run
    assert "nvfp4_suitability" in r.stages_run
    assert "routing_consistency" in r.stages_run
    assert "kv_budget" in r.stages_run
    assert "structural_fallback" in r.stages_run
    assert "pareto" in r.stages_run
    assert r.routing_consistency_passed
    j = json.dumps(v3_run_to_jsonable(r))
    assert j  # JSON-serializable (tuple keys normalized)


def test_output_contract_declares_v3_artifacts() -> None:
    v3_files = {
        "v3_run.json",
        "v3_candidate_graph.json",
        "v3_corpus_evidence.json",
        "shared_representation.json",
        "spectral_quality.json",
        "routing_consistency.json",
        "global_bit_budget.json",
        "kv_ledger.json",
        "pareto_frontier.json",
    }
    assert v3_files <= ATLAS_RUN_FILES
    enhanced = expected_run_files(EvidenceLevel.ENHANCED_ATLAS)
    assert "v3_run.json" in enhanced
    assert "kv_ledger.json" in enhanced
