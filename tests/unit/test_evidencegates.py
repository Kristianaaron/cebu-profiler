"""Phase 7: eval + Pareto evidence gates (measured only for benchmarked)."""

from model_atlas.evidencegates import FrontierRecorder, MeasuredGateStatus
from model_atlas.schemas.evidence import EvidenceKind


def test_gate_requires_full_chain_for_measured():
    g = MeasuredGateStatus(candidate_id="c1")
    assert g.is_measured() is False
    assert g.evidence_kind is EvidenceKind.PREDICTED

    g.materialized = True
    g.heldout_evaluated = True
    g.runtime_benchmarked = True
    g.compute()
    assert g.is_measured() is True
    assert g.evidence_kind is EvidenceKind.MEASURED


def test_gate_never_measured_partially():
    for md, he, rt in (
        (True, True, False),  # not runtime-benchmarked
        (True, False, True),  # not held-out evaluated
        (False, True, True),  # not materialized
        (True, True, True),  # full
    ):
        g = MeasuredGateStatus(
            candidate_id="x", materialized=md, heldout_evaluated=he, runtime_benchmarked=rt
        )
        got = g.is_measured()
        assert got == (md and he and rt)
        if not got:
            assert g.evidence_kind is EvidenceKind.PREDICTED


def test_frontier_recorder_separates_measured_and_predicted():
    r = FrontierRecorder()
    kind = r.add_candidate(
        "predicted-1",
        quality=0.99,
        resident_gib=214.0,
        decode_tps=21.0,
        context_tokens=256000,
        materialized=False,
        heldout_evaluated=False,
        runtime_benchmarked=False,
    )
    assert kind == EvidenceKind.PREDICTED.value
    assert len(r.measured_frontier()) == 0
    assert len(r.predicted_frontier()) == 1

    kind2 = r.add_candidate(
        "measured-1",
        quality=0.995,
        resident_gib=196.0,
        decode_tps=26.0,
        context_tokens=384000,
        materialized=True,
        heldout_evaluated=True,
        runtime_benchmarked=True,
        provenance="bench:GLM-5.2-NVFP4-2node window 2026-08-14",
    )
    assert kind2 == EvidenceKind.MEASURED.value
    assert len(r.measured_frontier()) == 1
    assert len(r.predicted_frontier()) == 1
    assert r.measured_frontier()[0]["candidate_id"] == "measured-1"


def test_frontier_to_dict_separates():
    r = FrontierRecorder()
    r.add_candidate("a", quality=0.9, resident_gib=10.0, decode_tps=1.0, context_tokens=1000)
    r.add_candidate(
        "b",
        quality=0.95,
        resident_gib=12.0,
        decode_tps=1.2,
        context_tokens=2000,
        materialized=True,
        heldout_evaluated=True,
        runtime_benchmarked=True,
    )
    d = r.to_dict()
    assert len(d["measured"]) == 1
    assert len(d["predicted"]) == 1
