"""Phase 3 round-3 review: torch scoring kernels + real-hook + measured-gate tests.

Covers D: causal ablation is a genuine baseline-vs-ablated-output scorer with
shape checks; RealActivationHook only measures with an explicit real-corpus run
id (offline replay never); TorchScoringResult rejects MEASURED with synthetic or
missing provenance.
"""

import math

import pytest

from model_atlas.schemas.evidence import EvidenceKind
from model_atlas.scoring.torch_scores import (
    RealActivationHook,
    TorchScoringResult,
    causal_ablation_scores,
    flexmoe_channel_ranking,
    grouped_taylor_surrogate,
    needs_for_real_scoring,
    tenp_importance,
)

torch = pytest.importorskip("torch")


def test_tenp_importance_matches_hand():
    down = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    z = torch.tensor([[1.0, 0.5], [1.0, 0.5]])
    s = tenp_importance(torch.zeros(1, 2), torch.zeros(1, 2), down, z)
    assert math.isclose(s[0], 1.0, rel_tol=1e-6)
    assert math.isclose(s[1], 1.0, rel_tol=1e-6)


def test_flexmoe_retains_all_experts():
    keep = flexmoe_channel_ranking({0: 0.1, 1: 0.9, 2: 0.5, 3: 0.7}, 4, 0.75)
    assert len(keep) == 3 and 1 in keep


def test_causal_ablation_genuine_diff_and_shape_check():
    b = torch.tensor([1.0, 1.0, 1.0])
    a = torch.tensor([1.0, 0.0, 0.5])  # ablated: channel1 fully, channel2 half
    out = causal_ablation_scores(b, a)
    assert abs(sum(out.values()) - 1.0) < 1e-6
    # channel1 (delta=1) > channel2 (delta=0.5) > channel0 (delta=0)
    assert out[1] > out[2] > out[0]
    # shape mismatch fails closed
    with pytest.raises(ValueError):
        causal_ablation_scores(torch.ones(2), torch.ones(3))
    # zero-delta must not fabricate nonzero contributions (epsilon must not add)
    z = causal_ablation_scores(torch.zeros(2), torch.zeros(2))
    assert all(abs(v) < 1e-9 for v in z.values())


def test_grouped_taylor_groups():
    out = grouped_taylor_surrogate(
        torch.zeros(1, 4), torch.zeros(1, 4), torch.eye(4), torch.ones(4, 4),
        group_size=2, lambda_=0.1,
    )
    assert set(out) == {0, 1, 2, 3}
    assert math.isclose(out[0], out[1], rel_tol=1e-6)


def test_real_hook_offline_replay_never_measured():
    hook = RealActivationHook(3, 0)
    hook.capture(torch.ones(2, 4))  # offline replay, no real run id
    assert hook.has_captured is True
    assert hook.is_measured() is False

    # marking AFTER a stale offline capture must NOT retroactively measure it
    hook.mark_real_corpus("run-abc")
    assert hook.is_measured() is False  # stale offline capture stays unmeasured

    # a NEW capture under an active run id is measurable
    hook2 = RealActivationHook(3, 0)
    hook2.mark_real_corpus("run-abc")
    hook2.capture(torch.ones(2, 4))
    assert hook2.is_measured() is True
    assert "run-abc" in hook2.evidence_provenance()


def test_real_hook_missing_run_id_raises():
    hook = RealActivationHook(0, 0)
    with pytest.raises(ValueError):
        hook.mark_real_corpus("")


def test_score_result_measured_rejected_for_synthetic():
    with pytest.raises(ValueError):
        TorchScoringResult(
            requirements=needs_for_real_scoring("bounded_cpu"),
            rows={(0, 0, 0): {"tenp": 1.0}},
            input_source="synthetic",
            evidence_kind=EvidenceKind.MEASURED,
        )


def test_score_result_measured_requires_provenance():
    with pytest.raises(ValueError):
        TorchScoringResult(
            requirements=needs_for_real_scoring("full_forward"),
            rows={(0, 0, 0): {"tenp": 1.0}},
            input_source="real_corpus_forward",
            evidence_kind=EvidenceKind.MEASURED,
            provenance="",
        )


def test_score_result_predicted_for_real_input_with_provenance_is_allowed():
    r = TorchScoringResult(
        requirements=needs_for_real_scoring("full_forward"),
        rows={(0, 0, 0): {"tenp": 1.0}},
        input_source="real_corpus_forward",
        evidence_kind=EvidenceKind.MEASURED,
        provenance="run:abc",
    )
    assert r.evidence_kind is EvidenceKind.MEASURED
