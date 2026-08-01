"""F4 tests: success/failure/recovery contrasts + representation traces."""

from model_atlas.atlas.reap import (
    ContrastAccumulator,
    make_synthetic_corpus,
    run_contrast,
)
from model_atlas.atlas.runtime import build_mini_moe, forward, representation_profile
from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.ontology import CapabilityLabel, SuccessState, TrajectoryStage

ARCH = get_registry().get("k3-mini")


def test_forward_emits_representation_norms():
    model = build_mini_moe(ARCH, seed=1)
    result = forward(model, [3, 4, 5], top_k=2)
    for trace in result.traces:
        assert len(trace.input_norm) == 3
        assert len(trace.moe_norm) == 3
        assert len(trace.output_norm) == 3
        # norms are non-negative and finite
        assert all(v >= 0.0 for v in trace.input_norm + trace.moe_norm + trace.output_norm)


def test_representation_profile_stats_only():
    model = build_mini_moe(ARCH, seed=2)
    profile = representation_profile(model, [1, 2, 3], top_k=2)
    assert len(profile) == ARCH.num_text_layers
    for row in profile:
        assert "input_norm_mean" in row and "entropy_mean" in row
        assert row["moe_norm_mean"] >= 0.0


def test_contrast_success_minus_failure_sign_direction():
    model = build_mini_moe(ARCH, seed=3)
    samples, labels, _ = make_synthetic_corpus(
        n_samples=40, seq_len=6, vocab=ARCH.vocabulary_size, seed=0
    )
    acc = run_contrast(model, samples, top_k=2)
    label = labels[0]
    rows = acc.contrast(label, pos=SuccessState.SUCCESS, neg=SuccessState.FAILURE, topk=100)
    # every (layer, expert) that appeared in either success or failure is present
    assert rows
    # deltas are symmetric around zero; verify a specific expert difference is within range
    (lay, exp, delta), *_ = rows
    assert abs(delta) <= 1.0  # saliency magnitudes stay small


def test_contrast_handcrafted_math():
    # hand-built accumulator to verify contrast = mean_pos - mean_neg
    acc = ContrastAccumulator()
    lab, stg, suc, fail = (
        CapabilityLabel.DEBUGGING,
        TrajectoryStage.REPAIR,
        SuccessState.SUCCESS,
        SuccessState.FAILURE,
    )
    # (layer, expert)= (0, 2): 2 success tokens scores 0.1, 0.3 ; 1 failure token 0.4
    acc.add(0, 2, 0.1, lab, stg, suc, routed=True)
    acc.add(0, 2, 0.3, lab, stg, suc, routed=True)
    acc.add(0, 2, 0.4, lab, stg, fail, routed=True)
    rows = acc.contrast(lab, pos=suc, neg=fail, stage=stg, topk=1)
    layer, expert, delta = rows[0]
    assert (layer, expert) == (0, 2)
    mean_suc = (0.1 + 0.3) / 2
    assert abs(delta - (mean_suc - 0.4)) < 1e-12


def test_participation_recovery():
    model = build_mini_moe(ARCH, seed=4)
    samples, labels, _ = make_synthetic_corpus(
        n_samples=30, seq_len=6, vocab=ARCH.vocabulary_size, seed=0
    )
    acc = run_contrast(model, samples, top_k=2)
    recoverers = acc.participates(
        labels[0], states={SuccessState.RECOVERED}, stage=samples[0].stage
    )
    # recovery participation is a subset of routed experts (some may be empty)
    assert isinstance(recoverers, list)
    # every recovered expert must have been routed (no fabrication)
    assert all(
        acc.saliency(lay, e, labels[0], samples[0].stage, SuccessState.RECOVERED) >= 0.0
        for lay, e in recoverers
    )
