"""F3 tests: mini-MoE forward, routing validity, determinism, REAP aggregation."""

import math

from cebu_profiler.profiler.reap import (
    make_synthetic_corpus,
    run_calibration,
)
from cebu_profiler.profiler.runtime import build_mini_moe, forward
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.ontology import CapabilityLabel, TrajectoryStage

ARCH = get_registry().get("k3-mini")


def test_forward_routing_is_valid_probability():
    model = build_mini_moe(ARCH, seed=1)
    result = forward(model, [5, 42, 900], top_k=2)
    for trace in result.traces:
        assert trace.layer == 0 or trace.layer >= 0
        for p in trace.probs_all:
            assert abs(sum(p) - 1.0) < 1e-9
            assert all(0.0 <= v <= 1.0 for v in p)
        for e in trace.entropy:
            assert 0.0 <= e <= math.log(model.n_exp) + 1e-9
        # top-k selection count and probabilities over selected sum to ~1
        for sel, selp in zip(trace.topk_ids, trace.topk_probs, strict=True):
            assert len(sel) == 2
            assert abs(sum(selp) - 1.0) < 1e-9


def test_forward_deterministic():
    a = forward(build_mini_moe(ARCH, seed=7), [1, 2, 3], top_k=2)
    b = forward(build_mini_moe(ARCH, seed=7), [1, 2, 3], top_k=2)
    assert a.traces[0].router_weighted == b.traces[0].router_weighted
    assert a.logits == b.logits


def test_model_produces_output_logits():
    model = build_mini_moe(ARCH, seed=2)
    result = forward(model, [0, 1], top_k=2)
    assert len(result.logits) == ARCH.vocabulary_size
    assert any(v != 0.0 for v in result.logits)


def test_reap_saliency_measured_and_positive():
    model = build_mini_moe(ARCH, seed=3)
    samples, labels, stages = make_synthetic_corpus(
        n_samples=24, seq_len=8, vocab=ARCH.vocabulary_size, seed=0
    )
    acc = run_calibration(model, samples, top_k=2)
    # aggregate mean across experts/layers for one label is positive (some expert routed)
    label = CapabilityLabel.CODE_GENERATION
    stage = TrajectoryStage.PLAN
    ranked = acc.rank(label, stage=stage, topk=100)
    assert ranked  # at least one scored pair
    # scores are non-negative (p * norm >= 0)
    assert all(s >= 0.0 for _, _, s in ranked)


def test_reap_score_equals_mean_of_router_weighted():
    # verification: rank score matches recomputed mean from accumulator
    model = build_mini_moe(ARCH, seed=3)
    samples, _, _ = make_synthetic_corpus(
        n_samples=4, seq_len=2, vocab=ARCH.vocabulary_size, seed=0
    )
    acc = run_calibration(model, samples, top_k=2)
    top = acc.rank(samples[0].labels[0], stage=samples[0].stage, topk=1)[0]
    layer, expert, score = top
    recomputed = acc.mean(layer, expert, samples[0].labels[0], samples[0].stage)
    assert abs(score - recomputed) < 1e-12


def test_rank_without_stage_has_unique_layer_expert():
    model = build_mini_moe(ARCH, seed=3)
    samples, labels, _ = make_synthetic_corpus(
        n_samples=30, seq_len=6, vocab=ARCH.vocabulary_size, seed=0
    )
    acc = run_calibration(model, samples, top_k=2)
    for label in labels[:3]:
        rows = acc.rank(label, stage=None, topk=1000)
        pairs = [(lay, e) for lay, e, _ in rows]
        assert len(pairs) == len(set(pairs)), f"duplicate (layer, expert) for {label}"


def test_routing_frequency_between_0_and_1():
    model = build_mini_moe(ARCH, seed=5)
    samples, labels, stages = make_synthetic_corpus(
        n_samples=12, seq_len=4, vocab=ARCH.vocabulary_size, seed=0
    )
    acc = run_calibration(model, samples, top_k=2)
    label = labels[0]
    stage = stages[0]
    # every expert should have a defined frequency after calibration
    for layer in range(ARCH.num_text_layers):
        for e in range(ARCH.moe.num_routed_experts):
            f = acc.frequency(layer, e, label, stage)
            assert 0.0 <= f <= 1.0
