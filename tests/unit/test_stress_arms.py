"""Tests for the causal prune-arm stress matrix (stress/arms)."""

from __future__ import annotations

import math

from cebu_profiler.profiler.reap import SaliencyAccumulator
from cebu_profiler.profiler.runtime import build_mini_moe, forward
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.ontology import CapabilityLabel, TrajectoryStage
from cebu_profiler.stress.arms import matrix_payload, run_prune_arms


def _mini_model_and_acc():
    spec = get_registry().get("k3-mini")
    model = build_mini_moe(spec, seed=0)
    acc = SaliencyAccumulator()
    # route a handful of synthetic probes through the accumulator
    label, stage = CapabilityLabel.PLANNING, TrajectoryStage.EXECUTE
    for i in range(4):
        tokens = [(i * 7 + j) % (spec.vocabulary_size or 1000) for j in range(4)]
        res = forward(model, tokens)
        for trace in res.traces:
            for t, sel in enumerate(trace.topk_ids):
                for e, p in zip(sel, trace.topk_probs[t], strict=True):
                    acc.add(
                        trace.layer,
                        e,
                        p * trace.expert_norm[t][e],
                        label,
                        stage,
                        routed=True,
                    )
    return model, acc


def test_identity_arm_is_exact_noop():
    model, acc = _mini_model_and_acc()
    probes = [[1, 2, 3], [4, 5, 6]]
    arms = run_prune_arms(model, acc, probes, fractions=(0.02,))
    identity = [a for a in arms if a.arm == "identity"]
    assert len(identity) == 1
    assert identity[0].sequence_exact == 1.0
    assert identity[0].mean_logit_kl == 0.0
    assert identity[0].output_cosine_mean == 1.0


def test_matrix_has_all_arms_and_controls():
    model, acc = _mini_model_and_acc()
    probes = [[7, 8, 9, 10]]
    arms = run_prune_arms(model, acc, probes, fractions=(0.02, 0.05))
    names = {a.arm for a in arms}
    assert names == {"identity", "low_reap", "random", "high_reap"}
    # one identity + 3 arms x 2 fractions
    assert len(arms) == 1 + 3 * 2
    low = sorted(a.fraction for a in arms if a.arm == "low_reap")
    assert low == [0.02, 0.05]


def test_low_reap_damage_le_high_reap_damage():
    """Averaged over enough probes, pruning the *lowest* scores costs no more
    than pruning the highest (single tiny probes can invert on noise)."""
    model, acc = _mini_model_and_acc()
    # 24 probes across the vocab so per-arm averaging reflects the ranking signal
    probes = [[(i * 5 + j) % 1000 for j in range(4)] for i in range(24)]
    arms = run_prune_arms(model, acc, probes, fractions=(0.10,))
    low = next(a for a in arms if a.arm == "low_reap")
    high = next(a for a in arms if a.arm == "high_reap")
    assert low.mean_logit_kl <= high.mean_logit_kl + 1e-6
    assert low.output_cosine_mean >= high.output_cosine_mean - 1e-6


def test_payload_carries_limitations_and_is_jsonable():
    model, acc = _mini_model_and_acc()
    arms = run_prune_arms(model, acc, [[21, 22]], fractions=(0.02,))
    bundle = matrix_payload(arms)
    assert bundle["limits"]["method"].startswith("frozen-model route ablation")
    assert isinstance(__import__("json").dumps(bundle), str)
    # every arm exposes the evidence fields
    for arm in bundle["arms"]:
        assert {"arm", "fraction", "mean_logit_kl", "sequence_exact"} <= set(arm)


def test_removed_slot_fraction_matches_scores_universe():
    model, acc = _mini_model_and_acc()
    probes = [[23, 24, 25]]
    arms = run_prune_arms(model, acc, probes, fractions=(0.05,))
    scored = {(ly, e) for ly in range(model.arch.num_text_layers) for e in range(model.n_exp)}
    for a in arms:
        if a.arm != "identity":
            assert a.removed_slot_fraction <= 0.06
            assert set(a.removed) <= scored or not a.removed


def test_all_damage_metrics_bounded():
    model, acc = _mini_model_and_acc()
    arms = run_prune_arms(model, acc, [[30, 31]], fractions=(0.10,))
    for a in arms:
        assert a.mean_logit_kl >= 0.0
        assert -1.0 <= a.output_cosine_mean <= 1.0 + 1e-9
        assert 0.0 <= a.sequence_exact <= 1.0
        assert math.isfinite(a.mean_logit_kl)
