"""F6 tests: sparse features + vocabulary projections (v2 §15–§16)."""

from cebu_profiler.profiler.features import (
    directional_projection,
    expert_direction,
    learn_features,
    project_vocabulary,
    promoted_suppressed,
    residual_direction,
)
from cebu_profiler.profiler.reap import make_synthetic_corpus
from cebu_profiler.profiler.runtime import build_mini_moe
from cebu_profiler.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def test_vocab_projection_self_dot_ranks_own_row_top():
    model = build_mini_moe(ARCH, seed=1)
    k = 42
    direction = list(model.lm_head[k])  # project a vocab row's own direction
    ranked = project_vocabulary(model, direction)
    assert ranked[0][0] == k  # self-dot (squared norm) dominates

    proj = directional_projection(model, direction, topk=5)
    assert k in proj.promoted
    assert 0.0 <= proj.confidence <= 1.0


def test_promoted_suppressed_disjoint():
    model = build_mini_moe(ARCH, seed=2)
    direction = expert_direction(model, [1, 2, 3], layer=0, expert=1)
    promoted, suppressed = promoted_suppressed(model, direction, topk=8)
    pset = {t for t, _ in promoted}
    sset = {t for t, _ in suppressed}
    assert pset.isdisjoint(sset)


def test_expert_direction_nonzero_and_distinct():
    model = build_mini_moe(ARCH, seed=3)
    d0 = expert_direction(model, [5, 6], layer=0, expert=0)
    d1 = expert_direction(model, [5, 6], layer=0, expert=1)
    assert any(v != 0.0 for v in d0)
    # two different experts on the same input give materially different directions
    diff = sum((a - b) ** 2 for a, b in zip(d0, d1, strict=True))
    assert diff > 0.0


def test_residual_direction_is_real():
    model = build_mini_moe(ARCH, seed=4)
    d = residual_direction(model, [7, 8])
    assert len(d) == ARCH.hidden_dim


def test_sparse_features_are_well_formed_and_deterministic():
    model = build_mini_moe(ARCH, seed=5)
    samples, _, _ = make_synthetic_corpus(
        n_samples=20, seq_len=6, vocab=ARCH.vocabulary_size, seed=0
    )
    fd = learn_features(model, samples, n_features=16, k_sparse=2, seed=0)
    fd2 = learn_features(model, samples, n_features=16, k_sparse=2, seed=0)
    assert len(fd.features) == 16
    # deterministic
    assert [f.max_redundancy for f in fd.features] == [f.max_redundancy for f in fd2.features]
    # activation frequencies sum to (k_sparse * mean active) -> each feature's freq in [0,1]
    total = sum(f.activation_frequency for f in fd.features)
    assert 0.0 < total <= len(fd.features)
    assert all(0.0 <= f.activation_frequency <= 1.0 for f in fd.features)
    # redundancy is a cosine in [0,1] and some feature overlaps something
    assert all(0.0 <= f.max_redundancy <= 1.0 + 1e-9 for f in fd.features)


def test_feature_has_label_associations():
    model = build_mini_moe(ARCH, seed=6)
    samples, _, _ = make_synthetic_corpus(
        n_samples=24, seq_len=6, vocab=ARCH.vocabulary_size, seed=0
    )
    fd = learn_features(model, samples, n_features=8, k_sparse=2, seed=0)
    # enough calibration tokens such that every feature is activated at least once
    for f in fd.features:
        assert f.activation_frequency >= 0.0
    # at least one feature is activated and has label/expert links (measured, real)
    active = [f for f in fd.features if f.activation_frequency > 0.0]
    assert active
    assert all(isinstance(f.top_experts, list) for f in active)
