import pytest

from model_atlas.prune.channel_saliency import (
    aggregate_group_saliency,
    expert_saliency,
    neuron_saliency_from_router_norm,
)
from model_atlas.prune.kl_gate import KLGateError, decide_kl_gate
from model_atlas.prune.ranked_keeper import KeepMapError, select_keep_map

# ---------------------------------------------------------------------------
# channel saliency (decision C: channel-level primary + expert-level fallback)
# ---------------------------------------------------------------------------

def test_group_aggregation_is_mean_of_members() -> None:
    per_neuron = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    groups = aggregate_group_saliency(per_neuron, group=4)
    assert groups == [5.0, 13.0]
    assert aggregate_group_saliency([1.0] * 16, group=16) == [1.0]


def test_group_aggregation_rejects_non_multiple() -> None:
    with pytest.raises(ValueError):
        aggregate_group_saliency([1.0, 2.0, 3.0], group=2)


def test_group_aggregation_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        aggregate_group_saliency([1.0, float("nan"), 1.0], group=2)


def test_router_norm_saliency() -> None:
    sal = neuron_saliency_from_router_norm(2.0, [1.0, 3.0, 5.0])
    assert sal == [2.0, 6.0, 10.0]


def test_expert_saliency_fallback() -> None:
    assert expert_saliency([2.0, 4.0, 6.0]) == 4.0


# ---------------------------------------------------------------------------
# saliency-ranked keep-map selection (decision A)
# ---------------------------------------------------------------------------

def _saliency_map() -> dict[tuple[int, int], list[float]]:
    # 1 layer, 1 expert, full=32 -> 2 sixteen-channel groups.
    return {(0, 0): [5.0, 20.0]}  # group 1 far more salient


def test_selection_keeps_highest_saliency_group() -> None:
    keep = select_keep_map(
        _saliency_map(), width=16, full=32, sparse_layers=[0], n_exp=1
    )
    # group 1 most salient -> channels 16..31 retained, ascending
    assert keep == {(0, 0): list(range(16, 32))}


def test_selection_is_16_aligned_ascending_union() -> None:
    keep = select_keep_map(
        {(0, 0): [20.0, 5.0, 10.0, 40.0]},
        width=32, full=64, sparse_layers=[0], n_exp=1,
    )
    channels = keep[(0, 0)]
    # groups ranked by saliency desc: 3,0,2,1 -> keep top-2 = groups 3 and 0,
    # ascending order -> channels 0..15 and 48..63
    assert channels == list(range(16)) + list(range(48, 64))
    assert len(channels) == 32
    assert sorted(channels) == channels


def test_selection_respects_width() -> None:
    keep = select_keep_map(
        {(0, 0): [20.0, 5.0, 10.0]}, width=32, full=48, sparse_layers=[0], n_exp=1
    )
    assert len(keep[(0, 0)]) == 32  # exactly `width` channels retained


def test_selection_fails_closed_on_missing_coverage() -> None:
    with pytest.raises(KeepMapError, match="complete coverage required"):
        select_keep_map({(0, 1): [1.0, 2.0]}, width=16, full=32,
                        sparse_layers=[0], n_exp=1)


def test_selection_rejects_unaligned_width() -> None:
    with pytest.raises(KeepMapError, match="multiple of group"):
        select_keep_map({(0, 0): [1.0, 2.0]}, width=17, full=32,
                        sparse_layers=[0], n_exp=1)


def test_selection_rejects_saliency_length_mismatch() -> None:
    with pytest.raises(KeepMapError, match="group saliency length"):
        select_keep_map({(0, 0): [1.0, 2.0, 100.0]}, width=16, full=32,
                        sparse_layers=[0], n_exp=1)


def test_selection_tie_breaks_deterministically() -> None:
    a = select_keep_map({(0, 0): [1.0, 1.0]}, width=16, full=32,
                        sparse_layers=[0], n_exp=1)
    b = select_keep_map({(0, 0): [1.0, 1.0]}, width=16, full=32,
                        sparse_layers=[0], n_exp=1)
    assert a == b  # equal saliency -> lowest group index wins


# ---------------------------------------------------------------------------
# KLD gate (decision A: the acceptance rule)
# ---------------------------------------------------------------------------

def test_kl_gate_accepts_when_within_budget() -> None:
    r = decide_kl_gate(
        mean_kld=0.02, worst_domain_kld=0.08, p99_kld=0.10, min_cka=0.95,
        mean_budget=0.05, worst_domain_budget=0.10, p99_budget=0.12,
        cka_floor=0.9,
    )
    assert r.accepted is True
    assert r.failures == ()


def test_kl_gate_rejects_when_mean_kld_exceeds() -> None:
    r = decide_kl_gate(
        mean_kld=0.12, worst_domain_kld=0.05, p99_kld=0.06, min_cka=0.99,
        mean_budget=0.05, worst_domain_budget=0.10, p99_budget=0.12,
        cka_floor=0.9,
    )
    assert r.accepted is False
    assert any("mean KLD" in f for f in r.failures)


def test_kl_gate_rejects_when_min_cka_below_floor() -> None:
    r = decide_kl_gate(
        mean_kld=0.01, worst_domain_kld=0.02, p99_kld=0.03, min_cka=0.5,
        mean_budget=0.05, worst_domain_budget=0.10, p99_budget=0.12,
        cka_floor=0.9,
    )
    assert r.accepted is False
    assert any("min CKA" in f for f in r.failures)


def test_kl_gate_lists_all_failures() -> None:
    r = decide_kl_gate(
        mean_kld=0.2, worst_domain_kld=0.2, p99_kld=0.2, min_cka=0.1,
        mean_budget=0.05, worst_domain_budget=0.10, p99_budget=0.12,
        cka_floor=0.9,
    )
    assert len(r.failures) == 4


def test_kl_gate_rejects_non_finite() -> None:
    with pytest.raises(KLGateError):
        decide_kl_gate(mean_kld=float("nan"), worst_domain_kld=0.0, p99_kld=0.0,
                       min_cka=1.0, mean_budget=0.1, worst_domain_budget=0.1,
                       p99_budget=0.1, cka_floor=0.9)
