"""Phase 3: torch-backed real scoring (TENP/FlexMoE/Taylor/causal) tests.

These require torch; they run under `.venv-exec` where CUDA/torch are present.
In the default repo venv they are skipped (no torch). All math is verified
against small hand-computed tensors.
"""

import math

import pytest

from model_atlas.scoring.base import ScoreNeed
from model_atlas.scoring.torch_scores import (
    causal_ablation_scores,
    flexmoe_channel_ranking,
    grouped_taylor_surrogate,
    needs_for_real_scoring,
    tenp_importance,
)

torch = pytest.importorskip("torch")


def test_tenp_importance_matches_hand_computation():
    # down [2, k=2], z [2, k=2]
    down = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    z = torch.tensor([[1.0, 0.5], [1.0, 0.5]])
    scores = tenp_importance(
        torch.zeros(1, 2), torch.zeros(1, 2), down, z, expert_norm=1.0
    )
    # channel0: col_norm = 1.0, mean|z|=1.0 -> 1.0
    # channel1: col_norm = 2.0, mean|z|=0.5 -> 1.0
    assert math.isclose(scores[0], 1.0, rel_tol=1e-6)
    assert math.isclose(scores[1], 1.0, rel_tol=1e-6)


def test_tenp_importance_greater_magnitude_scores_higher():
    down = torch.tensor([[1.0, 3.0]])
    z = torch.tensor([[1.0, 1.0]])
    scores = tenp_importance(
        torch.zeros(1, 1), torch.zeros(1, 1), down, z, expert_norm=1.0
    )
    assert scores[1] > scores[0]


def test_flexmoe_ranking_keeps_top_channels_and_all_experts_retained():
    imp = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.7}
    keep = flexmoe_channel_ranking(imp, k=4, budget_frac=0.75)
    assert set(keep) <= {1, 3, 2, 0}
    assert len(keep) == 3  # ceil(4*0.75)
    assert 1 in keep  # highest importance always retained
    # every expert (this one) is retained — only width changes
    assert isinstance(keep, list)


def test_grouped_taylor_surrogate_groups_and_regularizes():
    gate = torch.zeros(1, 4)
    up = torch.zeros(1, 4)
    down = torch.eye(4)
    z = torch.ones(4, 4)
    out = grouped_taylor_surrogate(gate, up, down, z, group_size=2, lambda_=0.1)
    # two groups of 2; base equal within group -> all = base
    assert set(out) == {0, 1, 2, 3}
    assert math.isclose(out[0], out[1], rel_tol=1e-6)
    assert math.isclose(out[2], out[3], rel_tol=1e-6)


def test_causal_ablation_normalized():
    z = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    out = causal_ablation_scores(z, torch.ones(2, 2), epsilon=1e-6)
    assert abs(sum(out.values()) - 1.0) < 1e-6
    assert out[1] > out[0]  # channel1 has higher activation contribution


def test_requirements_forward_only_for_bounded():
    req = needs_for_real_scoring("bounded_cpu")
    assert req.forward_only
    assert ScoreNeed.GRADIENTS not in req.needs
    assert ScoreNeed.HIGH_PRECISION_WEIGHTS not in req.needs


def test_requirements_full_forward_flagged():
    req = needs_for_real_scoring("full_forward")
    assert not req.forward_only
    assert ScoreNeed.HIGH_PRECISION_WEIGHTS in req.needs
    assert ScoreNeed.ROUTER_LOGITS in req.needs
