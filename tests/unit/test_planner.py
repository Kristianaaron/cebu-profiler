import pytest

from model_atlas.prune.planner import PlannerError, plan_uniform_width


def test_plan_halves_width_for_half_budget() -> None:
    # 400 GiB experts + 100 GiB protected; 300 GiB target -> half the experts
    w = plan_uniform_width(
        expert_source_gib=400.0, protected_gib=100.0, target_gib=300.0, full=64
    )
    # budget is 200 GiB -> frac 0.5 -> width 32
    assert w == 32
    assert w % 16 == 0


def test_plan_full_when_budget_allows() -> None:
    w = plan_uniform_width(
        expert_source_gib=400.0, protected_gib=100.0, target_gib=500.0, full=64
    )
    assert w == 64  # clamped to full


def test_plan_aligns_down_to_group() -> None:
    # frac = 0.63 -> 40.3 -> aligned down to 32 (group 16)
    w = plan_uniform_width(
        expert_source_gib=400.0, protected_gib=100.0, target_gib=352.4, full=64
    )
    assert w % 16 == 0
    assert 0 < w <= 64


def test_plan_respects_min_width() -> None:
    w = plan_uniform_width(
        expert_source_gib=1000.0, protected_gib=100.0, target_gib=120.0,
        full=64, min_width=16,
    )
    assert w == 16  # floor at min, never 0/negative


def test_plan_fails_closed_when_target_under_protected() -> None:
    with pytest.raises(PlannerError, match="leaves no room"):
        plan_uniform_width(
            expert_source_gib=400.0, protected_gib=100.0, target_gib=90.0, full=64
        )


def test_plan_rejects_bad_geometry() -> None:
    with pytest.raises(PlannerError):
        plan_uniform_width(
            expert_source_gib=400.0, protected_gib=0.0, target_gib=300.0, full=33
        )
