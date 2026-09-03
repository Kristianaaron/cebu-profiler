"""F17 tests: Milestone E quality-size experiment (blueprint §17).

Matched-budget uniform vs heterogeneous pruning, measured by held-out
representation drift / logit KL. Key claim: on a structured synthetic MoE where
importance is concentrated, measured heterogeneous allocation preserves the
representation better than a uniform-width control at the same retained budget.
"""

import random

from cebu_profiler.experiments.controls import (
    budget_for,
    channel_importance,
    hetero_clone,
    matched_budget_compare,
    uniform_clone,
)
from cebu_profiler.experiments.fidelity import measure_fidelity
from cebu_profiler.experiments.pareto import pareto_sweep
from cebu_profiler.experiments.structured import build_structured_model
from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, build_mini_moe
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.ontology import CapabilityLabel, TrajectoryStage

ARCH = get_registry().get("k3-mini")


def _samples_n(model: MiniMoE, seed: int, n: int) -> list[CalibrationSample]:
    rng = random.Random(seed)
    vocab = model.arch.vocabulary_size
    assert vocab is not None
    return [
        CalibrationSample(
            tokens=[rng.randrange(vocab) for _ in range(12)],
            labels=[CapabilityLabel.FACTUAL_KNOWLEDGE],
            stage=TrajectoryStage.UNDERSTAND,
        )
        for _ in range(n)
    ]


def _clone_widths(clone: MiniMoE) -> dict[tuple[int, int], int]:
    return {
        (layer, e): len(lw.experts[e]["gate"])
        for layer, lw in enumerate(clone.layers)
        for e in range(clone.n_exp)
    }


def test_full_width_clones_match_source() -> None:
    model = build_mini_moe(ARCH, seed=1)
    cal, held = _samples_n(model, 0, 24), _samples_n(model, 2, 20)
    imp = channel_importance(model, cal)
    full = budget_for(model, 1.0)
    for clone in (uniform_clone(model, imp, full), hetero_clone(model, imp, full)):
        rep = measure_fidelity(model, clone, held)
        assert rep.mean_hidden_drift < 1e-9
        assert abs(rep.retention - 1.0) < 1e-9


def test_budget_matched_and_nested() -> None:
    model = build_mini_moe(ARCH, seed=1)
    imp = channel_importance(model, _samples_n(model, 0, 24))
    budget = budget_for(model, 0.6)
    for clone in (uniform_clone(model, imp, budget), hetero_clone(model, imp, budget)):
        widths = _clone_widths(clone)
        assert sum(widths.values()) == budget
        # every retained expert keeps a non-empty prefix
        for layer, lw in enumerate(clone.layers):
            for e, exp in enumerate(lw.experts):
                assert widths[(layer, e)] >= 1
                # gate/up/down coupled to the same width
                assert len(exp["up"]) == len(exp["gate"])
                assert all(len(row) == len(exp["gate"]) for row in exp["down"])


def test_uniform_widths_are_flat() -> None:
    model = build_mini_moe(ARCH, seed=1)
    imp = channel_importance(model, _samples_n(model, 0, 24))
    widths = set(_clone_widths(uniform_clone(model, imp, budget_for(model, 0.7))).values())
    assert max(widths) - min(widths) <= 1  # equal width, remainder <=1


def test_heterogeneous_widths_vary_on_structure() -> None:
    model = build_structured_model(seed=1, n_strong=1, strong_scale=8.0, channels=6)
    imp = channel_importance(model, _samples_n(model, 0, 24))
    widths = set(_clone_widths(hetero_clone(model, imp, budget_for(model, 0.4))).values())
    assert len(widths) > 1  # allocation tracks per-expert importance


def test_hetero_beats_uniform_on_structured() -> None:
    # Milestone E: measured heterogeneous allocation preserves the representation
    # better than the equal-width control at the same budget, with concentrated
    # importance.
    model = build_structured_model(seed=1, n_strong=1, strong_scale=8.0, channels=6)
    cal, held = _samples_n(model, 0, 24), _samples_n(model, 2, 20)
    out = matched_budget_compare(model, cal, held, budget_for(model, 0.4))
    assert out.hetero.mean_hidden_drift <= out.uniform.mean_hidden_drift


def test_pareto_sweep_frontier() -> None:
    model = build_structured_model(seed=1, n_strong=1, strong_scale=8.0, channels=6)
    cal, held = _samples_n(model, 0, 24), _samples_n(model, 2, 16)
    pts = pareto_sweep(model, cal, held, fractions=(1.0, 0.6))
    assert pts[0].retain_fraction == 1.0
    assert pts[0].hetero_kl < 1e-9  # identity anchor
    assert pts[1].budget < pts[0].budget
    assert pts[0].uniform_topk == 1.0
