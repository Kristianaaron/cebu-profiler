"""F17 protection: coalition-driven channel protection (blueprint §8.2)."""

import random

from cebu_profiler.experiments.structured import build_structured_model
from cebu_profiler.planning.protection import (
    coalition_protected_experts,
    full_channel_protection,
)
from cebu_profiler.profiler.compress import run_compression_pipeline
from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, build_mini_moe
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.manifest import validate_manifest
from cebu_profiler.schemas.ontology import CapabilityLabel, TrajectoryStage

ARCH = get_registry().get("k3-mini")


def _samples(model: MiniMoE, n: int = 20, seed: int = 0) -> list[CalibrationSample]:
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


def test_coalition_protection_detects_persistent_experts() -> None:
    model = build_structured_model(seed=1, n_strong=2, strong_scale=8.0, channels=4)
    samples = _samples(model)
    protected = coalition_protected_experts(model, samples, top_k=2, min_coactivity=1)
    assert protected  # strong experts are persistently co-routed
    assert all(layer in range(len(model.layers)) and 0 <= e < model.n_exp for layer, e in protected)


def test_full_channel_protection_expands_experts() -> None:
    model = build_mini_moe(ARCH, seed=1)
    exp_set = {(0, 1), (0, 5)}
    prot = full_channel_protection(model, exp_set)
    assert prot == {k: set(range(model.mid)) for k in exp_set}


def test_pipeline_respects_protected_channels() -> None:
    model = build_mini_moe(ARCH, seed=1)
    samples = _samples(model)
    protected = {(0, 1): {0, 1, 2, 3, 4}}  # < mid=16; forces widening on that expert
    manifest, validation = run_compression_pipeline(
        model,
        samples,
        allowed_widths=[16, 12, 8, 4],
        coverage_target=0.6,
        n_stability_runs=2,
        protected=protected,
    )
    assert validation.ok, validation.errors
    plan = manifest.layers["0"].experts["1"]
    assert protected[(0, 1)].issubset(set(plan.keep_channels))
    assert plan.target_width >= len(protected[(0, 1)])
    # an unprotected expert may still be pruned
    assert validate_manifest(manifest).ok
