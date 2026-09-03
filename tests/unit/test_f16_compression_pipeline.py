"""F16 tests: end-to-end Cebu Profiler compression pipeline + structural executor.

Covers the blueprint's first end-to-end milestone (§25):
    trace -> TENP -> stability -> causal -> grouped Taylor
    -> width-bucket planner -> compression manifest
plus the six executor tests mandated by blueprint §12.2 (dry-run,
coupled slicing, permutation equivalence, topology, manifest replay,
protected channels).
"""

import random

from cebu_profiler.executor.structural import (
    apply_manifest,
    build_clone,
    dry_run,
    orders_from_manifest,
    reorder_channels,
)
from cebu_profiler.planning.width_buckets import SM121_WIDTH_BUCKETS
from cebu_profiler.planning.widths import build_manifest, estimate_params
from cebu_profiler.profiler.compress import run_compression_pipeline
from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, build_mini_moe, forward
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.manifest import validate_manifest
from cebu_profiler.schemas.ontology import CapabilityLabel, TrajectoryStage
from cebu_profiler.scoring.base import ChannelScore, ScoreNeed
from cebu_profiler.scoring.tenp import TenpScorer

ARCH = get_registry().get("k3-mini")


def _model(seed: int = 1) -> MiniMoE:
    return build_mini_moe(ARCH, seed=seed)


def _samples(model: MiniMoE, n: int = 24, seed: int = 0) -> list[CalibrationSample]:
    rng = random.Random(seed)
    vocab = model.arch.vocabulary_size
    assert vocab is not None
    return [
        CalibrationSample(
            tokens=[rng.randrange(vocab) for _ in range(16)],
            labels=[CapabilityLabel.FACTUAL_KNOWLEDGE],
            stage=TrajectoryStage.UNDERSTAND,
        )
        for _ in range(n)
    ]


def _concentrated_rows(full_width: int = 16) -> list[ChannelScore]:
    """Expert (0,0) with importance concentrated; all other experts uniform."""
    rows: list[ChannelScore] = []
    for e in range(8):
        for c in range(full_width):
            imp = (10.0 if c == 0 else (1.0 if 1 <= c <= 7 else 0.001)) if e == 0 else 0.01
            rows.append(ChannelScore(layer=0, expert=e, channel=c, tenp=imp))
    return rows


# --- pipeline ---------------------------------------------------------------


def test_pipeline_emits_valid_deterministic_manifest() -> None:
    model = _model()
    manifest, validation = run_compression_pipeline(model, _samples(model), n_stability_runs=3)
    assert validation.ok, validation.errors
    manifest2, validation2 = run_compression_pipeline(model, _samples(model), n_stability_runs=3)
    assert manifest2.model_dump() == manifest.model_dump()


def test_pipeline_widths_obey_bucket_vocab_and_cardinality() -> None:
    model = _model()
    manifest, validation = run_compression_pipeline(model, _samples(model))
    assert validation.ok
    allowed = set(manifest.allowed_widths)
    assert allowed.issubset(set(SM121_WIDTH_BUCKETS) | {model.mid})
    for layer in manifest.layers.values():
        for plan in layer.experts.values():
            assert plan.target_width <= plan.original_width
            assert len(plan.keep_channels) == plan.target_width
            assert plan.target_width in allowed


def test_decisions_trace_to_measured_scores() -> None:
    model = _model()
    manifest, _ = run_compression_pipeline(model, _samples(model), n_stability_runs=3)
    plan = list(manifest.layers["0"].experts.values())[0]
    # scores carry measured/aggregated views, never fabricated None-as-number
    assert plan.confidence >= 0.0 and plan.confidence <= 1.0
    assert plan.scores.tenp is not None


def test_tenp_scorer_is_forward_only() -> None:
    # blueprint §9.1: TENP runs immediately on NVFP4 (no gradients / high-precision)
    reqs = TenpScorer(_model()).requirements()
    assert reqs.forward_only
    assert ScoreNeed.FORWARD_ACTIVATIONS in reqs.needs
    assert ScoreNeed.RAW_EXPERT_TENSORS in reqs.needs
    assert ScoreNeed.GRADIENTS not in reqs.needs


# --- planner ----------------------------------------------------------------


def test_planner_variable_width_on_concentrated_importance() -> None:
    rows = _concentrated_rows()
    manifest = build_manifest(
        model_name="k3-mini",
        source_checkpoint="glm52-compression-v1",
        score_rows=rows,
        num_layers=1,
        num_experts=8,
        full_width=16,
        allowed_widths=[16, 12, 8, 4],
        coverage_target=0.9,
    )
    assert validate_manifest(manifest).ok
    plan = manifest.layers["0"].experts["0"]
    assert plan.original_width == 16
    assert plan.target_width == 8  # concentrated -> smaller bucket
    assert len(plan.keep_channels) == 8
    assert set(plan.keep_channels).issubset(range(16))


def test_protected_channels_cannot_be_pruned() -> None:
    rows = _concentrated_rows()
    protected = {(0, 0): {3, 4, 5, 6, 7, 8, 9, 10, 11}}  # 9 channels, beyond an 8-bucket
    manifest = build_manifest(
        model_name="k3-mini",
        source_checkpoint="glm52-compression-v1",
        score_rows=rows,
        num_layers=1,
        num_experts=8,
        full_width=16,
        allowed_widths=[16, 12, 8, 4],
        coverage_target=0.9,
        protected=protected,
    )
    assert validate_manifest(manifest).ok
    plan = manifest.layers["0"].experts["0"]
    assert protected[(0, 0)].issubset(set(plan.keep_channels))
    assert plan.protected_reasons  # provenance recorded


def test_planner_can_emit_full_uniform_control() -> None:
    # uniform-width control (blueprint §17, Priority 4 #6): all experts same width
    rows = _concentrated_rows(16)
    manifest = build_manifest(
        model_name="k3-mini",
        source_checkpoint="glm52-compression-v1",
        score_rows=rows,
        num_layers=1,
        num_experts=8,
        full_width=16,
        allowed_widths=[16, 8],
        coverage_target=1.0,  # keep everything -> full uniform control
    )
    widths = {p.target_width for lp in manifest.layers.values() for p in lp.experts.values()}
    assert widths == {16}


def test_estimate_params_matches_keep_cardinality() -> None:
    rows = _concentrated_rows()
    manifest = build_manifest(
        model_name="k3-mini",
        source_checkpoint="glm52-compression-v1",
        score_rows=rows,
        num_layers=1,
        num_experts=8,
        full_width=16,
        allowed_widths=[16, 12, 8, 4],
        coverage_target=0.9,
    )
    kept = sum(p.target_width for lp in manifest.layers.values() for p in lp.experts.values())
    assert estimate_params(manifest, hidden=128) == kept * 3 * 128


def test_unmeasured_expert_kept_full_and_valid() -> None:
    # only expert 0 has measured rows; the rest have no evidence and must stay
    # full-width (measure-before-cut), with an always-valid manifest
    rows = [ChannelScore(layer=0, expert=0, channel=c, tenp=1.0) for c in range(16)]
    manifest = build_manifest(
        model_name="k3-mini",
        source_checkpoint="glm52-compression-v1",
        score_rows=rows,
        num_layers=1,
        num_experts=8,
        full_width=16,
        allowed_widths=[16, 12, 8, 4],
        coverage_target=0.9,
    )
    assert validate_manifest(manifest).ok
    for e in range(1, 8):
        plan = manifest.layers["0"].experts[str(e)]
        assert plan.target_width == 16
        assert plan.keep_channels == list(range(16))


# --- executor (§12.2) -------------------------------------------------------


def test_dry_run_validates_against_source_shapes() -> None:
    model = _model()
    manifest, validation = run_compression_pipeline(model, _samples(model))
    assert validation.ok
    assert dry_run(model, manifest).ok


def test_executor_preserves_router_topology_and_forwards() -> None:
    model = _model()
    manifest, _ = run_compression_pipeline(model, _samples(model))
    clone = apply_manifest(model, manifest)
    assert clone.n_exp == model.n_exp
    assert len(clone.layers) == len(model.layers)
    assert all(len(lw.experts) == model.n_exp for lw in clone.layers)
    r = forward(clone, [1, 2, 3], top_k=2)
    assert len(r.logits) == model.arch.vocabulary_size


def test_coupled_slicing_keeps_gate_up_down_aligned() -> None:
    model = _model()
    manifest, _ = run_compression_pipeline(model, _samples(model))
    clone = apply_manifest(model, manifest)
    l0 = clone.layers[0]
    for e, exp in enumerate(l0.experts):
        tw = manifest.layers["0"].experts[str(e)].target_width
        assert len(exp["gate"]) == tw
        assert len(exp["up"]) == tw
        assert all(len(row) == tw for row in exp["down"])


def test_permutation_equivalence_exact() -> None:
    # §12.2 #1: consistent reorder of all channels is a numerical no-op
    model = _model(seed=3)
    mid = model.mid
    perm = list(reversed(range(mid)))  # non-trivial permutation of every channel
    orders = {(layer, e): perm for layer in range(len(model.layers)) for e in range(model.n_exp)}
    clone = build_clone(model, orders, default_width=mid)
    base = forward(model, [4, 5, 6], top_k=2).logits
    reo = forward(clone, [4, 5, 6], top_k=2).logits
    # reordering only relabels the summed channel index, so outputs agree up to
    # FP summation-order rounding -- assert to a tight tolerance
    assert max(abs(a - b) for a, b in zip(base, reo, strict=True)) < 1e-9


def test_manifest_replay_is_deterministic() -> None:
    # §12.2 #5: same manifest -> same keep sets (executor is a pure function)
    model = _model()
    manifest, _ = run_compression_pipeline(model, _samples(model))
    a = orders_from_manifest(manifest)
    b = orders_from_manifest(manifest)
    assert {k: list(v) for k, v in a.items()} == {k: list(v) for k, v in b.items()}


def test_reorder_channels_is_same_set_permutation() -> None:
    model = _model()
    manifest, _ = run_compression_pipeline(model, _samples(model))
    key = (0, 0)
    keep = orders_from_manifest(manifest)[key]
    perm = reorder_channels(model, manifest, key)
    if len(keep) >= 2:
        assert sorted(perm) == sorted(range(len(keep)))
        assert perm != keep  # genuinely reordered
