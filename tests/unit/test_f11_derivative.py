"""F11 tests: derivative builder + validation + registration (v2 §26)."""

from cebu_profiler.builder import build_derivative, register_derivative
from cebu_profiler.compression import expert_response_curve, get_backend_registry
from cebu_profiler.planning import SearchInputs, build_candidate
from cebu_profiler.profiler.reap import make_synthetic_corpus, run_calibration
from cebu_profiler.profiler.runtime import build_mini_moe, forward
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.model_asset import AssetType, ModelAsset

ARCH = get_registry().get("k3-mini")


def _plan(keep=4, seed=1):
    model = build_mini_moe(ARCH, seed=seed)
    samples = make_synthetic_corpus(n_samples=8, seq_len=5, vocab=ARCH.vocabulary_size, seed=0)[0]
    sal = run_calibration(model, samples, top_k=2)
    reg = get_backend_registry()
    response = {}
    for layer in range(model.arch.num_text_layers):
        for e in range(model.n_exp):
            response[(layer, e)] = expert_response_curve(
                model, [1, 2, 3], layer=layer, expert=e, backends=reg, formats=["int4", "int8"]
            )
    coalitions = {0: [(0, 2, 4)], 1: [(1, 3)]}
    plan = build_candidate(
        SearchInputs(model=model, saliency=sal, coalitions=coalitions, response=response),
        name=f"keep{keep}-value",
        keep_budget_per_layer=keep,
        strategy="value",
        node_budget_bytes=1e12,
        active_bytes_per_token=100.0,
    )
    return model, plan


def test_derivative_renumbers_and_remaps_router():
    model, plan = _plan(keep=4)
    result = build_derivative(model, plan)
    dv = result.model
    assert dv.n_exp == 4
    assert len(dv.layers[0].router) == 4
    assert len(dv.layers[0].experts) == 4
    # identity mapping: derivative slot -> source expert id (preserved)
    kept = sorted(plan.keep.kept(0))
    slots0 = [r for r in result.identity_map if r.layer == 0]
    assert [r.source_expert_id for r in slots0] == kept
    assert slots0[0].derivative_slot == 0


def test_derivative_routes_only_among_retained_experts():
    model, plan = _plan(keep=4)
    result = build_derivative(model, plan)
    kept_map = {r.layer: set(plan.keep.kept(r.layer)) for r in result.identity_map}
    out = forward(result.model, [1, 2, 3], top_k=2)
    for trace in out.traces:
        kept_layer = kept_map[trace.layer]
        for ids in trace.topk_ids:
            # ids are derivative-slot indices -> map back to source
            for slot in ids:
                src = [
                    r.source_expert_id
                    for r in result.identity_map
                    if r.layer == trace.layer and r.derivative_slot == slot
                ][0]
                assert src in kept_layer


def test_derivative_validation_passes():
    model, plan = _plan(keep=4)
    result = build_derivative(model, plan)
    assert result.validation.ok
    assert result.validation.tensor_coverage
    assert result.validation.inference_ok


def test_register_derivative_asset():
    model, plan = _plan(keep=4)
    result = build_derivative(model, plan)
    source = ModelAsset(
        model_asset_id="k3-src-001",
        display_name="K3 source",
        asset_type=AssetType.SOURCE_CHECKPOINT,
        checkpoint_path="/models/k3",
    )
    asset = register_derivative(
        result, display_name="deriv-a", source_asset=source, source_experiment_id="exp-1"
    )
    assert asset.asset_type == AssetType.DERIVATIVE_CHECKPOINT
    assert asset.parent_model_id == "k3-src-001"
    assert asset.source_experiment_id == "exp-1"
    assert asset.metadata["validated"] is True
