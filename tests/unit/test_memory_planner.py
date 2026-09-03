"""Memory-planner tests: go/no-go, resident vs stored vs active accounting."""

from cebu_profiler.census.census import build_manifest
from cebu_profiler.census.tensor_ownership import PhysicalLocation
from cebu_profiler.planning.memory_planner import (
    active_expert_bytes_per_token,
    assess,
    resident_bytes_by_node,
)
from cebu_profiler.registry.architectures import get_registry
from cebu_profiler.schemas.architecture import DTYPE_BYTES, DType, TensorRole


def test_safe_plan_when_budgets_are_large():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    result = assess(spec, manifest, budget_a_gb=100.0, budget_b_gb=100.0, runtime_reserve_gb=1.0)
    assert result.safe is True
    assert result.failures == []


def test_plan_rejects_over_budget():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    result = assess(spec, manifest, budget_a_gb=0.00001, budget_b_gb=100.0, runtime_reserve_gb=0.0)
    assert result.safe is False
    assert any("node A" in f for f in result.failures)


def test_resident_counts_match_manifest_locations():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    by_node = resident_bytes_by_node(manifest)
    loc = manifest.bytes_by_node()
    # mini uses no REPLICATED tier, so node resident == located bytes
    assert by_node[PhysicalLocation.NODE_A] == loc[PhysicalLocation.NODE_A]
    assert by_node[PhysicalLocation.NODE_B] == loc[PhysicalLocation.NODE_B]


def test_active_expert_bytes_per_token_formula():
    spec = get_registry().get("k3-mini")
    per_expert = spec.tensor_params[TensorRole.EXPERTS] * DTYPE_BYTES[DType.MXFP4]
    expected = spec.num_text_layers * spec.moe.top_k * per_expert
    assert active_expert_bytes_per_token(spec) == expected


def test_assess_active_and_stored():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    result = assess(spec, manifest, budget_a_gb=100.0, budget_b_gb=100.0, runtime_reserve_gb=1.0)
    assert result.stored_bytes == manifest.total_bytes()
    assert result.active_expert_bytes_per_token == active_expert_bytes_per_token(spec)


def test_k3_cannot_be_planned_without_measurement():
    spec = get_registry().get("k3")
    manifest = build_manifest(spec)
    result = assess(spec, manifest, budget_a_gb=200.0, budget_b_gb=200.0, runtime_reserve_gb=30.0)
    assert result.safe is False
    assert result.failures
