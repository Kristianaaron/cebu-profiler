"""Census + ownership tests: coverage, no unclassified, source identity."""

from model_atlas.census.census import build_manifest
from model_atlas.census.precision import PrecisionCensus, census_precision
from model_atlas.census.tensor_ownership import PhysicalLocation, TensorRole
from model_atlas.checkpoint.source_manifest import load_manifest
from model_atlas.checkpoint.synthetic import make_synthetic_checkpoint
from model_atlas.registry.architectures import get_registry

MINI_EXPECTED_RECORDS = 34  # per layer: 7 single + 8 experts + 1 shared = 16; x2 = 32; +2 global


def test_mini_manifest_counts():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    assert manifest.status == "synthetic"
    assert len(manifest.records) == MINI_EXPECTED_RECORDS


def test_no_unclassified_tensors():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    # every tensor maps to exactly one non-null role (invariant)
    assert all(rec.role is not None for rec in manifest.records)
    roles = {rec.role for rec in manifest.records}
    assert roles <= set(TensorRole)


def test_keys_unique():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    keys = [r.key for r in manifest.records]
    assert len(keys) == len(set(keys))


def test_routed_experts_split_expert_parallel():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    experts = [r for r in manifest.records if r.role == TensorRole.EXPERTS]
    assert len(experts) == spec.num_text_layers * spec.moe.num_routed_experts
    for r in experts:
        assert r.expert_index is not None
        expected = PhysicalLocation.NODE_A if r.expert_index % 2 == 0 else PhysicalLocation.NODE_B
        assert r.location == expected


def test_both_nodes_have_resident_data():
    spec = get_registry().get("k3-mini")
    manifest = build_manifest(spec)
    by_node = manifest.bytes_by_node()
    assert by_node[PhysicalLocation.NODE_A] > 0
    assert by_node[PhysicalLocation.NODE_B] > 0


def test_k3_manifest_needs_measurement():
    spec = get_registry().get("k3")
    manifest = build_manifest(spec)
    assert manifest.status == "needs_source_measurement"
    assert manifest.records == []


def test_precision_census_reports_achieved_bpw(tmp_path):
    ckpt = make_synthetic_checkpoint(tmp_path / "ckpt")
    manifest = load_manifest(ckpt)
    pc = census_precision(manifest)
    assert isinstance(pc, PrecisionCensus)
    assert pc.total_stored_bytes == manifest.total_bytes
    assert pc.overall_bpw is not None and pc.overall_bpw > 0
    experts = pc.role(TensorRole.EXPERTS)
    assert experts is not None
    # synthetic toggle fixture is F16 expert -> 16 bpw
    assert abs(experts.achieved_bpw - 16.0) < 1e-6
    # every role has a real dtype and positive byte count
    for r in pc.by_role:
        assert r.stored_bytes > 0
        assert r.dominant_dtype is not None
