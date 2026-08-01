"""F2 checkpoint tests: manifest enumeration, classification, structural graph."""

from model_atlas.checkpoint.classifier import classify_tensor
from model_atlas.checkpoint.source_manifest import load_manifest, shard_hashes
from model_atlas.checkpoint.structural_graph import build_structural_graph
from model_atlas.checkpoint.synthetic import UNCLASSIFIED_NAME, make_synthetic_checkpoint
from model_atlas.schemas.architecture import TensorRole

CLASSIFIED_TENSOR_COUNT = 20  # 18 per-layer + embed + lm_head


def test_manifest_enumerates_tensors(tmp_path):
    ckpt = make_synthetic_checkpoint(tmp_path / "ckpt")
    manifest = load_manifest(ckpt)
    assert manifest.tensor_count == CLASSIFIED_TENSOR_COUNT
    assert manifest.total_bytes > 0
    assert len(manifest.shards) == 1
    # every tensor has a positive byte size and a shard
    assert all(t.byte_size > 0 and t.shard for t in manifest.tensors)


def test_shard_hashes_computed(tmp_path):
    ckpt = make_synthetic_checkpoint(tmp_path / "ckpt")
    hashes = shard_hashes(ckpt)
    assert len(hashes) == 1
    assert len(next(iter(hashes.values()))) == 64  # sha256 hex


def test_classifier_roles(tmp_path):
    ckpt = make_synthetic_checkpoint(tmp_path / "ckpt")
    manifest = load_manifest(ckpt)
    by_role = {}
    for t in manifest.tensors:
        c = classify_tensor(t.name)
        assert not c.unclassified
        assert c.role is not None
        by_role.setdefault(c.role, 0)
        by_role[c.role] += 1
    assert TensorRole.EMBEDDING in by_role
    assert TensorRole.LM_HEAD in by_role
    assert by_role[TensorRole.EXPERTS] == 4  # experts.0 + experts.1 per layer
    # expert identity preserved
    exp = [classify_tensor(t.name) for t in manifest.tensors if "experts.1" in t.name]
    assert all(c.expert_index == 1 for c in exp)


def test_structural_graph_full_coverage(tmp_path):
    ckpt = make_synthetic_checkpoint(tmp_path / "ckpt")
    manifest = load_manifest(ckpt)
    graph = build_structural_graph(manifest)
    assert graph.coverage == 1.0
    assert graph.unclassified == []
    assert graph.valid is True
    assert graph.nodes  # nodes exist
    assert any(e.kind == "output" for e in graph.edges)


def test_structural_graph_fails_closed_on_unclassified(tmp_path):
    ckpt = make_synthetic_checkpoint(tmp_path / "ckpt", include_unclassified=True)
    manifest = load_manifest(ckpt)
    graph = build_structural_graph(manifest)
    assert graph.coverage < 1.0
    assert graph.valid is False
    assert UNCLASSIFIED_NAME in graph.unclassified
