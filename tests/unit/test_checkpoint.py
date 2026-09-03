"""F2 checkpoint tests: manifest enumeration, classification, structural graph."""

from pathlib import Path

from cebu_profiler.checkpoint.classifier import classify_tensor
from cebu_profiler.checkpoint.source_manifest import load_manifest, shard_hashes
from cebu_profiler.checkpoint.structural_graph import build_structural_graph
from cebu_profiler.checkpoint.synthetic import UNCLASSIFIED_NAME, make_synthetic_checkpoint
from cebu_profiler.schemas.architecture import TensorRole

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


def test_classifier_glm52_eh_proj_is_head():
    # GLM-5.2's final external-hidden output projection belongs to the head,
    # not a residual-layer role, and is a global tensor (no layer index).
    c = classify_tensor("model.layers.78.eh_proj.weight")
    assert c.role is TensorRole.LM_HEAD
    assert not c.unclassified
    assert c.layer_index is None


def test_load_manifest_skips_appledouble_junk(tmp_path):
    # Mac exFAT/NTFS source drives litter shard dirs with `._*` AppleDouble
    # metadata files. Reading one as a safetensors header yields a garbage
    # length -> MemoryError. Discovery must ignore them.
    ckpt = make_synthetic_checkpoint(tmp_path / "ckpt")
    junk = Path(ckpt) / "._model-00001-of-00047.safetensors"
    junk.write_bytes(b"\xff" * 8 + b"definitely not a safetensors header")
    manifest = load_manifest(ckpt)
    assert manifest.tensor_count == CLASSIFIED_TENSOR_COUNT
    assert len(manifest.shards) == 1
    assert manifest.config  # config.json still read
