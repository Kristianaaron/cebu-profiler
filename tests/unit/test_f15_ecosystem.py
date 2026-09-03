"""Ecosystem bridge tests: eval-lab task corpus -> Cebu Profiler calibration (deterministic)."""

from pathlib import Path

from cebu_profiler.ecosystem import (
    label_for_path,
    pipeline_summary,
    prompt_corpus,
    tokens_from_text,
)
from cebu_profiler.schemas.ontology import CapabilityLabel, DataPartition


def _make_tasks(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    (root / "coding").mkdir(parents=True)
    (root / "voxel").mkdir(parents=True)
    (root / "coding" / "prompt.md").write_text(
        "fix the parser and run the tests\nassert everything"
    )
    (root / "voxel" / "prompt.md").write_text("rotate the voxel scene around the y axis")
    return root


def test_label_for_path_domain_mapping():
    assert (
        label_for_path(Path("tasks/coding/refactor/prompt.md")) == CapabilityLabel.CODE_GENERATION
    )
    assert label_for_path(Path("tasks/voxel/sphere/prompt.md")) == CapabilityLabel.VOXEL_SPATIAL
    assert (
        label_for_path(Path("tasks/unknown_thing/prompt.md")) == CapabilityLabel.GENERAL_REASONING
    )


def test_tokens_from_text_deterministic_and_bounded():
    a = tokens_from_text("hello world hello world", vocab=1000, seed=0)
    b = tokens_from_text("hello world hello world", vocab=1000, seed=0)
    assert a == b
    assert all(0 <= t < 1000 for t in a)
    assert len(tokens_from_text("word " * 1000, vocab=1000)) <= 256  # capped


def test_prompt_corpus_ingests_real_prompts(tmp_path):
    root = _make_tasks(tmp_path)
    samples = prompt_corpus(str(root), vocab=1000, seed=0)
    assert len(samples) == 2
    labels = {s.labels[0].value for s in samples}
    assert "code_generation" in labels
    assert "voxel_spatial" in labels
    assert all(s.tokens for s in samples)
    # honors data partition
    assert all(s.stage is not None for s in samples)


def test_pipeline_summary_runs_over_eval_tasks(tmp_path):
    root = _make_tasks(tmp_path)
    summary = pipeline_summary(str(root), seed=0, partition=DataPartition.DEVELOPMENT_EVALUATION)
    assert summary["source"] == "eval-lab task corpus"
    assert summary["n_tasks"] == 2
    assert summary["partition"] == "development_evaluation"
    assert "code_generation" in summary["saliency_per_label"]
    assert "voxel_spatial" in summary["saliency_per_label"]
