"""Phase 2: corpus loader + resumable run manifest tests."""

import json

import pytest

from model_atlas.corpus import load_corpus
from model_atlas.runmanifest import (
    ChunkedRunner,
    ChunkProgress,
    RunManifest,
    load_run_manifest,
    save_run_manifest,
)
from model_atlas.schemas.ontology import DataPartition


def _write_corpus(tmp_path):
    cal = tmp_path / "calib" / "a.jsonl"
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text(
        json.dumps({"id": "c0", "token_ids": [1, 2, 3], "labels": ["code"]})
        + "\n"
        + json.dumps({"id": "c1", "token_ids": [4, 5], "labels": ["math"]})
        + "\n"
    )
    dev = tmp_path / "dev" / "d.txt"
    dev.parent.mkdir(parents=True, exist_ok=True)
    dev.write_text("some dev tokens\nsecond dev line\n")
    ho = tmp_path / "heldout" / "h.jsonl"
    ho.parent.mkdir(parents=True, exist_ok=True)
    ho.write_text(json.dumps({"id": "h0", "text": "heldout only"}) + "\n")
    return tmp_path


@pytest.mark.integration
def test_corpus_partitions_immutable(tmp_path):
    root = _write_corpus(tmp_path)
    cm = load_corpus(str(root), vocab=1000)
    assert cm.counts()[DataPartition.ATLAS_CALIBRATION.value] == 2
    assert cm.counts()[DataPartition.DEVELOPMENT_EVALUATION.value] == 2
    assert cm.counts()[DataPartition.HELD_OUT_EVALUATION.value] == 1
    # every sample belongs to exactly one partition (immutable, non-overlapping)
    all_ids = []
    for partition in DataPartition:
        for s in cm.samples(partition):
            all_ids.append((partition, s.sample_id))
    assert len(all_ids) == len({i for _, i in all_ids})


@pytest.mark.integration
def test_corpus_explicit_file_lists():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.jsonl"
        p.write_text(json.dumps({"token_ids": [7, 8]}) + "\n")
        cm = load_corpus(calibration=[str(p)], vocab=10)
        assert cm.samples(DataPartition.ATLAS_CALIBRATION)[0].tokens == [7, 8]


@pytest.mark.integration
def test_run_manifest_roundtrip(tmp_path):
    m = RunManifest(
        run_id="r1",
        model="glm-5.2",
        source_checkpoint="/media/glm52",
        seed=0,
    )
    m.record_chunk(
        ChunkProgress(
            chunk_id="L0",
            partition=DataPartition.ATLAS_CALIBRATION.value,
            layer_index=0,
            sample_count=16,
            completed=True,
            summary={"tenp_sum": 1.5},
        )
    )
    p = tmp_path / "run.json"
    save_run_manifest(m, p)
    loaded = load_run_manifest(p)
    assert loaded.run_id == "r1"
    assert loaded.chunks["L0"].completed
    assert loaded.chunks["L0"].summary["tenp_sum"] == 1.5


@pytest.mark.integration
def test_chunked_runner_resumes(tmp_path):
    calls = {"count": 0}

    def worker(chunk):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return {"sum": chunk.sample_count}

    m = RunManifest(run_id="r2", model="m", source_checkpoint="s", seed=1)
    m.record_chunk(
        ChunkProgress(
            chunk_id="L0", partition="calib", layer_index=0, sample_count=3, completed=True
        )
    )
    m.record_chunk(
        ChunkProgress(
            chunk_id="L1", partition="calib", layer_index=1, sample_count=5, completed=False
        )
    )
    p = tmp_path / "run.json"
    ChunkedRunner(m, p, worker).run()
    # L0 already completed -> skipped; only L1 recomputed
    assert calls["count"] == 1
    final = load_run_manifest(p)
    assert final.chunks["L1"].completed
    assert final.chunks["L1"].summary["sum"] == 5
    assert final.chunks["L0"].completed
