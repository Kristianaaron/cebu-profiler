"""Resumable run manifest at layer/chunk boundaries (Phase 2).

Captures per-chunk progress state for a long GLM-5.2 trace/corpus run so it can
be resumed at a layer or chunk boundary instead of restarting. Only the
cumulative per-(layer, chunk) aggregate results are persisted (never huge raw
activations), per the stream/aggregate contract.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from model_atlas.schemas.ontology import DataPartition


@dataclass
class ChunkProgress:
    chunk_id: str
    partition: str
    layer_index: int
    sample_count: int
    completed: bool = False
    started_at: float = 0.0
    completed_at: float | None = None
    summary: dict[str, object] = field(default_factory=dict)  # cumulative aggregates, small


@dataclass
class RunManifest:
    run_id: str
    model: str
    source_checkpoint: str
    seed: int
    start_time: float = field(default_factory=time.time)
    status: str = "running"  # running | complete | failed
    chunks: dict[str, ChunkProgress] = field(default_factory=dict)
    calibration_suite: str = "glm52-compression-v1"
    data_partition: DataPartition = DataPartition.ATLAS_CALIBRATION

    def record_chunk(self, chunk: ChunkProgress) -> None:
        self.chunks[chunk.chunk_id] = chunk

    def next_unfinished(self) -> list[ChunkProgress]:
        return [c for c in self.chunks.values() if not c.completed]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "source_checkpoint": self.source_checkpoint,
            "seed": self.seed,
            "start_time": self.start_time,
            "status": self.status,
            "calibration_suite": self.calibration_suite,
            "data_partition": self.data_partition.value,
            "chunks": [asdict(c) for c in sorted(self.chunks.values(), key=lambda c: c.chunk_id)],
        }


def save_run_manifest(manifest: RunManifest, path: str | Path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return str(path)


def load_run_manifest(path: str | Path) -> RunManifest:
    d = json.loads(Path(path).read_text())
    chunks = {}
    for cd in d.pop("chunks", []):
        chunk = ChunkProgress(**cd)
        chunks[chunk.chunk_id] = chunk
    return RunManifest(chunks=chunks, **d)


class ChunkedRunner:
    """Drives a per-`chunk` callable and persists a resumable run manifest.

    On resume, already-completed chunks are skipped (their cached summary is
    reused); only the first incomplete chunk onward is recomputed. The worker
    callable must return a small aggregate dict (never raw activations).
    """

    def __init__(
        self,
        manifest: RunManifest,
        manifest_path: str | Path,
        worker: Callable[[ChunkProgress], dict[str, object]],
    ) -> None:
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.worker = worker

    def run(self) -> RunManifest:
        for chunk in sorted(self.manifest.chunks.values(), key=lambda c: c.chunk_id):
            if chunk.completed:
                continue
            chunk.started_at = time.time()
            try:
                chunk.summary = self.worker(chunk)
                chunk.completed = True
                chunk.completed_at = time.time()
            except Exception:  # noqa: BLE001
                chunk.completed = False
                chunk.completed_at = time.time()
                self.manifest.status = "failed"
                save_run_manifest(self.manifest, self.manifest_path)
                raise
            self.manifest.chunks[chunk.chunk_id] = chunk
            save_run_manifest(self.manifest, self.manifest_path)
        return self.manifest
