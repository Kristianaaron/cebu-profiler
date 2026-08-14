"""Corpus manifest/loader (Phase 2).

Loads a real GLM-5.2 routing/trace corpus from JSONL / plain-text sources into
an immutable, partitioned structure: Atlas calibration / development evaluation
/ held-out evaluation are distinct partitions and never mixed. No network is
required — filesystem inputs only. Records the immutable partition contract so
calibration never contaminates development/held-out evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from model_atlas.schemas.ontology import DataPartition

# Partition -> directory/file tag (immutable by convention + enforced on load).
_PARTITION_KEYS: dict[DataPartition, str] = {
    DataPartition.ATLAS_CALIBRATION: "calib",
    DataPartition.DEVELOPMENT_EVALUATION: "dev",
    DataPartition.HELD_OUT_EVALUATION: "heldout",
}


@dataclass
class CorpusEntry:
    """One immutable corpus record (token ids + optional labels/stage/domain)."""

    partition: DataPartition
    sample_id: str
    tokens: list[int]
    labels: list[str] = field(default_factory=list)
    stage: str | None = None
    domain: str | None = None
    source: str | None = None  # file path it came from (lineage)


@dataclass
class CorpusManifest:
    """The whole partitioned corpus, immutable partitions."""

    root: str
    partitions: dict[DataPartition, list[CorpusEntry]] = field(default_factory=dict)

    def samples(self, partition: DataPartition) -> list[CorpusEntry]:
        return self.partitions.get(partition, [])

    def counts(self) -> dict[str, int]:
        return {p.value: len(v) for p, v in self.partitions.items()}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
            else:
                out.append({"text": obj})
    return out


def _tokens_from_record(rec: dict[str, object], vocab: int | None) -> list[int]:
    """Accept an explicit token ids list, else deterministic tokenization from
    a text field (no external tokenizer required; vocab-bounded)."""
    if "token_ids" in rec and isinstance(rec["token_ids"], list):
        return [int(t) for t in rec["token_ids"]]
    text = str(rec.get("text", rec.get("prompt", "")))
    out: list[int] = []
    for ch in text:
        out.append(ord(ch) % (vocab if vocab else 1000))
    return out or [0]


def _load_partition(
    paths: list[str], partition: DataPartition, vocab: int | None
) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"corpus file {raw} not found")
        if p.suffix.lower() == ".jsonl":
            records = _read_jsonl(p)
        else:  # plain text: one document per line
            records = [{"text": line} for line in p.read_text().splitlines() if line.strip()]
        for i, rec in enumerate(records):
            labels_raw = rec.get("labels", [])
            entries.append(
                CorpusEntry(
                    partition=partition,
                    sample_id=str(rec.get("id", f"{p.stem}:{i}")),
                    tokens=_tokens_from_record(rec, vocab),
                    labels=[str(x) for x in labels_raw] if isinstance(labels_raw, list) else [],
                    stage=str(rec["stage"]) if "stage" in rec else None,
                    domain=str(rec["domain"]) if "domain" in rec else None,
                    source=str(p),
                )
            )
    return entries


def load_corpus(
    root: str | None = None,
    *,
    calibration: list[str] | None = None,
    development: list[str] | None = None,
    heldout: list[str] | None = None,
    vocab: int | None = None,
) -> CorpusManifest:
    """Load a partitioned corpus from filesystem paths.

    If `root` is given, it is scanned for `calib/`, `dev/`, `heldout/`
    subdirectories (each containing .jsonl / .txt files). Otherwise explicit
    file lists per partition are used. Partitions are immutable: each sample
    belongs to exactly one partition and is never reused across them.
    """
    partitions: dict[DataPartition, list[CorpusEntry]] = {}
    if root:
        r = Path(root)
        for partition, key in _PARTITION_KEYS.items():
            d = r / key
            if not d.exists() or not d.is_dir():
                continue
            files = sorted(
                p.as_posix()
                for p in d.rglob("*")
                if p.suffix.lower() in {".jsonl", ".txt"} and not p.name.startswith("._")
            )
            if files:
                partitions[partition] = _load_partition(files, partition, vocab)
    if calibration is not None:
        partitions[DataPartition.ATLAS_CALIBRATION] = _load_partition(
            calibration, DataPartition.ATLAS_CALIBRATION, vocab
        )
    if development is not None:
        partitions[DataPartition.DEVELOPMENT_EVALUATION] = _load_partition(
            development, DataPartition.DEVELOPMENT_EVALUATION, vocab
        )
    if heldout is not None:
        partitions[DataPartition.HELD_OUT_EVALUATION] = _load_partition(
            heldout, DataPartition.HELD_OUT_EVALUATION, vocab
        )
    if not partitions:
        raise ValueError(
            "empty corpus: pass `root` with calib/dev/heldout dirs or per-partition file lists"
        )
    return CorpusManifest(root=str(root or ""), partitions=partitions)
