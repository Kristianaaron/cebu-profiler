"""Source checkpoint manifest: enumerate shards and tensor headers.

Metadata-first: reads `config.json`, discovers `.safetensors` shards, and reads
each shard's header (no tensor bodies). Produces a `CheckpointManifest` listing
every tensor with shape/dtype/byte-range/shard, plus optional per-shard hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.checkpoint.safetensors import read_safetensors_header


class TensorEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    shape: list[int]
    numel: int
    byte_size: int
    shard: str
    offset_start: int
    offset_end: int


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_dir: str
    config: dict[str, Any] = Field(default_factory=dict)
    shards: list[str] = Field(default_factory=list)
    tensors: list[TensorEntry] = Field(default_factory=list)
    total_bytes: int = 0
    tensor_count: int = 0


def _numel(shape: list[int]) -> int:
    return math.prod(shape) if shape else 0


def _discover_shards(checkpoint_dir: str) -> list[str]:
    root = Path(checkpoint_dir)
    return sorted(
        p.name
        for p in root.glob("*.safetensors")
        if not p.name.endswith(".index.json") and not p.name.startswith("._")
    )


def load_manifest(checkpoint_dir: str) -> CheckpointManifest:
    """Enumerate all tensors of a checkpoint directory from headers only."""
    root = Path(checkpoint_dir)
    config: dict[str, Any] = {}
    config_path = root / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    shards = _discover_shards(checkpoint_dir)
    tensors: list[TensorEntry] = []
    for shard in shards:
        header = read_safetensors_header(root / shard)
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            shape = list(spec["shape"])
            start, end = spec["data_offsets"]
            tensors.append(
                TensorEntry(
                    name=name,
                    dtype=spec["dtype"],
                    shape=shape,
                    numel=_numel(shape),
                    byte_size=end - start,
                    shard=shard,
                    offset_start=start,
                    offset_end=end,
                )
            )

    tensors.sort(key=lambda t: (t.shard, t.name))
    total = sum(t.byte_size for t in tensors)
    return CheckpointManifest(
        checkpoint_dir=str(root),
        config=config,
        shards=shards,
        tensors=tensors,
        total_bytes=total,
        tensor_count=len(tensors),
    )


def shard_hashes(checkpoint_dir: str) -> dict[str, str]:
    """sha256 of each safetensors shard (whole-file). Small for fixtures."""
    root = Path(checkpoint_dir)
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("*.safetensors")
        if not p.name.startswith("._")
    }
