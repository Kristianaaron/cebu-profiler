"""sha256 of each safetensors shard, chunked so a multi-hundred-GB checkpoint
never has a shard fully resident in memory.

Small files (<= chunk size) hash in one read; large ones stream in chunks.
Result map is identical to the previous whole-file implementation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 64 * 1024 * 1024  # 64 MiB


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_hashes(checkpoint_dir: str) -> dict[str, str]:
    """sha256 of each safetensors shard (streamed, constant memory)."""
    root = Path(checkpoint_dir)
    return {
        p.name: _hash_file(p) for p in root.glob("*.safetensors") if not p.name.startswith("._")
    }
