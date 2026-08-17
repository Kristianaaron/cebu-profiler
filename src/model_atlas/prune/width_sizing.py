"""Principled, header-level checkpoint sizing for width planning.

The width planner needs real ``expert_source_gib`` / ``protected_gib`` and the
full expert width (``moe_intermediate_size``). We derive these by reading each
shard's safetensors *header* (never the tensor payloads), classifying tensors
as routed-expert FFN vs protected backbone, and summing byte sizes. This is the
"census" the GLM architecture spec truthfully reports as
``needs_source_measurement`` — we supply the measurement read from disk.

Sizing is only used to *plan* retention width (recipe marks evidence PREDICTED);
it is not an identity/hash claim about the source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from model_atlas.checkpoint.safetensors import read_safetensors_header

_DTYPE_BYTES = {
    "U8": 1, "I8": 1, "I16": 2, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "BF8": 1,
}
# routed-expert FFN projection suffix inside a per-expert MLP submodule
_EXPERT_PROJ = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj")


class SizingError(RuntimeError):
    """Width sizing failed without weakening evidence."""


@dataclass(frozen=True)
class WidthSizing:
    full_width: int
    expert_bytes: int
    protected_bytes: int
    shards_scanned: int

    @property
    def expert_gib(self) -> float:
        return self.expert_bytes / (1024**3)

    @property
    def protected_gib(self) -> float:
        return self.protected_bytes / (1024**3)

    @property
    def total_gib(self) -> float:
        return self.expert_gib + self.protected_gib


def _tensor_bytes(dtype: str, shape: list[int]) -> int:
    try:
        width = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise SizingError(f"unknown dtype {dtype!r} in checkpoint header") from exc
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * width


def _classify(name: str) -> bool:
    """Return True if the tensor is part of a routed-expert FFN projection."""
    return _EXPERT_PROJ.search(name) is not None


def size_checkpoint_for_width(source: Path) -> WidthSizing:
    """Read GLM-style NVFP4 safetensors shard headers and size expert vs protected.

    ``source`` must contain ``config.json`` (for ``moe_intermediate_size``) and
    one or more ``*.safetensors`` shards. Fails closed on missing geometry or an
    empty/zero-byte sweep.
    """
    source = source.resolve()
    if not source.is_dir():
        raise SizingError(f"checkpoint source is not a directory: {source}")
    config = source / "config.json"
    if not config.is_file():
        raise SizingError("checkpoint config.json missing")
    try:
        import json

        cfg = json.loads(config.read_text(encoding="utf-8"))
        full_width = int(cfg["moe_intermediate_size"])
    except (KeyError, ValueError, OSError) as exc:
        raise SizingError(f"config moe_intermediate_size unreadable: {exc}") from exc
    if full_width <= 0 or full_width % 16 != 0:
        raise SizingError(f"moe_intermediate_size {full_width} must be a positive multiple of 16")

    shards = sorted(p for p in source.glob("*.safetensors") if p.is_file())
    if not shards:
        raise SizingError("checkpoint has no .safetensors shards")
    expert = 0
    protected = 0
    for shard in shards:
        header = read_safetensors_header(shard)
        if not isinstance(header, dict):
            raise SizingError(f"shard header is not an object: {shard.name}")
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(info, dict):
                continue
            dtype = info.get("dtype")
            shape = info.get("shape")
            if not isinstance(dtype, str) or not isinstance(shape, list):
                raise SizingError(f"malformed tensor record {name!r} in {shard.name}")
            size = _tensor_bytes(dtype, shape)
            if _classify(name):
                expert += size
            else:
                protected += size
    if expert <= 0:
        raise SizingError("no routed-expert FFN bytes found in checkpoint")
    return WidthSizing(
        full_width=full_width,
        expert_bytes=expert,
        protected_bytes=protected,
        shards_scanned=len(shards),
    )


__all__ = ["SizingError", "WidthSizing", "size_checkpoint_for_width"]
