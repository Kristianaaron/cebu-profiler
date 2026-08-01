"""Dependency-free Safetensors header reader and (test-only) writer.

The reader loads only the 8-byte header length and the JSON header — never the
tensor bodies — so we can census an oversized checkpoint without materializing
it. The writer exists for synthetic fixtures only.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

_HEADER_LEN = 8


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Return the Safetensors header dict without reading tensor payloads."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(_HEADER_LEN))
        header_bytes = f.read(header_len)
    return cast(dict[str, Any], json.loads(header_bytes.decode("utf-8")))


def write_safetensors(path: str | Path, tensors: dict[str, dict[str, Any]]) -> None:
    """Write a small Safetensors file. `tensors`: name -> {dtype, shape, bytes}.

    For synthetic fixtures only — not used to produce real checkpoints.
    """
    header: dict[str, Any] = {"__metadata__": {}}
    data = bytearray()
    for name, spec in tensors.items():
        body = spec["bytes"]
        start = len(data)
        end = start + len(body)
        header[name] = {
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "data_offsets": [start, end],
        }
        data += body
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(data))
