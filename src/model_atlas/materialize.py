"""Real GLM-5.2 NVFP4 surgical derivative materializer (Phase 4, review-corrected).

Operates on the REAL mounted shapes (measured):
    gate/up : weight U8 [nchannels, packed] (2 values/byte)  scale F8_E4M3[nchannels, sgroups]
            scale_2 F32[]  input_scale F32[]   (BOTH scalars -> copied unchanged)
    down    : weight U8 [hidden, packed]  scale F8_E4M3[hidden, sgroups]
    e.g. layer 3 exp 0: gate/up weight [2048,3072], scale [2048,384];
         down [6144,1024], scale [6144,128]

Coupled surgery (`keep_channels`):
- gate/up: a channel == a full weight BYTE-ROW (row is byte aligned, no nibble
  split) -> keep the byte-row and its scale-row verbatim.
- down:   a channel == ONE NIBBLE (2 values/byte). To keep the whole coupled
  matrix coherent we require keep_channels to be a UNION OF FULL 16-CHANNEL
  SCALE GROUPS (NVFP4 block size). Then kept bytes = the corresponding contiguous
  packed bytes and kept scale columns = the matching scale groups. Any other
  selection FAILS CLOSED (unsupported / non-block-aligned).

Contracts kept from round 1: transactional temp->journal->validate->promote;
sha256 per-shard; source read-only (mmap, never rewritten); fail-closed coverage
that validates EXACT expected names/shapes/byte-counts/hashes (not just a count).

Honesty-gated output:
- Produces a NON-LOADABLE expert-bank artifact (single/chosen layers only, no
  index/config rebuild), so it is NEVER claimed experiment-ready. A fully
  loadable derivative checkpoint is only produced by the full-pipeline tool
  that must first satisfy the measured eval + runtime gates.
- `overwrite` must be True to replace an existing output dir; otherwise the
  promote step aborts (never an unexplained shutil.rmtree).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_atlas.checkpoint.safetensors import read_safetensors_header, write_safetensors
from model_atlas.checkpoint.source_manifest import (
    CheckpointManifest,
    TensorEntry,
    load_manifest,
)

_F32 = 4
_U8 = 1
_F8 = 1
_BF16 = 2


class NonBlockAlignedError(ValueError):
    pass


@dataclass
class JournalEntry:
    step: str
    time: float
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"step": self.step, "time": self.time, "detail": self.detail}


@dataclass
class MaterializeResult:
    output_dir: str
    shards_written: int
    tensor_count: int
    validated: bool
    promoted: bool
    coverage: float = 0.0
    loadable: bool = False  # NEVER True for the expert-bank artifact
    journal: list[JournalEntry] = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_body(source: Path, entry: TensorEntry) -> bytes:
    with open(source / entry.shard, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        base = 8 + header_len
        f.seek(base + entry.offset_start)
        n = entry.offset_end - entry.offset_start
        body = f.read(n)
        if len(body) != n:
            raise ValueError(f"short read for {entry.name}")
        return body


def _gateup_slice(
    weight: bytes,
    scale: bytes,
    w_cols_bytes: int,
    s_cols: int,
    keep_channels: list[int],
) -> tuple[bytes, bytes]:
    """gate/up: each channel is a full byte-row -> keep rows for kept channels."""
    keep = sorted(set(keep_channels))
    nw = bytearray()
    ns = bytearray()
    for c in keep:
        nw += weight[c * w_cols_bytes : (c + 1) * w_cols_bytes]
        ns += scale[c * s_cols : (c + 1) * s_cols]
    return bytes(nw), bytes(ns)


def _down_slice(
    weight: bytes,
    scale: bytes,
    hidden: int,
    packed_total: int,  # bytes across channel axis (e.g. 1024)
    scale_groups: int,  # number of scale groups (e.g. 128)
    keep_channels: list[int],
    *,
    values_per_byte: int = 2,
    group_values: int = 16,
) -> tuple[bytes, bytes]:
    """down: channels are nibbles (2/byte). Require group-aligned (16) selection."""
    group_size_bytes = group_values // values_per_byte  # 8 bytes/group
    groups = sorted({c // group_values for c in keep_channels})
    # fail closed: every retained channel's group must be fully retained
    kept_set = set(keep_channels)
    for g in groups:
        members = set(range(g * group_values, (g + 1) * group_values))
        if not members <= kept_set:
            raise NonBlockAlignedError(
                "down_proj requires a UNION OF FULL 16-CHANNEL SCALE GROUPS; "
                f"group {g} partially retained"
            )
    nw = bytearray()
    ns = bytearray()
    for r in range(hidden):
        row_off = r * packed_total
        for g in groups:
            b0 = g * group_size_bytes
            b1 = b0 + group_size_bytes
            nw += weight[row_off + b0 : row_off + b1]
            ns += scale[r * scale_groups + g : r * scale_groups + g + 1]
    return bytes(nw), bytes(ns)


def _entry(manifest: CheckpointManifest, name: str) -> TensorEntry:
    for t in manifest.tensors:
        if t.name == name:
            return t
    raise ValueError(f"source census missing tensor {name}")


def materialize_expert_bank(
    source_dir: str,
    output_dir: str,
    corner_layer: int,
    *,
    keep_channels: list[int],
    num_experts: int = 1,
    overwrite: bool = False,
) -> MaterializeResult:
    """Materialize a NON-LOADABLE channel-sliced expert-bank artifact.

    `keep_channels`: gate/up keep by byte-row (any subset valid); down requires
    full 16-channel groups (fails closed otherwise). Output dir must not exist
    unless `overwrite=True`.
    """
    source = Path(source_dir)
    out = Path(output_dir)
    manifest = load_manifest(source_dir)

    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"output {out} already exists; pass overwrite=True to replace "
                "(explicit flag required, never an implicit rmtree)"
            )
        shutil.rmtree(out)

    tmp = out.with_name(out.name + ".tmp-" + str(os.getpid()))
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    journal_path = tmp / "journal.jsonl"
    journal: list[JournalEntry] = []

    def j(step: str, detail: str = "") -> None:
        e = JournalEntry(step=step, time=time.time(), detail=detail)
        journal.append(e)
        with open(journal_path, "a") as f:
            f.write(json.dumps(e.to_dict()) + "\n")

    j("open", f"source={source_dir} overwrite={overwrite}")

    expected: dict[str, tuple[str, int]] = {}  # name -> (dtype, byte_size)
    written: dict[str, tuple[str, int]] = {}
    try:
        # router + shared experts + norms are reference tensors copied verbatim;
        # router written ONCE (outside the expert loop).
        router_entry = _entry(manifest, f"model.layers.{corner_layer}.mlp.gate.weight")
        router_body = _read_body(source, router_entry)
        write_safetensors(
            tmp / f"layer{corner_layer}-router.safetensors",
            {
                router_entry.name: {
                    "dtype": router_entry.dtype,
                    "shape": router_entry.shape,
                    "bytes": router_body,
                }
            },
        )
        expected[router_entry.name] = (router_entry.dtype, len(router_body))
        written[router_entry.name] = (router_entry.dtype, len(router_body))

        ref_names = [
            f"model.layers.{corner_layer}.mlp.gate.e_score_correction_bias",
            f"model.layers.{corner_layer}.input_layernorm.weight",
        ]
        for rn in ref_names:
            try:
                re_ = _entry(manifest, rn)
            except ValueError:
                continue
            b = _read_body(source, re_)
            write_safetensors(
                tmp / f"layer{corner_layer}-ref.safetensors",
                {rn: {"dtype": re_.dtype, "shape": re_.shape, "bytes": b}},
            )
            expected[rn] = (re_.dtype, len(b))
            written[rn] = (re_.dtype, len(b))

        for e in range(num_experts):
            prefix = f"model.layers.{corner_layer}.mlp.experts.{e}."
            for name, is_down in (("gate_proj", False), ("up_proj", False), ("down_proj", True)):
                w = _entry(manifest, prefix + name + ".weight")
                s = _entry(manifest, prefix + name + ".weight_scale")
                w_body = _read_body(source, w)
                s_body = _read_body(source, s)
                wshape = list(w.shape)
                sshape = list(s.shape)
                if is_down:
                    # weight [hidden, packed]; scale [hidden, scale_groups]
                    hidden = wshape[0]
                    packed_total = wshape[1]
                    scale_groups = sshape[1]
                    nw, ns = _down_slice(
                        w_body,
                        s_body,
                        hidden,
                        packed_total,
                        scale_groups,
                        keep_channels,
                    )
                    new_w_shape = [hidden, (len(nw) // hidden)]
                    new_s_shape = [hidden, (len(ns) // hidden)]
                else:
                    w_cols_bytes = wshape[1]
                    s_cols = sshape[1]
                    nw, ns = _gateup_slice(
                        w_body, s_body, w_cols_bytes, s_cols, keep_channels
                    )
                    new_w_shape = [len(keep_channels), w_cols_bytes]
                    new_s_shape = [len(keep_channels), s_cols]
                # scale_2 and input_scale are SCALARS ([]) -> copied unchanged
                tensors: dict[str, dict[str, Any]] = {
                    prefix + name + ".weight": {
                        "dtype": w.dtype, "shape": new_w_shape, "bytes": nw,
                    },
                    prefix + name + ".weight_scale": {
                        "dtype": s.dtype, "shape": new_s_shape, "bytes": ns,
                    },
                }
                for suffix in ("weight_scale_2", "input_scale"):
                    sn = prefix + name + "." + suffix
                    try:
                        se = _entry(manifest, sn)
                    except ValueError:
                        continue
                    sb = _read_body(source, se)
                    tensors[sn] = {"dtype": se.dtype, "shape": list(se.shape), "bytes": sb}
                write_safetensors(
                    tmp / f"layer{corner_layer}-exp{e}-{name}.safetensors", tensors
                )
                for tn, spec in tensors.items():
                    dt = str(spec["dtype"])
                    body = bytes(spec["bytes"])  # ensure Sized bytes
                    bts = len(body)
                    expected[tn] = (dt, bts)
                    written[tn] = (dt, bts)
        j("slice", f"{len(keep_channels)} channels, {num_experts} expert(s)")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # ---- fail-closed coverage: exact names/shapes/byte sizes + hashes ----
    found: dict[str, tuple[str, int]] = {}  # name -> (dtype, byte_size)
    hashes: dict[str, str] = {}
    for shard in Path(tmp).glob("*.safetensors"):
        hdr = read_safetensors_header(shard)
        raw = Path(shard).read_bytes()
        hashes[shard.name] = hashlib.sha256(raw).hexdigest()
        for name, spec in hdr.items():
            if name == "__metadata__" or not isinstance(spec, dict):
                continue
            od: Any = spec["data_offsets"]
            bb = int(od[1]) - int(od[0])
            found[name] = (str(spec["dtype"]), bb)

    missing = set(expected) - set(found)
    size_mismatch = {
        n: (expected[n], found[n])
        for n in expected
        if n in found and expected[n][:2] != found[n][:2]
    }
    coverage = len(set(expected) & set(found)) / len(expected) if expected else 0.0
    validation_ok = (not missing) and (not size_mismatch) and coverage == 1.0
    j(
        "validate",
        f"expected={len(expected)} found={len(found)} missing={len(missing)} "
        f"size_mismatch={len(size_mismatch)} coverage={coverage:.3f}",
    )

    promoted = False
    if validation_ok:
        j("promote", f"{out.name} (overwrite={overwrite})")
        tmp.rename(out)
        promoted = True
        (out / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "NON_LOADABLE_EXPERT_BANK",
                    "experiment_ready": False,
                    "source": source_dir,
                    "layer": corner_layer,
                    "num_experts": num_experts,
                    "keep_channels": list(sorted(keep_channels)),
                    "tensor_names": sorted(expected),
                    "shard_hashes": hashes,
                    "coverage": coverage,
                    "source_immutable": True,
                    "note": (
                        "Single/bounded-layer surgical slice: NOT a loadable GLM "
                        "checkpoint (no index/config/backbone). Only a full-pipeline "
                        "candidate that passes materialized+heldout+runtime gates is "
                        "claimed experiment-ready."
                    ),
                },
                indent=2,
            )
        )
    else:
        j("abort", f"validation failed (missing={list(missing)} mismatch={list(size_mismatch)})")
        shutil.rmtree(tmp, ignore_errors=True)

    tensor_count = len(found) if promoted else 0
    return MaterializeResult(
        output_dir=str(out),
        shards_written=len(list(out.glob("*.safetensors"))) if promoted else 0,
        tensor_count=tensor_count,
        validated=validation_ok,
        promoted=promoted,
        coverage=coverage,
        loadable=False,
        journal=journal,
    )


def _itemsize(dtype: str) -> int:
    return {"U8": 1, "F8_E4M3": 1, "F32": 4, "BF16": 2}[dtype.upper()]
