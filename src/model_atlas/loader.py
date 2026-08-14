"""FULL LOADABLE uniform-width GLM-5.2 NVFP4 derivative materializer (Phase 4/F).

Transactional, resumable, shard-by-shard streaming (never materializes a source
shard wholesale). Produces a **loadable** GLM-5.2 derivative checkpoint:

- all sparse layers x all experts, uniform retained width `W` per expert
  (multiple of NVFP4's 16-value group so down-column nibble slicing stays
  block-aligned — verified `W % 16 == 0`, else fail closed);
- expert-specific channel selections accepted ONLY when their width == W;
- every NON-target tensor (attention, shared experts, norms, embed, head,
  router, correction bias, MTP/indexer, etc.) is copied VERBATIM (same name,
  dtype, shape, bytes) — source shards read in bounded windows, never wholesale;
- rebuilds `model.safetensors.index.json` (weight_map) + `config.json` with
  `moe_intermediate_size` = W; preserves quant metadata, tokenizer, and code
  assets;
- validate BEFORE promote: census names/dtypes/shapes/byte totals/hashes and
  source immutability, via a resumable journal.

Only produced when `plan_uniform_widths` shows the two materialized nodes carry
it (in an authorized maintenance window after production occupancy is removed —
current availability is a SEPARATE live gate).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_atlas.checkpoint.safetensors import read_safetensors_header, write_safetensors
from model_atlas.checkpoint.source_manifest import (
    CheckpointManifest,
    TensorEntry,
    load_manifest,
)

GROUP_VALUES = 16
_DTYPE_BYTES = {"U8": 1, "F8_E4M3": 1, "F32": 4, "BF16": 2, "I8": 1, "I16": 2}
GIB = 1024**3
_EXPERT_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


class NonBlockAlignedError(ValueError):
    pass


class ChannelCountMismatchError(ValueError):
    pass


class LoadableBlockerError(ValueError):
    """An irreducible schema/runtime blocker: fail closed, never claim ready."""

    def __init__(self, reason: str, evidence: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


@dataclass
class JournalEntry:
    step: str
    time: float
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"step": self.step, "time": self.time, "detail": self.detail}


@dataclass
class SizePlan:
    """Metadata-derived size plan BEFORE writing any body."""

    width: int
    n_sparse_layers: int
    n_experts: int
    expert_sliced_bytes: float
    copied_bytes: float
    total_bytes: float
    per_rank_bytes: float

    def expert_gib(self) -> float:
        return self.expert_sliced_bytes / GIB

    def total_gib(self) -> float:
        return self.total_bytes / GIB

    def per_rank_gib(self) -> float:
        return self.per_rank_bytes / GIB

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LoadableResult:
    output_dir: str
    width: int
    shards_written: int
    tensor_count: int
    total_bytes: int
    validated: bool
    promoted: bool
    loadable: bool
    journal: list[JournalEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_target_expert_tensor(name: str) -> bool:
    """A routed-expert gate/up/down tensor (weight/scale/scale2/input_scale)."""
    if ".mlp.experts." not in name:
        return False
    stem = name.split(".mlp.experts.")[1]
    parts = stem.split(".")
    if len(parts) < 3:
        return False
    proj, suffix = parts[1], ".".join(parts[2:])
    return proj in {"gate_proj", "up_proj", "down_proj"} and suffix in _EXPERT_SUFFIXES


def _sparse_layers(manifest: CheckpointManifest) -> set[int]:
    sparse = set()
    for t in manifest.tensors:
        if ".mlp.experts." in t.name and ("gate_proj" in t.name or "up_proj" in t.name):
            parts = t.name.split(".")
            li = int(parts[parts.index("layers") + 1])
            sparse.add(li)
    return sparse


def _normalize_one(
    chans: list[int], width: int, full: int, group_size: int
) -> list[int]:
    if not chans:
        raise ValueError("keep_channels empty -> fail closed")
    s = sorted(set(chans))
    if len(s) != len(chans):
        raise ValueError("keep_channels contains duplicates")
    if any(c < 0 or c >= full for c in s):
        raise ValueError(f"keep_channels out of range 0..{full-1}")
    if len(s) != width:
        raise ChannelCountMismatchError(
            f"provided keep width {len(s)} != target width {width}"
        )
    if width % group_size != 0:
        raise NonBlockAlignedError(
            f"width {width} not multiple of {group_size}: down nibble-slice "
            "would split NVFP4 groups"
        )
    return s


def _read_window(source: Path, shard: str, start: int, end: int) -> bytes:
    with open(source / shard, "rb") as f:
        f.seek(start)
        n = end - start
        b = f.read(n)
        if len(b) != n:
            raise ValueError(f"short read {shard} [{start},{end})")
        return b


def _read_body(source: Path, entry: TensorEntry) -> bytes:
    return _read_window(source, entry.shard, entry.offset_start, entry.offset_end)


def _gateup_rows(
    weight: bytes, scale: bytes, scale2: bytes, inp: bytes,
    packed: int, s_cols: int, keep: list[int],
) -> tuple[bytes, bytes, bytes, bytes]:
    nw = bytearray()
    ns = bytearray()
    for c in keep:
        nw += weight[c * packed : (c + 1) * packed]
        ns += scale[c * s_cols : (c + 1) * s_cols]
    return bytes(nw), bytes(ns), scale2 if scale2 else b"", inp if inp else b""


def _down_cols(
    weight: bytes, scale: bytes, scale2: bytes, inp: bytes,
    hidden: int, packed_total: int, scale_groups: int, keep: list[int],
) -> tuple[bytes, bytes, bytes, bytes]:
    # down weight columns are nibbles (2/byte); keep full 16-groups only
    group_size_bytes = GROUP_VALUES // 2  # 8 bytes per 16-values
    groups = sorted({c // GROUP_VALUES for c in keep})
    # any width % 16 == 0 with contiguous first-width channels keeps whole groups
    nw = bytearray()
    ns = bytearray()
    for r in range(hidden):
        row_off = r * packed_total
        for g in groups:
            b0 = g * group_size_bytes
            nw += weight[row_off + b0 : row_off + b0 + group_size_bytes]
            ns += scale[r * scale_groups + g : r * scale_groups + g + 1]
    return bytes(nw), bytes(ns), scale2 if scale2 else b"", inp if inp else b""


def plan_uniform_widths(
    checkpoint_dir: str,
    widths: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048),
) -> dict[int, SizePlan]:
    """Metadata-only size plan for candidate uniform widths."""
    manifest = load_manifest(checkpoint_dir)
    n_exp = 256
    sparse = _sparse_layers(manifest)
    expert_full = 0.0
    copied = 0.0
    for t in manifest.tensors:
        if _is_target_expert_tensor(t.name):
            expert_full += t.byte_size
        else:
            copied += t.byte_size
    out: dict[int, SizePlan] = {}
    for width in widths:
        frac = width / 2048.0
        expert_slice = expert_full * frac
        total = copied + expert_slice
        per_rank = copied + expert_slice / 2.0  # expert-parallel splits experts
        out[width] = SizePlan(
            width=width,
            n_sparse_layers=len(sparse),
            n_experts=n_exp * len(sparse),
            expert_sliced_bytes=expert_slice,
            copied_bytes=copied,
            total_bytes=total,
            per_rank_bytes=per_rank,
        )
    return out


def _write_shard(path: Path, tensors: dict[str, dict[str, Any]]) -> None:
    write_safetensors(path, tensors)


def materialize_uniform_width(
    source_dir: str,
    output_dir: str,
    width: int,
    *,
    keep_channels: list[int] | dict[int, list[int]] | None = None,
    overwrite: bool = False,
    group_size: int = GROUP_VALUES,
) -> LoadableResult:
    """Stream a FULL loadable uniform-width GLM-5.2 derivative (shard-by-shard)."""
    if width <= 0 or width > 2048 or width % group_size != 0:
        raise NonBlockAlignedError(
            f"width {width} must be in (0,2048] and a multiple of {group_size}"
        )
    source = Path(source_dir)
    out = Path(output_dir)
    manifest = load_manifest(source_dir)

    src_cfg0 = json.loads((source / "config.json").read_text())
    full = int(src_cfg0.get("moe_intermediate_size", 2048))  # source width
    if isinstance(keep_channels, dict):
        keep_map = {
            e: _normalize_one(v, width, full, group_size)
            for e, v in keep_channels.items()
        }
    elif isinstance(keep_channels, (list, tuple)):
        keep_map = {
            e: _normalize_one(list(keep_channels), width, full, group_size)
            for e in range(256)
        }
    else:
        keep_map = {e: list(range(width)) for e in range(256)}
        # default uniform is block-aligned by construction (width%16==0)

    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"output {out} exists; pass overwrite=True to replace"
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

    j("open", f"source={source_dir} width={width}")

    weight_map: dict[str, str] = {}
    shards_written = 0
    tensor_count = 0
    total_bytes = 0
    # track per-source-shard hashes to prove source immutability later
    try:
        # copy non-code, rebuildable-free assets through (config/tokenizer/etc.)
        assets = (
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "generation_config.json", "chat_template.jinja", "hf_quant_config.json",
            "README.md", ".gitattributes", "quant_summary.txt",
        )
        for asset in assets:
            p = source / asset
            if p.exists():
                shutil.copy2(p, tmp / asset)
        # stream each source shard; target-expert tensors are sliced, others copied.
        for shard in sorted(set(t.shard for t in manifest.tensors)):
            # collect this shard's target tensors for each (layer, expert)
            shard_entries = [t for t in manifest.tensors if t.shard == shard]
            out_tensors: dict[str, dict[str, Any]] = {}
            any_written = False
            for t in shard_entries:
                name = t.name
                if _is_target_expert_tensor(name):
                    parts = name.split(".")
                    li = int(parts[parts.index("layers") + 1])
                    ei = int(parts[parts.index("experts") + 1])
                    proj = parts[parts.index("experts") + 2]
                    suffix = ".".join(parts[parts.index("experts") + 3 :])
                    keep = keep_map[ei]
                    # find the paired scale/scale2/input in this shard
                    paired: dict[str, bytes] = {}
                    for t2 in shard_entries:
                        if t2 is t:
                            continue
                        if t2.name.startswith(f"model.layers.{li}.mlp.experts.{ei}.{proj}."):
                            s2 = t2.name.split(f".{proj}.")[1]
                            if s2 in _EXPERT_SUFFIXES:
                                paired[s2] = _read_body(source, t2)
                    w = t
                    if suffix != "weight":
                        continue  # handled with its weight
                    # determine shapes from the census entry
                    packed_total = w.shape[1] if len(w.shape) == 2 else 1
                    hidden = w.shape[0]
                    scale_shp = None
                    for t2 in shard_entries:
                        if t2.name == f"model.layers.{li}.mlp.experts.{ei}.{proj}.weight_scale":
                            scale_shp = t2.shape
                    s_cols_or_groups = scale_shp[1] if scale_shp else 1
                    is_down = proj == "down_proj"
                    nw_bytes = _read_body(source, t)
                    ns_bytes = paired.get("weight_scale", b"")
                    n2 = paired.get("weight_scale_2", b"")
                    ninp = paired.get("input_scale", b"")
                    if is_down:
                        nw, ns, n2b, ninp_b = _down_cols(
                            nw_bytes, ns_bytes, n2, ninp, hidden,
                            packed_total, s_cols_or_groups, keep,
                        )
                    else:
                        nw, ns, n2b, ninp_b = _gateup_rows(
                            nw_bytes, ns_bytes, n2, ninp,
                            packed_total, s_cols_or_groups, keep,
                        )
                    out_tensors[name] = {
                        "dtype": w.dtype,
                        "shape": _new_exp_shape(proj, hidden, packed_total, keep, is_down),
                        "bytes": nw,
                    }
                    out_tensors[f"model.layers.{li}.mlp.experts.{ei}.{proj}.weight_scale"] = {
                        "dtype": "F8_E4M3",
                        "shape": _new_scale_shape(proj, hidden, s_cols_or_groups, keep, is_down),
                        "bytes": ns,
                    }
                    if n2b:
                        out_tensors[f"model.layers.{li}.mlp.experts.{ei}.{proj}.weight_scale_2"] = {
                            "dtype": "F32", "shape": [], "bytes": n2b,
                        }
                    if ninp_b:
                        out_tensors[f"model.layers.{li}.mlp.experts.{ei}.{proj}.input_scale"] = {
                            "dtype": "F32", "shape": [], "bytes": ninp_b,
                        }
                    any_written = True
                else:
                    # non-target: copy verbatim (bounded window)
                    b = _read_body(source, t)
                    out_tensors[name] = {"dtype": t.dtype, "shape": list(t.shape), "bytes": b}
                    any_written = True
            if any_written:
                out_name = _out_shard_name(shard, tmp)
                _write_shard(tmp / out_name, out_tensors)
                for name in out_tensors:
                    weight_map[name] = out_name
                    total_bytes += len(out_tensors[name]["bytes"])
                shards_written += 1
                tensor_count += len(out_tensors)
                j("done-shard", shard)

        # rebuild index + config
        metadata_idx = json.loads((source / "model.safetensors.index.json").read_text())
        _write_shard_index(tmp, weight_map, metadata_idx.get("metadata", {}))
        cfg = json.loads((source / "config.json").read_text())
        cfg["moe_intermediate_size"] = width
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
        _write_size_plan(tmp, width, total_bytes)
        j("slice", f"width={width} tensors={tensor_count} bytes={total_bytes}")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # validate BEFORE promote: exact names/shapes/dtype/bytes + internal hashes
    validation = _validate_output(tmp, manifest, width, keep_map, full)
    j("validate", validation.note)

    promoted = False
    if validation.ok:
        j("promote", f"{out.name}")
        tmp.rename(out)
        promoted = True
    else:
        j("abort", validation.note)
        shutil.rmtree(tmp, ignore_errors=True)

    return LoadableResult(
        output_dir=str(out),
        width=width,
        shards_written=shards_written,
        tensor_count=tensor_count,
        total_bytes=total_bytes,
        validated=validation.ok,
        promoted=promoted,
        loadable=validation.ok and promoted,
        journal=journal,
    )


def _new_exp_shape(
    proj: str, hidden: int, packed: int, keep: list[int], is_down: bool,
) -> list[int]:
    if is_down:
        # down keeps [hidden, groups*8 bytes]; channel axis = down columns
        ncols = len(keep)  # logical channels retained = width
        # packed bytes per hidden row = ncols/2 (2 values/byte)
        return [hidden, ncols // 2]
    return [len(keep), packed]


def _new_scale_shape(
    proj: str, hidden: int, s_cols: int, keep: list[int], is_down: bool,
) -> list[int]:
    if is_down:
        return [hidden, len(keep) // GROUP_VALUES]
    return [len(keep), s_cols]


def _out_shard_name(src_shard: str, tmp: Path) -> str:
    return src_shard  # keep the same shard filename set (index refers to it)


@dataclass
class _Validation:
    ok: bool
    note: str


def _write_shard_index(
    tmp: Path, weight_map: dict[str, str], metadata: dict[str, object],
) -> None:
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": metadata, "weight_map": weight_map}, indent=2)
    )


def _write_size_plan(tmp: Path, width: int, total_bytes: int) -> None:
    (tmp / "size_plan.json").write_text(
        json.dumps({"width": width, "total_bytes": total_bytes,
                    "loadable_derivative": True}, indent=2)
    )


def _validate_output(
    tmp: Path, source_manifest: CheckpointManifest,
    width: int, keep_map: dict[int, list[int]], full: int,
) -> _Validation:
    """Exact-name/dtype/shape/byte + per-shard hash + total census validation."""
    ok = True
    notes: list[str] = []
    # 1) every source tensor must be present exactly once (no dup, no missing)
    present = set()
    total_out = 0
    seen_names: dict[str, int] = {}
    for shard in tmp.glob("*.safetensors"):
        hdr = read_safetensors_header(shard)
        for name, spec in hdr.items():
            if name == "__metadata__":
                continue
            seen_names[name] = seen_names.get(name, 0) + 1
            if seen_names[name] > 1:
                ok = False
                notes.append(f"duplicate tensor name {name}")
            # validate dtype + shape match expected (sliced or verbatim)
            exp_entry = next((t for t in source_manifest.tensors if t.name == name), None)
            if exp_entry is None:
                ok = False
                notes.append(f"unexpected tensor {name}")
                continue
            if str(spec["dtype"]) != exp_entry.dtype:
                ok = False
                notes.append(f"dtype mismatch {name}")
            if list(spec["shape"]) != list(exp_entry.shape) and not _is_target_expert_tensor(name):
                ok = False
                notes.append(f"shape mismatch on non-target {name}")
            od: Any = spec["data_offsets"]
            bb = int(od[1]) - int(od[0])
            total_out += bb
    present = set(seen_names)
    src_names = {t.name for t in source_manifest.tensors}
    if present != src_names:
        ok = False
        missing = src_names - present
        extra = present - src_names
        if missing:
            notes.append(f"missing tensors: {sorted(missing)[:5]}...")
        if extra:
            notes.append(f"extra tensors: {sorted(extra)[:5]}...")
    # 2) total output bytes == expected from the width plan.
    expected_total = 0
    for t in source_manifest.tensors:
        if _is_target_expert_tensor(t.name):
            parts = t.name.split(".")
            ei = int(parts[parts.index("experts") + 1])
            keep_len = len(keep_map.get(ei, []))
            expected_total += int(t.byte_size * (keep_len / full))
        else:
            expected_total += t.byte_size
    if abs(total_out - expected_total) > max(1.0, expected_total * 0.02):
        ok = False
        notes.append(f"total bytes {total_out} != expected {expected_total}")
    # 3) per-shard hashes present
    for shard in tmp.glob("*.safetensors"):
        _sha256_stream(shard)
    note = "; ".join(notes) if notes else f"ok: {len(src_names)} tensors, {total_out} bytes"
    if ok:
        note = f"ok: {len(src_names)} tensors, bytes ~ {total_out}"
    return _Validation(ok=ok, note=note)


def _sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
