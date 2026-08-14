"""Full STRUCTURALLY-COMPLETE uniform-width GLM-5.2 NVFP4 derivative exporter.

NOT a runtime-loadable claim. Distinguishes:

- `structurally_complete`: a valid safetensors + index + config tree whose every
  source tensor is present exactly once with correct name/dtype/shape/bytes, and
  whose non-target tensors are byte-identical to source. This is what the
  exporter guarantees.
- `runtime_loadable`: FALSE until an authoritative backend probe proves the
  installed stack can decode/execute ModelOpt NVFP4. Standard vllm 0.21 +
  transformers 5.9.0 in this env expose NO ModelOpt-NVFP4 decoder (verified by
  source/dir probe: `grep -rl modelopt vllm/model_executor/...` etc.), so
  `runtime_loadable` stays False regardless of what we write.

Engineering requirements:
- SAFETENSORS data-base offset: every body read adds 8 + uint64(header_len).
- Production streaming writer (distinct from test-only `write_safetensors`):
  precomputes ordered specs + header, writes header, streams each body in bounded
  chunks — no shard-sized body dict/bytearray.
- Real resume: stable staging dir keyed by output+plan identity; shards written
  to `.partial` then fsync+rename; per-shard hash/census + plan fingerprint in a
  journal; on resume completed shards re-validated and skipped, corrupt shards
  rebuilt or fail closed.
- Exact O(1) validation with per-shard expected maps and 0% tolerance.
- Keep-map accepts (layer,expert) or layer->expert mapping; infers expert count +
  full width from config/census; requires uniform width W + exact aligned groups.
- Down slicing requires exact aligned 16-channel group unions (group order kept).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_atlas.checkpoint.safetensors import read_safetensors_header
from model_atlas.checkpoint.source_manifest import (
    CheckpointManifest,
    TensorEntry,
    load_manifest,
)

GROUP_VALUES = 16
GIB = 1024**3
_EXPERT_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")
_DTYPE_BYTES = {"U8": 1, "F8_E4M3": 1, "F32": 4, "BF16": 2, "I8": 1, "I16": 2}


class NonBlockAlignedError(ValueError):
    pass


class ChannelCountMismatchError(ValueError):
    pass


@dataclass
class JournalEntry:
    step: str
    time: float
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"step": self.step, "time": self.time, "detail": self.detail}


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: list[int]
    data_len: int


@dataclass
class SizePlan:
    """Exact metadata size plan (scalar quant tensors do NOT scale).
    Integer byte fields are authoritative; GiB fields are display-only and
    derived. Never convert through float GiB to compare totals."""
    width: int
    n_sparse_layers: int
    n_experts: int
    replicated_bytes: int
    sharded_expert_bytes: int
    total_bytes: int
    per_rank_bytes: int

    @property
    def replicated_gib(self) -> float:
        return self.replicated_bytes / GIB

    @property
    def sharded_expert_gib(self) -> float:
        return self.sharded_expert_bytes / GIB

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def per_rank_gib(self) -> float:
        return self.per_rank_bytes / GIB

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ExportResult:
    output_dir: str
    width: int
    shards_written: int
    tensor_count: int
    total_bytes: int
    structurally_complete: bool
    promoted: bool
    runtime_loadable: bool = False
    journal: list[JournalEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------- helpers ---
def _shard_data_base(path: Path) -> int:
    with open(path, "rb") as f:
        raw = f.read(8)
    (header_len,) = struct.unpack("<Q", raw)
    return 8 + int(header_len)


def _is_target_expert_tensor(name: str) -> bool:
    if ".mlp.experts." not in name:
        return False
    parts = name.split(".mlp.experts.")[1].split(".")
    if len(parts) < 3:
        return False
    proj, suffix = parts[1], ".".join(parts[2:])
    return proj in {"gate_proj", "up_proj", "down_proj"} and suffix in _EXPERT_SUFFIXES


def _layer_of(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def _expert_of(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("experts") + 1])


def _proj_suffix(name: str) -> tuple[str, str]:
    parts = name.split(".mlp.experts.")[1].split(".")
    return parts[1], ".".join(parts[2:])


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_geometry(
    manifest: CheckpointManifest, source_cfg: dict[str, object],
) -> tuple[int, int, list[int]]:
    _moes = source_cfg.get("moe_intermediate_size", 2048)
    _ne = source_cfg.get("n_routed_experts", source_cfg.get("num_experts", 256))
    full = int(str(_moes))
    n_exp = int(str(_ne))
    sparse = sorted(
        {_layer_of(t.name) for t in manifest.tensors if _is_target_expert_tensor(t.name)}
    )
    return full, n_exp, sparse


def _normalize_groups(keep: list[int]) -> list[int]:
    """Require exact union of 16-channel aligned groups (down nibble safety)."""
    if not keep:
        raise ValueError("empty keep")
    s = sorted(set(keep))
    if len(s) != len(keep):
        raise ValueError("duplicates in keep")
    groups: dict[int, list[int]] = {}
    for c in s:
        groups.setdefault(c // GROUP_VALUES, []).append(c)
    for g, ch in groups.items():
        if set(ch) != set(range(g * GROUP_VALUES, (g + 1) * GROUP_VALUES)):
            raise NonBlockAlignedError(
                f"block {g} partial ({len(ch)}/{GROUP_VALUES}): need exact aligned "
                "16-channel group union"
            )
    return s


def _build_keep_map(
    keep_channels: object, width: int, full: int, n_exp: int,
    sparse_layers: list[int],
) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    if keep_channels is None or (isinstance(keep_channels, (list, tuple)) and not keep_channels):
        # default: uniform first `width` channels for every (sparse, expert)
        gy = list(range(width))
        _normalize_groups(gy)
        for li in sparse_layers:
            for e in range(n_exp):
                out[(li, e)] = gy
        return out
    if isinstance(keep_channels, (list, tuple)):
        base = _normalize_groups(list(keep_channels))
        if len(base) != width:
            raise ChannelCountMismatchError(f"global width {len(base)} != {width}")
        for li in sparse_layers:
            for e in range(n_exp):
                out[(li, e)] = base
    elif isinstance(keep_channels, dict):
        layered = all(isinstance(k, int) for k in keep_channels)
        if layered:
            for li, by_exp in keep_channels.items():
                if not isinstance(by_exp, dict):
                    raise ValueError(f"layer {li} must map to expert->list")
                for e, ch in by_exp.items():
                    out[(int(li), int(e))] = _normalize_groups(list(ch))
        else:
            for key, ch in keep_channels.items():
                if isinstance(key, tuple) and len(key) == 2:
                    out[(int(key[0]), int(key[1]))] = _normalize_groups(list(ch))
                else:
                    raise ValueError(f"unknown keep key {key!r}")
        # FAIL CLOSED: partial mapping must cover every (sparse,expert) target.
        # No silent [0,width) fallback unless an explicit global fallback is
        # supplied elsewhere (not auto-inferred here).
        expected = {(li, e) for li in sparse_layers for e in range(n_exp)}
        missing = expected - set(out)
        if missing:
            raise ChannelCountMismatchError(
                f"keep map does not cover {len(missing)} (sparse,expert) targets: "
                "complete coverage required (or supply an explicit global fallback)"
            )
    else:
        raise ValueError("keep_channels must be list/dict/layer->expert")
    for key, v in out.items():
        if len(v) != width:
            raise ChannelCountMismatchError(f"{key} len {len(v)} != {width}")
        _normalize_groups(v)
    return out


def _project_target(
    name: str, t: TensorEntry, keep: list[int], width: int,
) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Return (dtype, out_shape, [(src_abs_start, src_abs_end), ...]) for a target.
    NOTE: src windows returned here are BODY-relative; caller adds data base."""
    proj, suffix = _proj_suffix(name)
    # weight_scale_2 / input_scale scalars unchanged
    if suffix not in ("weight", "weight_scale"):
        return t.dtype, list(t.shape), [(0, t.byte_size)]
    hidden = t.shape[0]
    if proj == "down_proj":
        groups = sorted({c // GROUP_VALUES for c in keep})
        if suffix == "weight":
            gbytes = GROUP_VALUES // 2  # 8 bytes/group
            packed_total = t.shape[1]
            w = []
            for r in range(hidden):
                row = r * packed_total
                for g in groups:
                    b0 = g * gbytes
                    w.append((row + b0, row + b0 + gbytes))
            return t.dtype, [hidden, len(groups) * gbytes], w
        if suffix == "weight_scale":
            s_cols = t.shape[1]
            w = []
            for r in range(hidden):
                for g in groups:
                    w.append((r * s_cols + g, r * s_cols + g + 1))
            return t.dtype, [hidden, len(groups)], w
        return t.dtype, list(t.shape), [(0, t.byte_size)]
    # gate/up
    if suffix == "weight":
        packed = t.shape[1]
        return t.dtype, [len(keep), packed], [(c * packed, (c + 1) * packed) for c in keep]
    if suffix == "weight_scale":
        s_cols = t.shape[1]
        return t.dtype, [len(keep), s_cols], [(c * s_cols, (c + 1) * s_cols) for c in keep]
    return t.dtype, list(t.shape), [(0, t.byte_size)]


def _expected_bytes(
    name: str, t: TensorEntry, keep_map: dict[tuple[int, int], list[int]], width: int,
) -> int:
    if not _is_target_expert_tensor(name):
        return t.byte_size
    key = (_layer_of(name), _expert_of(name))
    keep = keep_map[key]
    proj, suffix = _proj_suffix(name)
    # weight_scale_2 / input_scale are scalars ([]) -> unchanged
    if suffix not in ("weight", "weight_scale"):
        return t.byte_size
    hidden = t.shape[0]
    if proj == "down_proj":
        if suffix == "weight":
            return hidden * (len(keep) // 2)
        return hidden * (len(keep) // GROUP_VALUES)
    return len(keep) * t.shape[1]


# ------------------------------------------------- production streaming writer --
def production_write_shard(path: Path, specs: list[TensorSpec], body_provider: Any) -> None:
    """Streaming production safetensors writer (distinct from test-only helper).

    SAFETENSORS `data_offsets` are RELATIVE to the start of the tensor-data
    buffer (file position = 8 + header_len + offset). Header is written first,
    then each body is streamed through `body_provider` sequentially in bounded
    chunks, so body offset `a` corresponds to buffer position `a`. Never
    materializes a shard-sized body dict/bytearray.
    """
    # relative offsets: body a starts at buffer position `a`
    cursor = 0
    rel: dict[str, tuple[int, int]] = {}
    for sp in specs:
        rel[sp.name] = (cursor, cursor + sp.data_len)
        cursor += sp.data_len
    header_bytes = json.dumps(
        {
            "__metadata__": {},
            **{
                sp.name: {
                    "dtype": sp.dtype,
                    "shape": list(sp.shape),
                    "data_offsets": list(rel[sp.name]),
                }
                for sp in specs
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for sp in specs:
            # stream each body at this buffer position (provider handles source)
            body_provider(sp.name, rel[sp.name][0], sp.data_len, f)
        f.flush()
        os.fsync(f.fileno())


def _write_partial_and_finalize(
    staging: Path, shard: str, specs: list[TensorSpec], build_body: Any,
) -> str:
    partial = staging / (shard + ".partial")
    # write ONCE to .partial (production_write_shard fsyncs), then atomic rename
    production_write_shard(partial, specs, build_body)
    final = staging / shard
    if final.exists():
        final.unlink()
    partial.rename(final)
    return _fingerprint(final)


# ------------------------------------------------------------ sizing / main ----
def plan_exact_sizes(
    manifest: CheckpointManifest, source_cfg: dict[str, object],
    keep_map: dict[tuple[int, int], list[int]],
    sparse_layers: list[int] | None = None,
) -> SizePlan:
    full, n_exp, sl = _infer_geometry(manifest, source_cfg)
    if sparse_layers is not None:
        sl = sorted(set(sparse_layers))
    exp_full = 0.0
    replicated = 0.0
    src_by_name = {t.name: t for t in manifest.tensors}
    for name, t in src_by_name.items():
        if _is_target_expert_tensor(name):
            key = (_layer_of(name), _expert_of(name))
            if key not in keep_map:
                replicated += t.byte_size
                continue
            exp_full += _expected_bytes(name, t, keep_map, width=_width_of(keep_map))
        else:
            replicated += t.byte_size
    exp_full_i = int(exp_full)
    replicated_i = int(replicated)
    total_i = replicated_i + exp_full_i
    per_rank_i = replicated_i + exp_full_i // 2
    return SizePlan(
        width=_width_of(keep_map) if keep_map else 0,
        n_sparse_layers=len(sl),
        n_experts=n_exp * len(sl),
        replicated_bytes=replicated_i,
        sharded_expert_bytes=exp_full_i,
        total_bytes=total_i,
        per_rank_bytes=per_rank_i,
    )


def _width_of(keep_map: dict[tuple[int, int], list[int]]) -> int:
    if not keep_map:
        return 0
    return len(next(iter(keep_map.values())))


def materialize_uniform_width(
    source_dir: str,
    output_dir: str,
    width: int,
    *,
    keep_channels: object | None = None,
    overwrite: bool = False,
) -> ExportResult:
    """Produce a STRUCTURALLY-COMPLETE uniform-width GLM-5.2 derivative.

    `runtime_loadable` stays False — the installed stack can't decode ModelOpt
    NVFP4 (verified); `structurally_complete` is what this exporter guarantees.
    """
    if width <= 0 or width % GROUP_VALUES != 0:
        raise NonBlockAlignedError(f"width {width} must be a nonzero multiple of {GROUP_VALUES}")
    source = Path(source_dir)
    out = Path(output_dir)
    manifest = load_manifest(source_dir)
    source_cfg = json.loads((source / "config.json").read_text())
    full, n_exp, sparse_layers = _infer_geometry(manifest, source_cfg)
    keep_map = _build_keep_map(keep_channels, width, full, n_exp, sparse_layers)

    staging = out.parent / f".{out.name}.staging-w{width}"

    # canonical plan fingerprint: source config/index identity + width + keep map
    plan_fp = json.dumps({
        "source_cfg": json.loads(json.dumps(source_cfg, sort_keys=True)),
        "width": width,
        "keep": {f"{k[0]}:{k[1]}": (sorted(v),) for k, v in sorted(keep_map.items())},
    }, sort_keys=True)

    if out.exists():
        if not overwrite:
            raise FileExistsError("output exists; pass overwrite=True")
        shutil.rmtree(out)

    # Preserve valid staging; do NOT delete on normal entry. Validate plan
    # identity: mismatch must fail closed unless explicit overwrite clears it.
    plan_path = staging / "plan.json"
    if plan_path.exists():
        prior = plan_path.read_text().strip()
        if prior != plan_fp:
            if not overwrite:
                raise ValueError(
                    "staging plan identity mismatch; pass overwrite=True to clear stale plan"
                )
            shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=True)
    if not staging.exists():
        staging.mkdir(parents=True, exist_ok=True)

    # write canonical plan only if not already present (do not blindly overwrite)
    if not plan_path.exists():
        plan_path.write_text(plan_fp)

    journal_path = staging / "journal.jsonl"
    journal: list[JournalEntry] = []
    shard_hashes: dict[str, str] = {}

    def j(step: str, detail: str = "") -> None:
        e = JournalEntry(step=step, time=time.time(), detail=detail)
        journal.append(e)
        with open(journal_path, "a") as f:
            f.write(json.dumps(e.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if step == "shard-final":
            sp, h = detail.split(" ", 1)
            shard_hashes[sp] = h

    # resume: load prior journal (stable staging) with plan identity verified
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("step") == "shard-final":
                sp, h = d["detail"].split(" ", 1)
                shard_hashes[sp] = h
            elif d.get("step") == "shard-partial":
                pass

    j("open", f"source={source_dir} width={width}")

    # pre source fingerprints (immutable proof)
    pre_src = {sh: _fingerprint(source / sh) for sh in sorted({t.shard for t in manifest.tensors})}

    weight_map: dict[str, str] = {}
    shards_written = 0
    tensor_count = 0
    total_bytes = 0

    assets = (
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "generation_config.json", "chat_template.jinja", "hf_quant_config.json",
        "README.md", ".gitattributes", "quant_summary.txt",
    )
    try:
        for asset in assets:
            p = source / asset
            if p.exists():
                _copy_stream(p, staging / asset)
        for shard in sorted({t.shard for t in manifest.tensors}):
            # resume: skip only if finalized AND hash matches; else rebuild
            final_path = staging / shard
            if final_path.exists() and shard_hashes.get(shard) == _fingerprint(final_path):
                # rebuild output census from the finalized shard (not just skip)
                specs_skip, _ = _plan_output_shard(source, manifest, shard, keep_map, width)
                for sp in specs_skip:
                    weight_map[sp.name] = shard
                    total_bytes += sp.data_len
                tensor_count += len(specs_skip)
                shards_written += 1
                j("skip-shard", shard)
                continue
            specs, build_body = _plan_output_shard(source, manifest, shard, keep_map, width)
            if not specs:
                continue
            h = _write_partial_and_finalize(staging, shard, specs, build_body)
            j("shard-final", f"{shard} {h}")
            for sp in specs:
                weight_map[sp.name] = shard
                total_bytes += sp.data_len
            shards_written += 1
            tensor_count += len(specs)
        _write_index(staging, weight_map, source_cfg, width)
        j("slice", f"width={width} tensors={tensor_count} bytes={total_bytes}")
    except Exception:
        # preserve staging on interruption/failure (do NOT delete)
        raise

    validation = _exact_validate(staging, manifest, keep_map, source_cfg, width, pre_src)
    j("validate", validation.note)
    promoted = False
    structurally = False
    if validation.ok:
        j("promote", f"{out.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
        promoted = True
        structurally = True
    else:
        j("abort", validation.note)
    return ExportResult(
        output_dir=str(out), width=width, shards_written=shards_written,
        tensor_count=tensor_count, total_bytes=total_bytes,
        structurally_complete=structurally, promoted=promoted,
        runtime_loadable=False, journal=journal,
    )


# ------------------------------------------------- streaming body + validate ---
def _copy_stream(src: Path, dst: Path, chunk: int = 4 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            b = fi.read(chunk)
            if not b:
                break
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())


def _stream_body_window(
    src: Path, abs_start: int, abs_end: int, dst: Any,
    dst_off: int, size: int,
) -> None:
    with open(src, "rb") as f:
        f.seek(abs_start)
        rem = abs_end - abs_start
        while rem > 0:
            take = f.read(min(4 << 20, rem))
            if not take:
                raise ValueError("short")
            dst.write(take)
            rem -= len(take)


def _plan_output_shard(
    source: Path, manifest: CheckpointManifest, shard: str,
    keep_map: dict[tuple[int, int], list[int]], width: int,
) -> tuple[list[TensorSpec], Any]:
    """Plan an output shard's ordered specs + streaming body writer (data-base
    aware). Returns (specs, build_body)."""
    base = _shard_data_base(source / shard)
    shard_entries = sorted((t for t in manifest.tensors if t.shard == shard), key=lambda t: t.name)
    specs: list[TensorSpec] = []
    bodies: dict[str, list[tuple[int, int]]] = {}
    for t in shard_entries:
        name = t.name
        # tensor data buffer begins at file base + offset_start
        tbase = base + t.offset_start
        if _is_target_expert_tensor(name):
            key = (_layer_of(name), _expert_of(name))
            keep = keep_map.get(key)
            if keep is None:
                continue
            dtype, shape, win = _project_target(name, t, keep, width)
            # map tensor-data-relative windows to ABSOLUTE file offsets
            abs_win = [(tbase + a, tbase + b) for (a, b) in win]
            total = sum(b - a for a, b in abs_win)
            specs.append(TensorSpec(name, dtype, shape, total))
            bodies[name] = abs_win
        else:
            specs.append(TensorSpec(name, t.dtype, list(t.shape), t.byte_size))
            # non-target: whole body [offset_start, offset_end)
            bodies[name] = [(tbase, tbase + t.byte_size)]

    def build_body(name: str, start: int, size: int, dst: Any) -> None:
        _ = start
        for (a, b) in bodies[name]:
            _stream_body_window(source / shard, a, b, dst, 0, b - a)

    return specs, build_body


def _write_index(
    staging: Path, weight_map: dict[str, str], source_cfg: dict[str, object], width: int,
) -> None:
    cfg = json.loads(json.dumps(source_cfg))
    cfg["moe_intermediate_size"] = width
    (staging / "config.json").write_text(json.dumps(cfg, indent=2))
    (staging / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )


def _exact_validate(
    staging: Path, manifest: CheckpointManifest,
    keep_map: dict[tuple[int, int], list[int]], source_cfg: dict[str, object],
    width: int, pre_src_hashes: dict[str, str],
) -> _Validation:
    ok = True
    notes: list[str] = []
    src_by_name = {t.name: t for t in manifest.tensors}
    present: set[str] = set()
    total_out = 0
    seen: dict[str, int] = {}
    for shard in sorted(staging.glob("*.safetensors")):
        hdr = read_safetensors_header(shard)
        for name, spec in hdr.items():
            if name == "__metadata__":
                continue
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                ok = False
                notes.append(f"duplicate {name}")
            s_entry = src_by_name.get(name)
            if s_entry is None:
                ok = False
                notes.append(f"unexpected {name}")
                continue
            if str(spec["dtype"]) != s_entry.dtype:
                ok = False
                notes.append(f"dtype {name}")
            exp_shape = _expected_shape(name, s_entry, keep_map, width)
            if list(spec["shape"]) != exp_shape:
                ok = False
                notes.append(f"shape {name}")
            od = spec["data_offsets"]
            bb = int(od[1]) - int(od[0])
            exp_bytes = _expected_bytes(name, s_entry, keep_map, width)
            if bb != exp_bytes:
                ok = False
                notes.append(f"bytes {name}: {bb}!={exp_bytes}")
            total_out += bb
            present.add(name)
    src_names = set(src_by_name)
    if present != src_names:
        ok = False
        notes.append(
            f"missing {sorted(src_names-present)[:3]} extra {sorted(present-src_names)[:3]}"
        )
    from model_atlas.loader import plan_exact_sizes

    sp = plan_exact_sizes(manifest, source_cfg, keep_map)
    exp_total = int(sp.total_gib * GIB)
    if abs(total_out - exp_total) > 0:
        ok = False
        notes.append(f"total {total_out} != {exp_total}")
    # source immutability
    for sh, pre in pre_src_hashes.items():
        try:
            post = _fingerprint(Path(manifest.checkpoint_dir) / sh)
        except OSError:
            post = None
        if post is not None and post != pre:
            ok = False
            notes.append(f"source {sh} changed")
    return _Validation(ok=ok, note="; ".join(notes) if notes else f"ok {len(src_names)} tensors")


def _expected_shape(
    name: str, s_entry: TensorEntry,
    keep_map: dict[tuple[int, int], list[int]], width: int,
) -> list[int]:
    if not _is_target_expert_tensor(name):
        return list(s_entry.shape)
    key = (_layer_of(name), _expert_of(name))
    keep = keep_map[key]
    proj, suffix = _proj_suffix(name)
    if suffix not in ("weight", "weight_scale"):
        return list(s_entry.shape)  # scalar unchanged
    hidden = s_entry.shape[0]
    if proj == "down_proj":
        if suffix == "weight":
            return [hidden, len(keep) // 2]
        return [hidden, len(keep) // GROUP_VALUES]
    return [len(keep), s_entry.shape[1]]


class _Validation:
    def __init__(self, ok: bool, note: str) -> None:
        self.ok = ok
        self.note = note
