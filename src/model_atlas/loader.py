"""Full STRUCTURALLY-COMPLETE uniform-width GLM-5.2 NVFP4 derivative exporter.

NOT a runtime-loadable claim. Distinguishes:

- `structurally_complete`: a valid safetensors + index + config tree whose every
  source tensor is present exactly once with correct name/dtype/shape/bytes, and
  whose non-target tensors are byte-identical to source. This is what the
  exporter guarantees.
- `runtime_compatibility`: typed status from the installed stack
  (e.g. `schema-supported-unvalidated`). The installed vLLM (0.21) DOES contain a
  ModelOpt-NVFP4 path (ModelOptNvFp4Config + Linear/FusedMoE methods + kernels/
  emulation), so this is NOT decoder-blocked. `runtime_validated` stays False until
  a real materialized derivative load/forward is validated.

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
- Keep-map accepts a global list (uniform), a layer->expert dict, or a
  (layer,expert)->list dict; expert count + full width inferred from config/
  census; requires uniform width W + complete coverage (partial mapping FAILS
  CLOSED unless an explicit global list is supplied). Channel groups are kept in
  CANONICAL ASCENDING order (sorted by group, then channel); not the caller's
  arbitrary order.
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
    runtime_compatibility: str = "schema-supported-unvalidated"
    runtime_validated: bool = False
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
    # GLM-5.2 has a final shared-head block (layer == num_hidden_layers) whose
    # experts are BF16 and carry NO NVFP4 weight_scale/input_scale -> not part of
    # the 256-routed-expert NVFP4 geometry. Exclude it from the target set.
    parts = name.split(".")
    if "layers" in parts:
        try:
            li = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            li = None
        if li is not None and li >= 78:  # final head block
            return False
    else:
        return False
    sp = name.split(".mlp.experts.")[1].split(".")
    if len(sp) < 3:
        return False
    proj, suffix = sp[1], ".".join(sp[2:])
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


def _hash_small(path: Path) -> str:
    """Hash a small metadata file (config/index); safe (bounded, metadata-sized).
    Never used on large shards."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        {
            _layer_of(t.name)
            for t in manifest.tensors
            if _is_target_expert_tensor(t.name) and t.name.endswith(".weight")
        }
    )
    return full, n_exp, sparse


def _normalize_groups(keep: list[int], full: int) -> list[int]:
    """Require exact union of 16-channel aligned groups within [0, full).
    Fails closed on empty / duplicates / negatives / >= full / partial blocks,
    and returns CANONICAL ASCENDING order (group order ascending; not the
    caller's arbitrary order)."""
    if not isinstance(keep, list) or not keep:
        raise ValueError("empty keep")
    if not all(isinstance(c, int) for c in keep):
        raise ValueError("keep channels must be int")
    s = sorted(set(keep))
    if len(s) != len(keep):
        raise ValueError("duplicates in keep")
    if any(c < 0 for c in s):
        raise NonBlockAlignedError(f"negative channel in keep: {s}")
    if any(c >= full for c in s):
        raise NonBlockAlignedError(
            f"channel(s) >= full({full}) in keep: {[c for c in s if c >= full]}"
        )
    # every block must be fully (and only) retained within [0, full)
    groups: dict[int, list[int]] = {}
    for c in s:
        g = c // GROUP_VALUES
        groups.setdefault(g, []).append(c)
    for g, ch in groups.items():
        if g * GROUP_VALUES >= full:
            raise NonBlockAlignedError(f"block {g} starts at >= full({full})")
        expected = set(range(g * GROUP_VALUES, (g + 1) * GROUP_VALUES))
        if set(ch) != expected:
            raise NonBlockAlignedError(
                f"block {g} partial ({len(ch)}/{GROUP_VALUES}): need exact aligned "
                "16-channel group union"
            )
    return s  # canonical ascending (sorted(group ascending))


def _precheck_geometry(
    manifest: CheckpointManifest,
    sparse_layers: list[int],
    n_exp: int,
    full: int,
    hidden: int,
) -> None:
    """Round-6 #1: verify every sparse (layer,expert,proj) weight/scale/scalars
    against the REAL mounted GLM NVFP4 layout, BEFORE any body write.

    Mounted facts: hidden=6144, full=2048, 2 FP4 values/byte, group=16 values.
      gate/up weight U8 [full, hidden/2]   (bytes: hidden/2 per channel row)
      gate/up scale  F8 [full, hidden/16]  (groups: hidden/16 per channel row)
      down  weight U8 [hidden, full/2]
      down  scale  F8 [hidden, full/16]
    Requires full % 16 == 0 and hidden % 16 == 0; weight U8 + scale F8_E4M3;
    both weight_scale_2 and input_scale scalars (shape []) of dtype F32 exist.
    Also rejects duplicate source tensor names."""
    if full <= 0 or full % GROUP_VALUES != 0:
        raise ValueError(f"full({full}) must be positive and a multiple of {GROUP_VALUES}")
    if hidden <= 0 or hidden % GROUP_VALUES != 0:
        raise ValueError(f"hidden({hidden}) must be positive and a multiple of {GROUP_VALUES}")

    # duplicate source tensor names
    seen_names: set[str] = set()
    for t in manifest.tensors:
        if t.name in seen_names:
            raise ValueError(f"duplicate source tensor name {t.name}")
        seen_names.add(t.name)

    by: dict[tuple[int, int, str], dict[str, TensorEntry]] = {}
    for t in manifest.tensors:
        if not _is_target_expert_tensor(t.name):
            continue
        key = (_layer_of(t.name), _expert_of(t.name), _proj_of(t.name))
        by.setdefault(key, {})[t.name.rsplit(".", 1)[1]] = t
    for li in sparse_layers:
        for ei in range(n_exp):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                rec = by.get((li, ei, proj))
                if rec is None:
                    raise ValueError(
                        f"missing projection {proj} for sparse layer {li} expert {ei}"
                    )
                w = rec.get("weight")
                s = rec.get("weight_scale")
                s2 = rec.get("weight_scale_2")
                isc = rec.get("input_scale")
                if w is None:
                    raise ValueError(f"missing weight for l{li}e{ei}.{proj}")
                if s is None:
                    raise ValueError(f"missing weight_scale for l{li}e{ei}.{proj}")
                if s2 is None:
                    raise ValueError(f"missing weight_scale_2 for l{li}e{ei}.{proj}")
                if isc is None:
                    raise ValueError(f"missing input_scale for l{li}e{ei}.{proj}")
                if w.dtype.upper() != "U8":
                    raise ValueError(f"{proj} {li}e{ei} weight dtype {w.dtype} != U8")
                if s.dtype.upper() not in ("F8_E4M3", "F8", "U8"):
                    raise ValueError(f"{proj} {li}e{ei} scale dtype {s.dtype} not F8")
                for scalar, nm in ((s2, "weight_scale_2"), (isc, "input_scale")):
                    if list(scalar.shape) != [] or scalar.dtype.upper() != "F32":
                        raise ValueError(
                            f"{proj} {li}e{ei} {nm} expected scalar F32, got "
                            f"shape {list(scalar.shape)} dtype {scalar.dtype}"
                        )
                wh = list(w.shape)
                sh_s = list(s.shape)
                if proj == "down_proj":
                    if len(wh) != 2 or wh[0] != hidden or wh[1] != full // 2:
                        raise ValueError(
                            f"down {li}e{ei} weight {wh} != [hidden={hidden}, full/2={full//2}]"
                        )
                    if len(sh_s) != 2 or sh_s[0] != hidden or sh_s[1] != full // GROUP_VALUES:
                        raise ValueError(
                            f"down {li}e{ei} scale {sh_s} != "
                            f"[hidden={hidden}, full/16={full//GROUP_VALUES}]"
                        )
                else:
                    if len(wh) != 2 or wh[0] != full or wh[1] != hidden // 2:
                        raise ValueError(
                            f"{proj} {li}e{ei} weight {wh} != [full={full}, hidden/2={hidden//2}]"
                        )
                    if len(sh_s) != 2 or sh_s[0] != full or sh_s[1] != hidden // GROUP_VALUES:
                        raise ValueError(
                            f"{proj} {li}e{ei} scale {sh_s} != "
                            f"[full={full}, hidden/16={hidden//GROUP_VALUES}]"
                        )


def _proj_of(name: str) -> str:
    return name.split(".mlp.experts.")[1].split(".")[1]


def _build_keep_map(
    keep_channels: object, width: int, full: int, n_exp: int,
    sparse_layers: list[int],
) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    if keep_channels is None or (isinstance(keep_channels, (list, tuple)) and not keep_channels):
        # default: uniform first `width` channels for every (sparse, expert)
        gy = list(range(width))
        _normalize_groups(gy, full)
        for li in sparse_layers:
            for e in range(n_exp):
                out[(li, e)] = gy
        return out
    if isinstance(keep_channels, (list, tuple)):
        base = _normalize_groups(list(keep_channels), full)
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
                    out[(int(li), int(e))] = _normalize_groups(list(ch), full)
        else:
            for key, ch in keep_channels.items():
                if isinstance(key, tuple) and len(key) == 2:
                    out[(int(key[0]), int(key[1]))] = _normalize_groups(list(ch), full)
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
        _normalize_groups(v, full)
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
    header: dict[str, object] = {"__metadata__": {}}
    for sp in specs:
        header[sp.name] = {
            "dtype": sp.dtype,
            "shape": list(sp.shape),
            "data_offsets": list(rel[sp.name]),
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    close = getattr(body_provider, "close", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for sp in specs:
                # stream each body at this buffer position (provider handles source)
                body_provider(sp.name, rel[sp.name][0], sp.data_len, f)
            f.flush()
            os.fsync(f.fileno())
    finally:
        # idempotently close any resource-bearing provider (plain callable test
        # providers have no close -> no-op). Guarantees a per-shard source handle
        # is released even when a body writer raises.
        if callable(close):
            close()


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
    exp_full = 0
    replicated = 0
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
    total_i = replicated + exp_full
    per_rank_i = replicated + exp_full // 2
    return SizePlan(
        width=_width_of(keep_map) if keep_map else 0,
        n_sparse_layers=len(sl),
        n_experts=n_exp * len(sl),
        replicated_bytes=replicated,
        sharded_expert_bytes=exp_full,
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
    _test_hook: Any = None,
) -> ExportResult:
    """Produce a STRUCTURALLY-COMPLETE uniform-width GLM-5.2 derivative.

    `runtime_compatibility` reports schema-support status from the installed
    stack (see runtimeprobe); `runtime_validated` stays False until a real
    materialized derivative load/forward is validated. A structurally complete
    export + passing config/schema probe is NOT decoder-blocked, but it is also
    not end-to-end runtime validated.
    """
    source = Path(source_dir)
    out = Path(output_dir)
    manifest = load_manifest(source_dir)
    source_cfg = json.loads((source / "config.json").read_text())
    full, n_exp, sparse_layers = _infer_geometry(manifest, source_cfg)
    hidden = int(str(source_cfg.get("hidden_size", 0)))
    if hidden <= 0:
        raise ValueError("config missing/zero hidden_size")
    if width <= 0 or width > full or width % GROUP_VALUES != 0:
        raise NonBlockAlignedError(
            f"width {width} must satisfy 0 < width <= full({full}) and be a "
            f"multiple of {GROUP_VALUES}"
        )
    keep_map = _build_keep_map(keep_channels, width, full, n_exp, sparse_layers)
    _precheck_geometry(manifest, sparse_layers, n_exp, full, hidden)

    staging = out.parent / f".{out.name}.staging-w{width}"

    # canonical plan fingerprint: source config/index identity + width + keep map
    idx_path = source / "model.safetensors.index.json"
    index_sha = _hash_small(idx_path) if idx_path.exists() else ""
    cfg_sha = _hash_small(source / "config.json")
    shards_meta = {}
    for sh in sorted({t.shard for t in manifest.tensors}):
        try:
            st = os.stat(source / sh)
            shards_meta[sh] = {"size": st.st_size, "mtime": st.st_mtime_ns}
        except OSError as exc:
            raise ValueError(f"source shard {sh} unreadable: {exc}") from exc
    plan_fp = json.dumps({
        "source_path": str(source.resolve()),
        "index_sha256": index_sha,
        "config_sha256": cfg_sha,
        "width": width,
        "keep": {f"{k[0]}:{k[1]}": (sorted(v),) for k, v in sorted(keep_map.items())},
        "shards": shards_meta,
    }, sort_keys=True)

    if out.exists() and not overwrite:
        raise FileExistsError("output exists; pass overwrite=True")
    # NOTE: with overwrite=True we DO NOT delete the prior output now. It stays
    # in place throughout materialization and is only swapped on successful
    # validation (atomic rename old -> backup, staging -> output, then remove
    # backup). On any failure/interruption the old output remains intact.

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

    j("open", f"source={source_dir} width={width} io_chunk={_IO_CHUNK}")

    # source immutability tracked via STAT identity (lightweight; no full-file
    # re-hash of the 503GB source). Optional full hashing is not the default.
    pre_src = {sh: os.stat(source / sh) for sh in sorted({t.shard for t in manifest.tensors})}

    weight_map: dict[str, str] = {}
    shards_written = 0
    tensor_count = 0
    total_bytes = 0

    # Rebuilt files (exporter writes them) + runtime artifacts that must not be
    # copied: config.json, model.safetensors.index.json, journal/plan/manifests.
    _REBUILT = {"config.json", "model.safetensors.index.json",
                "journal.jsonl", "plan.json", "completed_shards.json",
                "size_plan.json", "artifact_manifest.json", "derivative_manifest.json"}
    try:
        # Copy every SAFE top-level asset (all non-shard files) byte-for-byte,
        # except the files we deliberately rebuild above. Do not follow symlinks
        # that point outside the source directory.
        for asset in sorted(p.name for p in source.iterdir() if p.is_file()):
            if asset in _REBUILT or asset.endswith(".safetensors"):
                continue
            srcp = source / asset
            # refuse symlinks resolving outside the source root
            try:
                target = srcp.resolve()
                if not target.is_relative_to(source.resolve()):
                    continue  # symlink resolves outside source -> skip
            except OSError:
                continue
            _copy_stream(srcp, staging / asset)
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
            # test-only hook: raise after a finalized shard to simulate a genuine
            # interruption mid-export (staging is preserved for resume)
            if _test_hook is not None:
                _test_hook(len(journal))
            for sp in specs:
                weight_map[sp.name] = shard
                total_bytes += sp.data_len
            shards_written += 1
            tensor_count += len(specs)
        # validate source index weight_map matches manifest shard assignments
        src_idx = source / "model.safetensors.index.json"
        if src_idx.exists():
            try:
                src_imap = json.loads(src_idx.read_text()).get("weight_map", {})
                man_map = {t.name: t.shard for t in manifest.tensors}
                if set(src_imap) != set(man_map):
                    raise ValueError(
                        "source index weight_map keys != manifest census"
                    )
                for n, sh in src_imap.items():
                    if man_map.get(n) != sh:
                        raise ValueError(
                            f"source index assigns {n}->{sh} but manifest says {man_map.get(n)}"
                        )
            except ValueError as exc:
                raise ValueError(f"source index/object mismatch: {exc}") from exc
        _write_index(staging, weight_map, source_cfg, width, source, total_bytes)
        j("slice", f"width={width} tensors={tensor_count} bytes={total_bytes}")
    except Exception:
        # preserve staging on interruption/failure (do NOT delete)
        raise

    validation = _exact_validate(
        staging, manifest, keep_map, source_cfg, width, pre_src, shard_hashes,
    )
    j("validate", validation.note)
    promoted = False
    structurally = False
    if validation.ok:
        j("promote", f"{out.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        # atomic same-filesystem swap: old output -> backup, staging -> output
        backup = out.with_name(out.name + f".bak-{os.getpid()}")
        if out.exists():
            if backup.exists():
                shutil.rmtree(backup)
            out.rename(backup)
        try:
            staging.rename(out)
        except Exception:  # noqa: BLE001
            # restore old output if staging rename failed
            if backup.exists() and not out.exists():
                backup.rename(out)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        promoted = True
        structurally = True
    else:
        j("abort", validation.note)
        # old output (if any) left intact; staging kept for resume
    return ExportResult(
        output_dir=str(out), width=width, shards_written=shards_written,
        tensor_count=tensor_count, total_bytes=total_bytes,
        structurally_complete=structurally, promoted=promoted,
        runtime_compatibility="schema-supported-unvalidated",
        runtime_validated=False,
        journal=journal,
    )


# ------------------------------------------------- streaming body + validate ---
_IO_CHUNK = 4 << 20  # bounded I/O chunk used everywhere (recorded bound)


def _copy_stream(src: Path, dst: Path, chunk: int = _IO_CHUNK) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            b = fi.read(chunk)
            if not b:
                break
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())


def _read_exact(f: Any, n: int, what: str) -> bytes:
    out = bytearray()
    while len(out) < n:
        b = f.read(n - len(out))
        if not b:
            raise ValueError(f"short read {what}: wanted {n}, got {len(out)}")
        out += b
    return bytes(out)


def _copy_range(
    f: Any, abs_start: int, length: int, dst: Any, chunk: int = _IO_CHUNK,
) -> None:
    """Stream one contiguous source interval [abs_start, abs_start+length) into
    dst in <=chunk reads/writes. Used for a single contiguous run, including one
    longer than `chunk` (streamed in <=chunk chunks). Never materializes more
    than `chunk` bytes."""
    f.seek(abs_start)
    rem = length
    while rem > 0:
        take = f.read(min(chunk, rem))
        if not take:
            raise ValueError(f"short read at {abs_start} rem {rem}")
        dst.write(take)
        rem -= len(take)


def _bounded_spans(
    windows: list[tuple[int, int]], chunk: int = _IO_CHUNK,
) -> list[list[tuple[int, int]]]:
    """Group ordered windows into source spans whose EXTENT (first.a .. last.b,
    gaps included) <= `chunk`. One seek/read per span then extracts the requested
    subranges. Rejects overlapping/reversed windows; adjacent windows (a == prev_b)
    stay in the same span. A single contiguous window longer than `chunk` remains
    its own (oversized) span and is streamed by the caller in <=chunk chunks.
    Offsets are ABSOLUTE file offsets (may exceed `chunk`), so span formation
    compares derivative offsets (b - first_a), never the raw offsets."""
    spans: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    first_a: int | None = None
    prev_b: int | None = None
    for (a, b) in windows:
        if not isinstance(a, int) or not isinstance(b, int) or b <= a:
            raise ValueError(f"invalid/inverted window: {(a, b)}")
        if prev_b is not None and a < prev_b:
            raise ValueError(f"overlapping/reversed windows: {a}<{prev_b}")
        # flush cur once the current span would exceed the chunk extent (gaps
        # included), keyed on derivative extent only.
        if cur and first_a is not None and (b - first_a) > chunk:
            spans.append(cur)
            cur = []
            first_a = None
        if not cur:
            first_a = a
        cur.append((a, b))
        prev_b = b
    if cur:
        spans.append(cur)
    return spans


def _plan_output_shard(
    source: Path, manifest: CheckpointManifest, shard: str,
    keep_map: dict[tuple[int, int], list[int]], width: int,
) -> tuple[list[TensorSpec], Any]:
    """Plan an output shard's ordered specs + streaming body writer (data-base
    aware). Returns (specs, build_body).

    The returned body provider opens the source shard ONCE lazily and reuses that
    handle for every body in the shard. Its `close` is idempotent and always
    invoked by `production_write_shard` via `finally`, so resume-skipped shards
    (never written) never open a handle and no handle leaks on error. It also
    works as a plain callable (single body) for compatibility.

    Ordering/IO: ordered sparse windows are gathered in original order and
    grouped into source spans of extent <= _IO_CHUNK, read with one seek/read per
    span (gaps included), requested subranges emitted in order into a <=4MiB
    buffer and written once. A single contiguous interval >4MiB is streamed in
    <=4MiB chunks. Every read/write <=4MiB; no full tensor/shard is materialized.
    """
    base = _shard_data_base(source / shard)
    shard_entries = sorted((t for t in manifest.tensors if t.shard == shard), key=lambda t: t.name)
    src_path = source / shard
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

    state: dict[str, Any] = {"f": None, "closed": False}

    def _ensure() -> Any:
        if state["closed"]:
            raise ValueError("provider already closed")
        if state["f"] is None:
            state["f"] = open(src_path, "rb")  # noqa: SIM115 - lazy per-shard handle, closed by close()
        return state["f"]

    def build_body(name: str, start: int, size: int, dst: Any) -> None:
        _ = start
        wins = bodies[name]
        if not wins:
            return
        f = _ensure()
        blank = bytearray()
        for span in _bounded_spans(wins, _IO_CHUNK):
            first_a = span[0][0]
            last_b = span[-1][1]
            length = last_b - first_a
            if length <= 0:
                continue
            contiguous = sum((b - a) for (a, b) in span) == length
            if contiguous:
                # one interval (possibly > chunk): stream/slice in bounded chunks
                _copy_range(f, first_a, length, dst)
                continue
            # sparse span extent <= chunk: one seek/read, gather subranges in
            # original order into a bounded buffer, write once. Reads gap bytes
            # so the seek/read count is O(source_bytes/4MiB), not per-window.
            if length > _IO_CHUNK:  # defensive; formation keeps extent <= chunk
                raise ValueError(f"span extent {length} exceeds chunk")
            f.seek(first_a)
            raw = _read_exact(f, length, f"span {first_a}:{last_b}")
            blank.clear()
            for (a, b) in span:
                blank += raw[(a - first_a):(b - first_a)]
            dst.write(blank)

    def close() -> None:
        if state["closed"]:
            return
        state["closed"] = True
        f = state["f"]
        state["f"] = None
        if f is not None:
            f.close()

    build_body.close = close  # type: ignore[attr-defined]
    return specs, build_body


def _write_index(
    staging: Path, weight_map: dict[str, str], source_cfg: dict[str, object],
    width: int, source: Path, total_bytes: int,
) -> None:
    cfg = json.loads(json.dumps(source_cfg))
    cfg["moe_intermediate_size"] = width
    (staging / "config.json").write_text(json.dumps(cfg, indent=2))
    # Preserve any non-total_size source metadata (e.g. total_parameters),
    # but ALWAYS rebuild total_size to the exact output tensor DATA bytes. The
    # source total_size is the full 503 GB checkpoint and would be stale on a
    # width-reduced derivative.
    src_meta: dict[str, object] = {}
    src_idx = source / "model.safetensors.index.json"
    if src_idx.exists():
        try:
            src_idx_d = json.loads(src_idx.read_text())
            src_meta = dict(src_idx_d.get("metadata", {}))
        except ValueError:
            src_meta = {}
    src_meta["total_size"] = int(total_bytes)
    (staging / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": src_meta, "weight_map": weight_map}, indent=2)
    )


def _exact_validate(
    staging: Path, manifest: CheckpointManifest,
    keep_map: dict[tuple[int, int], list[int]], source_cfg: dict[str, object],
    width: int, pre_src_stats: dict[str, os.stat_result],
    write_hashes: dict[str, str],
) -> _Validation:
    ok = True
    notes: list[str] = []
    src_by_name = {t.name: t for t in manifest.tensors}
    present: set[str] = set()
    total_out = 0
    seen: dict[str, int] = {}
    out_hashes: dict[str, str] = {}  # shard_name -> sha256 (current)
    shard_of: dict[str, str] = {}  # tensor_name -> containing shard
    for shard in sorted(staging.glob("*.safetensors")):
        h = _fingerprint(shard)
        out_hashes[shard.name] = h
        # write-time journal hash must MATCH current (post-write corruption caught)
        if (
            write_hashes.get(shard.name) is not None
            and write_hashes[shard.name] != h
        ):
            ok = False
            notes.append(f"shard {shard.name} current hash != write-time journal hash")
        hdr = read_safetensors_header(shard)
        for name, spec in hdr.items():
            if name == "__metadata__":
                continue
            shard_of[name] = shard.name
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
    if total_out != sp.total_bytes:
        ok = False
        notes.append(f"total {total_out} != expected {sp.total_bytes} (exact)")
    # index weight_map: key set == source census; each name -> containing shard
    index_path = staging / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        imap = idx.get("weight_map", {})
        if set(imap) != set(src_by_name):
            ok = False
            notes.append(
                f"index weight_map keys != census ({len(imap)} vs {len(src_by_name)})"
            )
        for n, sh in imap.items():
            if shard_of.get(n) != sh:
                ok = False
                notes.append(f"index assigns {n}->{sh} but contained in {shard_of.get(n)}")
        missing_refs = [sh for sh in imap.values() if not (staging / sh).exists()]
        if missing_refs:
            ok = False
            notes.append(f"index refs missing shards: {sorted(set(missing_refs))[:3]}")
    else:
        ok = False
        notes.append("missing model.safetensors.index.json")
    # persist a machine-readable completed-shard manifest (hashes)
    (staging / "completed_shards.json").write_text(
        json.dumps(out_hashes, indent=2, sort_keys=True)
    )
    # source immutability via STAT identity (lightweight; no full-file re-hash)
    for sh, pre in pre_src_stats.items():
        try:
            post = os.stat(Path(manifest.checkpoint_dir) / sh)
        except OSError:
            ok = False
            notes.append(f"source {sh} stat failed")
            continue
        if post.st_size != pre.st_size or post.st_mtime_ns != pre.st_mtime_ns:
            ok = False
            notes.append(f"source {sh} changed (size/mtime)")
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
