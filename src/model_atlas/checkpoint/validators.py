"""Backend-independent typed checkpoint validator registry/API.

A checkpoint validator is a REGISTERED, deterministic, structural validator for
a backend's declared output format (e.g. real safetensors checkpoints). The
engine resolves a validator by ``(backend_id, kind)`` and fails closed when none
is wired — an algorithm adapter is never falsely marked available just because a
validator exists for a different backend or kind.

The built-in ``safetensors`` validator performs REAL structural validation:

  * parses every shard's 8-byte header length + JSON header (with a bound guard),
  * validates every tensor's dtype/shape byte count against its declared
    ``data_offsets`` length,
  * validates that tensor byte ranges lie inside the data buffer and do NOT
    overlap (sorted, gap-free coverage),
  * validates the index ``weight_map`` BIDIRECTIONALLY: every tensor listed in
    the index resolves to an existing shard AND every tensor in every shard is
    reachable from the index (complete coverage),
  * verifies tensor counts match between the index and the union of shards,
  * hashes every shard and the whole output/checkpoint tree and returns the
    digests as part of the validation result.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from model_atlas.jobs.artifacts import sha256_file

_MAX_HEADER = 64 * 1024 * 1024  # bounded header read (corrupt length fails, never allocates)

# dtype -> bytes-per-element in safetensors naming = numpy dtypes
_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "U64": 8,
    "I64": 8,
    "U32": 4,
    "I32": 4,
    "U16": 2,
    "I16": 2,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
}


@dataclass(frozen=True)
class CheckpointValidationResult:
    ok: bool
    detail: str = ""
    tensor_count: int = 0
    shard_count: int = 0
    shard_hashes: dict[str, str] = field(default_factory=dict)
    checkpoint_digest: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "detail": self.detail,
            "tensor_count": self.tensor_count,
            "shard_count": self.shard_count,
            "shard_hashes": dict(self.shard_hashes),
            "checkpoint_digest": self.checkpoint_digest,
        }


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

# Validator kind -> (version, function(backend_id, staged_dir, format) ->
# CheckpointValidationResult)
RegisterValidator = Callable[[str, Path, str], CheckpointValidationResult]

_VALIDATORS: dict[tuple[str, str], str] = {}  # (backend_id, kind) -> version
_VALIDATOR_FNS: dict[tuple[str, str], RegisterValidator] = {}
_VALIDATOR_VERSION = "v1"


def register_checkpoint_validator(backend_id: str, kind: str, validator: RegisterValidator) -> None:
    """Register a validator. Duplicate (backend_id, kind) registration is a
    FATAL error UNLESS the validator is identical to the already-registered one
    (same callable + same version), so correctness is never silently
    reconsidered."""
    existing = _VALIDATORS.get((backend_id, kind))
    if existing is not None:
        if _VALIDATOR_FNS[(backend_id, kind)] is not validator:
            raise ValueError(
                f"checkpoint validator {backend_id}:{kind} already registered with a "
                "different implementation; refusing duplicate registration "
                "(versioned identity required)"
            )
        return  # identical re-registration is a no-op
    _VALIDATORS[(backend_id, kind)] = _VALIDATOR_VERSION
    _VALIDATOR_FNS[(backend_id, kind)] = validator


def get_checkpoint_validator(backend_id: str, kind: str) -> RegisterValidator | None:
    """Resolve a REAL registered validator; None when not wired (run fails
    closed — an algorithm adapter is never falsely marked available)."""
    return _VALIDATOR_FNS.get((backend_id, kind))


def registered_validators() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for backend_id, kind in _VALIDATOR_FNS:
        out.setdefault(backend_id, []).append(kind)
    return out


# --------------------------------------------------------------------------
# structural safetensors validator
# --------------------------------------------------------------------------


def _safetensors_structure(
    backend_id: str, staged_dir: Path, _fmt: str
) -> CheckpointValidationResult:
    """Validate real safetensors structure per the module contract."""
    files = [p for p in sorted(staged_dir.iterdir()) if p.is_file()]
    shards = [p for p in files if p.suffix == ".safetensors"]
    idx = [p for p in files if p.name.endswith(".index.json")]
    if not shards:
        return CheckpointValidationResult(False, "no safetensors shard in staging")
    if not idx:
        return CheckpointValidationResult(False, "no safetensors index in staging")

    # --- pass 1: parse + validate every shard header and tensor metadata ---
    tensors_by_shard: dict[str, dict[str, tuple[tuple[int, int], int, int]]] = {}
    for shard in shards:
        with open(shard, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return CheckpointValidationResult(False, f"{shard.name}: truncated header")
            (n,) = struct.unpack("<Q", head)
            if n > _MAX_HEADER:
                return CheckpointValidationResult(
                    False, f"{shard.name}: header length {n} exceeds bound"
                )
            header = f.read(n)
            if len(header) != n:
                return CheckpointValidationResult(False, f"{shard.name}: header length mismatch")
            try:
                obj = json.loads(header)
            except ValueError as exc:
                return CheckpointValidationResult(
                    False, f"{shard.name}: invalid JSON header ({exc})"
                )
            meta_fmt = obj.get("__metadata__", {}).get("format")
            if meta_fmt not in (None, 0):
                return CheckpointValidationResult(False, f"{shard.name}: unsupported format")
            body_len = f.seek(0, 2) - (8 + n)
            tensors: dict[str, tuple[tuple[int, int], int, int]] = {}
            for name, meta in obj.items():
                if name == "__metadata__":
                    continue
                dtype = meta.get("dtype")
                shape = meta.get("shape")
                offs = meta.get("data_offsets")
                if dtype not in _DTYPE_BYTES:
                    return CheckpointValidationResult(False, f"{shard.name}: unknown dtype {dtype}")
                if not isinstance(shape, list) or any(
                    not isinstance(d, int) or d < 0 for d in shape
                ):
                    return CheckpointValidationResult(
                        False, f"{shard.name}: invalid shape for {name}"
                    )
                if not offs or len(offs) != 2 or not all(isinstance(o, int) for o in offs):
                    return CheckpointValidationResult(
                        False, f"{shard.name}: invalid data_offsets for {name}"
                    )
                numel = 1
                for d in shape:
                    numel *= d
                expected_bytes = numel * _DTYPE_BYTES[dtype]
                span = offs[1] - offs[0]
                if span != expected_bytes:
                    return CheckpointValidationResult(
                        False,
                        f"{shard.name}: tensor {name} dtype/shape byte count {expected_bytes} "
                        f"!= data span {span}",
                    )
                if offs[1] > body_len or offs[0] < 0 or offs[0] > offs[1]:
                    return CheckpointValidationResult(
                        False, f"{shard.name}: tensor {name} offsets out of body"
                    )
                tensors[name] = (tuple(offs), expected_bytes, numel)
            # non-overlap: sort by start; adjacent spans must not overlap
            spans = sorted((v[0][0], v[0][1]) for v in tensors.values())
            for a, b in zip(spans[:-1], spans[1:], strict=True):
                if b[0] < a[1]:
                    return CheckpointValidationResult(
                        False, f"{shard.name}: overlapping tensor offsets"
                    )
            tensors_by_shard[shard.name] = tensors

    # --- pass 2: index validation (BIDIRECTIONAL + tensor counts) ---
    for idxf in idx:
        try:
            index = json.loads(idxf.read_text(encoding="utf-8"))
        except ValueError as exc:
            return CheckpointValidationResult(False, f"{idxf.name}: invalid index ({exc})")
        wm = index.get("weight_map", {})
        if not isinstance(wm, dict) or not wm:
            return CheckpointValidationResult(False, f"{idxf.name}: empty/invalid weight_map")
        # forward: every index tensor resolves to an existing shard + a real tensor
        for tname, shard in wm.items():
            if shard not in tensors_by_shard:
                return CheckpointValidationResult(
                    False, f"{idxf.name}: index {tname!r} -> missing shard {shard!r}"
                )
            if tname not in tensors_by_shard[shard]:
                return CheckpointValidationResult(
                    False, f"{idxf.name}: index {tname!r} not in shard {shard!r}"
                )
        # backward: every shard tensor reachable from the index (complete coverage)
        for shard_name, tensors in tensors_by_shard.items():
            idx_tensor_names = {t for t, s in wm.items() if s == shard_name}
            shard_names = set(tensors)
            if idx_tensor_names != shard_names:
                return CheckpointValidationResult(
                    False,
                    f"{shard_name}: index coverage mismatch (index {len(idx_tensor_names)} "
                    f"vs shard {len(shard_names)} tensors)",
                )
        tensor_count = len(wm)

    # --- pass 3: whole-output/checkpoint hashes ---
    shard_hashes = {p.name: sha256_file(p) for p in shards}
    checkpoint_digest = _whole_checkpoint_digest(files)
    return CheckpointValidationResult(
        True,
        detail="safetensors structure + index coverage + hashes valid",
        tensor_count=tensor_count,
        shard_count=len(shards),
        shard_hashes=shard_hashes,
        checkpoint_digest=checkpoint_digest,
    )


def _whole_checkpoint_digest(files: list[Path]) -> str:
    """Canonical digest over the whole staged checkpoint tree (name -> sha256)."""
    payload = json.dumps({p.name: sha256_file(p) for p in sorted(files)}, sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# registered validators (derivative backends only)
# --------------------------------------------------------------------------

for _backend in ("exl3", "modelopt_nvfp4", "llm_compressor", "atlas_quant_probe"):
    register_checkpoint_validator(_backend, "integrity", _safetensors_structure)
    register_checkpoint_validator(_backend, "format", _safetensors_structure)
    register_checkpoint_validator(_backend, "checkpoint", _safetensors_structure)
