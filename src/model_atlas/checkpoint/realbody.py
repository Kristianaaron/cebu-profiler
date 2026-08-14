"""Real GLM-5.2 NVFP4 bounded body validation (Phase 1).

Runs the mmap-backed bounded reader over a *bounded slice* of the real
NVIDIA GLM-5.2 NVFP4 checkpoint: it reads a handful of reference (BF16) tensors
and the constituent tensors of one NVFP4 expert (U8 weights + F8_E4M3 scales),
records each byte range and peak resident bytes, and reports the NVFP4
token/scale layout precisely — without materializing the ~465 GB source and
without touching the immutable parent (read-only mmap).

Also reports the measured dtype histogram and per-role classification coverage
(no-unclassified invariant).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from model_atlas.checkpoint.classifier import classify_tensor
from model_atlas.checkpoint.source_manifest import CheckpointManifest, TensorEntry
from model_atlas.checkpoint.streaming import CheckpointStream

# Hard cap on how many tensor bodies a validation run may read, so a misconfigured
# call can never balloon into a full-checkpoint materialization.
MAX_BODIES = 64


@dataclass
class TensorBodyReport:
    name: str
    dtype: str
    shape: list[int]
    byte_size: int
    decoded: bool  # True when reference float dtype decoded successfully
    peak_bytes: int = 0


@dataclass
class RealBodyScan:
    checkpoint_dir: str
    tensors_total: int
    shards: int
    total_bytes: int
    dtypes: dict[str, int] = field(default_factory=dict)
    unclassified_count: int = 0
    bodies_read: list[TensorBodyReport] = field(default_factory=list)
    peak_resident_bytes: int = 0
    nvfp4_expert_layout: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "tensors_total": self.tensors_total,
            "shards": self.shards,
            "total_bytes": self.total_bytes,
            "dtypes": self.dtypes,
            "unclassified_count": self.unclassified_count,
            "coverage": round(
                (self.tensors_total - self.unclassified_count) / self.tensors_total, 6
            )
            if self.tensors_total
            else 0.0,
            "bodies_read": [
                {
                    "name": b.name,
                    "dtype": b.dtype,
                    "shape": b.shape,
                    "byte_size": b.byte_size,
                    "decoded": b.decoded,
                }
                for b in self.bodies_read
            ],
            "peak_resident_bytes": self.peak_resident_bytes,
            "nvfp4_expert_layout": self.nvfp4_expert_layout,
        }


def _pick_bodies(
    manifest: CheckpointManifest,
    reference_max: int,
    nvfp4_experts: int,
) -> tuple[list[TensorEntry], list[TensorEntry]]:
    """Choose a bounded slice: small reference BF16/F32 tensors (norms, gate,
    input projections) and the constituent tensors of a few NVFP4 experts.

    Body sizes are capped so decoding stays fast and memory-bounded; the giant
    embed/lm_head (1.9 GB each) are census-but-not-decoded, just like the whole
    source which is never materialized.
    """
    MAX_REF_BYTES = 4 * 1024 * 1024  # 4 MiB — decode-stays-fast + bounded
    refs = [
        t
        for t in manifest.tensors
        if t.dtype.upper() in {"BF16", "F16", "F32"} and t.byte_size <= MAX_REF_BYTES
    ]
    refs.sort(key=lambda t: -t.byte_size)
    ref_bodies = refs[:reference_max]

    nvfp4_bodies: list[TensorEntry] = []
    seen_experts: set[str] = set()
    for t in manifest.tensors:
        if ".mlp.experts." not in t.name:
            continue
        expert_key = t.name.split(".experts.")[1].split(".")[0]
        layer_key = t.name.split(".layers.")[1].split(".")[0]
        key = f"L{layer_key}:E{expert_key}"
        if key in seen_experts:
            continue
        if len(nvfp4_bodies) >= nvfp4_experts:
            break
        seen_experts.add(key)
        for t2 in manifest.tensors:
            if f"model.layers.{layer_key}.mlp.experts.{expert_key}." in t2.name:
                nvfp4_bodies.append(t2)
    return ref_bodies, nvfp4_bodies


def validate_real_bodies(
    checkpoint_dir: str,
    *,
    reference_max: int = 4,
    nvfp4_experts: int = 1,
) -> RealBodyScan:
    """Bounded, read-only body validation over the real GLM-5.2 NVFP4 checkpoint.

    Reads at most ``reference_max`` reference tensors and the constituent
    tensors of ``nvfp4_experts`` experts (capped by MAX_BODIES). Never
    materializes the source; reports peak resident bytes.
    """
    from model_atlas.checkpoint.source_manifest import load_manifest

    manifest = load_manifest(checkpoint_dir)
    ref_bodies, nvfp4_bodies = _pick_bodies(manifest, reference_max, nvfp4_experts)
    chosen = list(ref_bodies) + list(nvfp4_bodies)

    bodies: list[TensorBodyReport] = []
    peak = 0
    with CheckpointStream(checkpoint_dir) as stream:
        for entry in chosen[:MAX_BODIES]:
            br = stream.get(entry.name)
            if br is None:
                continue
            decoded = bool(br.values)
            if not decoded and br.dtype.upper() in {"BF16", "F16", "F32"}:
                # reference float dtype present but decode failed -> surface it
                decoded = False
            bodies.append(
                TensorBodyReport(
                    name=br.name,
                    dtype=br.dtype,
                    shape=br.shape,
                    byte_size=br.byte_size,
                    decoded=decoded,
                )
            )
        peak = stream.stats.peak_bytes

    unclassified = sum(1 for t in manifest.tensors if classify_tensor(t.name).unclassified)
    dtypes = dict(Counter(t.dtype for t in manifest.tensors))

    return RealBodyScan(
        checkpoint_dir=checkpoint_dir,
        tensors_total=manifest.tensor_count,
        shards=len(manifest.shards),
        total_bytes=manifest.total_bytes,
        dtypes=dtypes,
        unclassified_count=unclassified,
        bodies_read=bodies,
        peak_resident_bytes=peak,
        nvfp4_expert_layout=_nvfp4_layout(manifest),
    )


def _nvfp4_layout(manifest: CheckpointManifest) -> dict[str, list[str]]:
    """Describe the NVFP4 expert constituent layout (dtype+shape strings)."""
    out: dict[str, list[str]] = {}
    for t in manifest.tensors:
        if ".mlp.experts.0." not in t.name or ".layers.3." not in t.name:
            continue
        short = t.name.split(".mlp.experts.0.")[1]
        out.setdefault(short, [])
        out[short].append(f"{t.dtype}{t.shape}")
    return out
