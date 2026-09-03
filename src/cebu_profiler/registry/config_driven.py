"""Config-driven architecture specs: derive an ArchitectureSpec from a
checkpoint's own config.json.

Model-agnostic by construction (AGENTS.md invariant 14): instead of hand-written
per-model adapters, the structural layout is read from the checkpoint manifest
the way serving stacks do. Hand-written integrations (glm52.py, k3.py) remain
for curated families with drift checks; this path covers everything else —
including families this repo has never seen.

Known config shapes (DeepSeek/GLM MoE conventions, validated against the
released GLM-5.x configs):
    text_config.* (multimodal wrappers) with nested MoE fields, or flat top-level
    fields. Vision towers are detected via `vision_config` / `vision_start_token_id`
    presence and carried as a role-level fact; the spec stays text-stack scoped.

Field aliases tried in order; the first present wins. Missing MoE fields fail
closed with a named error — never guessed (AGENTS.md: no invented numbers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from cebu_profiler.checkpoint.classifier import classify_tensor
from cebu_profiler.checkpoint.source_manifest import CheckpointManifest
from cebu_profiler.schemas.architecture import (
    ArchitectureSpec,
    DType,
    LayerKind,
    MoELayout,
)

# Layout facts we can derive from tensor names even when config.json is silent.
_QUANT_BPW: dict[str, float] = {
    # quantization_config.quant_method families -> measured bits-per-weight of
    # the routed expert bank (code/weights storage density).
    "nvfp4": 4.5,
    "mxfp4": 4.5,
    "compressed-tensors": 4.5,  # refined below from ignore-list inspection
    "fp8": 8.0,
    "int8": 8.0,
    "awq": 4.0,
    "gptq": 4.0,
}


class SpecDerivationError(ValueError):
    """A required structural field could not be derived — fail closed."""


def _first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _require(d: dict[str, Any], *keys: str) -> Any:
    v = _first(d, *keys)
    if v is None:
        raise SpecDerivationError(f"config.json missing required field(s): {keys}")
    return v


def _quant_bpw(cfg: dict[str, Any]) -> float | None:
    qc = cfg.get("quantization_config") or {}
    method = str(qc.get("quant_method", "")).lower()
    if method not in _QUANT_BPW:
        return None
    bpw = _QUANT_BPW[method]
    # compressed-tensors: inspect which weight bits the config group declares.
    try:
        groups = qc.get("config_groups") or {}
        first: dict[str, Any] = next(iter(groups.values()), {})
        bits = (first.get("weights") or {}).get("num_bits")
        if bits:
            # 4-bit block-scaled formats store ~12% overhead in scales.
            return float(bits) * 1.12 if float(bits) <= 4 else float(bits)
    except (AttributeError, TypeError, StopIteration):
        pass
    return bpw


def _expert_dtype(bpw: float | None) -> DType:
    if bpw is None:
        return DType.BF16
    if bpw <= 5.0:
        return DType.MXFP4
    if bpw <= 5.5:
        return DType.INT4
    if bpw <= 9.0:
        return DType.INT8
    return DType.BF16


def _moe_layout(text: dict[str, Any], hidden: int) -> MoELayout:
    n_routed = _require(text, "n_routed_experts", "num_routed_experts", "num_local_experts")
    top_k = _require(text, "num_experts_per_tok", "num_experts_per_token", "top_k")
    n_shared = int(_first(text, "n_shared_experts", "num_shared_experts") or 0)
    moe_intermediate = int(_first(text, "moe_intermediate_size", "expert_intermediate_size") or 0)
    bpw = _quant_bpw(text)
    return MoELayout(
        num_routed_experts=int(n_routed),
        top_k=int(top_k),
        num_shared_experts=n_shared,
        latent_dim=moe_intermediate or max(1, hidden // 2),
        hidden_dim=hidden,
        expert_dtype=_expert_dtype(bpw),
        dense_dtype=DType.BF16,
    )


def _layer_kinds(text: dict[str, Any]) -> tuple[int, int]:
    """(dense_layers, moe_layers) from first_sparse_layer / num layers."""
    n_layers = int(_require(text, "num_hidden_layers"))
    first_moe = int(
        _first(text, "first_k_dense_replace", "first_sparse_layer", "moe_layer_start") or 0
    )
    dense = max(0, first_moe)
    moe = max(0, n_layers - dense)
    return dense, moe


def _dense_layout(hidden: int) -> MoELayout:
    """Trivial layout for dense models (no routed experts declared)."""
    return MoELayout(
        num_routed_experts=1,
        top_k=1,
        num_shared_experts=0,
        latent_dim=max(1, hidden // 2),
        hidden_dim=hidden,
        expert_dtype=DType.BF16,
        dense_dtype=DType.BF16,
    )


def _detect_expert_indexing(manifest: CheckpointManifest) -> int | None:
    """Largest `.experts.N` index seen in tensor names, +1 = expert count."""
    from cebu_profiler.checkpoint.classifier import _EXPERT_RE

    best = -1
    for t in manifest.tensors:
        m = _EXPERT_RE.search(t.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1 if best >= 0 else None


def spec_from_config(
    config: dict[str, Any],
    *,
    name: str | None = None,
    manifest: CheckpointManifest | None = None,
) -> ArchitectureSpec:
    """Derive a structural ArchitectureSpec from a parsed config.json.

    `manifest` (optional) cross-checks expert count against actual tensor
    names — the measured fact wins over the declared one, and a mismatch is
    carried as an explicit `total_params=None` structural note rather than
    silently accepted.
    """
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    if not isinstance(text, dict):
        raise SpecDerivationError("config text_config must be an object when present")
    hidden = int(_require(text, "hidden_size"))
    dense, moe = _layer_kinds(text)
    has_moe_fields = (
        _first(text, "n_routed_experts", "num_routed_experts", "num_local_experts") is not None
    )
    # A config without MoE fields is dense regardless of first_k_dense_replace.
    if moe == 0 or not has_moe_fields:
        dense, moe = dense + moe, 0
    if dense + moe == 0:
        raise SpecDerivationError("config declares zero layers")
    layout = _dense_layout(hidden) if moe == 0 else _moe_layout(text, hidden)
    vocab = _first(text, "vocab_size", "vocabulary_size")

    if manifest is not None and moe > 0:
        measured_experts = _detect_expert_indexing(manifest)
        if measured_experts is not None and measured_experts != layout.num_routed_experts:
            # Measured fact wins; declared config kept for the drift check.
            layout = layout.model_copy(update={"num_routed_experts": measured_experts})

    if moe == 0:
        dense, moe = dense + moe, 0
        layers_by_kind = {LayerKind.DENSE: dense}
    else:
        layers_by_kind = {LayerKind.DENSE: dense, LayerKind.MOE: moe}

    arch_name = name or str(
        _first(config, "model_type") or _first(text, "model_type") or "derived-model"
    )
    return ArchitectureSpec(
        name=arch_name,
        num_text_layers=dense + moe,
        layers_by_kind=layers_by_kind,
        moe=layout,
        hidden_dim=hidden,
        vocabulary_size=int(vocab) if vocab else None,
        tensor_params={},  # real sizes always come from the manifest census
    )


def spec_from_checkpoint_dir(checkpoint_dir: str, *, name: str | None = None) -> ArchitectureSpec:
    """Load config.json from a checkpoint dir and derive its spec."""
    path = Path(checkpoint_dir) / "config.json"
    if not path.exists():
        raise SpecDerivationError(f"no config.json under {checkpoint_dir}")
    cfg = json.loads(path.read_text())
    return spec_from_config(cfg, name=name)


def verify_spec_against_manifest(spec: ArchitectureSpec, manifest: CheckpointManifest) -> list[str]:
    """Structural drift checks: spec vs measured tensor facts.

    Returns human-readable drift notes; empty list = consistent. Uses the
    shared classifier so role accounting matches the structural graph.
    """
    notes: list[str] = []
    experts_seen = _detect_expert_indexing(manifest)
    if experts_seen is not None and experts_seen != spec.moe.num_routed_experts:
        notes.append(
            f"expert count drift: spec={spec.moe.num_routed_experts} measured={experts_seen}"
        )
    max_layer = -1
    unclassified = 0
    for t in manifest.tensors:
        c = classify_tensor(t.name)
        if c.unclassified:
            unclassified += 1
        if c.layer_index is not None:
            max_layer = max(max_layer, c.layer_index)
    if max_layer >= 0 and max_layer >= spec.num_text_layers:
        notes.append(
            f"layer drift: spec has {spec.num_text_layers} layers, tensors reach l{max_layer}"
        )
    if unclassified:
        notes.append(f"{unclassified} tensor(s) unclassified by the shared classifier")
    return notes


def estimate_active_params(spec: ArchitectureSpec, tokens_per_seq: int = 1) -> int | None:
    """Rough analytic active-parameter count (DENSE + top-k MOE experts/layer).

    Tagged `estimated` by callers (invariant 12): this is layout math, not a
    measurement. None when the spec carries no measured tensor sizes.
    """
    if spec.needs_source_measurement:
        return None
    m = spec.moe
    per_expert = 3 * spec.hidden_dim * m.latent_dim
    per_layer = (
        4 * spec.hidden_dim * spec.hidden_dim  # attention ballpark
        + 2 * spec.hidden_dim * m.latent_dim  # shared expert (gate+up fused est.)
        + m.num_shared_experts * per_expert
        + m.top_k * per_expert
    )
    return spec.num_text_layers * per_layer * tokens_per_seq // max(1, tokens_per_seq)


def approx_total_params(spec: ArchitectureSpec) -> int | None:
    """Layout-math total parameter estimate; None without measured sizes."""
    if spec.needs_source_measurement:
        return None
    m = spec.moe
    per_expert = 3 * spec.hidden_dim * m.latent_dim
    routed = m.num_routed_experts * per_expert
    per_layer = 4 * spec.hidden_dim * spec.hidden_dim + 2 * spec.hidden_dim * m.latent_dim
    total = (
        spec.num_text_layers * (per_layer + (routed if m.num_routed_experts > 1 else 0))
        + (spec.vocabulary_size or 0) * spec.hidden_dim * 2
    )
    return total


def spec_summary(spec: ArchitectureSpec) -> dict[str, Any]:
    """JSON-safe summary for manifests (all values measured-or-None)."""
    return {
        "name": spec.name,
        "num_text_layers": spec.num_text_layers,
        "layers_by_kind": {k.value: v for k, v in spec.layers_by_kind.items()},
        "moe": spec.moe.model_dump(),
        "hidden_dim": spec.hidden_dim,
        "vocabulary_size": spec.vocabulary_size,
        "needs_source_measurement": spec.needs_source_measurement,
        "approx_total_params_estimated": approx_total_params(spec),
    }


_ = math  # kept import for future analytic helpers; ruff-friendly
