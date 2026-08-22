"""Real GLM-5.2 config/facts + BOUNDED routing probe that uses REAL ROUTER with
SYNTHETIC INPUT (Phase 3, review-corrected).

This is NOT a measured corpus/activation trace. It reads a sparse layer's real
router weights (BF16 gate + F32 correction bias) via the bounded streaming
substrate, but feeds DETERMINISTIC GAUSSIAN hidden-state placeholders (not real
forward activations). Every result is therefore labelled
`REAL_ROUTER_SYNTHETIC_INPUT_PROBE` / `PREDICTED` — a noise-injection/coverage
probe, never measured routing-on-corpus evidence. A real corpus forward remains
service-window-gated.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_atlas.checkpoint.streaming import CheckpointStream
from model_atlas.schemas.evidence import EvidenceKind

# NVFP4 checkpoint was produced by this ModelOpt version (measured).
MODELOPT_PRODUCER = "0.46.0.dev65+g977d34dc3"
# This probe's honest evidence/input label.
PROBE_EVIDENCE_KIND = EvidenceKind.PREDICTED
PROBE_INPUT_LABEL = "REAL_ROUTER_SYNTHETIC_INPUT_PROBE"


@dataclass
class Glm52Facts:
    """Measured structural facts from the mounted GLM-5.2 config + census."""

    checkpoint_dir: str | None = None
    n_layers: int = 0
    n_dense_layers: int = 3
    n_sparse_layers: int = 0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    top_k: int = 8
    hidden: int = 0
    moe_intermediate: int = 0
    vocab: int = 0
    quant_algo: str | None = None
    kv_cache_quant: str | None = None
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    v_head_dim: int | None = None
    num_mtp_layers: int | None = None
    group_size: int | None = None
    model_type: str | None = None
    architectures: list[str] = field(default_factory=list)
    normalized_layer_types: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_mlp(layer_index: int, mlp_layer_types: list[str] | None) -> str:
    """Whether layer `layer_index` routes MoE (default: first 3 dense -> sparse)."""
    if mlp_layer_types:
        return mlp_layer_types[layer_index] if layer_index < len(mlp_layer_types) else "sparse"
    return "dense" if layer_index < 3 else "sparse"


def load_glm52_facts(checkpoint_dir: str) -> Glm52Facts:
    """Measure GLM-5.2 NVFP4 structural facts from config.json (no bodies)."""
    root = Path(checkpoint_dir)
    cfg_path = root / "config.json"
    cfg: dict[str, Any] = json.loads(cfg_path.read_text())
    n_layers_cfg = int(cfg.get("num_hidden_layers", 78))
    mlp = cfg.get("mlp_layer_types") or (
        ["dense"] * min(3, n_layers_cfg) + ["sparse"] * max(0, n_layers_cfg - 3)
    )
    n_sparse = sum(1 for t in mlp if t == "sparse")
    qconf = cfg.get("quantization_config", {}) or {}
    return Glm52Facts(
        checkpoint_dir=checkpoint_dir,
        n_layers=int(cfg.get("num_hidden_layers", 78)),
        n_dense_layers=sum(1 for t in mlp if t == "dense"),
        n_sparse_layers=n_sparse,
        n_routed_experts=int(cfg.get("n_routed_experts", cfg.get("num_experts", 256))),
        n_shared_experts=int(cfg.get("n_shared_experts", 1)),
        top_k=int(cfg.get("num_experts_per_tok", 8)),
        hidden=int(cfg.get("hidden_size", 6144)),
        moe_intermediate=int(cfg.get("moe_intermediate_size", 2048)),
        vocab=int(cfg.get("vocab_size", 154880)),
        quant_algo=qconf.get("quant_algo"),
        kv_cache_quant=(
            qconf.get("kv_cache_scheme", {}).get("type")
            or (cfg.get("hf_quant_config", {}) or {})
            .get("quantization", {})
            .get("kv_cache_quant_algo")
        ),
        kv_lora_rank=cfg.get("kv_lora_rank"),
        q_lora_rank=cfg.get("q_lora_rank"),
        v_head_dim=cfg.get("v_head_dim"),
        num_mtp_layers=cfg.get("num_nextn_predict_layers"),
        group_size=(
            (qconf.get("config_groups", {}).get("group_0", {}).get("weights", {}) or {}).get(
                "group_size"
            )
        ),
        model_type=cfg.get("model_type"),
        architectures=list(cfg.get("architectures", []) or []),
        normalized_layer_types=True,
    )


def normalized_glm52_config(checkpoint_dir: str) -> dict[str, Any]:
    """Return a transformers-loadable GLM-5.2 config dict.

    The mounted `config.json` serializes `layer_types` with values the native
    `GlmMoeDsaConfig` does not recognize (`deepseek_sparse_attention`). We DO NOT
    claim DSA->sparse or silently replace attention kinds; instead we DROP the
    incompatible serialized `layer_types` key so the native config falls back to
    its own validated defaults (and `mlp_layer_types`/`indexer_types` are kept).
    The native config semantics are then tested directly by the caller.
    """
    root = Path(checkpoint_dir)
    cfg: dict[str, Any] = json.loads((root / "config.json").read_text())
    cfg.pop("layer_types", None)
    return cfg


def _topk_rank(scores: list[float], k: int) -> tuple[list[int], list[float]]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    probs: list[float] = []
    m = max(scores[i] for i in order)
    e = [math.exp(scores[i] - m) for i in order]
    s = sum(e)
    probs = [x / s for x in e]
    return order, probs


def _sum_p_log(probs: list[float]) -> float:
    """Shannon entropy of a categorical distribution (>= 0)."""
    return -sum(p * math.log(p) for p in probs if p > 0)


@dataclass
class RoutingRecord:
    layer: int
    token_height_index: int
    selected_experts: list[int]
    gate_weights: list[float]
    entropy: float


@dataclass
class GlmRoutingTrace:
    layer: int
    n_experts: int
    top_k: int
    hidden: int
    input_label: str = PROBE_INPUT_LABEL
    evidence_kind: EvidenceKind = PROBE_EVIDENCE_KIND
    provenance: str = (
        "REAL_ROUTER_SYNTHETIC_INPUT_PROBE: real router weights, deterministic "
        "Gaussian placeholder hidden states; NOT measured corpus/activation evidence"
    )
    records: list[RoutingRecord] = field(default_factory=list)
    frequency: dict[int, int] = field(default_factory=dict)
    coactivation: dict[tuple[int, int], int] = field(default_factory=dict)
    gate_bias_values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_label": self.input_label,
            "evidence_kind": self.evidence_kind.value,
            "provenance": self.provenance,
            "layer": self.layer,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "hidden": self.hidden,
            "records": [asdict(r) for r in self.records],
            "frequency": self.frequency,
            "coactivation": {f"{a}:{b}": c for (a, b), c in self.coactivation.items()},
        }


def stream_routing_trace(
    checkpoint_dir: str,
    layer: int = 3,
    *,
    n_hidden_rows: int = 8,
    hidden_scale: float = 0.5,
) -> GlmRoutingTrace:
    """Bounded REAL_ROUTER_SYNTHETIC_INPUT_PROBE over a real GLM-5.2 sparse layer.

    Reads the layer's real router (BF16 gate + F32 correction bias), synthesizes
    deterministic Gaussian hidden-state placeholders (NOT measured activations),
    and runs top-k gating. Output is tagged PREDICTED (synthetic input), so it is
    a coverage/probe only — never measured routing-on-corpus evidence.
    """
    from model_atlas.checkpoint.source_manifest import load_manifest

    facts = load_glm52_facts(checkpoint_dir)
    manifest = load_manifest(checkpoint_dir)
    gate_name = f"model.layers.{layer}.mlp.gate.weight"
    bias_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    gate_entry = next((t for t in manifest.tensors if t.name == gate_name), None)
    if gate_entry is None:
        raise ValueError(f"layer {layer} is not a sparse (routed) layer: no {gate_name}")

    with CheckpointStream(checkpoint_dir) as stream:
        gate_body = stream.get(gate_name)
        bias_body = stream.get(bias_name)
    if gate_body is None or not gate_body.values:
        raise ValueError(f"could not decode router gate for layer {layer}")
    bias_values = bias_body.values if bias_body else [0.0] * facts.n_routed_experts
    # gate.weight is [n_exp, hidden] row-major
    gate_rows = gate_body.values
    hidden = facts.hidden
    router = [
        gate_rows[hidden * e : hidden * (e + 1)] for e in range(facts.n_routed_experts)
    ]

    # deterministic Gaussian placeholder hidden states (synthetic input)
    import random

    rng = random.Random(7)
    hs: list[list[float]] = [
        [hidden_scale * rng.gauss(0.0, 1.0) for _ in range(hidden)]
        for _ in range(n_hidden_rows)
    ]

    records: list[RoutingRecord] = []
    freq: dict[int, int] = {}
    coact: dict[tuple[int, int], int] = {}
    for row in hs:
        scores = []
        for e, rw in enumerate(router):
            dot = sum(rw[j] * row[j] for j in range(hidden))
            bias = bias_values[e] if e < len(bias_values) else 0.0
            scores.append(dot + bias)
        sel, probs = _topk_rank(scores, facts.top_k)
        H = _sum_p_log(probs)  # entropy is already positive; do NOT negate
        for e, _p in zip(sel, probs, strict=True):
            freq[e] = freq.get(e, 0) + 1
        # distinct unordered expert COMBINATIONS exactly once per token
        seen: set[tuple[int, int]] = set()
        for i in range(facts.top_k):
            for j in range(i + 1, facts.top_k):
                a, b = sel[i], sel[j]
                key = (min(a, b), max(a, b))
                if key not in seen:
                    coact[key] = coact.get(key, 0) + 1
                    seen.add(key)
        records.append(
            RoutingRecord(
                layer=layer,
                token_height_index=len(records),
                selected_experts=sel,
                gate_weights=[round(x, 6) for x in probs],
                entropy=round(H, 6),
            )
        )

    return GlmRoutingTrace(
        layer=layer,
        n_experts=facts.n_routed_experts,
        top_k=facts.top_k,
        hidden=hidden,
        records=records,
        frequency=freq,
        coactivation=coact,
        gate_bias_values=[round(x, 6) for x in bias_values],
    )
