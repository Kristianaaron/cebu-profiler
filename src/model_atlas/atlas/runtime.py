"""A genuine synthetic mini-MoE and its streaming layerwise forward pass.

This is a real (tiny) model: seeded random weights, real router top-k gating,
softmax probabilities, per-expert gated outputs, router-weighted combination,
and per-expert output norms computed from actual activations. Nothing here is
fabricated — it is the honest substitute for the oversized checkpoint that
fully exercises the layerwise tracing + REAP machinery (fork A).

The forward iterates one layer at a time and keeps only the running hidden
state, so it plays the role of the "stream one layer, free it" REAP loop.

Numerics: the hot per-token × per-expert matmuls run on NumPy (BLAS). The
public model structures stay plain Python lists (scorers, the derivative
builder, and the checkpoint tooling read them directly); arrays are derived
once per model into a lazily-built cache and traces are converted back to
lists at the boundary, so results stay JSON-serializable and byte-comparable
with the previous pure-Python engine up to float summation order.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


from model_atlas.atlas.collector import ChannelStatsAccumulator
from model_atlas.schemas.architecture import ArchitectureSpec


@dataclass
class LayerWeights:
    ln_w: list[float]  # [hidden]
    router: list[list[float]]  # [n_exp, hidden]
    experts: list[dict[str, list[list[float]]]]  # e -> {gate, up, down}


@dataclass
class MiniMoE:
    arch: ArchitectureSpec
    hidden: int
    n_exp: int
    mid: int
    embed: list[list[float]]  # [vocab, hidden]
    lm_head: list[list[float]]  # [vocab, hidden]
    layers: list[LayerWeights] = field(default_factory=list)
    # lazily-built NumPy mirror of the weights above (never serialized)
    np_cache: Any = field(default=None, repr=False, compare=False)


def build_mini_moe(arch: ArchitectureSpec, seed: int = 0) -> MiniMoE:
    """Deterministic synthetic MoE from an ArchitectureSpec."""
    rng = random.Random(seed)
    hidden = arch.hidden_dim
    n_exp = arch.moe.num_routed_experts
    mid = max(1, hidden // 8)

    def rmat(c: int, r: int, scale: float) -> list[list[float]]:
        return [[rng.gauss(0.0, scale / math.sqrt(r)) for _ in range(r)] for _ in range(c)]

    vocab = arch.vocabulary_size or 1000
    model = MiniMoE(
        arch=arch,
        hidden=hidden,
        n_exp=n_exp,
        mid=mid,
        # scale=1.0 -> each embedding row has ~unit norm, so hidden states and
        # saliency magnitudes are O(1) rather than microscopic (math unchanged)
        embed=rmat(vocab, hidden, 1.0),
        lm_head=rmat(vocab, hidden, 1.0),
    )
    for _ in range(arch.num_text_layers):
        expert_w: list[dict[str, list[list[float]]]] = []
        for _ in range(n_exp):
            expert_w.append(
                {
                    "gate": rmat(mid, hidden, 1.0),
                    "up": rmat(mid, hidden, 1.0),
                    "down": rmat(hidden, mid, 1.0),
                }
            )
        model.layers.append(
            LayerWeights(
                ln_w=[1.0] * hidden,
                router=rmat(n_exp, hidden, 1.0),
                experts=expert_w,
            )
        )
    return model


# NumPy is loaded lazily on first engine use so that merely importing this
# module (e.g. via model_atlas.evaluation) stays dependency-light.
np = None


def _ensure_np():
    global np
    if np is None:
        import numpy as _numpy

        np = _numpy
    return np


def _model_np(model: MiniMoE) -> dict[str, Any]:
    _ensure_np()
    """Build (once) the array mirror of every layer's weights."""
    cache = getattr(model, "np_cache", None)
    if cache is not None:
        return cache
    layers = []
    for lw in model.layers:
        layers.append(
            {
                "ln_w": np.asarray(lw.ln_w, dtype=np.float64),
                "router": np.asarray(lw.router, dtype=np.float64),
                # per-expert 2D arrays: experts may have different channel
                # widths after pruning, so they are kept unstacked here and
                # grouped by width at compute time.
                "gate": [np.asarray(e["gate"], dtype=np.float64) for e in lw.experts],
                "up": [np.asarray(e["up"], dtype=np.float64) for e in lw.experts],
                "down": [np.asarray(e["down"], dtype=np.float64) for e in lw.experts],
            }
        )
    cache = {
        "embed": np.asarray(model.embed, dtype=np.float64),
        "lm_head": np.asarray(model.lm_head, dtype=np.float64),
        "layers": layers,
    }
    model.np_cache = cache
    return cache


@dataclass
class LayerTrace:
    layer: int
    logits: list[list[float]]  # [T, n_exp]
    probs_all: list[list[float]]  # [T, n_exp]
    topk_ids: list[list[int]]  # [T, k]
    topk_probs: list[list[float]]  # [T, k]
    expert_norm: list[list[float]]  # [T, n_exp]
    router_weighted: list[list[float]]  # [T, n_exp]
    entropy: list[float]  # [T]
    input_norm: list[float]  # [T] representation stats (v2 §11)
    moe_norm: list[float]  # [T]
    output_norm: list[float]  # [T]
    combined: list[list[float]]  # [T, hidden] MoE-output vector per token (v2 §11)


@dataclass
class ForwardResult:
    traces: list[LayerTrace]
    final_hidden: list[float]
    logits: list[float]  # [vocab]
    final_hidden_states: list[list[float]] = field(
        default_factory=list
    )  # per-token hidden (v2 §11)
    deviations_used: int = 0  # count of per-expert computations (none fabricated)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    _ensure_np()
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def representation_profile(
    model: MiniMoE, tokens: list[int], top_k: int | None = None
) -> list[dict[str, float]]:
    """Per-layer statistics-only representation profile (v2 §11 storage granularity).

    Stores means of input/moe/output norms and routing entropy per layer — no
    full tensors. Deterministic for the same model + tokens.
    """
    result = forward(model, tokens, top_k=top_k)
    profile: list[dict[str, float]] = []
    for trace in result.traces:
        t = len(trace.input_norm)
        profile.append(
            {
                "layer": float(trace.layer),
                "input_norm_mean": sum(trace.input_norm) / t if t else 0.0,
                "moe_norm_mean": sum(trace.moe_norm) / t if t else 0.0,
                "output_norm_mean": sum(trace.output_norm) / t if t else 0.0,
                "entropy_mean": sum(trace.entropy) / t if t else 0.0,
            }
        )
    return profile


def _forward_layer(
    model: MiniMoE,
    layer_weights: LayerWeights,
    layer_idx: int,
    hidden: list[list[float]],
    top_k: int,
    route_override: dict[tuple[int, int], list[int]] | None = None,
    excluded: dict[int, frozenset[int]] | None = None,
    channel_stats: ChannelStatsAccumulator | None = None,
) -> tuple[LayerTrace, list[list[float]]]:
    _ensure_np()
    W = _model_np(model)["layers"][layer_idx]
    H = np.asarray(hidden, dtype=np.float64)  # [T, hidden]
    T = H.shape[0]
    E = model.n_exp

    # (simplified) pre-MoE normalization in place of full attention stack
    ln = H * W["ln_w"]  # [T, hidden]
    logits = ln @ W["router"].T  # [T, n_exp]

    excluded_set = excluded.get(layer_idx, frozenset()) if excluded else frozenset()
    allowed = [e for e in range(E) if e not in excluded_set]
    if not allowed:
        raise ValueError(f"ablation excluded all experts at layer {layer_idx}")
    sub = logits[:, allowed]  # [T, A]

    # route among `allowed` (all experts normally, minus `excluded` on ablation)
    k_eff = min(top_k, len(allowed))
    order = np.argsort(-sub, axis=1, kind="stable")[:, :k_eff]  # [T, k] cols into allowed
    sel_logits = np.take_along_axis(sub, order, axis=1)  # [T, k]
    topk_p = _softmax_rows(sel_logits)  # softmax over the selected set
    # full softmax over the allowed set, scattered back over all experts
    p_mat = np.zeros((T, E), dtype=np.float64)
    p_mat[:, allowed] = _softmax_rows(sub)

    # counterfactual: force a specific equal-compute expert set; keep the
    # frozen router's gate values over that set (no router retraining)
    override = route_override or {}
    overridden = {t: list(sel) for (l, t), sel in override.items() if l == layer_idx}
    for t, sel in overridden.items():
        sel_logits_t = np.asarray([logits[t, e] for e in sel], dtype=np.float64)
        topk_p[t] = _softmax_rows(sel_logits_t)
        p_mat[t] = _softmax_rows(logits[t])  # full softmax over ALL experts

    # per-expert FFN for every token at once: [T,E,mid_e] intermediates -> [T,E,hidden].
    # experts may have different channel widths after pruning (variable mid_e), so
    # group experts by width and batch each group; results land in expert_out.
    expert_out = np.zeros((T, E, model.hidden), dtype=np.float64)
    groups: dict[int, list[int]] = {}
    for e in range(E):
        groups.setdefault(W["gate"][e].shape[0], []).append(e)
    Z_by_expert: dict[int, Any] = {}
    for width, es in groups.items():
        gs = np.stack([W["gate"][e] for e in es])          # [g, width, hidden]
        us = np.stack([W["up"][e] for e in es])            # [g, width, hidden]
        ds = np.stack([W["down"][e] for e in es])          # [g, hidden, width]
        gate_g = np.einsum("th,gwh->tgw", ln, gs)
        gate_g = gate_g / (1.0 + np.exp(-gate_g))          # SiLU on the gate branch
        up_g = np.einsum("th,gwh->tgw", ln, us)
        Zg = gate_g * up_g
        out_g = np.einsum("tgw,ghw->tgh", Zg, ds)  # [T, g, hidden]
        for gi, e in enumerate(es):
            Z_by_expert[e] = Zg[:, gi, :]
            expert_out[:, e, :] = out_g[:, gi, :]
    if channel_stats is not None:
        for e in range(E):
            channel_stats.observe_expert(layer_idx, e, Z_by_expert[e])
    expert_norm = np.linalg.norm(expert_out, axis=2)  # [T, E]
    router_weighted = p_mat * expert_norm  # [T, E]

    # combine only the selected experts, weighted by their routing probability.
    # overridden tokens take their weights over the forced expert set instead.
    contrib = np.zeros((T, E), dtype=np.float64)
    allowed_arr = np.asarray(allowed)
    for t in range(T):
        if t in overridden:
            for idx, e in enumerate(overridden[t]):
                contrib[t, e] += topk_p[t, idx]
        else:
            cols = allowed_arr[order[t]] if len(allowed) != E else order[t]
            contrib[t, cols] += topk_p[t]
    combined = np.einsum("te,teh->th", contrib, expert_out)  # [T, hidden]

    out = H + combined  # residual add
    pos = p_mat > 0.0
    entropy = -np.where(pos, p_mat * np.log(np.where(pos, p_mat, 1.0)), 0.0).sum(axis=1)

    ids_rows: list[list[int]] = []
    for t in range(T):
        if t in overridden:
            ids_rows.append(list(overridden[t]))
        else:
            ids_rows.append([int(allowed[j]) for j in order[t]])

    trace = LayerTrace(
        layer=layer_idx,
        logits=logits.tolist(),
        probs_all=p_mat.tolist(),
        topk_ids=ids_rows,
        topk_probs=topk_p.tolist(),
        expert_norm=expert_norm.tolist(),
        router_weighted=router_weighted.tolist(),
        entropy=entropy.tolist(),
        input_norm=np.linalg.norm(H, axis=1).tolist(),
        moe_norm=np.linalg.norm(combined, axis=1).tolist(),
        output_norm=np.linalg.norm(out, axis=1).tolist(),
        combined=combined.tolist(),
    )
    return trace, out.tolist()


def forward(
    model: MiniMoE,
    tokens: list[int],
    top_k: int | None = None,
    route_override: dict[tuple[int, int], list[int]] | None = None,
    excluded: dict[int, frozenset[int]] | None = None,
    channel_stats: ChannelStatsAccumulator | None = None,
) -> ForwardResult:
    _ensure_np()
    """Streaming forward across layers; returns per-layer traces + final output.

    `route_override` forces specific expert sets at (layer, token) for
    counterfactual routing. `excluded` removes experts from routing at a layer
    (ablation), renormalizing the router over the rest. Both are frozen-model
    interventions.
    """
    if top_k is None:
        top_k = model.arch.moe.top_k
    cache = _model_np(model)
    hidden = cache["embed"][list(tokens)].tolist()  # [T, hidden] as lists
    traces: list[LayerTrace] = []
    for idx, layer_w in enumerate(model.layers):
        trace, hidden = _forward_layer(
            model, layer_w, idx, hidden, top_k, route_override, excluded, channel_stats
        )
        traces.append(trace)
    final = hidden[-1]  # last token hidden state
    logits = (cache["lm_head"] @ np.asarray(final, dtype=np.float64)).tolist()  # [vocab]
    return ForwardResult(
        traces=traces,
        final_hidden=final,
        logits=logits,
        final_hidden_states=hidden,  # per-token after all layers
    )
