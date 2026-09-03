"""A genuine synthetic mini-MoE and its streaming layerwise forward pass.

This is a real (tiny) model: seeded random weights, real router top-k gating,
softmax probabilities, per-expert gated outputs, router-weighted combination,
and per-expert output norms computed from actual activations. Nothing here is
fabricated — it is the honest substitute for the oversized checkpoint that
fully exercises the layerwise tracing + REAP machinery (fork A).

The forward iterates one layer at a time and keeps only the running hidden
state, so it plays the role of the "stream one layer, free it" REAP loop.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from cebu_profiler.profiler.collector import ChannelStatsAccumulator
from cebu_profiler.schemas.architecture import ArchitectureSpec


# a: [n, m] x b: [m, p] -> [n, p]
def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n, m = len(a), len(a[0]) if a else 0
    p = len(b[0]) if b else 0
    out: list[list[float]] = [[0.0] * p for _ in range(n)]
    for i in range(n):
        ai = a[i]
        for j in range(m):
            aij = ai[j]
            if aij == 0.0:
                continue
            bj = b[j]
            oi = out[i]
            for q in range(p):
                oi[q] += aij * bj[q]
    return out


def _matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(ai[q] * x[q] for q in range(len(x))) for ai in a]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


def _silu_vec(x: list[float]) -> list[float]:
    return [v / (1.0 + math.exp(-v)) for v in x]


def _dot(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def _l2_norm(x: list[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def _softmax_full(x: list[float]) -> list[float]:
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def _softmax_topk(logits: list[float], k: int) -> tuple[list[int], list[float]]:
    """Top-k selected indices + softmax probabilities over the selected set."""
    order = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    selected = order[:k]
    sel_logits = [logits[i] for i in selected]
    probs = _softmax_full(sel_logits)
    return selected, probs


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
    T = len(hidden)
    logits: list[list[float]] = []
    probs_all: list[list[float]] = []
    topk_ids: list[list[int]] = []
    topk_probs: list[list[float]] = []
    expert_norm: list[list[float]] = []
    router_weighted: list[list[float]] = []
    entropy: list[float] = []
    input_norm: list[float] = []
    moe_norm: list[float] = []
    output_norm: list[float] = []
    combined_out: list[list[float]] = []
    out: list[list[float]] = []
    override = route_override or {}
    excluded_set = excluded.get(layer_idx, frozenset()) if excluded else frozenset()
    allowed = [e for e in range(model.n_exp) if e not in excluded_set]
    if not allowed:
        raise ValueError(f"ablation excluded all experts at layer {layer_idx}")

    for t in range(T):
        h = hidden[t]
        # (simplified) pre-MoE normalization in place of full attention stack
        ln = [(v * w) for v, w in zip(h, layer_weights.ln_w, strict=True)]
        i_norm = _l2_norm(h)
        logits_l = _matvec(layer_weights.router, ln)  # [n_exp]
        sub = [logits_l[e] for e in allowed]
        if (layer_idx, t) in override:
            # counterfactual: force a specific equal-compute expert set; keep the
            # frozen router's gate values over that set (no router retraining)
            sel = list(override[(layer_idx, t)])
            sel_p = _softmax_full([logits_l[e] for e in sel])
            p = _softmax_full(logits_l)
        else:
            # route among `allowed` (all experts normally, minus `excluded` on ablation)
            sel_idx, sel_p = _softmax_topk(sub, top_k)
            sel = [allowed[i] for i in sel_idx]
            sub_p = _softmax_full(sub)
            p = [0.0] * model.n_exp
            for idx, e in enumerate(allowed):
                p[e] = sub_p[idx]

        norm_e: list[float] = []
        weighted: list[float] = []
        combined = [0.0] * model.hidden
        for e, we in enumerate(layer_weights.experts):
            gate = _silu_vec(_matvec(we["gate"], ln))
            up = _matvec(we["up"], ln)
            if channel_stats is not None:
                channel_stats.observe_expert(layer_idx, e, gate, up)
            expert_out = _matvec(we["down"], [g * u for g, u in zip(gate, up, strict=True)])
            nrm = _l2_norm(expert_out)
            norm_e.append(nrm)
            weighted.append(p[e] * nrm)
            if e in sel:
                idx = sel.index(e)
                combined = _vec_add(combined, [sel_p[idx] * v for v in expert_out])
        # residual add
        nt = _vec_add(h, combined)
        out.append(nt)
        combined_out.append(combined)

        H = -sum(pi * math.log(pi) for pi in p if pi > 0.0)
        entropy.append(H)
        logits.append(logits_l)
        input_norm.append(i_norm)
        moe_norm.append(_l2_norm(combined))
        output_norm.append(_l2_norm(nt))
        probs_all.append(p)
        topk_ids.append(sel)
        topk_probs.append(sel_p)
        expert_norm.append(norm_e)
        router_weighted.append(weighted)

    trace = LayerTrace(
        layer=layer_idx,
        logits=logits,
        probs_all=probs_all,
        topk_ids=topk_ids,
        topk_probs=topk_probs,
        expert_norm=expert_norm,
        router_weighted=router_weighted,
        entropy=entropy,
        input_norm=input_norm,
        moe_norm=moe_norm,
        output_norm=output_norm,
        combined=combined_out,
    )
    return trace, out


def forward(
    model: MiniMoE,
    tokens: list[int],
    top_k: int | None = None,
    route_override: dict[tuple[int, int], list[int]] | None = None,
    excluded: dict[int, frozenset[int]] | None = None,
    channel_stats: ChannelStatsAccumulator | None = None,
) -> ForwardResult:
    """Streaming forward across layers; returns per-layer traces + final output.

    `route_override` forces specific expert sets at (layer, token) for
    counterfactual routing. `excluded` removes experts from routing at a layer
    (ablation), renormalizing the router over the rest. Both are frozen-model
    interventions.
    """
    if top_k is None:
        top_k = model.arch.moe.top_k
    emb = model.embed
    hidden = [list(emb[t]) for t in tokens]
    traces: list[LayerTrace] = []
    for idx, layer_w in enumerate(model.layers):
        trace, hidden = _forward_layer(
            model, layer_w, idx, hidden, top_k, route_override, excluded, channel_stats
        )
        traces.append(trace)
    final = hidden[-1]  # last token hidden state
    logits = _matvec(model.lm_head, final)  # [vocab]
    return ForwardResult(
        traces=traces,
        final_hidden=final,
        logits=logits,
        final_hidden_states=hidden,  # per-token after all layers
    )
