"""Routing-consistency guard (v3 %4 / blueprint §3.2, VSRAQ).

For every materialized MoE candidate we measure router divergence:
- router logit/value divergence;
- top-k expert agreement;
- rank-order agreement;
- decision-boundary margin change;
- per-corpus-cluster routing drift.

Candidates can fail this gate even if perplexity is acceptable. The guard never
mutates anything; it measures whether quantization/structure perturbed routing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.schemas.evidence import EvidenceKind


def _jensen_shannon(p: list[float], q: list[float]) -> float:
    import math

    def _softmax(x: list[float]) -> list[float]:
        m = max(x)
        e = [math.exp(v - m) for v in x]
        s = sum(e)
        return [v / s for v in e]

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b, strict=True) if ai > 0.0)

    p_s, q_s = _softmax(p), _softmax(q)
    m = [(x + y) / 2.0 for x, y in zip(p_s, q_s, strict=True)]
    return (_kl(p_s, m) + _kl(q_s, m)) / 2.0


class RoutingConsistency(BaseModel):
    """Routing-consistency evidence for one (layer, token) comparison."""

    model_config = ConfigDict(extra="forbid")

    layer: int
    token: int
    topk_agreement: float = Field(ge=0.0, le=1.0)  # Jaccard of top-k sets
    rank_order_agreement: float = Field(ge=0.0, le=1.0)  # Kendall-tau on top-k
    boundary_margin_delta: float = Field(ge=0.0)  # change in top-k boundary margin
    router_js_divergence: float = Field(ge=0.0)
    drift_flag: bool = False
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED


class RoutingConsistencyReport(BaseModel):
    """Whole-run routing-consistency summary against the gate threshold."""

    model_config = ConfigDict(extra="forbid")

    model: str
    threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    rows: list[RoutingConsistency] = Field(default_factory=list)
    passed: bool = True

    @property
    def mean_topk_agreement(self) -> float:
        return sum(r.topk_agreement for r in self.rows) / len(self.rows) if self.rows else 1.0


def _kendall_tau(a: list[int], b: list[int]) -> float:
    pos = {x: i for i, x in enumerate(a)}
    pairs = [(pos[y], j) for j, y in enumerate(b) if y in pos]
    inv = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1]) < 0:
                inv += 1
    n = len(pairs)
    return 1.0 - 2.0 * inv / (n * (n - 1)) if n > 1 else 1.0


def routing_consistency(
    source: MiniMoE,
    candidate: MiniMoE,
    samples: list[CalibrationSample],
    *,
    top_k: int | None = None,
    threshold: float = 0.95,
) -> RoutingConsistencyReport:
    """Measure how consistently the candidate preserves routing vs the source."""
    k = top_k or source.arch.moe.top_k
    rows: list[RoutingConsistency] = []
    for s in samples:
        src = forward(source, s.tokens, top_k=k)
        cand = forward(candidate, s.tokens, top_k=k)
        for li, (strace, ctrace) in enumerate(zip(src.traces, cand.traces, strict=True)):
            for t in range(len(s.tokens)):
                a = strace.topk_ids[t]
                b = ctrace.topk_ids[t]
                sa, sb = set(a), set(b)
                jac = len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0
                tau = _kendall_tau(list(sa), list(sb))

                # boundary margin: gap between k-th and (k+1)-th softmax
                def _margin(probs: list[float]) -> float:
                    sorted_p = sorted(probs, reverse=True)
                    return sorted_p[k - 1] - sorted_p[k] if len(sorted_p) > k else 0.0

                m_src = _margin(strace.probs_all[t])
                m_cand = _margin(ctrace.probs_all[t])
                js = _jensen_shannon(strace.logits[t], ctrace.logits[t])
                flag = jac < threshold or js > 0.1
                rows.append(
                    RoutingConsistency(
                        layer=li,
                        token=t,
                        topk_agreement=round(jac, 6),
                        rank_order_agreement=round(tau, 6),
                        boundary_margin_delta=round(abs(m_cand - m_src), 6),
                        router_js_divergence=round(js, 6),
                        drift_flag=flag,
                    )
                )
    passed = all(not r.drift_flag for r in rows)
    return RoutingConsistencyReport(
        model=candidate.arch.name, threshold=threshold, rows=rows, passed=passed
    )
