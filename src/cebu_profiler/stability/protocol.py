"""Rank-trust protocol: split-half stability, Jaccard@k, and proxy controls.

Pattern after the split-half reliability practice used in published evidence
bundles (e.g. alesha-pro/atlas GLM-5.3-Flash `stability` block: shard-parity
split-half exact-REAP Spearman + keep-set Jaccard at fixed sizes + three named
proxy controls), re-implemented natively for the Cebu Profiler scorer interface.

Evidence-typed per AGENTS.md invariant 12: the verdict never upgrades a proxy
agreement to 'measured' — `measured` is reserved for split-half on real runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cebu_profiler.scoring.stability import _spearman

# A run's scores keyed by slot (any hashable; e.g. (layer, expert) or (layer, expert, channel)).
ScoreMap = dict[tuple[int, ...], float]

# Default keep-set sizes for Jaccard@k (mirrors the published 72/108/144/216 pattern,
# scaled for Cebu's per-expert channel and per-layer expert slots).
DEFAULT_KS = (72, 108, 144, 216)

# Proxy-control names, in published order.
CONTROL_NAMES = ("count", "mass", "proxy")


@dataclass
class SplitHalf:
    """Split-half agreement between two score maps over their common slots."""

    spearman: float
    n_common: int

    def payload(self) -> dict[str, Any]:
        return {"spearman": self.spearman, "n_common": self.n_common}


@dataclass
class JaccardAtK:
    """Keep-set overlap between two rankings, at each requested size k."""

    jaccard: dict[int, float] = field(default_factory=dict)

    def payload(self) -> dict[int, float]:
        return dict(self.jaccard)


@dataclass
class ProtocolResult:
    split_half: SplitHalf
    jaccard: JaccardAtK
    controls: dict[str, float]
    verdict: str  # "measured" | "proxy" | "insufficient"
    meta: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "halves": {
                "spearman": self.split_half.spearman,
                "n_common": self.split_half.n_common,
            },
            "keep_set_jaccard": self.jaccard.payload(),
            "controls_vs_reference": dict(self.controls),
            "verdict": self.verdict,
            "meta": dict(self.meta),
        }


def _top_k_slots(scores: ScoreMap, k: int) -> set[tuple[int, ...]]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {slot for slot, _ in ordered[:k]}


def _jaccard(a: set[tuple[int, ...]], b: set[tuple[int, ...]]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def split_half_agreement(
    half_a: ScoreMap, half_b: ScoreMap, ks: Sequence[int] = DEFAULT_KS
) -> tuple[SplitHalf, JaccardAtK]:
    """Spearman over common slots + Jaccard of top-k keep-sets between halves."""
    common = sorted(set(half_a) & set(half_b))
    rho = 1.0
    if len(common) >= 2:
        rho = _spearman([half_a[s] for s in common], [half_b[s] for s in common])
    jac = JaccardAtK()
    for k in ks:
        max_k = min(k, len(half_a), len(half_b))
        if max_k == 0:
            jac.jaccard[k] = 0.0
            continue
        jac.jaccard[k] = _jaccard(_top_k_slots(half_a, max_k), _top_k_slots(half_b, max_k))
    return SplitHalf(spearman=rho, n_common=len(common)), jac


def proxy_controls(reference: ScoreMap, proxies: dict[str, ScoreMap]) -> dict[str, float]:
    """Spearman of each named proxy against the reference ranking."""
    out: dict[str, float] = {}
    common = sorted(set(reference) & set().union(*proxies.values())) if proxies else []
    for name, proxy in proxies.items():
        slots = sorted(set(reference) & set(proxy))
        if len(slots) < 2:
            out[name] = 0.0
            continue
        out[name] = _spearman([reference[s] for s in slots], [proxy[s] for s in slots])
    if not common and proxies:
        # no overlap at all: explicit zeros rather than fabricated agreement
        for name in proxies:
            out.setdefault(name, 0.0)
    return out


def run_rank_trust_protocol(
    half_a: ScoreMap,
    half_b: ScoreMap,
    *,
    proxies: dict[str, ScoreMap] | None = None,
    reference: ScoreMap | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    meta: dict[str, Any] | None = None,
) -> ProtocolResult:
    """Full rank-trust protocol over one calibration split pair.

    Verdict rule (evidence-typed):
      - "measured": split-half Spearman >= 0.9 on >= 100 common slots
      - "proxy": split exists but thin agreement/coverage; controls decide
      - "insufficient": fewer than 2 common slots — no ranking claim at all
    """
    sh, jac = split_half_agreement(half_a, half_b, ks=ks)
    ctrls = proxy_controls(reference or half_a, proxies or {})
    if sh.n_common < 2:
        verdict = "insufficient"
    elif sh.spearman >= 0.9 and sh.n_common >= 100:
        verdict = "measured"
    else:
        verdict = "proxy"
    return ProtocolResult(
        split_half=sh,
        jaccard=jac,
        controls=ctrls,
        verdict=verdict,
        meta=dict(meta or {}),
    )
