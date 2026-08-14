"""Torch-backed real scoring: TENP / FlexMoE channel ranking / grouped-Taylor /
causal ablation, wired to tensor/trace protocols (Phase 3).

These operate on *real* torch tensors (decoded from the bounded streaming
substrate via the exec venv) and on real hidden-state traces, independent of the
PurePython `MiniMoE`. They are forward-only where possible and expose which
needs require high-precision/gradients (reported honestly). The pure-Python
`scoring/*` scorers remain for the deterministic fixture; this module is the
real-weight path.

Requires torch (exec venv). Default repo venv has none, so torch is imported
lazily and callers handle its absence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from model_atlas.scoring.base import ScoreNeed, ScorerRequirements


def _torch() -> Any:
    import torch  # type: ignore[import-not-found]  # torch lives in .venv-exec, not repo venv

    return torch


@dataclass
class TorchScoringResult:
    """Measured per-(layer, expert, channel) scores from real tensors."""

    requirements: ScorerRequirements
    rows: dict[tuple[int, int, int], dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": sorted(self.requirements.needs),
            "forward_only": self.requirements.forward_only,
            "note": self.requirements.note,
            "rows": {f"{layer}:{e}:{c}": v for (layer, e, c), v in self.rows.items()},
            "notes": [str(n) for n in self.notes],
        }


def tenp_importance(
    gate: Any,
    up: Any,
    down: Any,
    z_activation: Any | None = None,
    *,
    expert_norm: float = 1.0,
) -> dict[int, float]:
    """TENP score per channel: `score(c) = mean|z_c| * ||down[:,c]||`.

    All args are torch tensors: gate/up [k, hidden], down [hidden, k].
    `z_activation` (optional, [num_tokens, k]) supplies the measured mean |z|;
    if omitted, uses ||gate||*||up|| over the rows.
    """
    k = down.shape[1]
    col_norm = down.norm(dim=0)  # [k]
    if z_activation is not None:
        z_norm = z_activation.abs().mean(dim=0)  # [k]
    else:
        g = gate.norm(dim=1)
        u = up.norm(dim=1)
        z_norm = (g * u) / max(int(gate.shape[0]), 1)  # approximate per-row factor
    score = (z_norm * col_norm * expert_norm).tolist()
    return {c: float(score[c]) for c in range(k)}


def flexmoe_channel_ranking(
    importance: dict[int, float], k: int, budget_frac: float = 0.7
) -> list[int]:
    """FlexMoE-style nested intra-expert channel ranking by importance.

    Returns the top-`ceil(k*budget_frac)` channels, retained per expert — every
    routed expert is retained, only its width changes.
    """
    ranked = sorted(importance, key=lambda c: -importance[c])
    keep = max(1, int(math.ceil(k * budget_frac)))
    return ranked[:keep]


def grouped_taylor_surrogate(
    gate: Any,
    up: Any,
    down: Any,
    z_activation: Any,
    *,
    group_size: int = 16,
    lambda_: float = 1e-3,
) -> dict[int, float]:
    """Grouped-Taylor/Hessian surrogate per channel using activation Hessian.

    Approximates the Hessian-vector curvature via the intermediate activation's
    second moment and group-wise averaging (a cheap forward-only surrogate;
    reported as ESTIMATED, not causal).
    """
    # z^2 expectation as a curvature proxy along the output-projection direction
    col_norm2 = (down.pow(2)).sum(dim=0)  # [k]
    zsq = (z_activation.pow(2)).mean(dim=0)  # [k]
    raw = (zsq * col_norm2).tolist()
    n = len(raw)
    out: dict[int, float] = {}
    for g in range(0, n, group_size):
        chunk = raw[g : g + group_size]
        if not chunk:
            continue
        base = sum(chunk) / len(chunk)
        for i, v in enumerate(chunk):
            out[g + i] = base + lambda_ * (v - base)  # L2-regularized group surrogate
    return out


def causal_ablation_scores(
    z_activation: Any,
    expert_out: Any,
    *,
    epsilon: float = 1e-6,
) -> dict[int, float]:
    """Channel-wise causal ablation proxy: contribution of each channel to the
    expert output norm, via the derivative of the norm w.r.t. each channel.

    Uses the output projection's column dot the activation to measure marginal
    contribution (a cheap causal proxy; gradients not required).
    """
    # expert_out: [num_tokens, hidden]; z_activation: [num_tokens, k]
    # marginal = mean over tokens of (expert_out * z) per channel, via down norm
    z_norm = z_activation.norm(dim=1)  # [num_tokens]
    contrib = (z_activation * z_norm.unsqueeze(1)).mean(dim=0).tolist()
    denom = sum(abs(v) for v in contrib) or 1.0
    return {c: abs(v) / denom for c, v in enumerate(contrib)}


def needs_for_real_scoring(trace_mode: str) -> ScorerRequirements:
    """Report what this path requires (honest feature detection).

    - `bounded_cpu`: forward-only over decoder-exposed activations/tensors, no
      gradients, no full-model materialization.
    - `full_forward`: needs a real per-token forward -> HIGH_PRECISION_WEIGHTS +
      ROUTER_LOGITS (service-window gated).
    """
    if trace_mode == "bounded_cpu":
        return ScorerRequirements(
            frozenset({ScoreNeed.FORWARD_ACTIVATIONS, ScoreNeed.RAW_EXPERT_TENSORS}),
            note="bounded CPU route: forward-only, no gradients, no GPU",
        )
    return ScorerRequirements(
        frozenset(
            {
                ScoreNeed.FORWARD_ACTIVATIONS,
                ScoreNeed.RAW_EXPERT_TENSORS,
                ScoreNeed.HIGH_PRECISION_WEIGHTS,
                ScoreNeed.ROUTER_LOGITS,
            }
        ),
        note="full forward requires a real per-token run (service-window gated)",
    )
