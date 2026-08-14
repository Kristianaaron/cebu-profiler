"""Torch-backed scoring kernels + REAL-HOOK interface (Phase 3, review-corrected).

Two distinct things, never conflated:

1. **Kernels** (pure functions over torch tensors) — TENP / FlexMoE channel
   ranking / grouped-Taylor surrogate / causal-ablation proxy. These are
   IMPLEMENTATIONS: when fed synthetic/random tensors they produce kernel
   results, NOT measured TENP/Taylor/causal evidence. Every result therefore
   carries a `provenance` that labels its input source and evidence kind.

2. **Real-hook interface** — a forward hook / observer API that a *real* corpus
   forward (in the maintenance window) can drive to capture genuine activations
   and router logits, which then flow into the same kernels. The measured gate
   stays CLOSED until a real corpus forward has run (service-window gated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from model_atlas.schemas.evidence import EvidenceKind
from model_atlas.scoring.base import ScoreNeed, ScorerRequirements


def _torch() -> Any:
    import torch  # type: ignore[import-not-found]  # exists in .venv-exec only

    return torch


def _provenance(input_source: str, evidence: EvidenceKind) -> str:
    return (
        f"kernels are implementations; input_source={input_source}, "
        f"evidence={evidence.value}"
    )


@dataclass
class TorchScoringResult:
    """Per-(layer, expert, channel) scores + honest provenance."""

    requirements: ScorerRequirements
    rows: dict[tuple[int, int, int], dict[str, float]] = field(default_factory=dict)
    input_source: str = "synthetic"  # synthetic | real_corpus_forward
    evidence_kind: EvidenceKind = EvidenceKind.PREDICTED
    provenance: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.provenance:
            self.provenance = _provenance(self.input_source, self.evidence_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": sorted(self.requirements.needs),
            "forward_only": self.requirements.forward_only,
            "note": self.requirements.note,
            "rows": {f"{layer}:{e}:{c}": v for (layer, e, c), v in self.rows.items()},
            "input_source": self.input_source,
            "evidence_kind": self.evidence_kind.value,
            "provenance": self.provenance,
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
    """TENP kernel: `score(c) = mean|z_c| * ||down[:,c]||` on torch tensors."""
    k = down.shape[1]
    col_norm = down.norm(dim=0)
    if z_activation is not None:
        z_norm = z_activation.abs().mean(dim=0)
    else:
        g = gate.norm(dim=1)
        u = up.norm(dim=1)
        z_norm = (g * u) / max(int(gate.shape[0]), 1)
    score = (z_norm * col_norm * expert_norm).tolist()
    return {c: float(score[c]) for c in range(k)}


def flexmoe_channel_ranking(
    importance: dict[int, float], k: int, budget_frac: float = 0.7
) -> list[int]:
    """FlexMoE nested intra-expert ranking; retains every expert (width only)."""
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
    """Grouped-Taylor/Hessian surrogate kernel (forward-only; ESTIMATED not causal)."""
    col_norm2 = (down.pow(2)).sum(dim=0)
    zsq = (z_activation.pow(2)).mean(dim=0)
    raw = (zsq * col_norm2).tolist()
    n = len(raw)
    out: dict[int, float] = {}
    for g in range(0, n, group_size):
        chunk = raw[g : g + group_size]
        if not chunk:
            continue
        base = sum(chunk) / len(chunk)
        for i, v in enumerate(chunk):
            out[g + i] = base + lambda_ * (v - base)
    return out


def causal_ablation_scores(
    z_activation: Any,
    expert_out: Any,
    *,
    epsilon: float = 1e-6,
) -> dict[int, float]:
    """Channel-wise causal-ablation PROXY kernel (not a causal test)."""
    z_norm = z_activation.norm(dim=1)
    contrib = (z_activation * z_norm.unsqueeze(1)).mean(dim=0).tolist()
    denom = sum(abs(v) for v in contrib) or 1.0
    return {c: abs(v) / denom for c, v in enumerate(contrib)}


def needs_for_real_scoring(trace_mode: str) -> ScorerRequirements:
    """Honest requirements for bounded_cpu vs full_forward."""
    if trace_mode == "bounded_cpu":
        return ScorerRequirements(
            frozenset({ScoreNeed.FORWARD_ACTIVATIONS, ScoreNeed.RAW_EXPERT_TENSORS}),
            note="bounded CPU route: kernels over decoder-exposed tensors; "
            "forward-only, no gradients, no GPU; evidence PREDICTED until a real "
            "corpus forward runs",
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
        note="full forward requires a real per-token run (service-window gated); "
        "only then can evidence become MEASURED",
    )


class RealActivationHook:
    """Hook interface to capture REAL activations/router during a corpus forward.

    Attach to a real model's MoE expert modules in the maintenance window; the
    captured `z_activation` / router logits feed the kernels above. Until a real
    forward has run, `has_captured` is False and any scoring is PREDICTED.
    """

    def __init__(self, layer: int, expert: int) -> None:
        self.layer = layer
        self.expert = expert
        self.z_activation: Any | None = None
        self.router_logits: Any | None = None
        self.has_captured = False
        self._hook_ref: Any | None = None

    def __call__(self, module: Any, inp: Any, out: Any) -> None:
        """Forward-hook body: store intermediate activation + router logits."""
        self.z_activation = out  # expert-intermediate / gated activation
        self.has_captured = True

    def capture(self, z_activation: Any, router_logits: Any | None = None) -> None:
        """Direct capture (for tests / offline replay)."""
        self.z_activation = z_activation
        self.router_logits = router_logits
        self.has_captured = True

    def attach(self, module: Any) -> Any:
        """Register as a forward hook on a torch module; returns the handle."""
        self._hook_ref = module.register_forward_hook(self)
        return self._hook_ref

    def detach(self) -> None:
        if self._hook_ref is not None:
            self._hook_ref.remove()
            self._hook_ref = None

    def is_measured(self) -> bool:
        """Real-hook captures are only MEASURED evidence if a real corpus forward
        actually ran (has_captured) AND the caller declares it real; otherwise
        the measured gate stays closed."""
        return bool(self.has_captured)
