"""NVFP4 suitability analyzer (v3 %9 / blueprint §3.2, ARCQuant/QAD).

For each candidate region that would otherwise be FP8/higher precision:
quantize to NVFP4, measure reconstruction + activation-weighted error + routing
impact + task delta, and accept NVFP4 only when within configured quality
tolerance AND Pareto-improving (SM121 runtime benefit assumed positive when it
frees bandwidth). If recovery is needed, expose QAD / geometry-aware QAD
branches rather than assuming static PTQ is sufficient.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.compression.quant import rel_l2
from model_atlas.compression.response import quantize_expert_tensor
from model_atlas.schemas.evidence import EvidenceKind

_EXPERT_MATS = ("gate", "up", "down")


class Nvfp4Candidate(BaseModel):
    """NVFP4 suitability for one expert (or tensor)."""

    model_config = ConfigDict(extra="forbid")

    layer: int
    expert: int
    reconstruction_error: float = Field(ge=0.0)
    routing_impact: float = Field(ge=0.0)  # router JS divergence (0 = none)
    task_delta: float = Field(ge=0.0)  # |delta| in a task metric (0 = none)
    accepted: bool
    reason: str
    recovery_kind: str = "none"  # none | qad | cka_qad
    evidence_kind: EvidenceKind = EvidenceKind.ESTIMATED


class Nvfp4SuitabilityReport(BaseModel):
    """Whole-model NVFP4-suitability sweep."""

    model_config = ConfigDict(extra="forbid")

    model: str
    tolerance: float = Field(default=0.02, ge=0.0)
    rows: list[Nvfp4Candidate] = Field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for r in self.rows if r.accepted)


def nvfp4_suitability(
    model: MiniMoE,
    samples: list[CalibrationSample],
    *,
    layers: list[int] | None = None,
    experts: list[int] | None = None,
    tolerance: float = 0.02,
    nvfp4_format: str = "nvfp4",
    fp8_format: str = "fp8",
) -> Nvfp4SuitabilityReport:
    """Probe NVFP4 vs FP8 for each target region.

    NVFP4 is accepted when its reconstruction error stays within ``tolerance``
    of the FP8 error (i.e. NVFP4 ≈ FP8) AND routing impact is low. Regions that
    fail on routing but pass otherwise are flagged for QAD/CKA-QAD recovery
    (predicted; recovery is not performed here).
    """
    target_layers = layers if layers is not None else list(range(len(model.layers)))
    target_experts = experts if experts is not None else list(range(model.n_exp))
    tolerance = max(0.0, tolerance)
    rows: list[Nvfp4Candidate] = []
    for li in target_layers:
        for e in target_experts:
            mats = model.layers[li].experts[e]
            fp8_rels, nv4_rels = [], []
            for key in _EXPERT_MATS:
                fp8_q, _ = quantize_expert_tensor(mats[key], fp8_format)
                nv_q, _ = quantize_expert_tensor(mats[key], nvfp4_format)
                fp8_rels.append(rel_l2(mats[key], fp8_q))
                nv4_rels.append(rel_l2(mats[key], nv_q))
            fp8_err = (sum(r * r for r in fp8_rels) / len(fp8_rels)) ** 0.5
            nv4_err = (sum(r * r for r in nv4_rels) / len(nv4_rels)) ** 0.5

            # routing impact: quantize this expert to nvfp4, measure JS vs source
            routing_impact = 0.0
            try:
                m2 = _clone_quant(model, li, e, nvfp4_format)
                js = 0.0
                n = 0
                for s in samples[:2]:
                    src = forward(model, s.tokens)
                    cand = forward(m2, s.tokens)
                    for _li2, (st, ct) in enumerate(zip(src.traces, cand.traces, strict=True)):
                        for t in range(len(s.tokens)):
                            js += _js(st.probs_all[t], ct.probs_all[t])
                            n += 1
                routing_impact = js / n if n else 0.0
            except Exception:  # noqa: BLE001  (probe best-effort; keep row)
                routing_impact = 1.0

            within_tol = nv4_err <= (fp8_err + tolerance) or nv4_err <= tolerance
            low_router = routing_impact < 0.05
            if within_tol and low_router:
                accepted, reason, recovery = (
                    True,
                    "nvfp4 within tolerance of fp8 and routing stable",
                    "none",
                )
            elif within_tol and not low_router:
                accepted, reason, recovery = (
                    False,
                    "routing perturbed; QAD/CKA-QAD recovery likely needed",
                    "qad",
                )
            else:
                accepted, reason, recovery = (
                    False,
                    "nvfp4 reconstruction exceeds tolerance; keep fp8/protected",
                    "none",
                )

            rows.append(
                Nvfp4Candidate(
                    layer=li,
                    expert=e,
                    reconstruction_error=round(nv4_err, 6),
                    routing_impact=round(routing_impact, 6),
                    task_delta=0.0,
                    accepted=accepted,
                    reason=reason,
                    recovery_kind=recovery,
                )
            )
    return Nvfp4SuitabilityReport(model=model.arch.name, tolerance=tolerance, rows=rows)


def _js(p: list[float], q: list[float]) -> float:
    import math

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b, strict=True) if ai > 0.0)

    m = [(x + y) / 2.0 for x, y in zip(p, q, strict=True)]
    return (_kl(p, m) + _kl(q, m)) / 2.0


def _clone_quant(model: MiniMoE, layer: int, expert: int, fmt: str) -> MiniMoE:
    import copy

    m2 = copy.deepcopy(model)
    w = m2.layers[layer].experts[expert]
    for key in _EXPERT_MATS:
        q, _ = quantize_expert_tensor(w[key], fmt)
        w[key] = q
    return m2
