"""KLD-gated acceptance decision for width-pruned candidates.

A thin, pure decision rule over teacher-relative evidence. It consumes the
aggregates produced by ``model_atlas.evaluation.capture_metrics`` (or the
pure ``model_atlas.evaluation.kld`` report) and decides pass/fail against an
explicit quality budget. It never invents evidence and never "auto-passes" a
candidate above budget.

Metric direction: smaller KLD is better (little drift from the NVFP4 source
teacher); larger CKA is better (intermediate representation preserved).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


class KLGateError(ValueError):
    """Raised when gate inputs are malformed / non-finite."""


@dataclass(frozen=True)
class KLGateBudget:
    mean_kld: float
    worst_domain_kld: float
    p99_kld: float
    cka_floor: float


@dataclass(frozen=True)
class KLGateResult:
    accepted: bool
    mean_kld: float
    worst_domain_kld: float
    p99_kld: float
    min_cka: float
    failures: tuple[str, ...] = field(default_factory=tuple)


def _finite_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise KLGateError(f"{name} must be finite and > 0")


def decide_kl_gate(
    *,
    mean_kld: float,
    worst_domain_kld: float,
    p99_kld: float,
    min_cka: float,
    mean_budget: float,
    worst_domain_budget: float,
    p99_budget: float,
    cka_floor: float,
) -> KLGateResult:
    """Return an acceptance decision under KL(teacher||candidate) budgets.

    Accepts only if every constraint holds: token-weighted mean KLD <= budget,
    worst per-domain KLD <= budget, p99 KLD <= budget, and min layer CKA >=
    floor. On any violation, returns ``accepted=False`` with the exact failing
    constraints — never a blanket pass.
    """
    for name, value, budget in (
        ("mean_kld", mean_kld, mean_budget),
        ("worst_domain_kld", worst_domain_kld, worst_domain_budget),
        ("p99_kld", p99_kld, p99_budget),
    ):
        if not isfinite(value):
            raise KLGateError(f"{name} must be finite")
        if not isfinite(budget) or budget < 0.0:
            raise KLGateError(f"{name}_budget must be finite and >= 0")
    if not isfinite(min_cka):
        raise KLGateError("min_cka must be finite")
    if not isfinite(cka_floor):
        raise KLGateError("cka_floor must be finite")

    failures: list[str] = []
    if mean_kld > mean_budget:
        failures.append(
            f"mean KLD {mean_kld:.6g} > budget {mean_budget:g}"
        )
    if worst_domain_kld > worst_domain_budget:
        failures.append(
            f"worst-domain KLD {worst_domain_kld:.6g} > budget "
            f"{worst_domain_budget:g}"
        )
    if p99_kld > p99_budget:
        failures.append(f"p99 KLD {p99_kld:.6g} > budget {p99_budget:g}")
    if min_cka < cka_floor:
        failures.append(f"min CKA {min_cka:.6g} < floor {cka_floor:g}")

    return KLGateResult(
        accepted=not failures,
        mean_kld=mean_kld,
        worst_domain_kld=worst_domain_kld,
        p99_kld=p99_kld,
        min_cka=min_cka,
        failures=tuple(failures),
    )


__all__ = [
    "KLGateBudget",
    "KLGateError",
    "KLGateResult",
    "decide_kl_gate",
]
