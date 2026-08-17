"""Teacher-relative KLD/CKA quality gate, separate from identity control.

The identity-capture runner establishes determinism of the boundary
(source versus an identical re-capture); it explicitly does NOT claim quality
(``quality_claim=False``). This module is the *distinct* adjudication that
accepts or rejects a width-slice candidate by teacher-relative distance to the
NVFP4 source: ``KL(source || candidate)`` per token, worst-domain, p99, and the
minimum layer CKA, under explicit budgets via ``decide_kl_gate``.

It is fail-closed: requires identity control to have passed first, requires at
least one CKA layer, and produces an accept/reject — never a blanket pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from model_atlas.prune.kl_gate import KLGateBudget, KLGateError, KLGateResult, decide_kl_gate


class _Aggregate(Protocol):
    token_weighted_mean: float
    p99: float


class _Domain(Protocol):
    token_weighted_mean: float


class _DomainReport(Protocol):
    overall: _Aggregate
    by_domain: list[_Domain]


class _KLDCapture(Protocol):
    report: _DomainReport


class _LayerCKA(Protocol):
    score: float


class _MetricReport(Protocol):
    identity_control_passed: bool | None
    kld: _KLDCapture
    layer_cka: list[_LayerCKA]


@dataclass(frozen=True)
class QualityGateRejection(Exception):
    """Raised when the quality gate does NOT accept the candidate."""

    result: KLGateResult


def gate_inputs_from_report(report: _MetricReport) -> dict[str, float]:
    """Map a capture metric report to the gate's scalar inputs.

    ``identity_control_passed`` must be true (the boundary was proven stable
    before making any quality claim); otherwise the gate refuses to run.
    """
    if report.identity_control_passed is not True:
        raise KLGateError(
            "quality gate requires identity control to have passed first"
        )
    overall = report.kld.report.overall
    worst_domain = max(
        (d.token_weighted_mean for d in report.kld.report.by_domain),
        default=overall.token_weighted_mean,
    )
    if not report.layer_cka:
        raise KLGateError("quality gate requires at least one CKA layer")
    min_cka = min(layer.score for layer in report.layer_cka)
    return {
        "mean_kld": overall.token_weighted_mean,
        "worst_domain_kld": worst_domain,
        "p99_kld": overall.p99,
        "min_cka": min_cka,
    }


def run_kl_quality_gate(
    *,
    report: _MetricReport,
    budgets: KLGateBudget,
) -> KLGateResult:
    """Adjudicate the candidate under KL/CKA budgets; raises on rejection."""
    inputs = gate_inputs_from_report(report)
    result = decide_kl_gate(
        mean_kld=inputs["mean_kld"],
        worst_domain_kld=inputs["worst_domain_kld"],
        p99_kld=inputs["p99_kld"],
        min_cka=inputs["min_cka"],
        mean_budget=budgets.mean_kld,
        worst_domain_budget=budgets.worst_domain_kld,
        p99_budget=budgets.p99_kld,
        cka_floor=budgets.cka_floor,
    )
    if not result.accepted:
        raise QualityGateRejection(result)
    return result


__all__ = [
    "QualityGateRejection",
    "gate_inputs_from_report",
    "run_kl_quality_gate",
]
