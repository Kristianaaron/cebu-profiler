from types import SimpleNamespace

import pytest

from model_atlas.evaluation.quality_gate import (
    QualityGateRejection,
    gate_inputs_from_report,
    run_kl_quality_gate,
)
from model_atlas.prune.kl_gate import KLGateBudget, KLGateError


def _report(
    *,
    mean: float = 0.5,
    p99: float = 1.2,
    domains: list[float] | None = None,
    cka: list[float] | None = None,
    identity: bool = True,
) -> SimpleNamespace:
    overall = SimpleNamespace(token_weighted_mean=mean, p99=p99)
    by_domain = [
        SimpleNamespace(token_weighted_mean=d) for d in (domains if domains is not None else [mean])
    ]
    layer_cka = [SimpleNamespace(score=c) for c in (cka if cka is not None else [0.9])]
    return SimpleNamespace(
        identity_control_passed=identity,
        kld=SimpleNamespace(report=SimpleNamespace(overall=overall, by_domain=by_domain)),
        layer_cka=layer_cka,
    )


def test_gate_inputs_map_worst_domain_and_min_cka() -> None:
    report = _report(mean=0.5, p99=1.2, domains=[0.4, 0.6], cka=[0.9, 0.85])
    inputs = gate_inputs_from_report(report)
    assert inputs["mean_kld"] == 0.5
    assert inputs["worst_domain_kld"] == 0.6
    assert inputs["p99_kld"] == 1.2
    assert inputs["min_cka"] == 0.85


def test_gate_accepts_within_budget() -> None:
    report = _report(mean=0.5, p99=1.2, domains=[0.4, 0.6], cka=[0.9, 0.85])
    budgets = KLGateBudget(mean_kld=1.0, worst_domain_kld=1.0, p99_kld=2.0, cka_floor=0.8)
    result = run_kl_quality_gate(report=report, budgets=budgets)
    assert result.accepted is True
    assert result.mean_kld == 0.5
    assert result.worst_domain_kld == 0.6  # worst per-domain mean
    assert result.p99_kld == 1.2
    assert result.min_cka == 0.85


def test_gate_rejects_on_mean_budget_violation() -> None:
    report = _report(mean=3.0)
    budgets = KLGateBudget(mean_kld=1.0, worst_domain_kld=5.0, p99_kld=5.0, cka_floor=0.5)
    with pytest.raises(QualityGateRejection) as exc:
        run_kl_quality_gate(report=report, budgets=budgets)
    assert exc.value.result.accepted is False
    assert any("mean KLD" in f for f in exc.value.result.failures)


def test_gate_rejects_on_cka_floor_violation() -> None:
    report = _report(cka=[0.4])
    budgets = KLGateBudget(mean_kld=5.0, worst_domain_kld=5.0, p99_kld=5.0, cka_floor=0.8)
    with pytest.raises(QualityGateRejection) as exc:
        run_kl_quality_gate(report=report, budgets=budgets)
    assert any("CKA" in f for f in exc.value.result.failures)


def test_gate_requires_identity_control_first() -> None:
    report = _report(identity=False)
    budgets = KLGateBudget(mean_kld=5.0, worst_domain_kld=5.0, p99_kld=5.0, cka_floor=0.8)
    with pytest.raises(KLGateError, match="identity control"):
        run_kl_quality_gate(report=report, budgets=budgets)


def test_gate_requires_at_least_one_cka_layer() -> None:
    report = _report(cka=[])
    budgets = KLGateBudget(mean_kld=5.0, worst_domain_kld=5.0, p99_kld=5.0, cka_floor=0.8)
    with pytest.raises(KLGateError, match="CKA layer"):
        run_kl_quality_gate(report=report, budgets=budgets)
