#!/usr/bin/env python3
"""Explicit teacher-relative KLD/CKA quality gate (GLM canary).

Distinct step AFTER identity capture. The identity-capture runner proves the
boundary is deterministic (source vs identical re-capture); this gate alone
adjudicates whether a width-slice CANDIDATE is acceptable by teacher-relative
KL(source || candidate) under explicit budgets.

Fail-closed: the metric report is re-validated from JSON (report_id is a
content digest), budgets are explicit CLI inputs, and any rejection writes a
decision artifact and exits non-zero — never a blanket pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from model_atlas.evaluation.capture_metrics import CaptureMetricReport
from model_atlas.evaluation.quality_gate import (
    QualityGateRejection,
    run_kl_quality_gate,
)
from model_atlas.prune.kl_gate import KLGateBudget

SCHEMA_VERSION = 1
_MAX_REPORT_BYTES = 256 * 1024 * 1024


def _read_bounded(path: Path, limit: int = _MAX_REPORT_BYTES) -> bytes:
    info = os.stat(path)
    if not info.st_size or info.st_size > limit:
        raise RuntimeError("quality-gate metric report is not a bounded regular file")
    with open(path, "rb") as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("quality-gate metric report exceeds its bounded read")
    return data


def _exclusive_json(path: Path, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    with os.fdopen(fd, "wb") as fh:
        fh.write(encoded)
        fh.flush()
        os.fsync(fh.fileno())
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-report", type=Path, required=True)
    parser.add_argument("--mean-budget", type=float, required=True)
    parser.add_argument("--worst-domain-budget", type=float, required=True)
    parser.add_argument("--p99-budget", type=float, required=True)
    parser.add_argument("--cka-floor", type=float, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report-sha256", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    raw = _read_bounded(args.metric_report)
    if args.report_sha256 and hashlib.sha256(raw).hexdigest() != args.report_sha256:
        raise RuntimeError("quality-gate metric report drifted from its claimed sha256")
    report = CaptureMetricReport.model_validate_json(raw.decode("utf-8"))
    budgets = KLGateBudget(
        mean_kld=args.mean_budget,
        worst_domain_kld=args.worst_domain_budget,
        p99_kld=args.p99_budget,
        cka_floor=args.cka_floor,
    )
    try:
        result = run_kl_quality_gate(report=report, budgets=budgets)
        decision = {
            "schema_version": SCHEMA_VERSION,
            "status": "accepted",
            "quality_claim": True,
            "metric_report_id": report.report_id,
            "budgets": {
                "mean_kld": budgets.mean_kld,
                "worst_domain_kld": budgets.worst_domain_kld,
                "p99_kld": budgets.p99_kld,
                "cka_floor": budgets.cka_floor,
            },
            "gate": {
                "mean_kld": result.mean_kld,
                "worst_domain_kld": result.worst_domain_kld,
                "p99_kld": result.p99_kld,
                "min_cka": result.min_cka,
                "accepted": True,
                "failures": [],
            },
        }
        _exclusive_json(args.result, decision)
        return 0
    except QualityGateRejection as rejection:
        result = rejection.result
        _exclusive_json(args.result, {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected",
            "quality_claim": False,
            "metric_report_id": report.report_id,
            "budgets": {
                "mean_kld": budgets.mean_kld,
                "worst_domain_kld": budgets.worst_domain_kld,
                "p99_kld": budgets.p99_kld,
                "cka_floor": budgets.cka_floor,
            },
            "gate": {
                "mean_kld": result.mean_kld,
                "worst_domain_kld": result.worst_domain_kld,
                "p99_kld": result.p99_kld,
                "min_cka": result.min_cka,
                "accepted": False,
                "failures": list(result.failures),
            },
        })
        print(f"quality gate REJECTED: {'; '.join(result.failures)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
