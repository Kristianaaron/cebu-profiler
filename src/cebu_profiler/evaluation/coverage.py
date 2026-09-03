"""Evidence coverage gate + limitations ledger for profiler runs.

Two fail-closed additions to the run contract, in the spirit of published
evidence-bundle practice (the alesha-pro/atlas GLM bundle ships a machine-checkable
`coverage.passed`/`coverage.failures` block and an explicit `limitations` map —
both are good discipline this repo adopts and hardens):

- coverage: every metric family a run at its evidence level promised vs what is
  actually on disk (zero-byte / missing = failure), machine-checkable.
- limitations: typed notes that keep measured/estimated/causal claims honest
  (invariant 12), so no consumer ever has to guess what a number is.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.profiler.output_layout import expected_run_files
from cebu_profiler.schemas.evidence import EvidenceLevel


class LimitationKind(StrEnum):
    EMULATED_BACKEND = "emulated_backend"  # measured via emulation, not native kernel
    ROUTE_ABLATION_ONLY = "route_ablation_only"  # reversible ablation, not physical prune
    ESTIMATED_NOT_MEASURED = "estimated_not_measured"
    SAMPLED_NOT_EXHAUSTIVE = "sampled_not_exhaustive"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    HARDWARE_DEPENDENT = "hardware_dependent"
    CALIBRATION_DEPENDENT = "calibration_dependent"


class Limitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: LimitationKind
    subject: str  # what the limitation applies to (metric family / stage id)
    note: str


class CoverageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    present: bool
    nonempty: bool
    bytes: int = 0


class CoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_level: EvidenceLevel
    passed: bool
    metric_count: int  # artifacts checked
    failures: list[str] = Field(default_factory=list)
    entries: list[CoverageEntry] = Field(default_factory=list)


def check_coverage(
    run_dir: str | Path,
    evidence_level: EvidenceLevel,
    *,
    metric_count: int = 0,
) -> CoverageReport:
    """Verify every artifact the evidence level promises exists and is non-empty.

    Fail-closed: a promised-but-empty artifact is a failure, never silently
    tolerated (extends the no-unclassified tensor invariant to evidence files).
    """
    root = Path(run_dir)
    expected = expected_run_files(evidence_level)
    entries: list[CoverageEntry] = []
    failures: list[str] = []
    for name in sorted(expected):
        p = root / name
        size = p.stat().st_size if p.is_file() else 0
        present = p.exists()
        nonempty = size > 0
        entries.append(CoverageEntry(artifact=name, present=present, nonempty=nonempty, bytes=size))
        if not present:
            failures.append(f"missing: {name}")
        elif not nonempty:
            failures.append(f"empty: {name}")
    return CoverageReport(
        evidence_level=evidence_level,
        passed=not failures,
        metric_count=metric_count or len(entries),
        failures=failures,
        entries=entries,
    )


def coverage_payload(report: CoverageReport) -> dict[str, Any]:
    """JSON-safe coverage block for run manifests (atlas-bundle compatible shape)."""
    return {
        "evidence_level": report.evidence_level.value,
        "passed": report.passed,
        "metric_count": report.metric_count,
        "failures": report.failures,
    }


def default_limitations(evidence_level: EvidenceLevel) -> list[Limitation]:
    """Limitations every run of this codebase carries (honest by default)."""
    lims = [
        Limitation(
            kind=LimitationKind.CALIBRATION_DEPENDENT,
            subject="all saliency/contrast metrics",
            note=(
                "scores depend on the calibration corpus composition; per-domain "
                "views are reported where available"
            ),
        ),
        Limitation(
            kind=LimitationKind.SAMPLED_NOT_EXHAUSTIVE,
            subject="counterfactual route scans",
            note="alternative routes are sampled, not enumerated",
        ),
    ]
    if evidence_level is EvidenceLevel.BASIC_SALIENCY:
        lims.append(
            Limitation(
                kind=LimitationKind.ESTIMATED_NOT_MEASURED,
                subject="compression projections",
                note="byte/quality projections at this level are estimates, not measured runs",
            )
        )
    return lims


def limitations_payload(lims: list[Limitation]) -> dict[str, str]:
    """Flat kind->note map per subject, the shape the GLM bundle popularized."""
    return {f"{lim.subject} [{lim.kind.value}]": lim.note for lim in lims}
