"""Tests for the evidence coverage gate and limitations ledger."""

from __future__ import annotations

from pathlib import Path

from cebu_profiler.evaluation.coverage import (
    Limitation,
    LimitationKind,
    check_coverage,
    coverage_payload,
    default_limitations,
    limitations_payload,
)
from cebu_profiler.schemas.evidence import EvidenceLevel


def test_missing_run_dir_fails_closed():
    report = check_coverage("/nonexistent/run", EvidenceLevel.BASIC_SALIENCY)
    assert report.passed is False
    assert report.failures
    assert all(f.startswith("missing:") for f in report.failures)


def test_empty_artifact_is_failure(tmp_path: Path):
    # discover one expected artifact name and create it empty
    from cebu_profiler.profiler.output_layout import expected_run_files

    expected = sorted(expected_run_files(EvidenceLevel.BASIC_SALIENCY))
    assert expected
    (tmp_path / expected[0]).write_text("")
    report = check_coverage(tmp_path, EvidenceLevel.BASIC_SALIENCY)
    assert report.passed is False
    assert any(f == f"empty: {expected[0]}" for f in report.failures)


def test_full_run_dir_passes(tmp_path: Path):
    from cebu_profiler.profiler.output_layout import expected_run_files

    for name in expected_run_files(EvidenceLevel.BASIC_SALIENCY):
        (tmp_path / name).write_text("x")
    report = check_coverage(tmp_path, EvidenceLevel.BASIC_SALIENCY)
    assert report.passed is True
    assert report.failures == []
    assert report.metric_count == len(report.entries)


def test_causal_level_demands_more(tmp_path: Path):
    from cebu_profiler.profiler.output_layout import expected_run_files

    basic = expected_run_files(EvidenceLevel.BASIC_SALIENCY)
    causal = expected_run_files(EvidenceLevel.CAUSAL_PROFILER)
    assert len(causal) > len(basic)
    for name in basic:
        (tmp_path / name).write_text("x")
    report = check_coverage(tmp_path, EvidenceLevel.CAUSAL_PROFILER)
    assert report.passed is False  # causal artifacts missing


def test_payload_shape_matches_bundle_convention():
    report = check_coverage("/nonexistent", EvidenceLevel.BASIC_SALIENCY)
    payload = coverage_payload(report)
    assert set(payload) == {"evidence_level", "passed", "metric_count", "failures"}
    assert payload["passed"] is False


def test_limitations_payload_roundtrip():
    lims = default_limitations(EvidenceLevel.CAUSAL_PROFILER)
    assert lims
    flat = limitations_payload(lims)
    assert all("[" in k and "]" in k for k in flat)
    lim = Limitation(
        kind=LimitationKind.EMULATED_BACKEND, subject="fc2", note="emulated, not native"
    )
    assert limitations_payload([lim]) == {"fc2 [emulated_backend]": "emulated, not native"}
