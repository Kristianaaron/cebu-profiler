import pytest
from pydantic import ValidationError

from model_atlas.evaluation.contracts import (
    CorpusSlice,
    DomainKLDAggregate,
    DomainKLDReport,
    EvaluationIdentity,
    EvaluationReport,
    EvidenceKind,
    MetricEvidence,
    ReproducibilityManifest,
    RouterDivergenceRecord,
    RouterDivergenceSummary,
    SampleAlignment,
    TokenKLDResult,
    TokenKLDRow,
    canonical_evaluation_identity,
    identity_digest,
)

_SHA = "a" * 64  # valid 64-lowercase-hex digest


def _identity() -> EvaluationIdentity:
    return EvaluationIdentity(teacher_id="teacher-a", candidate_id="candidate-b")


def _corpus() -> CorpusSlice:
    return CorpusSlice(
        manifest_hash=_SHA,
        held_out_partition="heldout-1",
        ordered_sample_id_hash=_SHA,
        tokenizer_hash=_SHA,
        n_samples=10,
    )


def _repro() -> ReproducibilityManifest:
    return ReproducibilityManifest(
        source_manifest_hash=_SHA,
        candidate_manifest_hash=_SHA,
        corpus_hash=_SHA,
        tokenizer_hash=_SHA,
        config_hash=_SHA,
        harness_revision="rev",
        adapter_version="a1",
        backend_version="b1",
        seed=1,
        dtype="float32",
        device="cpu",
        topology="single",
        input_hash=_SHA,
        argv=[],
    )


def _measured_evidence() -> MetricEvidence:
    return MetricEvidence(
        kind=EvidenceKind.MEASURED,
        value=0.5,
        artifact_digest=_SHA,
        producer="producer-x",
        producer_version="1.0",
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        identity=_identity(),
        corpus=_corpus(),
        reproducibility=_repro(),
    )


def test_contracts_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluationIdentity(teacher_id="a", candidate_id="b", bogus=1)
    with pytest.raises(ValidationError):
        MetricEvidence(kind=EvidenceKind.MEASURED, value=0.5, bogus=True)


def test_invalid_ranges_rejected() -> None:
    with pytest.raises(ValidationError):
        CorpusSlice(
            manifest_hash="m", held_out_partition="p",
            ordered_sample_id_hash="s", tokenizer_hash="t", n_samples=0,
        )
    with pytest.raises(ValidationError):
        TokenKLDRow(sample_id="x", position=0, token_id=0, kld=-0.1)
    with pytest.raises(ValidationError):
        DomainKLDAggregate(
            domain="d", n_tokens=1, token_weighted_mean=float("nan"),
            p50=0.0, p95=0.0, p99=0.0, max=0.0,
        )
    with pytest.raises(ValidationError):
        RouterDivergenceRecord(
            layer_index=0, expert_count=2, matched_tokens=5,
            route_agreement=1.5, kl_divergence=0.0,
        )


def test_nonfinite_numeric_metrics_rejected() -> None:
    # NaN, +inf, and -inf rejected for every numeric metric field.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            MetricEvidence(kind=EvidenceKind.INFERRED, value=bad)
        with pytest.raises(ValidationError):
            DomainKLDAggregate(
                domain="d", n_tokens=1, token_weighted_mean=0.0,
                p50=0.0, p95=0.0, p99=0.0, max=bad,
            )
        with pytest.raises(ValidationError):
            TokenKLDRow(sample_id="x", position=0, token_id=0, kld=bad)


def test_measured_evidence_requires_artifact_and_producer() -> None:
    with pytest.raises(ValidationError):
        MetricEvidence(
            kind=EvidenceKind.MEASURED, value=0.3,
            artifact_digest="abc", producer="p", producer_version=None,
        )
    with pytest.raises(ValidationError):
        MetricEvidence(kind=EvidenceKind.MEASURED, value=0.3)
    # Empty (whitespace) provenance is also rejected.
    with pytest.raises(ValidationError):
        MetricEvidence(
            kind=EvidenceKind.MEASURED, value=0.3,
            artifact_digest="  ", producer="p", producer_version="1",
        )
    m = MetricEvidence(
        kind=EvidenceKind.MEASURED, value=0.3,
        artifact_digest=_SHA, producer="p", producer_version="1.0",
    )
    assert m.kind is EvidenceKind.MEASURED
    e = MetricEvidence(
        kind=EvidenceKind.ESTIMATED, value=0.3,
        artifact_digest=None, producer=None, producer_version=None,
    )
    assert e.value == 0.3


def test_sha256_digests_validated() -> None:
    # Bad digest syntax rejected: wrong length, uppercase, non-hex.
    for bad in ("abc", "A" * 64, "z" * 64, "1" * 63):
        with pytest.raises(ValidationError):
            CorpusSlice(
                manifest_hash=bad, held_out_partition="p",
                ordered_sample_id_hash=_SHA, tokenizer_hash=_SHA, n_samples=1,
            )


def test_version_999_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationReport(
            report_version=999,
            identity=_identity(),
            corpus=_corpus(),
            reproducibility=_repro(),
        )
    with pytest.raises(ValidationError):
        EvaluationReport(
            schema_version=999,
            identity=_identity(),
            corpus=_corpus(),
            reproducibility=_repro(),
        )
    with pytest.raises(ValidationError):
        ReproducibilityManifest(
            manifest_version=999,
            source_manifest_hash=_SHA, candidate_manifest_hash=_SHA,
            corpus_hash=_SHA, tokenizer_hash=_SHA, config_hash=_SHA,
            harness_revision="r", adapter_version="a", backend_version="b",
            seed=1, dtype="f", device="c", topology="t", input_hash=_SHA,
        )


def test_required_meta_rejected_when_empty() -> None:
    with pytest.raises(ValidationError):
        EvaluationIdentity(teacher_id="", candidate_id="c")
    with pytest.raises(ValidationError):
        ReproducibilityManifest(
            source_manifest_hash=_SHA, candidate_manifest_hash=_SHA,
            corpus_hash=_SHA, tokenizer_hash=_SHA, config_hash=_SHA,
            harness_revision="", adapter_version="a", backend_version="b",
            seed=1, dtype="f", device="c", topology="t", input_hash=_SHA,
        )
    # topology / device required (not optional) and nonempty.
    with pytest.raises(ValidationError):
        ReproducibilityManifest(
            source_manifest_hash=_SHA, candidate_manifest_hash=_SHA,
            corpus_hash=_SHA, tokenizer_hash=_SHA, config_hash=_SHA,
            harness_revision="r", adapter_version="a", backend_version="b",
            seed=1, dtype="f", device="", topology="t", input_hash=_SHA,
        )


def test_token_kld_result_row_identity() -> None:
    row = TokenKLDRow(sample_id="s", position=0, token_id=1, kld=0.1)
    result = TokenKLDResult(
        sample_ids=["s"],
        rows=[row],
        report=DomainKLDReport(
            overall=DomainKLDAggregate(
                domain="overall", n_tokens=1, token_weighted_mean=0.1,
                p50=0.1, p95=0.1, p99=0.1, max=0.1,
            ),
            by_domain=[],
        ),
        evidence=_measured_evidence(),
    )
    assert result.rows[0].sample_id == "s"

    # Duplicate (sample_id, position) identity rejected.
    with pytest.raises(ValidationError):
        TokenKLDResult(
            sample_ids=["s"],
            rows=[row, row.model_copy()],
            report=result.report,
            evidence=_measured_evidence(),
        )
    # Row referencing an unknown sample rejected.
    with pytest.raises(ValidationError):
        TokenKLDResult(
            sample_ids=["s"],
            rows=[TokenKLDRow(sample_id="other", position=0, token_id=1, kld=0.1)],
            report=result.report,
            evidence=_measured_evidence(),
        )


def test_router_summary_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        RouterDivergenceSummary(
            records=[], matched_tokens_total=0, overall_agreement=0.0
        )


def test_sample_alignment_validation() -> None:
    # Ascending unique nonnegative positions required.
    SampleAlignment(sample_id="s", positions=(0, 1), token_ids=(5, 6))
    with pytest.raises(ValueError):
        SampleAlignment(sample_id="s", positions=(1, 1), token_ids=(5, 6))
    with pytest.raises(ValueError):
        SampleAlignment(sample_id="s", positions=(1, 0), token_ids=(5, 6))
    with pytest.raises(ValueError):
        SampleAlignment(sample_id="s", positions=(0, 1), token_ids=(-1, 2))
    with pytest.raises(ValueError):
        SampleAlignment(sample_id="s", positions=(0,), token_ids=(1, 2))


def test_canonical_identity_is_stable_and_excludes_metric_results() -> None:
    r1 = _report()
    r2 = _report()
    d1 = canonical_evaluation_identity(r1)
    d2 = canonical_evaluation_identity(r2)
    assert d1 == d2
    assert len(d1) == 64

    r1.kld = TokenKLDResult(
        sample_ids=["s"],
        rows=[TokenKLDRow(sample_id="s", position=0, token_id=1, kld=0.5)],
        report=DomainKLDReport(
            overall=DomainKLDAggregate(
                domain="overall", n_tokens=1, token_weighted_mean=0.5,
                p50=0.5, p95=0.5, p99=0.5, max=0.5,
            ),
            by_domain=[],
        ),
        evidence=_measured_evidence(),
    )
    assert canonical_evaluation_identity(r1) == d1

    r2.corpus.manifest_hash = _SHA[:-1] + "b"
    assert canonical_evaluation_identity(r2) != d1


def test_timestamps_do_not_affect_canonical_identity() -> None:
    r1 = _report()
    r2 = _report()
    assert identity_digest(r1) == identity_digest(r2)
    assert canonical_evaluation_identity(r1) == canonical_evaluation_identity(r2)
    for model in (ReproducibilityManifest, EvaluationReport):
        flds = set(model.model_fields)
        assert "created_at" not in flds, "created_at leaks into identity"
        assert "timestamp" not in flds, "timestamp leaks into identity"
        assert "run_time" not in flds, "run_time leaks into identity"
        assert "executed_at" not in flds, "executed_at leaks into identity"
