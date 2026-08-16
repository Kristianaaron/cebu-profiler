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
    TokenKLDRow,
    canonical_evaluation_identity,
    identity_digest,
)


def _identity() -> EvaluationIdentity:
    return EvaluationIdentity(
        teacher_id="teacher-a", candidate_id="candidate-b"
    )


def _corpus() -> CorpusSlice:
    return CorpusSlice(
        manifest_hash="m", held_out_partition="heldout-1",
        ordered_sample_id_hash="s", tokenizer_hash="t",
        n_samples=10,
    )


def _repro() -> ReproducibilityManifest:
    return ReproducibilityManifest(
        source_manifest_hash="s", candidate_manifest_hash="c",
        corpus_hash="cor", tokenizer_hash="tok", config_hash="cfg",
        harness_revision="rev", adapter_version="a1", backend_version="b1",
        seed=1, dtype="float32", device="cpu", topology="single",
        input_hash="in", argv=[],
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        identity=_identity(),
        corpus=_corpus(),
        reproducibility=_repro(),
    )


def test_contracts_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluationIdentity(
            teacher_id="a", candidate_id="b", bogus=1
        )
    with pytest.raises(ValidationError):
        MetricEvidence(kind=EvidenceKind.MEASURED, value=0.5, bogus=True)


def test_invalid_ranges_rejected() -> None:
    # n_samples must be > 0 (nonempty held-out corpus).
    with pytest.raises(ValidationError):
        CorpusSlice(
            manifest_hash="m", held_out_partition="p",
            ordered_sample_id_hash="s", tokenizer_hash="t", n_samples=0,
        )
    # Negative / non-finite KLD rejected.
    with pytest.raises(ValidationError):
        TokenKLDRow(
            sample_id="x", position=0, token_id=0, kld=-0.1,
        )
    with pytest.raises(ValidationError):
        DomainKLDAggregate(
            domain="d", n_tokens=1, token_weighted_mean=float("nan"),
            p50=0.0, p95=0.0, p99=0.0, max=0.0,
        )
    # Route agreement must be in [0, 1].
    with pytest.raises(ValidationError):
        RouterDivergenceRecord(
            layer_index=0, expert_count=2, matched_tokens=5,
            route_agreement=1.5, kl_divergence=0.0,
        )


def test_measured_evidence_requires_artifact_and_producer() -> None:
    # MEASURED without digest / producer / version rejected.
    with pytest.raises(ValidationError):
        MetricEvidence(
            kind=EvidenceKind.MEASURED, value=0.3,
            artifact_digest="abc", producer="p", producer_version=None,
        )
    with pytest.raises(ValidationError):
        MetricEvidence(kind=EvidenceKind.MEASURED, value=0.3)
    # A complete MEASURED evidence validates.
    m = MetricEvidence(
        kind=EvidenceKind.MEASURED, value=0.3,
        artifact_digest="abc", producer="p", producer_version="1.0",
    )
    assert m.kind is EvidenceKind.MEASURED
    # Non-measured kinds explicitly carry no artifact requirement.
    e = MetricEvidence(
        kind=EvidenceKind.ESTIMATED, value=0.3,
        artifact_digest=None, producer=None, producer_version=None,
    )
    assert e.value == 0.3


def test_canonical_identity_is_stable_and_excludes_metric_results() -> None:
    r1 = _report()
    r2 = _report()
    # Same content => identical digest regardless of instance/method.
    d1 = canonical_evaluation_identity(r1)
    d2 = canonical_evaluation_identity(r2)
    assert d1 == d2
    assert len(d1) == 64  # sha256 hex

    # Adding metric results must NOT change canonical identity.
    r1.kld = DomainKLDReport(
        overall=DomainKLDAggregate(
            domain="overall", n_tokens=1, token_weighted_mean=0.5,
            p50=0.5, p95=0.5, p99=0.5, max=0.5,
        ),
        by_domain=[],
    )
    assert canonical_evaluation_identity(r1) == d1

    # Changing a pinned hash changes identity.
    r2.corpus.manifest_hash = "different"
    assert canonical_evaluation_identity(r2) != d1


def test_timestamps_do_not_affect_canonical_identity() -> None:
    # No identity-bearing contract carries a wall-clock field, so the
    # canonical digest is deterministic and construction-independent.
    r1 = _report()
    r2 = _report()
    assert identity_digest(r1) == identity_digest(r2)
    assert canonical_evaluation_identity(r1) == canonical_evaluation_identity(r2)
    # The reproducibility manifest pins content, never a timestamp.
    for model in (ReproducibilityManifest, EvaluationReport):
        flds = set(model.model_fields)
        assert "created_at" not in flds, "created_at leaks into identity"
        assert "timestamp" not in flds, "timestamp leaks into identity"
        assert "run_time" not in flds, "run_time leaks into identity"
        assert "executed_at" not in flds, "executed_at leaks into identity"
