import math
import random
import subprocess
import sys

import pytest

from model_atlas.evaluation.cka import centered_linear_cka
from model_atlas.evaluation.contracts import (
    EvidenceKind,
    MetricEvidence,
    SampleAlignment,
)
from model_atlas.evaluation.kld import (
    KLDMismatchError,
    build_domain_report,
)
from model_atlas.evaluation.kld import token_kld as _token_kld_kernel

_SHA = "b" * 64


def _evidence(value: float = 0.5) -> MetricEvidence:
    return MetricEvidence(
        kind=EvidenceKind.MEASURED,
        value=value,
        artifact_digest=_SHA,
        producer="producer-x",
        producer_version="1.0",
    )


def _logits(batch: int, seq: int, vocab: int, *, seed: int = 0) -> list[list[list[float]]]:
    rng = random.Random(seed)
    return [[[rng.gauss(0.0, 1.0) for _ in range(vocab)] for _ in range(seq)] for _ in range(batch)]


def _binary_logits(*, p: float) -> list[list[list[float]]]:
    """Two-class logit pair giving class-0 probability ``p``."""
    return [[[math.log(p / (1.0 - p)), 0.0]]]


def _binary_kl(p: float, q: float) -> float:
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def token_kld(teacher: object, candidate: object, **kwargs: object) -> tuple[object, object]:
    """Test helper that supplies explicit identity alignment by default."""
    if (
        "teacher_alignment" not in kwargs
        and "candidate_alignment" not in kwargs
        and isinstance(teacher, (list, tuple))
    ):
        alignments = []
        requested_ids = kwargs.get("sample_ids")
        for index, sample in enumerate(teacher):
            sequence = sample if isinstance(sample, (list, tuple)) else []
            positions = tuple(range(len(sequence)))
            sample_id = f"sample#{index}"
            if isinstance(requested_ids, list) and index < len(requested_ids):
                sample_id = str(requested_ids[index])
            alignments.append(SampleAlignment(sample_id, positions, positions))
        kwargs["teacher_alignment"] = alignments
        kwargs["candidate_alignment"] = list(alignments)
    return _token_kld_kernel(teacher, candidate, **kwargs)  # type: ignore[arg-type]


def test_identical_logits_kld_zero() -> None:
    logits = _logits(8, 5, 4, seed=0)
    rows, result = token_kld(logits, logits, evidence=_evidence())
    assert len(rows) == 40
    assert all(r.kld <= 1e-12 for r in rows)
    assert result.report.overall.token_weighted_mean <= 1e-12
    assert len(result.sample_ids) == 8


def test_analytic_binary_kld() -> None:
    p, q = 0.3, 0.6
    rows, result = token_kld(_binary_logits(p=p), _binary_logits(p=q), evidence=_evidence())
    assert len(rows) == 1
    assert rows[0].kld == pytest.approx(_binary_kl(p, q), abs=1e-9)
    assert result.report.overall.n_tokens == 1
    assert result.report.overall.token_weighted_mean == pytest.approx(_binary_kl(p, q), abs=1e-9)


def test_batch_bound_domains_and_token_position_reset() -> None:
    # Two samples, batch-bound domains; positions reset to 0 per sample.
    teacher = _logits(2, 3, 4, seed=1)
    candidate = [[s[:] for s in sample] for sample in teacher]
    candidate[1][1:] = [[v + 0.5 for v in tok] for tok in candidate[1][1:]]

    pos_mask = [[False, True, False], [False, False, False]]
    token_ids_0 = [7, 8, 9]
    token_ids_1 = [10, 11, 12]
    sample_ids = ["s0", "s1"]
    domains = ["code", "math"]

    rows, result = token_kld(
        teacher,
        candidate,
        mask=pos_mask,
        sample_ids=sample_ids,
        domains=domains,
        teacher_alignment=[
            SampleAlignment("s0", (0, 1, 2), tuple(token_ids_0)),
            SampleAlignment("s1", (0, 1, 2), tuple(token_ids_1)),
        ],
        candidate_alignment=[
            SampleAlignment("s0", (0, 1, 2), tuple(token_ids_0)),
            SampleAlignment("s1", (0, 1, 2), tuple(token_ids_1)),
        ],
        evidence=_evidence(),
    )
    assert [r.masked for r in rows] == [False, True, False, False, False, False]
    # Positions reset per sample: s0 positions 0,1,2 then s1 positions 0,1,2.
    assert [(r.sample_id, r.position) for r in rows] == [
        ("s0", 0),
        ("s0", 1),
        ("s0", 2),
        ("s1", 0),
        ("s1", 1),
        ("s1", 2),
    ]
    assert [(r.sample_id, r.token_id) for r in rows[:3]] == [
        ("s0", 7),
        ("s0", 8),
        ("s0", 9),
    ]
    # Masked token (s0,pos1) excluded from overall.
    assert result.report.overall.n_tokens == 5
    assert result.report.overall.token_weighted_mean == pytest.approx(
        sum(r.kld for r in rows if not r.masked) / 5.0
    )
    by = {d.domain: d for d in result.report.by_domain}
    assert set(by) == {"code", "math"}
    assert by["code"].n_tokens == 2  # s0 tokens 0 and 2 (token1 masked)
    assert by["math"].n_tokens == 3

    # Unknown domain explicit when none given.
    rows2, result2 = token_kld(teacher, candidate, evidence=_evidence())
    assert result2.report.by_domain[0].domain == "unknown"
    assert result2.report.by_domain[0].n_tokens == 6
    assert result2.report.overall.domain == "overall"


def test_domain_perturbation_is_nonzero_and_nonuniform() -> None:
    """Perturb selected vocab coordinates; domain KLD must be nonzero and
    must vary across domains (token-weighted aggregation independent)."""
    # Two samples, one token each, four vocab coords. Perturb only s0's
    # logits by shifting a specific coordinate so its distribution moves.
    teacher = _logits(2, 1, 6, seed=7)
    candidate = [[s[:] for s in sample] for sample in teacher]
    # s0: nudge vocab coords 1 and 3 => noticeable KLD.
    candidate[0][0][1] += 2.0
    candidate[0][0][3] -= 1.5
    # s1: leave identical => zero KLD.
    sample_ids = ["s0", "s1"]
    domains = ["code", "math"]

    rows, result = token_kld(
        teacher,
        candidate,
        sample_ids=sample_ids,
        domains=domains,
        evidence=_evidence(),
    )
    by = {d.domain: d for d in result.report.by_domain}
    assert by["code"].n_tokens == 1
    assert by["math"].n_tokens == 1
    assert by["code"].token_weighted_mean > 0.1  # perturbed => nonzero
    assert by["math"].token_weighted_mean == pytest.approx(0.0, abs=1e-12)
    # Nonuniform across domains.
    assert by["code"].token_weighted_mean != by["math"].token_weighted_mean

    # Overall is the token-weighted aggregation across both tokens.
    assert result.report.overall.n_tokens == 2
    expected = (rows[0].kld + rows[1].kld) / 2.0
    assert result.report.overall.token_weighted_mean == pytest.approx(expected)
    # Overall sits strictly between the domain means (weighted average).
    assert (
        min(by["code"].token_weighted_mean, by["math"].token_weighted_mean)
        <= result.report.overall.token_weighted_mean
        <= max(by["code"].token_weighted_mean, by["math"].token_weighted_mean)
    )


def test_shape_and_order_errors() -> None:
    a = _logits(3, 4, 2, seed=2)
    b = _logits(3, 4, 3, seed=3)
    with pytest.raises(KLDMismatchError):
        token_kld(a, b, evidence=_evidence())
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, mask=[[0, 0]], evidence=_evidence())  # mask shape mismatch
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, sample_ids=["only-one"], evidence=_evidence())
    with pytest.raises(ValueError):
        token_kld(a, a, mask=[[2, 0, 0, 0] for _ in range(3)], evidence=_evidence())
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, token_ids=[1, 2], evidence=_evidence())  # unsupported direct
    with pytest.raises(ValueError):
        token_kld(a, a, evidence=_evidence(), temperature=0.0)  # temperature > 0


def test_batch_and_alignment_mismatches_rejected() -> None:
    a = _logits(2, 2, 4, seed=4)
    # Batch mismatch between teacher and candidate.
    with pytest.raises(KLDMismatchError):
        token_kld(a, _logits(3, 2, 4, seed=5), evidence=_evidence())
    # sample_ids length must equal batch size.
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, sample_ids=["only-one"], evidence=_evidence())
    # domains length must equal batch size.
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, domains=["code"], evidence=_evidence())

    # Teacher/candidate alignment order mismatch (candidate positions differ).
    teacher_align = [
        SampleAlignment("s0", (0, 1), (7, 8)),
        SampleAlignment("s1", (0, 1), (9, 10)),
    ]
    cand_align_mismatched = [
        SampleAlignment("s0", (0, 1), (7, 8)),
        SampleAlignment("s1", (0, 1), (99, 100)),  # different token ids
    ]
    with pytest.raises(KLDMismatchError):
        token_kld(
            a,
            a,
            teacher_alignment=teacher_align,
            candidate_alignment=cand_align_mismatched,
            evidence=_evidence(),
        )
    # one-sided alignment rejected.
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, teacher_alignment=teacher_align, evidence=_evidence())

    with pytest.raises(KLDMismatchError, match="are required"):
        _token_kld_kernel(a, a, evidence=_evidence())

    # The same IDs in a different batch order must not be silently realigned.
    reversed_candidate = list(reversed(teacher_align))
    with pytest.raises(KLDMismatchError):
        token_kld(
            a,
            a,
            teacher_alignment=teacher_align,
            candidate_alignment=reversed_candidate,
            evidence=_evidence(),
        )

    # Explicit sample IDs bind every alignment row to its batch position.
    with pytest.raises(KLDMismatchError):
        token_kld(
            a,
            a,
            sample_ids=["s1", "s0"],
            teacher_alignment=teacher_align,
            candidate_alignment=teacher_align,
            evidence=_evidence(),
        )


def test_nonfinite_logits_rejected() -> None:
    a = _logits(1, 1, 3, seed=6)
    bad = [[[float("inf"), 0.0, 0.0]]]
    with pytest.raises(KLDMismatchError):
        token_kld(bad, a, evidence=_evidence())
    with pytest.raises(KLDMismatchError):
        token_kld(a, [[[float("nan"), 0.0, 0.0]]], evidence=_evidence())


def test_empty_sequence_and_vocab_rejected_cleanly() -> None:
    with pytest.raises(KLDMismatchError, match="sequence dimension"):
        token_kld([[]], [[]], evidence=_evidence())
    with pytest.raises(KLDMismatchError, match="vocab dimension"):
        token_kld([[[]]], [[[]]], evidence=_evidence())


def test_evidence_required() -> None:
    a = _logits(1, 1, 3, seed=8)
    with pytest.raises(ValueError):
        token_kld(a, a)


def test_build_domain_report_handles_mask_and_quantiles() -> None:
    from model_atlas.evaluation.contracts import TokenKLDRow

    def _mk(seq: int, kld: float, masked: bool) -> TokenKLDRow:
        return TokenKLDRow(
            sample_id="s",
            position=seq,
            token_id=0,
            kld=kld,
            masked=masked,
            domain="d",
        )

    rows = [_mk(0, 1.0, False), _mk(1, 3.0, False), _mk(2, 2.0, False), _mk(3, 999.0, True)]
    report = build_domain_report(rows)
    assert report.overall.n_tokens == 3
    assert report.overall.token_weighted_mean == pytest.approx(2.0)
    assert report.overall.max == pytest.approx(3.0)
    assert report.overall.p50 == pytest.approx(2.0)


# ---- CKA ----


def _matrix(n: int, f: int, *, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(f)] for _ in range(n)]


def test_cka_identity_is_one() -> None:
    x = _matrix(20, 5, seed=2)
    r = centered_linear_cka(x, x)
    assert r.valid is True
    assert r.score == pytest.approx(1.0, abs=1e-9)


def test_cka_known_linear_transform_relation() -> None:
    # Orthogonal transform via Gram-Schmidt on random 5x5.
    rng = random.Random(3)
    a = [[rng.gauss(0.0, 1.0) for _ in range(5)] for _ in range(5)]
    # column-orthonormalize a -> q
    cols: list[list[float]] = []
    for col in range(5):
        v = [a[r][col] for r in range(5)]
        for u in cols:
            dot = sum(v[i] * u[i] for i in range(5))
            norm2 = sum(x * x for x in u)
            for i in range(5):
                v[i] -= dot / norm2 * u[i]
        norm = math.sqrt(sum(x * x for x in v))
        for i in range(5):
            v[i] /= norm
        cols.append(v)
    q = cols  # 5 orthonormal length-5 vectors
    x = _matrix(20, 5, seed=3)
    y = [[sum(x[r][f] * q[f][c] for f in range(5)) for c in range(5)] for r in range(20)]
    r = centered_linear_cka(x, y)
    assert r.valid is True
    assert r.score == pytest.approx(1.0, abs=1e-6)


def test_cka_orthogonal_inputs_lower() -> None:
    x = _matrix(30, 4, seed=4)
    y = _matrix(30, 4, seed=5)
    r = centered_linear_cka(x, y)
    assert r.valid is True
    assert -1.0 <= r.score <= 1.0


def test_cka_degenerate_blockers() -> None:
    x = [[1.0, 2.0], [1.0, 2.0]]
    y = [[0.0, 1.0], [1.0, 3.0]]
    r = centered_linear_cka(x, y)
    assert r.valid is False
    assert r.score is None

    r2 = centered_linear_cka([[0.0, 0.0]], [[0.0, 0.0]])
    assert r2.valid is False
    assert r2.score is None

    with pytest.raises(ValueError):
        centered_linear_cka([[0.0] * 4] * 3, [[0.0] * 4] * 4)

    with pytest.raises(ValueError, match="feature dimension"):
        centered_linear_cka([[], []], [[], []])


def test_cka_supports_different_feature_widths() -> None:
    x = _matrix(24, 3, seed=10)
    y = _matrix(24, 7, seed=11)
    result = centered_linear_cka(x, y)
    assert result.valid is True
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


def test_cka_wide_hidden_states_use_exact_observation_gram() -> None:
    # Regression for the GLM capture shape: features greatly exceed rows.
    # The former feature-covariance loop scaled as 4096^2 * rows; the exact
    # observation-Gram identity scales as rows^2 * 4096.
    x = _matrix(8, 4096, seed=101)
    result = centered_linear_cka(x, x)
    assert result.valid is True
    assert result.score == pytest.approx(1.0, abs=1e-9)


def test_eager_package_import_brings_no_numpy() -> None:
    # The evaluation package must be importable without pulling NumPy onto the
    # path (dependency-free pure kernels). Checked in a fresh interpreter so
    # the assertion measures what THIS package imports -- other engine tests
    # may legitimately load numpy earlier in the shared pytest process.
    proc = subprocess.run(
        [sys.executable, "-c", "import sys, model_atlas.evaluation; assert 'numpy' not in sys.modules"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
