import numpy as np
import pytest

from model_atlas.evaluation.cka import centered_linear_cka
from model_atlas.evaluation.kld import KLDMismatchError, token_kld


def _binary_logits(*, p: float) -> np.ndarray:
    """Two-class logit pair giving class-0 probability ``p``."""
    return np.array([[np.log(p / (1.0 - p)), 0.0]])


def _binary_kl(p: float, q: float) -> float:
    """Analytic KL of two binary Bernoulli distributions."""
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def test_identical_logits_kld_zero() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(8, 5))
    rows, report = token_kld(logits, logits)
    assert len(rows) == 8
    assert all(r.kld <= 1e-12 for r in rows)
    assert report.overall.token_weighted_mean <= 1e-12


def test_analytic_binary_kld() -> None:
    p, q = 0.3, 0.6
    rows, report = token_kld(_binary_logits(p=p), _binary_logits(p=q))
    assert len(rows) == 1
    assert rows[0].kld == pytest.approx(_binary_kl(p, q), abs=1e-9)
    assert report.overall.n_tokens == 1
    assert report.overall.token_weighted_mean == pytest.approx(
        _binary_kl(p, q), abs=1e-9
    )


def test_padding_mask_and_token_weighted_domains() -> None:
    # Three tokens, two domains; middle token masked out.
    rng = np.random.default_rng(1)
    teacher = rng.normal(size=(3, 4))
    candidate = teacher.copy()
    candidate[1:] += 0.5  # perturb tokens 1,2

    pos_mask = np.array([False, True, False])  # token 1 masked

    token_ids = np.array([7, 8, 9])
    domains = ["code", "math"]
    per_position_domains = np.array([0, 1, 1])

    rows, report = token_kld(
        teacher,
        candidate,
        mask=pos_mask,
        token_ids=token_ids,
        domains=domains,
        per_position_domains=per_position_domains,
    )
    assert [r.masked for r in rows] == [False, True, False]
    # Masked token 1 excluded from overall.
    assert report.overall.n_tokens == 2
    assert report.overall.token_weighted_mean == pytest.approx(
        (rows[0].kld + rows[2].kld) / 2.0
    )
    by = {d.domain: d for d in report.by_domain}
    assert set(by) == {"code", "math"}
    assert by["code"].n_tokens == 1
    assert by["math"].n_tokens == 1
    # Unknown domain explicit when no mapping given.
    rows2, _ = token_kld(teacher, candidate)
    assert rows2[0].domain == "unknown"


def test_shape_and_order_errors() -> None:
    a = np.zeros((3, 4))
    b = np.zeros((3, 5))
    with pytest.raises(KLDMismatchError):
        token_kld(a, b)
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, mask=np.zeros((3, 2)))  # mask shape mismatch
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, sample_ids=["only-one"])
    with pytest.raises(KLDMismatchError):
        token_kld(a, a, token_ids=np.zeros((2,)))
    with pytest.raises(ValueError):
        token_kld(a, a, mask=np.full((3, 4), 2))  # not boolean/0-1
    with pytest.raises(KLDMismatchError):
        token_kld(np.zeros(4), np.zeros(4))  # 1D not allowed


# ---- CKA ----

def test_cka_identity_is_one() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 5))
    r = centered_linear_cka(x, x)
    assert r.valid is True
    assert r.score == pytest.approx(1.0, abs=1e-9)


def test_cka_known_linear_transform_relation() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 5))
    # y = X @ Q for an orthogonal Q: linear CKA is invariant to orthogonal
    # feature transforms, so CKA(x, y) == 1. Non-orthogonal invertible maps
    # distort correlation structure and are NOT CKA-invariant.
    a = rng.normal(size=(5, 5))
    q, _ = np.linalg.qr(a)  # q is orthonormal (5x5)
    y = x @ q
    r = centered_linear_cka(x, y)
    assert r.valid is True
    assert r.score == pytest.approx(1.0, abs=1e-6)


def test_cka_orthogonal_inputs_lower() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 4))
    y = rng.normal(size=(30, 4))
    r = centered_linear_cka(x, y)
    assert r.valid is True
    assert -1.0 <= r.score <= 1.0


def test_cka_degenerate_blockers() -> None:
    # Constant column => zero variance => invalid, no numeric score.
    x = np.array([[1.0, 2.0], [1.0, 2.0]])
    y = np.array([[0.0, 1.0], [1.0, 3.0]])
    r = centered_linear_cka(x, y)
    assert r.valid is False
    assert r.score is None

    # Fewer than two observations => invalid.
    r2 = centered_linear_cka(np.zeros((1, 4)), np.zeros((1, 4)))
    assert r2.valid is False
    assert r2.score is None

    # Observation-count mismatch => error.
    with pytest.raises(ValueError):
        centered_linear_cka(np.zeros((3, 4)), np.zeros((4, 4)))
