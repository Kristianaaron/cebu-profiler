"""Pure teacher-relative token KLD kernel with no external evaluator.

Implements a stable ``KL(teacher || candidate)`` over matching teacher-forced
logits with temperature, an explicit boolean/0-1 padding mask, strict
shape/order validation, float64 logsumexp reduction, and no fabricated
epsilon mass. Produces per-token rows plus token-weighted overall and domain
aggregates.
"""

from __future__ import annotations

import math

import numpy as np

from model_atlas.evaluation.contracts import (
    DomainKLDAggregate,
    DomainKLDReport,
    TokenKLDRow,
)

# Tight tolerance for identical logits hitting zero within numerical noise.
_IDENTITY_TOL = 1e-12


class KLDMismatchError(ValueError):
    """Raised on shape, mask, or alignment mismatches."""


def _softmax_log_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Float64, numerically stable log-softmax over the last axis."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / float(temperature)
    logits_float = np.asarray(scaled, dtype=np.float64)
    max_log = np.asarray(np.max(logits_float, axis=-1, keepdims=True), dtype=np.float64)
    logsumexp = np.asarray(
        np.log(
            np.asarray(np.sum(np.exp(logits_float - max_log), axis=-1, keepdims=True))
        ),
        dtype=np.float64,
    )
    return np.asarray(logits_float - max_log - logsumexp, dtype=np.float64)


def _normalise_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Accept a bool array or a 0/1 float/int array; validate shape."""
    arr = np.asarray(mask)
    if arr.dtype == bool:
        pass
    elif np.issubdtype(arr.dtype, np.number):
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("padding mask must be boolean or 0/1 numeric")
        arr = arr.astype(bool)
    else:
        raise ValueError("padding mask must be boolean or 0/1 numeric")
    if arr.shape != shape:
        raise KLDMismatchError(f"mask shape {arr.shape} != logits shape {shape}")
    return arr


def build_domain_report(rows: list[TokenKLDRow]) -> DomainKLDReport:
    """Token-weighted KLD report from validated rows.

    Masked rows are excluded from aggregates but retained in the caller's row
    list. Each unmasked token gets equal weight (purely token-weighted);
    percentiles and max are computed over the same unweighted per-token KLD
    values for observationally consistent quantiles.
    """
    active = [r for r in rows if not r.masked]

    def _empty(domain: str) -> DomainKLDAggregate:
        return DomainKLDAggregate(
            domain=domain,
            n_tokens=0,
            token_weighted_mean=0.0,
            p50=0.0,
            p95=0.0,
            p99=0.0,
            max=0.0,
        )

    if not active:
        return DomainKLDReport(overall=_empty("overall"), by_domain=[])

    def _agg(group: list[TokenKLDRow], domain: str) -> DomainKLDAggregate:
        vals = np.array([r.kld for r in group], dtype=np.float64)
        return DomainKLDAggregate(
            domain=domain,
            n_tokens=len(group),
            token_weighted_mean=float(np.mean(vals)),
            p50=float(np.percentile(vals, 50)),
            p95=float(np.percentile(vals, 95)),
            p99=float(np.percentile(vals, 99)),
            max=float(np.max(vals)),
        )

    domains: list[str] = []
    for r in active:
        if r.domain not in domains:
            domains.append(r.domain)
    by_domain = [_agg([r for r in active if r.domain == d], d) for d in domains]
    return DomainKLDReport(overall=_agg(active, "overall"), by_domain=by_domain)


def token_kld(
    teacher_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    temperature: float = 1.0,
    mask: np.ndarray | None = None,
    sample_ids: list[str] | None = None,
    token_ids: np.ndarray | None = None,
    domains: list[str] | None = None,
    per_position_domains: np.ndarray | None = None,
) -> tuple[list[TokenKLDRow], object]:
    """Compute KL(teacher || candidate) over teacher-forced positions.

    ``teacher_logits`` and ``candidate_logits`` are ``[..., vocab]`` arrays
    with strictly identical shape (the outer leading dimensions are flattened
    into a token sequence in row-major order). ``mask`` must be the same shape
    as the logits; masked positions contribute nothing to aggregates but still
    yield a row marked ``masked=True``.

    Returns ``(rows, report)`` where ``report`` is a ``DomainKLDReport`` with
    token-weighted overall and per-domain aggregates.
    """
    teacher = np.asarray(teacher_logits)
    candidate = np.asarray(candidate_logits)
    if teacher.shape != candidate.shape:
        raise KLDMismatchError(
            f"logits shape mismatch: teacher {teacher.shape} != candidate "
            f"{candidate.shape}"
        )
    if teacher.ndim < 2:
        raise KLDMismatchError("logits must have at least 2 dimensions ([..., vocab])")
    if teacher.shape[-1] < 2:
        raise KLDMismatchError("vocab dimension must be >= 2")

    mask_arr = (
        _normalise_mask(mask, teacher.shape[:-1]) if mask is not None else None
    )

    leading = teacher.shape[:-1]
    n_tokens = int(np.prod(leading)) if leading else 0
    vocab = teacher.shape[-1]

    t_logp = _softmax_log_probs(teacher, temperature).reshape(n_tokens, vocab)
    c_logp = _softmax_log_probs(candidate, temperature).reshape(n_tokens, vocab)

    if sample_ids is not None and len(sample_ids) != n_tokens:
        raise KLDMismatchError(
            f"sample_ids length {len(sample_ids)} != token count {n_tokens}"
        )
    if token_ids is not None:
        tid = np.asarray(token_ids)
        if tid.shape != (n_tokens,):
            raise KLDMismatchError(
                f"token_ids shape {tid.shape} != token count {n_tokens}"
            )
    if per_position_domains is not None and np.asarray(per_position_domains).shape != (
        n_tokens,
    ):
        raise KLDMismatchError("per_position_domains must have one entry per token")
    if domains and per_position_domains is None:
        raise KLDMismatchError("domains provided without per_position_domains mapping")

    if sample_ids is None:
        sample_ids = [f"tok#{i}" for i in range(n_tokens)]
    if token_ids is None:
        token_ids = np.arange(n_tokens, dtype=np.int64)

    tid_arr = np.asarray(token_ids)

    flat_mask = (
        np.zeros(n_tokens, dtype=bool)
        if mask_arr is None
        else mask_arr.reshape(n_tokens)
    )

    rows: list[TokenKLDRow] = []
    for i in range(n_tokens):
        t = t_logp[i]
        c = c_logp[i]
        # KL(teacher || candidate) = sum_v p(v) * (log p(v) - log q(v)).
        # p==0 terms contribute 0; we never fabricate epsilon mass.
        kld = float(np.sum(np.exp(t) * (t - c)))
        if not math.isfinite(kld):
            raise FloatingPointError("KLD produced non-finite value")
        if kld < 0.0 and kld > -_IDENTITY_TOL:
            kld = 0.0

        domain = "unknown"
        if per_position_domains is not None:
            didx = int(np.asarray(per_position_domains)[i])
            if domains is not None and 0 <= didx < len(domains):
                domain = domains[didx]
            else:
                domain = "unknown"

        rows.append(
            TokenKLDRow(
                sample_id=sample_ids[i],
                position=i,
                token_id=int(tid_arr[i]),
                kld=kld,
                masked=bool(flat_mask[i]),
                domain=domain,
            )
        )

    return rows, build_domain_report(rows)


__all__ = ["token_kld", "KLDMismatchError"]
