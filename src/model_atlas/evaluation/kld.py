"""Pure teacher-relative token KLD kernel with no external evaluator.

Implements a stable ``KL(teacher || candidate)`` over matching teacher-forced
logits with temperature, an explicit boolean/0-1 padding mask, strict
shape/order validation, float64 logsumexp reduction, and no fabricated
epsilon mass. Logits are modelled explicitly as ``[batch][sequence][vocab]``.
``sample_ids`` and ``domains`` are batch-bound; token positions reset per
sample. Teacher and candidate each carry an ordered alignment identity and
any mismatch is rejected before math. Produces a strict ``TokenKLDResult``
with rows (unique ``(sample_id, position)``) plus token-weighted overall and
domain aggregates, all bound to immutable evidence.
"""

from __future__ import annotations

import math

from model_atlas.evaluation.contracts import (
    DomainKLDAggregate,
    DomainKLDReport,
    MetricEvidence,
    SampleAlignment,
    TokenKLDResult,
    TokenKLDRow,
    _alignment_key,
)

# Tight tolerance for identical logits hitting zero within numerical noise.
_IDENTITY_TOL = 1e-12


class KLDMismatchError(ValueError):
    """Raised on shape, mask, or alignment mismatches."""


def _log_softmax(logits_row: list[float], temperature: float) -> list[float]:
    """Numerically stable log-softmax over one vocab row using math/log."""
    max_log = max(logits_row)
    scaled = [(v - max_log) / temperature for v in logits_row]
    logsumexp = math.log(sum(math.exp(v) for v in scaled))
    return [v - logsumexp for v in scaled]


def _validate_logits(name: str, logits: object) -> list[list[list[float]]]:
    """Validate nested logits and return a rectangular [batch][seq][vocab] list."""
    if not isinstance(logits, (list, tuple)):
        raise KLDMismatchError(f"{name} logits must be a nested sequence")
    batch = list(logits)
    if not batch:
        raise KLDMismatchError(f"{name} logits must be a nonempty batch")
    seq_len: int | None = None
    vocab: int | None = None
    out: list[list[list[float]]] = []
    for bi, sample in enumerate(batch):
        if not isinstance(sample, (list, tuple)):
            raise KLDMismatchError(
                f"{name} logits sample {bi} must be a sequence"
            )
        seq = list(sample)
        if seq_len is None:
            seq_len = len(seq)
        elif len(seq) != seq_len:
            raise KLDMismatchError(
                f"{name} logits rectangularity violated at batch {bi}: "
                f"expected seq len {seq_len}, got {len(seq)}"
            )
        row: list[list[float]] = []
        for si, tok in enumerate(seq):
            if not isinstance(tok, (list, tuple)):
                raise KLDMismatchError(
                    f"{name} logits token at batch {bi}, seq {si} must be a sequence"
                )
            vec = list(tok)
            if vocab is None:
                vocab = len(vec)
            elif len(vec) != vocab:
                raise KLDMismatchError(
                    f"{name} logits vocab mismatch at batch {bi}, seq {si}: "
                    f"expected {vocab}, got {len(vec)}"
                )
            if vocab < 2:
                raise KLDMismatchError(f"{name} logits vocab dimension must be >= 2")
            flat: list[float] = []
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise KLDMismatchError(
                        f"{name} logits must be numeric at batch {bi}, seq {si}"
                    )
                fv = float(v)
                if not math.isfinite(fv):
                    raise KLDMismatchError(
                        f"{name} logits must be finite at batch {bi}, seq {si}"
                    )
                flat.append(fv)
            row.append(flat)
        out.append(row)
    return out


def _validate_bool_mask(
    mask: object, batch: int, seq_len: int
) -> list[list[bool]]:
    """Accept a bool or 0/1 numeric nested mask matching [batch][seq]."""
    if not isinstance(mask, (list, tuple)):
        raise KLDMismatchError("mask must be a nested sequence")
    mbatch = list(mask)
    if len(mbatch) != batch:
        raise KLDMismatchError(
            f"mask batch length {len(mbatch)} != logits batch {batch}"
        )
    out: list[list[bool]] = []
    for bi, sample in enumerate(mbatch):
        if not isinstance(sample, (list, tuple)):
            raise KLDMismatchError(f"mask sample {bi} must be a sequence")
        seq = list(sample)
        if len(seq) != seq_len:
            raise KLDMismatchError(
                f"mask shape at batch {bi}: {len(seq)} != seq_len {seq_len}"
            )
        row: list[bool] = []
        for v in seq:
            if isinstance(v, bool):
                row.append(v)
            elif isinstance(v, (int, float)) and v in (0, 1):
                row.append(bool(v))
            else:
                raise KLDMismatchError("mask must be boolean or 0/1 numeric")
        out.append(row)
    return out


def _align_sample(
    *,
    batch_index: int,
    seq_len: int,
    alignment: SampleAlignment | None,
    sample_ids: list[str] | None,
    domains: list[str] | None,
) -> tuple[str, list[tuple[int, int]], str]:
    """Resolve a sample's identity, per-position (position, token_id), domain.

    Positions reset per sample (0..seq_len-1). token_ids default to the
    position values when no alignment was supplied. Domain is batch-bound.
    Returns ``(sample_id, positions, domain)``.
    """
    if alignment is not None:
        positions = list(
            zip(alignment.positions, alignment.token_ids, strict=True)
        )
        if len(positions) != seq_len:
            raise KLDMismatchError(
                f"alignment {alignment.sample_id!r} has {len(positions)} "
                f"positions but sequence length is {seq_len}"
            )
        if sample_ids is not None and alignment.sample_id not in sample_ids:
            raise KLDMismatchError(
                f"teacher/candidate alignment sample {alignment.sample_id!r} "
                "is not present in ordered sample_ids"
            )
        sid = alignment.sample_id
    else:
        if sample_ids is None:
            sid = f"sample#{batch_index}"
        else:
            if batch_index >= len(sample_ids):
                raise KLDMismatchError(
                    f"batch index {batch_index} exceeds sample_ids length "
                    f"{len(sample_ids)}"
                )
            sid = sample_ids[batch_index]
        positions = list(zip(range(seq_len), range(seq_len), strict=True))

    domain = "unknown"
    if domains is not None:
        if batch_index >= len(domains):
            raise KLDMismatchError(
                f"batch index {batch_index} exceeds domains length {len(domains)}"
            )
        domain = domains[batch_index]
    return sid, positions, domain


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

    def _quantile(sorted_vals: list[float], q: float) -> float:
        """Linear-interpolation percentile over a sorted ascending list."""
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        pos = q * (len(sorted_vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_vals[lo]
        frac = pos - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    def _agg(group: list[TokenKLDRow], domain: str) -> DomainKLDAggregate:
        vals = [r.kld for r in group]
        mean = sum(vals) / len(vals)
        sorted_vals = sorted(vals)
        return DomainKLDAggregate(
            domain=domain,
            n_tokens=len(group),
            token_weighted_mean=mean,
            p50=_quantile(sorted_vals, 0.50),
            p95=_quantile(sorted_vals, 0.95),
            p99=_quantile(sorted_vals, 0.99),
            max=sorted_vals[-1],
        )

    domains: list[str] = []
    for r in active:
        if r.domain not in domains:
            domains.append(r.domain)
    by_domain = [_agg([r for r in active if r.domain == d], d) for d in domains]
    return DomainKLDReport(overall=_agg(active, "overall"), by_domain=by_domain)


def token_kld(
    teacher_logits: object,
    candidate_logits: object,
    *,
    temperature: float = 1.0,
    mask: object | None = None,
    sample_ids: list[str] | None = None,
    token_ids: object | None = None,
    domains: list[str] | None = None,
    per_position_domains: object | None = None,
    teacher_alignment: list[SampleAlignment] | None = None,
    candidate_alignment: list[SampleAlignment] | None = None,
    evidence: MetricEvidence | None = None,
) -> tuple[list[TokenKLDRow], TokenKLDResult]:
    """Compute KL(teacher || candidate) over teacher-forced positions.

    ``teacher_logits`` and ``candidate_logits`` are ``[batch][seq][vocab]``
    nested sequences with strictly identical, rectangular shapes. ``mask`` is
    ``[batch][seq]`` boolean/0-1. ``sample_ids`` and ``domains`` are
    batch-bound; token positions reset per sample. Independent teacher and
    candidate alignment identities (ordered sample IDs plus per-sample valid
    token positions/token IDs) may be passed via ``teacher_alignment`` /
    ``candidate_alignment`` and any mismatch is rejected before math.

    Returns ``(rows, result)`` where ``result`` is a strict ``TokenKLDResult``
    carrying unique ``(sample_id, position)`` rows, token-weighted overall and
    per-domain aggregates, and immutable evidence.
    """
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError("temperature must be finite and > 0")

    teacher = _validate_logits("teacher", teacher_logits)
    candidate = _validate_logits("candidate", candidate_logits)

    batch = len(teacher)
    if len(candidate) != batch:
        raise KLDMismatchError(
            f"batch mismatch: teacher has {batch}, candidate has {len(candidate)}"
        )
    seq_len = len(teacher[0])
    vocab = len(teacher[0][0])
    for bi in range(batch):
        if len(candidate[bi]) != seq_len:
            raise KLDMismatchError(
                f"sequence length mismatch at batch {bi}: teacher {seq_len}, "
                f"candidate {len(candidate[bi])}"
            )
        if len(candidate[bi][0]) != vocab:
            raise KLDMismatchError(
                f"vocab mismatch at batch {bi}: teacher {vocab}, "
                f"candidate {len(candidate[bi][0])}"
            )

    if sample_ids is not None and len(sample_ids) != batch:
        raise KLDMismatchError(
            f"sample_ids length {len(sample_ids)} != batch size {batch}"
        )
    if domains is not None and len(domains) != batch:
        raise KLDMismatchError(
            f"domains length {len(domains)} != batch size {batch}"
        )
    if per_position_domains is not None:
        raise KLDMismatchError(
            "per_position_domains is not supported; use batch-bound 'domains'"
        )
    if token_ids is not None:
        # token_ids are expressed via per-sample alignment / positions only.
        raise KLDMismatchError(
            "token_ids must be supplied via teacher_alignment/candidate_alignment"
        )

    if teacher_alignment is None and candidate_alignment is not None:
        raise KLDMismatchError("candidate_alignment provided without teacher_alignment")
    if candidate_alignment is None and teacher_alignment is not None:
        raise KLDMismatchError("teacher_alignment provided without candidate_alignment")
    if teacher_alignment is not None and candidate_alignment is not None:
        if len(teacher_alignment) != batch:
            raise KLDMismatchError(
                f"teacher_alignment length {len(teacher_alignment)} != batch {batch}"
            )
        if len(candidate_alignment) != batch:
            raise KLDMismatchError(
                f"candidate_alignment length {len(candidate_alignment)} != batch {batch}"
            )
        t_keys = [_alignment_key(a) for a in teacher_alignment]
        c_keys = [_alignment_key(a) for a in candidate_alignment]
        if sorted(t_keys) != sorted(c_keys):
            raise KLDMismatchError(
                "teacher/candidate alignment mismatch: sample ids, positions, "
                "or token ids differ between teacher and candidate"
            )

    if mask is not None:
        mask_arr = _validate_bool_mask(mask, batch, seq_len)
    else:
        mask_arr = [[False] * seq_len for _ in range(batch)]

    rows: list[TokenKLDRow] = []
    for bi in range(batch):
        sid, positions, domain = _align_sample(
            batch_index=bi,
            seq_len=seq_len,
            alignment=teacher_alignment[bi] if teacher_alignment else None,
            sample_ids=sample_ids,
            domains=domains,
        )
        for si in range(seq_len):
            t_logp = _log_softmax(teacher[bi][si], temperature)
            c_logp = _log_softmax(candidate[bi][si], temperature)
            kld = sum(
                math.exp(t) * (t - c) for t, c in zip(t_logp, c_logp, strict=True)
            )
            if not math.isfinite(kld):
                raise FloatingPointError("KLD produced non-finite value")
            if kld < 0.0 and kld > -_IDENTITY_TOL:
                kld = 0.0
            position, token_id = positions[si]
            rows.append(
                TokenKLDRow(
                    sample_id=sid,
                    position=position,
                    token_id=token_id,
                    kld=kld,
                    masked=mask_arr[bi][si],
                    domain=domain,
                )
            )

    if evidence is None:
        raise ValueError(
            "token_kld requires evidence binding an artifact digest, producer, "
            "and producer version"
        )

    sample_ids_resolved = _ordered_sample_ids(rows)
    report = build_domain_report(rows)
    result = TokenKLDResult(
        sample_ids=sample_ids_resolved,
        rows=rows,
        report=report,
        evidence=evidence,
    )
    return rows, result


def _ordered_sample_ids(rows: list[TokenKLDRow]) -> list[str]:
    """Preserve first-seen batch order of sample ids."""
    seen: list[str] = []
    for r in rows:
        if r.sample_id not in seen:
            seen.append(r.sample_id)
    return seen


__all__ = ["token_kld", "KLDMismatchError", "build_domain_report"]
