"""Saliency-ranked keep-map derivation for the GLM-5.2 width-slice runner.

Bridges the honest ``channel_saliency`` evidence in a source profile to the
per-``(layer, expert)`` keep-map consumed by
``model_atlas.loader.materialize_uniform_width(..., keep_channels=...)``.

Two deterministic, unit-testable pieces:

1. :func:`parse_saliency_basis` — parses the profile's ``channel_saliency``
   ``StageEvidence.detail`` (the ``key=value;key=value`` lineage format
   emitted by ``glm52_source_profile.build_glm52_mixed_gguf_profile``) and
   derives a deterministic per-group ranking basis. Until real activation
   profiling runs (``activation_profiling=not_yet_run``), the only truthful
   basis is the header-derived structural census: group index ascending
   (the exporter's keep-first-N default), which the runner records honestly
   as ``basis=structural_group_order``. A future measured artifact can
   replace the detail with measured per-group scores without any runner
   change: any ``detail`` carrying ``saliency=<json>`` (a mapping of
   ``"(layer,expert)"`` to a finite per-group score list) upgrades the basis
   to measured ranking data, fail-closed on malformed input.
2. :func:`build_keep_map` — turns the basis into the complete keep-map via
   :func:`model_atlas.prune.ranked_keeper.select_keep_map` (top-saliency
   aligned 16-channel groups, deterministic tie-break, complete coverage).

Fail-closed everywhere: malformed lineage, non-finite scores, wrong widths,
or incomplete ``(layer, expert)`` coverage raise instead of silently falling
back to a fabricated ranking.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from model_atlas.prune.channel_saliency import GROUP_VALUES
from model_atlas.prune.ranked_keeper import KeepMapError, select_keep_map

_BASIS_STRUCTURAL = "structural_group_order"


class KeepMapBasisError(ValueError):
    """The profile's channel-saliency basis cannot yield a truthful ranking."""


@dataclass(frozen=True)
class SaliencyBasis:
    """Parsed channel-saliency basis from a profile evidence detail.

    ``measured`` is False until a real activation-profiling artifact supplies
    per-group scores; the structural census alone never claims measurement.
    """

    basis_artifact_sha256: str
    activation_profiling_run: bool
    # "(layer,expert)" -> per-group saliency scores (measured artifacts only)
    group_scores: dict[str, list[float]] | None

    @property
    def ranking_basis(self) -> str:
        return "measured_group_saliency" if self.group_scores else _BASIS_STRUCTURAL


def _parse_detail_fields(detail: str) -> dict[str, str]:
    """Split the ``k=v;k=v`` lineage detail into a strict string map.

    Empty keys/values and whitespace are rejected — the emitter never
    produces them, so accepting them here would hide drift.
    """
    fields: dict[str, str] = {}
    for part in detail.split(";"):
        key, sep, value = part.partition("=")
        if not sep or not key or not value or key != key.strip() or value != value.strip():
            raise KeepMapBasisError(f"malformed saliency detail component {part!r}")
        fields[key] = value
    return fields


def parse_saliency_basis(detail: str) -> SaliencyBasis:
    """Parse a profile ``channel_saliency`` detail into a typed basis.

    Required fields (mirroring the emitter): ``basis``,
    ``basis_artifact_sha256``, ``activation_profiling``. An optional
    ``saliency=<json>`` carries measured per-group scores
    (``{"(layer,expert)": [float, ...], ...}``); any present value must be
    finite. ``activation_profiling=not_yet_run`` with a ``saliency`` payload
    is contradictory and rejected — measured scores imply profiling ran.
    """
    fields = _parse_detail_fields(detail)
    for required in ("basis", "basis_artifact_sha256", "activation_profiling"):
        if required not in fields:
            raise KeepMapBasisError(f"saliency detail missing {required!r}")
    sha = fields["basis_artifact_sha256"]
    if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
        raise KeepMapBasisError("basis_artifact_sha256 is not a lowercase sha256")
    ran = fields["activation_profiling"]
    if ran not in ("not_yet_run", "completed"):
        raise KeepMapBasisError(f"unknown activation_profiling state {ran!r}")
    raw_scores = fields.get("saliency")
    group_scores: dict[str, list[float]] | None = None
    if raw_scores is not None:
        if ran != "completed":
            raise KeepMapBasisError(
                "saliency scores present but activation_profiling is not completed"
            )
        try:
            decoded: Any = json.loads(raw_scores)
        except ValueError as exc:
            raise KeepMapBasisError(f"saliency payload is not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict) or not decoded:
            raise KeepMapBasisError("saliency payload must be a nonempty object")
        parsed: dict[str, list[float]] = {}
        for key, values in decoded.items():
            if not isinstance(key, str) or not key:
                raise KeepMapBasisError("saliency keys must be nonempty strings")
            if not isinstance(values, list) or not values:
                raise KeepMapBasisError(f"saliency scores for {key!r} must be a nonempty list")
            parsed_values: list[float] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise KeepMapBasisError(f"saliency score for {key!r} is not numeric")
                score = float(value)
                if score != score or score in (float("inf"), float("-inf")):
                    raise KeepMapBasisError(f"saliency score for {key!r} is not finite")
                parsed_values.append(score)
            parsed[key] = parsed_values
        group_scores = parsed
    return SaliencyBasis(
        basis_artifact_sha256=sha,
        activation_profiling_run=ran == "completed",
        group_scores=group_scores,
    )


def _expert_key(layer: int, expert: int) -> str:
    return f"({layer},{expert})"


def _layer_expert_of(key: str) -> tuple[int, int]:
    try:
        layer_raw, expert_raw = key.strip().strip("()").split(",")
        return int(layer_raw.strip()), int(expert_raw.strip())
    except ValueError as exc:
        raise KeepMapBasisError(f"malformed saliency key {key!r}") from exc


def build_keep_map(
    basis: SaliencyBasis,
    *,
    width: int,
    full: int,
    sparse_layers: Sequence[int],
    n_exp: int,
    group: int = GROUP_VALUES,
) -> dict[tuple[int, int], list[int]]:
    """Return the complete ``{(layer, expert): [ascending kept channels]}`` map.

    With a measured basis the per-expert score vectors rank groups directly
    (length must equal ``full // group`` for every covered expert, and
    coverage must be COMPLETE over all sparse-layer/expert targets). With the
    structural census basis the ranking is ascending group index — exactly
    the exporter's keep-first-N default, recorded honestly as such.
    """
    try:
        if basis.group_scores is None:
            if not sparse_layers or n_exp <= 0:
                raise KeepMapError("geometry has no (layer, expert) targets")
            # structural census: ascending group index is the whole ranking
            uniform = [float(g) for g in range(full // group)]
            return select_keep_map(
                {(li, e): uniform for li in sparse_layers for e in range(n_exp)},
                width=width,
                full=full,
                sparse_layers=sparse_layers,
                n_exp=n_exp,
                group=group,
            )
        expected = {_expert_key(li, e) for li in sparse_layers for e in range(n_exp)}
        provided = set(basis.group_scores)
        missing = sorted(expected - provided)
        if missing:
            raise KeepMapError(
                f"saliency basis does not cover {len(missing)} (layer,expert) "
                f"targets (e.g. {missing[0]}); complete coverage required"
            )
        extra = sorted(provided - expected)
        if extra:
            raise KeepMapError(f"saliency basis names unknown (layer,expert) targets: {extra[:3]}")
        score_map = {_layer_expert_of(key): values for key, values in basis.group_scores.items()}
        return select_keep_map(
            score_map,
            width=width,
            full=full,
            sparse_layers=sparse_layers,
            n_exp=n_exp,
            group=group,
        )
    except KeepMapError as exc:
        raise KeepMapBasisError(str(exc)) from exc


__all__ = [
    "KeepMapBasisError",
    "SaliencyBasis",
    "build_keep_map",
    "parse_saliency_basis",
]
