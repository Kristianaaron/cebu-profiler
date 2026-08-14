"""Real GLM-5.2 two-node fit plan (round-4-corrected).

`plan_exact_sizes` yields a per-width exact-size plan for the candidate widths.
`fit_plan` compares against a WINDOW physical capacity budget — read dynamically
from measured host unified memory inputs, or a caller-passed value; the hardcoded
115 GiB is only a DEFAULT when nothing is measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from model_atlas.checkpoint.source_manifest import CheckpointManifest, load_manifest
from model_atlas.loader import SizePlan, plan_exact_sizes

DEFAULT_WIDTHS = (64, 128, 256, 512, 1024, 2048)
# Default only; prefer measured host physical capacity minus OS margin.
DEFAULT_WINDOW_GIB = 115.0
OS_HEADROOM_GIB = 5.0


@dataclass
class WidthFits:
    width: int
    total_bytes: int
    total_gib: float
    per_rank_bytes: int
    per_rank_gib: float
    fits_window: bool
    window_headroom_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_plan(
    checkpoint_dir: str,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    window_physical_gib: float | None = None,
    measured_per_node_gib: float | None = None,
) -> dict[int, WidthFits]:
    """Fit each candidate width against a window physical budget.

    `window_physical_gib` is the maintenance-window per-node budget; if None it
    defaults to `measured_per_node_gib - OS_HEADROOM_GIB`, else DEFAULT_WINDOW_GIB.
    Exact integer byte fields are preserved (no float-GiB round-trip for totals).
    """
    manifest = load_manifest(checkpoint_dir)
    source_cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    budget_gib = window_physical_gib
    if budget_gib is None:
        budget_gib = (measured_per_node_gib or DEFAULT_WINDOW_GIB) - OS_HEADROOM_GIB
    budget_bytes = int(budget_gib * (1024**3))
    out: dict[int, WidthFits] = {}
    for w in widths:
        keep_map = _uniform_keep(manifest, source_cfg, w)
        sp: SizePlan = plan_exact_sizes(manifest, source_cfg, keep_map)
        pr = sp.per_rank_bytes
        fits = pr <= budget_bytes
        out[w] = WidthFits(
            width=w,
            total_bytes=sp.total_bytes,
            total_gib=sp.total_gib,
            per_rank_bytes=pr,
            per_rank_gib=sp.per_rank_gib,
            fits_window=fits,
            window_headroom_bytes=budget_bytes - pr,
        )
    return out


def _uniform_keep(
    manifest: CheckpointManifest, source_cfg: dict[str, object], width: int,
) -> dict[tuple[int, int], list[int]]:
    from model_atlas.loader import _build_keep_map, _infer_geometry

    full, n_exp, sl = _infer_geometry(manifest, source_cfg)
    return _build_keep_map(None, width, full, n_exp, sl)


def choose_display(
    checkpoint_dir: str,
    measured_per_node_gib: float | None = None,
) -> dict[str, object]:
    """Largest uniform width fitting both nodes in the (measured) window."""
    fits = fit_plan(checkpoint_dir, measured_per_node_gib=measured_per_node_gib)
    fit = {w: f for w, f in fits.items() if f.fits_window}
    best = max(fit) if fit else None
    return {
        "recommended_width": best,
        "fits": {str(w): f.to_dict() for w, f in fits.items()},
        "note": (
            "fit computed against WINDOW physical capacity (production occupancy "
            "removed); measured_per_node_gib passed dynamically or defaulted. "
            "Current live availability is a SEPARATE gate."
        ),
    }


def to_json(checkpoint_dir: str) -> str:
    return json.dumps(choose_display(checkpoint_dir), indent=2, sort_keys=True)
