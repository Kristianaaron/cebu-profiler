"""Real GLM-5.2 two-node fit plan (Phase 4/G).

Computes, from the mounted 464.8 GB census, a measured size plan for aligned
uniform widths and finds which fit the two physical nodes after an explicitly
authorized maintenance-window removal of production occupancy. Current live
availability (production still running) is tracked separately as a live gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from model_atlas.loader import SizePlan, plan_uniform_widths

# Physical per-node capacity (host unified memory measured on both nodes).
# After an authorized maintenance window production occupancy is removed, so
# ~120 GiB physical minus a ~5 GiB OS/headroom margin is the binding budget.
WINDOW_PHYSICAL_GIB = 115.0
DEFAULT_WIDTHS = (64, 128, 256, 512, 1024, 2048)


@dataclass
class WidthFits:
    width: int
    total_gib: float
    per_rank_gib: float
    fits_window: bool
    window_headroom_gib: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_plan(
    checkpoint_dir: str,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    window_physical_gib: float = WINDOW_PHYSICAL_GIB,
) -> dict[int, WidthFits]:
    """Fits each candidate width against window physical budget (occupancy-free)."""
    plans: dict[int, SizePlan] = plan_uniform_widths(checkpoint_dir, widths=widths)
    out: dict[int, WidthFits] = {}
    for w, p in plans.items():
        fits_window = p.per_rank_gib() <= window_physical_gib
        out[w] = WidthFits(
            width=w,
            total_gib=p.total_gib(),
            per_rank_gib=p.per_rank_gib(),
            fits_window=fits_window,
            window_headroom_gib=window_physical_gib - p.per_rank_gib(),
        )
    return out


def choose_display(checkpoint_dir: str) -> dict[str, object]:
    """Best uniform width: largest that still fits both nodes in the window."""
    fits = fit_plan(checkpoint_dir)
    fit = {w: f for w, f in fits.items() if f.fits_window}
    best = max(fit) if fit else None
    return {
        "recommended_width": best,
        "fits": {str(w): f.to_dict() for w, f in fits.items()},
        "note": (
            "fit computed against WINDOW physical capacity (production occupancy "
            "removed); current live availability is a SEPARATE gate and is NOT "
            "this planning number"
        ),
    }


def to_json(checkpoint_dir: str) -> str:
    return json.dumps(choose_display(checkpoint_dir), indent=2, sort_keys=True)
