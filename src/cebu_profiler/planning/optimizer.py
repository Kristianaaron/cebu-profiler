"""Budget-constrained rate-distortion allocation (blueprint Priority 4 #5, §15).

The bucket planner picks width per expert by a coverage target; this is a
complementary *constrained* view: given an explicit deployment budget (bytes or
channels) it greedily keeps the highest-value channels up to the budget
(nested per-expert prefixes, FlexMoE-style), then rounds each expert to the
nearest allowed SM121 bucket. Used to search the two-Spark fit envelope instead
of hardcoding a prune ratio (§15 / §17 Stage 2).
"""

from __future__ import annotations

from dataclasses import dataclass

from model_atlas.atlas.runtime import MiniMoE
from model_atlas.schemas.manifest import (
    BudgetSpec,
    CompressionManifest,
    ExpertPlan,
    LayerPlan,
    QuantRecommendation,
)


def allocate_under_budget(
    model: MiniMoE,
    importance: dict[tuple[int, int, int], float],
    budget_channels: int,
) -> dict[tuple[int, int], list[int]]:
    """Greedy top-value channel allocation up to ``budget_channels`` (nested).

    Returns per-expert kept channel prefixes whose total cardinality is the
    largest number <= ``budget_channels`` (all experts are non-empty).
    """
    n_slots = len(model.layers) * model.n_exp
    budget_channels = min(max(budget_channels, n_slots), n_slots * model.mid)
    # global order by importance desc
    def _channels() -> list[tuple[int, int, int, float]]:
        return [
            (layer, e, c, importance.get((layer, e, c), 0.0))
            for layer in range(len(model.layers))
            for e in range(model.n_exp)
            for c in range(model.mid)
        ]

    all_chan = sorted(_channels(), key=lambda x: -x[3])
    selected = set(all_chan[idx][:3] for idx in range(budget_channels))
    orders: dict[tuple[int, int], list[int]] = {}
    for layer in range(len(model.layers)):
        for e in range(model.n_exp):
            ranked = sorted(
                (c for c in range(model.mid)),
                key=lambda c: -importance.get((layer, e, c), 0.0),
            )
            orders[(layer, e)] = [c for c in ranked if (layer, e, c) in selected]
    return orders


def _next_bucket(count: int, allowed: list[int], full: int) -> int:
    for b in sorted(allowed):
        if b >= count:
            return min(b, full)
    return full


@dataclass
class BudgetPlan:
    manifest: CompressionManifest
    kept_channels: int
    estimated_params: int


def rate_distortion_manifest(
    model: MiniMoE,
    importance: dict[tuple[int, int, int], float],
    budget_channels: int,
    source_checkpoint: str = "glm52-compression-v1",
    allowed_widths: list[int] | None = None,
    quant_bpw: dict[tuple[int, int], float] | None = None,
) -> BudgetPlan:
    """Build a CompressionManifest from a channel budget, rounded to allowed buckets."""
    from model_atlas.planning.width_buckets import SM121_WIDTH_BUCKETS

    allowed = [b for b in (allowed_widths or SM121_WIDTH_BUCKETS) if b <= model.mid] or [
        model.mid
    ]
    orders = allocate_under_budget(model, importance, budget_channels)
    kept = sum(len(o) for o in orders.values())
    # per-expert top-`width` ranked channels so keep cardinality == bucket width
    ranked_by_expert: dict[tuple[int, int], list[int]] = {}
    for layer in range(len(model.layers)):
        for e in range(model.n_exp):
            ranked_by_expert[(layer, e)] = sorted(
                (c for c in range(model.mid)),
                key=lambda c: -importance.get((layer, e, c), 0.0),
            )

    layers: dict[str, LayerPlan] = {}
    for layer in range(len(model.layers)):
        experts: dict[str, ExpertPlan] = {}
        for e in range(model.n_exp):
            count = len(orders[(layer, e)])
            width = _next_bucket(count, allowed, model.mid)
            experts[str(e)] = ExpertPlan(
                original_width=model.mid,
                target_width=width,
                keep_channels=ranked_by_expert[(layer, e)][:width],
                quant_recommendation=QuantRecommendation(
                    format="exl3", bpw=(quant_bpw or {}).get((layer, e), 3.25)
                ),
            )
        layers[str(layer)] = LayerPlan(experts=experts)

    manifest = CompressionManifest(
        model=model.arch.name,
        source_checkpoint=source_checkpoint,
        allowed_widths=allowed,
        budget=BudgetSpec(deployment="2x-dgx-spark-sm121"),
        layers=layers,
    )
    est = sum(
        p.target_width * 3 * model.hidden
        for lp in layers.values()
        for p in lp.experts.values()
    )
    return BudgetPlan(manifest=manifest, kept_channels=kept, estimated_params=est)
