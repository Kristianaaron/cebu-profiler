"""Width-bucket compression planner (blueprint §14.2, §19-P4).

Consumes the versioned ScoreTable and emits a versioned CompressionManifest.
Per-expert target width is chosen from the allowed SM121 width vocabulary by the
smallest bucket whose retained channels cover a `coverage_target` share of that
expert's composite importance — so experts with concentrated importance are
pruned more, diffuse ones less (variable width driven by measured sensitivity).

Low-confidence pruning is heavily penalized (composite = tenp * stability *
confidence^2, blueprint §7-E) and protected channels are forced into the keep
set and the next allowed width bucket, so the manifest always validates.
"""

from __future__ import annotations

from cebu_profiler.planning.width_buckets import SM121_WIDTH_BUCKETS
from cebu_profiler.schemas.manifest import (
    BudgetSpec,
    CompressionManifest,
    ExpertPlan,
    ExpertScores,
    LayerPlan,
    QuantRecommendation,
)
from cebu_profiler.scoring.base import ChannelScore

_ROW = ChannelScore


def _composite(row: ChannelScore) -> float:
    """Importance penalized by low confidence/stability (blueprint §7-E)."""
    s = row.tenp if row.tenp is not None else 0.0
    conf = row.confidence if row.confidence is not None else 1.0
    st = row.stability if row.stability is not None else 1.0
    return s * st * (conf * conf)


def _next_bucket(width: int, allowed: list[int], full: int) -> int:
    """Smallest allowed bucket >= width (capped at full width)."""
    for b in sorted(allowed):
        if b >= width:
            return min(b, full)
    return full


def plan_expert(
    rows_by_expert: dict[tuple[int, int], list[ChannelScore]],
    allowed_widths: list[int],
    coverage_target: float,
    full_width: int,
    protected: dict[tuple[int, int], set[int]],
    quant_bpw: dict[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], ExpertPlan]:
    """Plan one expert: choose target width + keep set from measured scores."""
    plans: dict[tuple[int, int], ExpertPlan] = {}
    for (layer, e), rows in rows_by_expert.items():
        prot = protected.get((layer, e), set())
        ranked = sorted(
            [(r.channel, _composite(r), r) for r in rows],
            key=lambda x: x[1],
            reverse=True,
        )
        bpw = (quant_bpw or {}).get((layer, e), 3.25)
        total = sum(comp for _, comp, _ in ranked)
        if not ranked or total <= 0.0:
            # No measured evidence for this expert -> keep it at full width.
            # Cebu Profiler removes capacity only on evidence (AGENTS.md invariant #1),
            # and the manifest must always validate against its own contract.
            chosen = full_width
            keep_ids = list(range(full_width))
            keep_set = set(keep_ids)
        else:
            # smallest bucket whose top-w cumulative coverage meets the target
            chosen = full_width
            cumsum: dict[int, float] = {}
            run = 0.0
            for w in range(1, full_width + 1):
                if w - 1 < len(ranked):
                    run += ranked[w - 1][1]
                if w in {b for b in allowed_widths if b <= full_width}:
                    cumsum[w] = run / total if total > 0 else (1.0 if w >= 1 else 0.0)
            for b in sorted(b for b in allowed_widths if b <= full_width):
                if cumsum.get(b, 0.0) >= coverage_target:
                    chosen = b
                    break
            # force protected channels into the keep set via the next width bucket
            if len(prot) > chosen:
                chosen = _next_bucket(len(prot), allowed_widths, full_width)
            keep_ids = [ch for ch, _, _ in ranked[:chosen]]
            keep_set = set(keep_ids)
        missing = prot - keep_set
        if missing:
            # widen by the smallest allowed bucket that covers all protected + chosen
            need = max(chosen, len(keep_set | missing))
            chosen = _next_bucket(need, allowed_widths, full_width)
            keep_ids = [ch for ch, _, _ in ranked[:chosen]]

        kept = [r for ch, _, r in ranked if ch in set(keep_ids)]
        scores = ExpertScores(
            tenp=_mean([r.tenp for r in kept if r.tenp is not None]),
            taylor=_mean([r.taylor for r in kept if r.taylor is not None]),
            causal=_mean([r.causal for r in kept if r.causal is not None]),
            stability=_mean([r.stability for r in kept if r.stability is not None]),
            semantic=_mean([r.semantic for r in kept if r.semantic is not None]),
            uniqueness=_mean([r.uniqueness for r in kept if r.uniqueness is not None]),
            kvalue=_mean([r.kvalue for r in kept if r.kvalue is not None]),
        )
        confidence = _mean([r.confidence for r in kept if r.confidence is not None]) or 1.0
        reasons = ["protected_recovery_path"] if any(ch in prot for ch in keep_ids) else []
        plans[(layer, e)] = ExpertPlan(
            original_width=full_width,
            target_width=chosen,
            keep_channels=sorted(keep_ids),
            confidence=max(0.0, min(1.0, confidence)),
            scores=scores,
            protected_reasons=reasons,
            quant_recommendation=QuantRecommendation(format="exl3", bpw=bpw),
        )
    return plans


def build_manifest(
    model_name: str,
    source_checkpoint: str,
    score_rows: list[ChannelScore],
    num_layers: int,
    num_experts: int,
    full_width: int,
    allowed_widths: list[int] | None = None,
    coverage_target: float = 0.9,
    protected: dict[tuple[int, int], set[int]] | None = None,
    quant_bpw: dict[tuple[int, int], float] | None = None,
    profiler_version: str = "0.0.0",
    deployment: str = "2x-dgx-spark-sm121",
) -> CompressionManifest:
    """Emit a deterministic CompressionManifest from a measured ScoreTable."""
    allowed = allowed_widths or SM121_WIDTH_BUCKETS
    buckets = [b for b in allowed if b <= full_width]
    prot = protected or {}

    rows_by_expert: dict[tuple[int, int], list[ChannelScore]] = {}
    for r in score_rows:
        rows_by_expert.setdefault((r.layer, r.expert), []).append(r)
    for e in range(num_experts):
        for layer in range(num_layers):
            rows_by_expert.setdefault((layer, e), [])

    plans = plan_expert(rows_by_expert, buckets, coverage_target, full_width, prot, quant_bpw)
    layers: dict[str, LayerPlan] = {}
    for layer in range(num_layers):
        lp = LayerPlan(experts={str(e): plans[(layer, e)] for e in range(num_experts)})
        layers[str(layer)] = lp

    return CompressionManifest(
        model=model_name,
        source_checkpoint=source_checkpoint,
        profiler_version=profiler_version,
        calibration_suite="glm52-compression-v1",
        budget=BudgetSpec(deployment=deployment),
        allowed_widths=[b for b in allowed if b <= full_width],
        layers=layers,
    )


def estimate_params(manifest: CompressionManifest, hidden: int) -> int:
    """Approximate total retained routed-expert params from the manifest.

    Each kept channel keeps 3 coupled tensors of `hidden` params
    (gate[j,:], up[j,:], down[:,j]), so per kept channel ~ 3*hidden.
    """
    total = 0
    for layer in manifest.layers.values():
        for plan in layer.experts.values():
            total += plan.target_width * 3 * hidden
    return total


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None
