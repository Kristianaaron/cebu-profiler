"""End-to-end compression planning pipeline (blueprint §25).

NVFP4-equivalent path runnable today on the synthetic MiniMoE:
    trace (streaming channel collector) -> TENP -> stability
    -> targeted causal -> grouped-Taylor surrogate -> width-bucket planner
    -> compression manifest

This is the blueprint's first end-to-end milestone. It does NOT require EXL3,
a BF16 parent, or SM121 kernels. On a real GLM-5.2 it would run unchanged
against the same contracts (the adapter supplies the geometry).
"""

from __future__ import annotations

from model_atlas.atlas.collector import ChannelStatsAccumulator
from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.planning.width_buckets import SM121_WIDTH_BUCKETS
from model_atlas.planning.widths import build_manifest
from model_atlas.schemas.manifest import (
    CompressionManifest,
    ManifestValidation,
    validate_manifest,
)
from model_atlas.scoring.base import ChannelScore
from model_atlas.scoring.causal import causal_scores
from model_atlas.scoring.stability import StabilityAggregator
from model_atlas.scoring.taylor_grouped import score_grouped_surrogate
from model_atlas.scoring.tenp import tenp_rank

_COVERAGE_TARGET = 0.9


def _collect(model: MiniMoE, samples: list[CalibrationSample]) -> ChannelStatsAccumulator:
    acc = ChannelStatsAccumulator()
    for sample in samples:
        forward(model, sample.tokens, channel_stats=acc)
    return acc


def _split_corpora(samples: list[CalibrationSample], n_runs: int) -> list[list[CalibrationSample]]:
    """Deterministic n-way split of the calibration corpus for stability runs."""
    if n_runs <= 1 or len(samples) < n_runs:
        return [samples]
    out: list[list[CalibrationSample]] = [[] for _ in range(n_runs)]
    for i, s in enumerate(samples):
        out[i % n_runs].append(s)
    return out


def run_compression_pipeline(
    model: MiniMoE,
    samples: list[CalibrationSample],
    allowed_widths: list[int] | None = None,
    coverage_target: float = _COVERAGE_TARGET,
    n_stability_runs: int = 3,
    source_checkpoint: str = "glm52-compression-v1",
    protected: dict[tuple[int, int], set[int]] | None = None,
) -> tuple[CompressionManifest, ManifestValidation]:
    """Trace -> score -> plan -> manifest over the synthetic MiniMoE."""
    buckets = [b for b in (allowed_widths or SM121_WIDTH_BUCKETS) if b <= model.mid] or [
        model.mid
    ]

    # stability: independent per-split runs of channel importance
    splits = _split_corpora(samples, n_stability_runs)
    run_maps: list[dict[tuple[int, int, int], float]] = []
    for split in splits:
        acc = _collect(model, split)
        run_maps.append(tenp_rank(model, acc))

    mean_tenp: dict[tuple[int, int, int], float] = {}
    for key in {k for m in run_maps for k in m}:
        mean_tenp[key] = sum(m.get(key, 0.0) for m in run_maps) / len(run_maps)

    base_acc = _collect(model, samples)
    causal = causal_scores(model, base_acc)
    taylor, _ = score_grouped_surrogate(model, base_acc)
    stability_rows = {
        (r.layer, r.expert, r.channel): r for r in StabilityAggregator(run_maps).aggregate()
    }

    keys = set(mean_tenp) | set(causal) | set(taylor) | set(stability_rows)
    score_rows: list[ChannelScore] = []
    for key in sorted(keys):
        layer, e, c = key
        st = stability_rows.get(key)
        score_rows.append(
            ChannelScore(
                layer=layer,
                expert=e,
                channel=c,
                tenp=mean_tenp.get(key),
                causal=causal.get(key),
                taylor=taylor.get(key),
                stability=st.stability if st else None,
                rank_stability=st.rank_stability if st else None,
                confidence=st.confidence if st else None,
            )
        )

    full_width = model.mid
    manifest = build_manifest(
        model_name=model.arch.name,
        source_checkpoint=source_checkpoint,
        score_rows=score_rows,
        num_layers=len(model.layers),
        num_experts=model.n_exp,
        full_width=full_width,
        allowed_widths=buckets,
        coverage_target=coverage_target,
        protected=protected,
        atlas_version="0.1.0",
    )
    validation = validate_manifest(manifest)
    return manifest, validation
