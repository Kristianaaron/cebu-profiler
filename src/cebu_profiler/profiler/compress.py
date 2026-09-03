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

from cebu_profiler.planning.width_buckets import SM121_WIDTH_BUCKETS
from cebu_profiler.planning.widths import build_manifest
from cebu_profiler.profiler.collector import ChannelStatsAccumulator
from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, forward
from cebu_profiler.schemas.manifest import (
    CompressionManifest,
    ManifestValidation,
    validate_manifest,
)
from cebu_profiler.scoring.base import ChannelScore
from cebu_profiler.scoring.causal import causal_scores
from cebu_profiler.scoring.quant_sensitivity import (
    expert_quant_sensitivity,
    recommend_bpw,
)
from cebu_profiler.scoring.redundancy import channel_kvalue, channel_uniqueness
from cebu_profiler.scoring.semantic import expert_semantic_score
from cebu_profiler.scoring.stability import StabilityAggregator
from cebu_profiler.scoring.taylor_grouped import score_grouped_surrogate
from cebu_profiler.scoring.tenp import tenp_rank

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
    buckets = [b for b in (allowed_widths or SM121_WIDTH_BUCKETS) if b <= model.mid] or [model.mid]

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

    # §8.1 semantic, §8.3 uniqueness / KEEP_VALUE, §8.4 quant-sensitivity views
    uniqueness_map = channel_uniqueness(model)
    st_vals = {key: st.stability for key, st in stability_rows.items() if st.stability is not None}
    kvalue_map = channel_kvalue(mean_tenp, uniqueness_map, causal, st_vals)
    sem_map = expert_semantic_score(model, samples)
    quant_bpw = recommend_bpw(expert_quant_sensitivity(model))

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
                semantic=sem_map.get((layer, e), 0.0),
                uniqueness=uniqueness_map.get(key),
                kvalue=kvalue_map.get(key),
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
        quant_bpw=quant_bpw,
        profiler_version="0.1.0",
    )
    validation = validate_manifest(manifest)
    return manifest, validation
