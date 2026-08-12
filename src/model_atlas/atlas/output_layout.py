"""The `atlas_runs/<id>/` output contract (v2 §27).

Declares the canonical machine-readable file set, which files each evidence
level is expected to guarantee, and validation that a run's produced evidence
matches its declared evidence level (so nothing is reported as present that was
not written, and no required output silently vanishes).
"""

from __future__ import annotations

from typing import Any

from model_atlas.schemas.atlas_run import AtlasRun
from model_atlas.schemas.evidence import EvidenceLevel

# Canonical §27 file set (as declared by the contract).
ATLAS_RUN_FILES: frozenset[str] = frozenset(
    {
        "run_manifest.json",
        "source_checkpoint_manifest.json",
        "source_checkpoint_manifest.parquet",
        "structural_model_graph.json",
        "calibration_manifest.json",
        "task_label_manifest.parquet",
        "token_trace_index.parquet",
        "routing_traces.parquet",
        "route_counterfactuals.parquet",
        "route_regret.parquet",
        "expert_activation.parquet",
        "expert_contribution.parquet",
        "representation_trace_manifest.json",
        "sparse_features.parquet",
        "feature_expert_links.parquet",
        "feature_path_links.parquet",
        "vocabulary_projection.parquet",
        "layer_saliency.parquet",
        "label_expert_saliency.parquet",
        "success_failure_contrasts.parquet",
        "recovery_contrasts.parquet",
        "expert_coactivation.parquet",
        "expert_transitions.parquet",
        "expert_similarity.parquet",
        "expert_substitutes.parquet",
        "expert_coalitions.parquet",
        "multi_component_causal_results.parquet",
        "cross_layer_paths.parquet",
        "path_branch_points.parquet",
        "quantization_probes.parquet",
        "quantization_compatibility.json",
        "expert_response_curves.parquet",
        "channel_map.parquet",
        "neuron_map.parquet",
        "tile_map.parquet",
        "compression_manifest.json",
        "hierarchy_map.json",
        "projection_sensitivity.parquet",
        "ablation_results.parquet",
        "negative_controls.parquet",
        "evidence_registry.parquet",
        "uncertainty_report.json",
        "resource_telemetry.parquet",
        "warnings.json",
        "atlas_summary.json",
        "reproducibility_command.sh",
    }
)

# Files guaranteed present at the lowest evidence level (routing + contribution).
ASSOCIATION_FILES: frozenset[str] = frozenset(
    {
        "run_manifest.json",
        "source_checkpoint_manifest.json",
        "calibration_manifest.json",
        "task_label_manifest.parquet",
        "routing_traces.parquet",
        "expert_activation.parquet",
        "expert_contribution.parquet",
        "layer_saliency.parquet",
        "label_expert_saliency.parquet",
        "expert_coactivation.parquet",
        "expert_transitions.parquet",
        "expert_similarity.parquet",
        "evidence_registry.parquet",
        "warnings.json",
        "atlas_summary.json",
        "reproducibility_command.sh",
    }
)


def expected_run_files(evidence_level: EvidenceLevel) -> frozenset[str]:
    """Files a run at the given evidence level must guarantee."""

    def _level_at_least(level: EvidenceLevel) -> bool:
        order = [
            EvidenceLevel.BASIC_SALIENCY,
            EvidenceLevel.ENHANCED_ATLAS,
            EvidenceLevel.CAUSAL_ATLAS,
        ]
        return order.index(evidence_level) >= order.index(level)

    base: set[str] = set(ASSOCIATION_FILES)
    if _level_at_least(EvidenceLevel.ENHANCED_ATLAS):
        base |= {
            "representation_trace_manifest.json",
            "route_counterfactuals.parquet",
            "route_regret.parquet",
            "sparse_features.parquet",
            "vocabulary_projection.parquet",
            "expert_substitutes.parquet",
            "projection_sensitivity.parquet",
            "compression_manifest.json",
            "hierarchy_map.json",
        }
    if _level_at_least(EvidenceLevel.CAUSAL_ATLAS):
        base |= {
            "multi_component_causal_results.parquet",
            "ablation_results.parquet",
            "expert_coalitions.parquet",
            "negative_controls.parquet",
            "cross_layer_paths.parquet",
            "path_branch_points.parquet",
        }
    return frozenset(base)


def validate_evidence_present(run: AtlasRun, evidence_present: list[str]) -> list[str]:
    """Return warnings for expected outputs (for the run's evidence level) that
    are not in `evidence_present`. Unknown/extra files are not errors."""
    present = set(evidence_present)
    expected = expected_run_files(run.evidence_level)
    missing = sorted(expected - present)
    warnings = [f"missing expected output: {f}" for f in missing]
    # Guard: any file claimed present must be one we know about (no invented files).
    unknown = sorted(set(evidence_present) - ATLAS_RUN_FILES)
    warnings += [f"unknown output not in atlas contract: {f}" for f in unknown]
    return warnings


def build_run_manifest(run: AtlasRun, evidence_present: list[str]) -> dict[str, Any]:
    """Compose the run_manifest.json content for this run."""
    return {
        "atlas_run_id": run.atlas_run_id,
        "source_model_asset_id": run.source_model_asset_id,
        "source_checkpoint_revision": run.source_checkpoint_revision,
        "calibration_suite_id": run.calibration_suite_id,
        "data_partition": run.data_partition.value,
        "evidence_level": run.evidence_level.value,
        "status": run.status.value,
        "configuration_hash": run.configuration_hash,
        "evidence_present": sorted(evidence_present),
        "warnings": list(run.warnings) + validate_evidence_present(run, evidence_present),
    }
