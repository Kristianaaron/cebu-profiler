"""Output-contract tests: evidence-level file guarantees + validation."""

from model_atlas.atlas.output_layout import (
    build_run_manifest,
    expected_run_files,
    validate_evidence_present,
)
from model_atlas.schemas.atlas_run import AtlasRun
from model_atlas.schemas.evidence import EvidenceLevel


def test_basic_level_files():
    files = expected_run_files(EvidenceLevel.BASIC_SALIENCY)
    assert "routing_traces.parquet" in files
    assert "expert_contribution.parquet" in files
    # deeper-only files are NOT guaranteed at basic
    assert "route_counterfactuals.parquet" not in files
    assert "ablation_results.parquet" not in files


def test_causal_level_adds_files():
    files = expected_run_files(EvidenceLevel.CAUSAL_ATLAS)
    assert "ablation_results.parquet" in files
    assert "negative_controls.parquet" in files
    assert "multi_component_causal_results.parquet" in files


def test_validate_catches_missing_and_unknown():
    run = AtlasRun(
        atlas_run_id="r1",
        source_model_asset_id="k3",
        calibration_suite_id="s",
        evidence_level=EvidenceLevel.BASIC_SALIENCY,
    )
    # claims one real file + one invented file; rest of the guaranteed set missing
    warnings = validate_evidence_present(run, ["routing_traces.parquet", "made_up.parquet"])
    assert any("missing expected output" in w for w in warnings)
    assert any("made_up.parquet" in w for w in warnings)


def test_build_run_manifest_includes_warnings():
    run = AtlasRun(
        atlas_run_id="r1",
        source_model_asset_id="k3",
        calibration_suite_id="s",
        evidence_level=EvidenceLevel.BASIC_SALIENCY,
    )
    manifest = build_run_manifest(run, ["routing_traces.parquet"])
    assert manifest["atlas_run_id"] == "r1"
    assert "routing_traces.parquet" in manifest["evidence_present"]
    assert any("missing expected output" in w for w in manifest["warnings"])
