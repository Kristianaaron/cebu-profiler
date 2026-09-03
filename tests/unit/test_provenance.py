"""Tests for the method-provenance registry and watermark embedding."""

from __future__ import annotations

import json

import pytest

from cebu_profiler.evaluation.provenance import (
    METHOD_PROVENANCE,
    WATERMARK,
    provenance_manifest,
    require_provenance,
)


def test_every_method_record_is_citation_shaped():
    for method_id, rec in METHOD_PROVENANCE.items():
        assert rec["kind"] in {"research", "spec", "community", "inspiration"}, method_id
        assert rec.get("title"), method_id
        # community/inspiration entries name their source project
        if rec["kind"] in {"community", "inspiration"}:
            assert rec.get("url") or rec["kind"] == "community", method_id


def test_require_provenance_fails_closed_on_unknown():
    with pytest.raises(KeyError) as err:
        require_provenance("nonexistent_method")
    assert "no provenance record" in str(err.value)


def test_provenance_manifest_embeds_watermark_and_origin():
    manifest = provenance_manifest(["reap", "gptq", "fp8_e4m3", "atlas_evidence_bundle"])
    assert manifest["watermark"] == WATERMARK
    assert "github.com/Kristianaaron/cebu-profiler" in manifest["origin"]
    assert set(manifest["methods"]) == {
        "reap",
        "gptq",
        "fp8_e4m3",
        "atlas_evidence_bundle",
    }
    assert manifest["methods"]["atlas_evidence_bundle"]["kind"] == "inspiration"
    assert isinstance(json.dumps(manifest), str)


def test_provenance_manifest_rejects_unknown_method():
    with pytest.raises(KeyError):
        provenance_manifest(["reap", "totally_made_up"])


def test_run_manifest_carries_method_provenance_block():
    from cebu_profiler.profiler.output_layout import build_run_manifest
    from cebu_profiler.schemas.evidence import EvidenceLevel
    from cebu_profiler.schemas.profiler_run import ProfilerRun, ProfilerRunStatus

    run = ProfilerRun(
        profiler_run_id="prov-test",
        source_model_asset_id="asset",
        calibration_suite_id="suite",
        evidence_level=EvidenceLevel.BASIC_SALIENCY,
        status=ProfilerRunStatus.COMPLETED,
    )
    manifest = build_run_manifest(
        run,
        evidence_present=[],
        methods_used=["reap", "kld_teacher_gate", "atlas_evidence_bundle"],
    )
    block = manifest["method_provenance"]
    assert block["watermark"] == WATERMARK
    assert set(block["methods"]) == {"reap", "kld_teacher_gate", "atlas_evidence_bundle"}


def test_watermark_names_origin_and_license():
    assert "Kristianaaron/cebu-profiler" in WATERMARK
    assert "Apache-2.0" in WATERMARK
    assert "README" in WATERMARK
