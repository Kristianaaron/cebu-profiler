"""F1 schema tests: ontology, evidence discipline, model assets, traces, runs."""

import pytest
from pydantic import ValidationError

from model_atlas.schemas.atlas_run import AtlasRun, AtlasRunStatus
from model_atlas.schemas.atlas_trace import AtlasTrace, Contribution, Intervention
from model_atlas.schemas.evidence import EvidenceClaim, EvidenceGrade, EvidenceKind, is_direct_kind
from model_atlas.schemas.model_asset import AssetType, ModelAsset
from model_atlas.schemas.ontology import (
    CapabilityLabel,
    TraceFamily,
    TrajectoryStage,
)


def test_ontology_cardinality():
    assert len(CapabilityLabel) == 21  # v2 §8 capability labels
    assert len(TrajectoryStage) == 10  # v2 §8 trajectory stages


def test_evidence_causal_grade_rejects_inference():
    with pytest.raises(ValidationError):
        EvidenceClaim(
            statement="X causes Y",
            grade=EvidenceGrade.DOWNSTREAM_CAUSAL_EFFECT,
            kind=EvidenceKind.INFERRED,
        )
    # direct causal test is fine
    EvidenceClaim(
        statement="X causes Y",
        grade=EvidenceGrade.DOWNSTREAM_CAUSAL_EFFECT,
        kind=EvidenceKind.CAUSALLY_TESTED,
    )


def test_association_can_be_inferred():
    EvidenceClaim(
        statement="X is associated with Y",
        grade=EvidenceGrade.OBSERVED_ASSOCIATION,
        kind=EvidenceKind.INFERRED,
    )


def test_is_direct_kind_classification():
    assert is_direct_kind(EvidenceKind.MEASURED) is True
    assert is_direct_kind(EvidenceKind.CAUSALLY_TESTED) is True
    assert is_direct_kind(EvidenceKind.PREDICTED) is False
    assert is_direct_kind(EvidenceKind.ESTIMATED) is False


def test_model_asset_requires_location():
    with pytest.raises(ValidationError):
        ModelAsset(model_asset_id="a", display_name="a", asset_type=AssetType.LOCAL_ENDPOINT)
    # endpoint-only local endpoint is fine
    ModelAsset(
        model_asset_id="a",
        display_name="a",
        asset_type=AssetType.LOCAL_ENDPOINT,
        endpoint="http://x",
    )


def test_source_checkpoint_requires_path():
    with pytest.raises(ValidationError):
        ModelAsset(
            model_asset_id="s",
            display_name="k3",
            asset_type=AssetType.SOURCE_CHECKPOINT,
            endpoint="http://x",  # endpoint alone is invalid for a source checkpoint
        )


def test_trace_family_must_match_payload():
    with pytest.raises(ValidationError):
        AtlasTrace(
            atlas_run_id="r1",
            family=TraceFamily.ROUTING,
            source_model_id="k3",
            payload=Contribution(),  # contribution under a "routing" family
        )


def test_intervention_trace_requires_layer():
    with pytest.raises(ValidationError):
        AtlasTrace(
            atlas_run_id="r1",
            family=TraceFamily.INTERVENTION,
            source_model_id="k3",
            payload=Intervention(intervention_type="expert_suppression"),
            # no layer_index -> should fail
        )
    # with layer_index it is valid
    AtlasTrace(
        atlas_run_id="r1",
        family=TraceFamily.INTERVENTION,
        source_model_id="k3",
        layer_index=3,
        payload=Intervention(intervention_type="expert_suppression"),
    )


def test_routing_trace_happy_path():
    trace = AtlasTrace(
        atlas_run_id="r1",
        family=TraceFamily.ROUTING,
        source_model_id="k3",
        capability_labels=[CapabilityLabel.DEBUGGING],
        payload={
            "kind": "routing",  # discriminator required when building from a dict
            "selected_expert_ids": [7, 2],
            "router_logits": [1.0, 0.5],
            "router_probabilities": [0.8, 0.2],
        },
    )
    assert trace.payload.selected_expert_ids == [7, 2]


def test_atlas_run_state_machine():
    run = AtlasRun(atlas_run_id="r1", source_model_asset_id="k3", calibration_suite_id="s")
    assert run.status == AtlasRunStatus.DRAFT
    assert run.is_pausable is False
    run.status = AtlasRunStatus.TRACING
    assert run.is_pausable is True
    run.status = AtlasRunStatus.COMPLETED
    assert run.is_terminal is True
