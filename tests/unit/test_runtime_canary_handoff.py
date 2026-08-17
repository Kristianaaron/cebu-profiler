from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_atlas.fit_telemetry import (
    CanaryPlan,
    CandidateBinding,
    FitSummary,
    NodeFitSummary,
    NodeRole,
    StopReason,
    build_base_canary_plan,
)
from model_atlas.runtime_canary_handoff import (
    CanaryHandoffError,
    load_verified_runtime_canary_handoff,
    publish_runtime_canary_handoff,
)
from model_atlas.schemas.evidence import EvidenceKind
from model_atlas.two_node_canary_executor import (
    CanaryExecutionReceipt,
    CanaryExecutionResult,
    JsonlEvidenceStore,
)


def _plan() -> CanaryPlan:
    return build_base_canary_plan(
        CandidateBinding(
            artifact_path="/artifacts/model.gguf",
            artifact_sha256="a" * 64,
            runtime_config_sha256="b" * 64,
            llama_server_sha256="c" * 64,
            worker_rpc_server_sha256="d" * 64,
            head_argv=("llama-server", "--model", "/artifacts/model.gguf"),
            worker_argv=("rpc-server",),
            producer_run_id="run-" + "1" * 24,
            producer_plan_id="recipe-" + "2" * 24,
            producer_recipe_sha256="3" * 64,
            producer_profile_id="profile-" + "4" * 24,
            producer_recommendation_id="rec-" + "5" * 24,
            producer_handoff_sha256="6" * 64,
        )
    )


def _node(node: NodeRole) -> NodeFitSummary:
    return NodeFitSummary(
        node=node,
        minimum_mem_available_bytes=16 * 1024**3,
        minimum_gpu_free_bytes=8 * 1024**3,
        peak_gpu_used_bytes=1,
        peak_process_pss_bytes=1,
        peak_process_private_bytes=1,
        peak_process_swap_bytes=0,
        new_pswpin_pages=0,
        new_pswpout_pages=0,
        rdma_rx_delta_bytes=1,
        rdma_tx_delta_bytes=1,
        disk_read_delta_bytes=1,
        disk_write_delta_bytes=1,
    )


def _execution(*, complete_base: bool = True) -> CanaryExecutionResult:
    plan = _plan()
    completed = tuple(step.step_id for step in (plan.steps if complete_base else plan.steps[:3]))
    fitted = complete_base
    stop_reason = StopReason.COMPLETED if complete_base else StopReason.OOM
    summary = FitSummary(
        plan_sha256=plan.canonical_sha256(),
        candidate=plan.candidate,
        both_nodes_measured=True,
        fitted=fitted,
        last_passing_context_tokens=65536 if complete_base else 4096,
        minimum_mem_headroom_bytes=8 * 1024**3,
        stop_reason=stop_reason,
        nodes=(_node(NodeRole.HEAD), _node(NodeRole.WORKER)),
        evidence_kind=EvidenceKind.MEASURED,
    )
    receipt = CanaryExecutionReceipt(
        plan_sha256=plan.canonical_sha256(),
        completed_step_ids=completed,
        stop_reason=stop_reason,
        runtime_claim_validated=True,
        runtime_claim_reason="validated",
        evidence_kind=EvidenceKind.MEASURED,
    )
    return CanaryExecutionResult(receipt=receipt, summary=summary)


def _evidence(path: Path, execution: CanaryExecutionResult) -> None:
    plan = _plan()
    store = JsonlEvidenceStore(path)
    store.append("canary_plan", plan)
    store.append("fit_summary", execution.summary)
    store.append_dict(
        "runtime_validation_claim",
        {"validated": True, "runtime_compat": ["llamacpp-rpc-two-spark"], "reason": "validated"},
    )
    store.append("canary_execution_receipt", execution.receipt)


def test_publish_and_load_measured_canary_handoff(tmp_path: Path) -> None:
    execution = _execution()
    evidence = (tmp_path / "evidence.jsonl").resolve()
    result = (tmp_path / "result.json").resolve()
    _evidence(evidence, execution)
    with evidence.open("a", encoding="utf-8") as handle:
        repeated = {
            "schema_version": 1,
            "record_type": "telemetry_sample",
            "payload": {"sample_set_id": "sample-1"},
        }
        handle.write(json.dumps(repeated) + "\n")
        handle.write(json.dumps(repeated) + "\n")
    published = publish_runtime_canary_handoff(
        result, plan=_plan(), execution=execution, evidence_path=evidence
    )
    assert published.validated_for_evaluation is True
    loaded = load_verified_runtime_canary_handoff(
        result, expected_plan=_plan(), require_evaluation_ready=True
    )
    assert loaded == published


def test_evidence_drift_and_plan_mismatch_fail_closed(tmp_path: Path) -> None:
    execution = _execution()
    evidence = (tmp_path / "evidence.jsonl").resolve()
    result = (tmp_path / "result.json").resolve()
    _evidence(evidence, execution)
    publish_runtime_canary_handoff(
        result, plan=_plan(), execution=execution, evidence_path=evidence
    )
    evidence.write_text(evidence.read_text() + "{}\n")
    with pytest.raises(CanaryHandoffError):
        load_verified_runtime_canary_handoff(result, expected_plan=_plan())


def test_incomplete_base_canary_cannot_authorize_evaluation(tmp_path: Path) -> None:
    execution = _execution(complete_base=False)
    evidence = (tmp_path / "evidence.jsonl").resolve()
    result = (tmp_path / "result.json").resolve()
    _evidence(evidence, execution)
    published = publish_runtime_canary_handoff(
        result, plan=_plan(), execution=execution, evidence_path=evidence
    )
    assert published.validated_for_evaluation is False
    with pytest.raises(CanaryHandoffError, match="evaluation-ready"):
        load_verified_runtime_canary_handoff(result, require_evaluation_ready=True)


def test_result_publication_is_exclusive(tmp_path: Path) -> None:
    execution = _execution()
    evidence = (tmp_path / "evidence.jsonl").resolve()
    result = (tmp_path / "result.json").resolve()
    _evidence(evidence, execution)
    result.write_text("occupied")
    with pytest.raises(FileExistsError):
        publish_runtime_canary_handoff(
            result, plan=_plan(), execution=execution, evidence_path=evidence
        )


def test_duplicate_evidence_records_and_parent_traversal_fail_closed(tmp_path: Path) -> None:
    execution = _execution()
    evidence = (tmp_path / "evidence.jsonl").resolve()
    result = (tmp_path / "result.json").resolve()
    _evidence(evidence, execution)
    with evidence.open("a", encoding="utf-8") as handle:
        handle.write(evidence.read_text(encoding="utf-8").splitlines()[0] + "\n")
    with pytest.raises(CanaryHandoffError, match="unique"):
        publish_runtime_canary_handoff(
            result, plan=_plan(), execution=execution, evidence_path=evidence
        )
    evidence.unlink()
    _evidence(evidence, execution)
    traversal = tmp_path / "child" / ".." / "result.json"
    with pytest.raises(CanaryHandoffError, match="absolute"):
        publish_runtime_canary_handoff(
            traversal, plan=_plan(), execution=execution, evidence_path=evidence
        )
