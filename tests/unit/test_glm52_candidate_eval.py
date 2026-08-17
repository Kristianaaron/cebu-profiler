from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from model_atlas.evaluation import glm52_candidate_eval as candidate_eval_module
from model_atlas.evaluation.eval_lab import (
    DataPartition,
    EndpointConfigIdentity,
    EndpointTransport,
    EvalLabAdapter,
    EvalLabRequest,
    EvalParameters,
    FrozenHeldOutManifest,
    canonical_directory_sha256,
)
from model_atlas.evaluation.glm52_candidate_eval import (
    CandidateEvalError,
    build_candidate_eval_plan,
    build_glm52_candidate_eval_plan,
    build_task_evidence,
    gguf_embedded_template_identity,
    parse_candidate_eval_runs,
)
from model_atlas.fit_telemetry import (
    CandidateBinding,
    FitSummary,
    NodeFitSummary,
    NodeRole,
    StopReason,
    build_base_canary_plan,
)
from model_atlas.runtime_artifact_handoff import CompressionHandoff
from model_atlas.runtime_canary_handoff import RuntimeCanaryHandoff
from model_atlas.schemas.evidence import EvidenceKind
from model_atlas.two_node_canary_executor import (
    CanaryExecutionReceipt,
    CanaryExecutionResult,
)


def _compression() -> CompressionHandoff:
    return CompressionHandoff(
        artifact_path="/artifacts/glm52-candidate.gguf",
        artifact_sha256="a" * 64,
        artifact_size_bytes=1,
        evidence_sha256="b" * 64,
        evidence_size_bytes=1,
        evidence_relpath="evidence.jsonl",
        producer_run_id="run-" + "1" * 24,
        producer_plan_id="recipe-" + "2" * 24,
        producer_recipe_sha256="3" * 64,
        producer_profile_id="profile-" + "4" * 24,
        producer_profile_sha256="5" * 64,
        producer_recommendation_id="rec-" + "6" * 24,
        handoff_sha256="7" * 64,
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


def _canary(compression: CompressionHandoff) -> RuntimeCanaryHandoff:
    plan = build_base_canary_plan(
        CandidateBinding(
            artifact_path=compression.artifact_path,
            artifact_sha256=compression.artifact_sha256,
            runtime_config_sha256="8" * 64,
            llama_server_sha256="9" * 64,
            worker_rpc_server_sha256="a" * 64,
            head_argv=("llama-server", "--model", compression.artifact_path),
            worker_argv=("rpc-server",),
            producer_run_id=compression.producer_run_id,
            producer_plan_id=compression.producer_plan_id,
            producer_recipe_sha256=compression.producer_recipe_sha256,
            producer_profile_id=compression.producer_profile_id,
            producer_recommendation_id=compression.producer_recommendation_id,
            producer_handoff_sha256=compression.handoff_sha256,
        )
    )
    summary = FitSummary(
        plan_sha256=plan.canonical_sha256(),
        candidate=plan.candidate,
        both_nodes_measured=True,
        fitted=True,
        last_passing_context_tokens=65536,
        minimum_mem_headroom_bytes=8 * 1024**3,
        stop_reason=StopReason.COMPLETED,
        nodes=(_node(NodeRole.HEAD), _node(NodeRole.WORKER)),
        evidence_kind=EvidenceKind.MEASURED,
    )
    receipt = CanaryExecutionReceipt(
        plan_sha256=plan.canonical_sha256(),
        completed_step_ids=tuple(step.step_id for step in plan.steps),
        stop_reason=StopReason.COMPLETED,
        runtime_claim_validated=True,
        runtime_claim_reason="validated",
        evidence_kind=EvidenceKind.MEASURED,
    )
    return RuntimeCanaryHandoff(
        plan=plan,
        execution=CanaryExecutionResult(receipt=receipt, summary=summary),
        evidence_path="/evidence/canary.jsonl",
        evidence_sha256="b" * 64,
        evidence_size_bytes=1,
        validated_for_evaluation=True,
    )


def _request(tmp_path: Path, compression: CompressionHandoff) -> EvalLabRequest:
    root = (tmp_path / "eval-lab").resolve()
    suite = root / "configs" / "suites" / "heldout.yaml"
    tasks = root / "tasks"
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text(
        "a20da6c6b9cbf872f7c083bffe66afde40c2c8f2\n"
    )
    executable = root / ".venv" / "bin" / "eval-lab"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    suite.parent.mkdir(parents=True)
    suite.write_text("tasks:\n  - task_id: synthetic.math\n  - task_id: synthetic.logic\n")
    tasks.mkdir()
    for task_id in ("synthetic.math", "synthetic.logic"):
        (tasks / f"{task_id}.yaml").write_text(
            f"id: {task_id}\n"
            "data_partition: held_out_evaluation\n"
            "execution:\n  runner: direct\n  timeout_seconds: 30\n"
        )
    held_out = FrozenHeldOutManifest(
        data_partition=DataPartition.HELD_OUT_EVALUATION,
        task_suite_id="atlas-glm52-heldout",
        task_suite_revision="c" * 40,
        task_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
        task_definitions_sha256=canonical_directory_sha256(tasks),
        tracked_task_ids=["synthetic.math", "synthetic.logic"],
        corpus_sha256="d" * 64,
        tokenizer_sha256="e" * 64,
        template_sha256="f" * 64,
        evaluation_sample_ids=["heldout-1", "heldout-2"],
    )
    return EvalLabRequest(
        candidate_artifact_path=compression.artifact_path,
        candidate_artifact_sha256=compression.artifact_sha256,
        endpoint=EndpointConfigIdentity(
            endpoint_id="candidate",
            transport=EndpointTransport.OPENAI_COMPATIBLE,
            endpoint_url="http://127.0.0.1:8080/v1",
            config_sha256="8" * 64,
        ),
        held_out=held_out,
        tasks=["synthetic.math", "synthetic.logic"],
        parameters=EvalParameters(seed=17, temperature=0.0, max_tokens=96, timeout_seconds=30),
        eval_lab_root=str(root),
        suite_ref=str(suite),
        tasks_dir=str(tasks),
        runs_root=str((tmp_path / "runs").resolve()),
        db_path=str((tmp_path / "runs.db").resolve()),
        model_id="glm52-candidate",
        model_name="glm52-candidate",
    )


def _adapter(request: EvalLabRequest) -> EvalLabAdapter:
    return EvalLabAdapter(executable=str(Path(request.eval_lab_root) / ".venv/bin/eval-lab"))


def _run(request: EvalLabRequest, task_id: str, run_id: str) -> Path:
    root = Path(request.runs_root) / run_id
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task_id": task_id,
                "model_id": request.model_id,
                "random_seed": request.parameters.seed,
                "sampling": {"temperature": 0.0, "max_tokens": 96},
                "budgets": {"timeout_seconds": 30.0, "http_timeout_seconds": 30.0},
                "result_status": "completed",
            }
        )
    )
    (root / "result.json").write_text(
        json.dumps({"run_id": run_id, "error": None, "duration_s": 1.5})
    )
    (root / "scores.jsonl").write_text(json.dumps({"scorer_id": "exact", "score": 1.0}) + "\n")
    return root


def test_plan_binds_verified_lineage_and_exact_argv(tmp_path: Path) -> None:
    compression = _compression()
    request = _request(tmp_path, compression)
    plan = build_candidate_eval_plan(
        compression_handoff=compression,
        runtime_canary_handoff=_canary(compression),
        eval_request=request,
        adapter=_adapter(request),
    )
    assert plan.compression_handoff_sha256 == compression.handoff_sha256
    assert plan.held_out_manifest_id == request.held_out.manifest_id
    assert plan.task_definitions_sha256 == request.held_out.task_definitions_sha256
    assert plan.tokenizer_sha256 == request.held_out.tokenizer_sha256
    assert plan.parameters == request.parameters
    assert plan.argv == (
        str(Path(request.eval_lab_root) / ".venv/bin/eval-lab"),
        "run",
        "suite",
        request.suite_ref,
        "--model",
        request.model_id,
        "--endpoint",
        "http://127.0.0.1:8080/v1",
        "--model-name",
        request.model_name,
        "--tasks-dir",
        request.tasks_dir,
        "--runs-root",
        request.runs_root,
        "--db",
        request.db_path,
        "--seed",
        "17",
        "--temperature",
        "0.0",
        "--max-tokens",
        "96",
        "--http-timeout-seconds",
        "30.0",
        "--task-timeout-seconds",
        "30.0",
        "--require-held-out",
        "--json",
    )


def test_parser_requires_current_evidence_and_frozen_manifest(tmp_path: Path) -> None:
    compression = _compression()
    request = _request(tmp_path, compression)
    plan = build_candidate_eval_plan(
        compression_handoff=compression,
        runtime_canary_handoff=_canary(compression),
        eval_request=request,
        adapter=_adapter(request),
    )
    evidence = tuple(
        build_task_evidence(task, _run(request, task, run_id))
        for task, run_id in zip(request.tasks, ("0" * 12, "1" * 12), strict=True)
    )
    parsed = parse_candidate_eval_runs(plan, evidence)
    assert parsed.plan_sha256 == plan.plan_sha256
    assert [score.task_id for score in parsed.report.task_scores] == request.tasks
    assert parsed.report.token_kld is None and parsed.report.cka is None
    Path(evidence[0].result_path).write_text("{}")
    with pytest.raises(CandidateEvalError, match="bytes drifted"):
        parse_candidate_eval_runs(plan, evidence)


def test_plan_rejects_unverified_or_mismatched_handoffs(tmp_path: Path) -> None:
    compression = _compression()
    request = _request(tmp_path, compression)
    with pytest.raises(CandidateEvalError, match="artifact differs"):
        build_candidate_eval_plan(
            compression_handoff=compression,
            runtime_canary_handoff=_canary(compression),
            eval_request=request.model_copy(update={"candidate_artifact_sha256": "0" * 64}),
            adapter=_adapter(request),
        )
    with pytest.raises(CandidateEvalError, match="not evaluation-ready"):
        build_candidate_eval_plan(
            compression_handoff=compression,
            runtime_canary_handoff=_canary(compression).model_copy(
                update={"validated_for_evaluation": False}
            ),
            eval_request=request,
            adapter=_adapter(request),
        )


def _real_default_checkout(tmp_path: Path) -> Path:
    root = (tmp_path / "eval-lab").resolve()
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text(
        "a20da6c6b9cbf872f7c083bffe66afde40c2c8f2\n"
    )
    executable = root / ".venv" / "bin" / "eval-lab"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    suite = root / "configs" / "suites" / "atlas-glm52-heldout.yaml"
    suite.parent.mkdir(parents=True)
    suite.write_text(
        "id: suite.atlas-glm52-heldout.001\n"
        "tasks:\n"
        "  - task_id: atlas.math\n"
        "  - task_id: atlas.reasoning\n"
    )
    tasks = root / "tasks" / "atlas_glm52_heldout"
    for directory, task_id, prompt in (
        ("math", "atlas.math", "What is 2 + 3?\n"),
        ("reasoning", "atlas.reasoning", "Reply with A.\n"),
    ):
        task_root = tasks / directory
        task_root.mkdir(parents=True)
        (task_root / "task.yaml").write_text(
            f"id: {task_id}\n"
            "data_partition: held_out_evaluation\n"
            "execution:\n  runner: direct\n  timeout_seconds: 300\n"
        )
        (task_root / "prompt.md").write_text(prompt)
    return root


def test_real_default_builder_derives_all_identities_from_fake_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _real_default_checkout(tmp_path)
    operation_root = (tmp_path / "operation").resolve()
    operation_root.mkdir()
    monkeypatch.setattr(candidate_eval_module, "GLM52_EVAL_LAB_ROOT", root)
    compression = _compression()

    plan = build_glm52_candidate_eval_plan(
        compression_handoff=compression,
        runtime_canary_handoff=_canary(compression),
        eval_output_root=operation_root,
        verified_tokenizer_sha256="e" * 64,
    )

    assert plan.eval_request.tasks == ["atlas.math", "atlas.reasoning"]
    assert str(plan.eval_request.endpoint.endpoint_url) == "http://127.0.0.1:8892/v1"
    assert plan.eval_request.endpoint.config_sha256 == "8" * 64
    assert plan.eval_request.parameters == EvalParameters(
        seed=17, temperature=0.0, max_tokens=96, timeout_seconds=300
    )
    assert plan.eval_request.model_id == plan.eval_request.model_name == "glm52-mixed-gguf"
    assert plan.eval_request.runs_root == str(operation_root / "runs")
    assert plan.eval_request.db_path == str(operation_root / "runstore.db")
    assert plan.template_sha256 == gguf_embedded_template_identity("a" * 64)
    assert plan.argv[0] == str(root / ".venv/bin/eval-lab")
    assert plan.plan_sha256 is not None


def test_real_default_builder_fails_closed_on_revision_and_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _real_default_checkout(tmp_path)
    operation_root = (tmp_path / "operation").resolve()
    operation_root.mkdir()
    monkeypatch.setattr(candidate_eval_module, "GLM52_EVAL_LAB_ROOT", root)
    compression = _compression()
    canary = _canary(compression)

    (root / ".git/refs/heads/main").write_text("0" * 40 + "\n")
    with pytest.raises(ValueError, match="pinned revision"):
        build_glm52_candidate_eval_plan(
            compression_handoff=compression,
            runtime_canary_handoff=canary,
            eval_output_root=operation_root,
            verified_tokenizer_sha256="e" * 64,
        )
    (root / ".git/refs/heads/main").write_text("a20da6c6b9cbf872f7c083bffe66afde40c2c8f2\n")
    plan = build_glm52_candidate_eval_plan(
        compression_handoff=compression,
        runtime_canary_handoff=canary,
        eval_output_root=operation_root,
        verified_tokenizer_sha256="e" * 64,
    )
    (root / "tasks/atlas_glm52_heldout/math/prompt.md").write_text("changed\n")
    with pytest.raises(ValidationError, match="tasks tree does not match"):
        type(plan).model_validate(plan.model_dump(mode="json"))


def test_real_default_builder_rejects_symlink_output_and_executable_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _real_default_checkout(tmp_path)
    operation_root = (tmp_path / "operation").resolve()
    operation_root.mkdir()
    linked_operation = tmp_path / "linked-operation"
    linked_operation.symlink_to(operation_root, target_is_directory=True)
    monkeypatch.setattr(candidate_eval_module, "GLM52_EVAL_LAB_ROOT", root)
    compression = _compression()
    canary = _canary(compression)

    with pytest.raises(CandidateEvalError, match="must not traverse symlinks"):
        build_glm52_candidate_eval_plan(
            compression_handoff=compression,
            runtime_canary_handoff=canary,
            eval_output_root=linked_operation,
            verified_tokenizer_sha256="e" * 64,
        )
    plan = build_glm52_candidate_eval_plan(
        compression_handoff=compression,
        runtime_canary_handoff=canary,
        eval_output_root=operation_root,
        verified_tokenizer_sha256="e" * 64,
    )
    (root / ".venv/bin/eval-lab").write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(ValidationError, match="executable identity changed"):
        type(plan).model_validate(plan.model_dump(mode="json"))


def test_plan_validation_rejects_forged_argv_and_endpoint_binding(tmp_path: Path) -> None:
    compression = _compression()
    request = _request(tmp_path, compression)
    plan = build_candidate_eval_plan(
        compression_handoff=compression,
        runtime_canary_handoff=_canary(compression),
        eval_request=request,
        adapter=_adapter(request),
    )
    payload = plan.model_dump(mode="json")
    payload["plan_sha256"] = None
    payload["argv"] = [*plan.argv, "--forged"]
    with pytest.raises(ValidationError, match="argv differs"):
        type(plan).model_validate(payload)

    mismatched_request = request.model_copy(
        update={"endpoint": request.endpoint.model_copy(update={"config_sha256": "0" * 64})}
    )
    with pytest.raises(CandidateEvalError, match="endpoint config differs"):
        build_candidate_eval_plan(
            compression_handoff=compression,
            runtime_canary_handoff=_canary(compression),
            eval_request=mismatched_request,
            adapter=_adapter(request),
        )
