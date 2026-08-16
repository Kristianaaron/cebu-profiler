import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from model_atlas.evaluation.eval_lab import (
    EVAL_LAB_REVISION,
    CandidateTaskReport,
    DataPartition,
    EndpointConfigIdentity,
    EndpointTransport,
    EvalLabAdapter,
    EvalLabRequest,
    EvalLabResult,
    EvalLabStatus,
    EvalParameters,
    FrozenHeldOutManifest,
    HandoffBlocker,
    PerformanceReport,
    TaskScore,
    TeacherRelativeBlocker,
    canonical_directory_sha256,
    eval_lab_output_layout,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REVISION = "d" * 40


def _heldout(**changes: object) -> FrozenHeldOutManifest:
    payload: dict[str, object] = {
        "data_partition": DataPartition.HELD_OUT_EVALUATION,
        "task_suite_id": "atlas-glm52-canary",
        "task_suite_revision": REVISION,
        "task_suite_sha256": SHA_A,
        "task_definitions_sha256": SHA_B,
        "tracked_task_ids": ["arc_easy", "gsm8k"],
        "corpus_sha256": SHA_C,
        "tokenizer_sha256": SHA_A,
        "template_sha256": SHA_B,
        "evaluation_sample_ids": ["eval-1", "eval-2"],
        "calibration_sample_ids": ["cal-1", "cal-2"],
    }
    payload.update(changes)
    return FrozenHeldOutManifest.model_validate(payload)


def _request(**changes: object) -> EvalLabRequest:
    payload: dict[str, object] = {
        "candidate_artifact_path": "/artifacts/glm52-q1.gguf",
        "candidate_artifact_sha256": SHA_A,
        "endpoint": EndpointConfigIdentity(
            endpoint_id="spark-pair-candidate",
            transport=EndpointTransport.OPENAI_COMPATIBLE,
            endpoint_url="http://127.0.0.1:8100/v1",
            config_sha256=SHA_B,
        ),
        "held_out": _heldout(),
        "tasks": ["arc_easy", "gsm8k"],
        "parameters": EvalParameters(
            seed=None, temperature=0.0, max_tokens=4096, timeout_seconds=300.0
        ),
        "eval_lab_root": "/opt/eval-lab",
        "suite_ref": "/opt/eval-lab/configs/suites/atlas.yaml",
        "tasks_dir": "/opt/eval-lab/tasks",
        "runs_root": "/var/lib/eval-lab/runs",
        "db_path": "/var/lib/eval-lab/runstore.db",
        "model_id": "glm52-atlas-candidate",
        "model_name": "glm-5.2",
    }
    payload.update(changes)
    return EvalLabRequest.model_validate(payload)


def _report(request: EvalLabRequest) -> CandidateTaskReport:
    return CandidateTaskReport(
        request_id=request.request_id,
        data_partition=DataPartition.HELD_OUT_EVALUATION,
        task_scores=[
            TaskScore(task_id="arc_easy", scores={"accuracy": 0.5}),
            TaskScore(task_id="gsm8k", scores={"exact_match": 0.25}),
        ],
        performance=PerformanceReport(
            requests=2,
            successful_requests=2,
            elapsed_seconds=3.0,
            tokens_per_second=22.0,
            latency_p50_ms=50.0,
            latency_p95_ms=70.0,
        ),
        teacher_relative_blockers=list(TeacherRelativeBlocker),
    )


def _write_json(path: Path, model: object) -> None:
    path.write_text(model.model_dump_json(indent=2))  # type: ignore[attr-defined]


def test_request_and_manifest_ids_are_stable_and_timestamp_free() -> None:
    first = _request()
    second = _request()
    assert first.request_id == second.request_id
    assert first.held_out.manifest_id == second.held_out.manifest_id
    assert "timestamp" not in first.model_dump(mode="json")
    drifted = _request(
        parameters=EvalParameters(
            seed=18, temperature=0.0, max_tokens=4096, timeout_seconds=300.0
        )
    )
    assert drifted.request_id != first.request_id


def test_declared_identity_digest_drift_is_rejected() -> None:
    request = _request()
    payload = request.model_dump(mode="json")
    payload["candidate_artifact_sha256"] = SHA_C
    with pytest.raises(ValidationError, match="request_id"):
        EvalLabRequest.model_validate(payload)

    heldout = _heldout()
    manifest_payload = heldout.model_dump(mode="json")
    manifest_payload["corpus_sha256"] = SHA_A
    with pytest.raises(ValidationError, match="manifest_id"):
        FrozenHeldOutManifest.model_validate(manifest_payload)


@pytest.mark.parametrize(
    "partition", [DataPartition.UNSET, DataPartition.CALIBRATION]
)
def test_unset_or_calibration_partition_is_rejected(partition: DataPartition) -> None:
    with pytest.raises(ValidationError, match="held_out_evaluation"):
        _heldout(data_partition=partition)


def test_sample_leakage_and_untracked_tasks_are_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        _heldout(calibration_sample_ids=["cal-1", "eval-2"])
    with pytest.raises(ValidationError, match="not pinned"):
        _request(tasks=["arc_easy", "untracked_task"])
    with pytest.raises(ValidationError):
        _heldout(task_suite_revision="main")
    with pytest.raises(ValidationError):
        _heldout(task_suite_sha256="unpinned")


def test_candidate_report_cannot_claim_teacher_relative_metrics() -> None:
    report = _report(_request())
    assert report.teacher_relative is False
    assert report.token_kld is None
    assert report.cka is None
    payload = report.model_dump(mode="json")
    payload["teacher_relative"] = True
    payload["token_kld"] = 0.1
    payload["cka"] = 0.9
    with pytest.raises(ValidationError):
        CandidateTaskReport.model_validate(payload)

    payload = report.model_dump(mode="json")
    payload["data_partition"] = DataPartition.CALIBRATION
    with pytest.raises(ValidationError):
        CandidateTaskReport.model_validate(payload)


def test_candidate_report_requires_explicit_teacher_blockers() -> None:
    report = _report(_request())
    payload = report.model_dump(mode="json")
    payload["teacher_relative_blockers"] = ["missing_bf16_teacher"]
    with pytest.raises(ValidationError, match="all teacher-relative blockers"):
        CandidateTaskReport.model_validate(payload)


def _pinned_checkout(tmp_path: Path, *, held_out: bool = True) -> EvalLabRequest:
    root = (tmp_path / "eval-lab").resolve()
    suite = root / "configs" / "suites" / "atlas.yaml"
    tasks = root / "tasks"
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text(f"{EVAL_LAB_REVISION}\n")
    suite.parent.mkdir(parents=True)
    suite.write_text(
        "tasks:\n"
        "  - task_id: arc_easy\n"
        "  - task_id: gsm8k\n"
    )
    partition = "held_out_evaluation" if held_out else "unset"
    for name in ("arc_easy", "gsm8k"):
        task_dir = tasks / name
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(
            f"id: {name}\n"
            f"data_partition: {partition}\n"
            "execution:\n"
            "  timeout_seconds: 300\n"
        )
    heldout_manifest = _heldout(
        task_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
        task_definitions_sha256=canonical_directory_sha256(tasks),
    )
    return _request(
        held_out=heldout_manifest,
        eval_lab_root=str(root),
        suite_ref=str(suite),
        tasks_dir=str(tasks),
        runs_root=str((tmp_path / "runs").resolve()),
        db_path=str((tmp_path / "runstore.db").resolve()),
    )


def test_adapter_emits_exact_pinned_cli_argv_and_validates_result(tmp_path: Path) -> None:
    request = _pinned_checkout(tmp_path)
    layout = eval_lab_output_layout(tmp_path.resolve(), request.request_id)
    layout["root"].mkdir(parents=True)
    _write_json(layout["request"], request)

    handoff = EvalLabAdapter("/opt/eval-lab/bin/eval-lab").emit_argv(request)
    assert handoff.eval_lab_revision == EVAL_LAB_REVISION
    assert handoff.executable is False
    assert handoff.blockers == [HandoffBlocker.REQUEST_PARAMETERS_NOT_CLI_BOUND]
    assert handoff.argv == (
        "/opt/eval-lab/bin/eval-lab",
        "run",
        "suite",
        request.suite_ref,
        "--model",
        "glm52-atlas-candidate",
        "--endpoint",
        "http://127.0.0.1:8100/v1",
        "--model-name",
        "glm-5.2",
        "--tasks-dir",
        request.tasks_dir,
        "--runs-root",
        request.runs_root,
        "--db",
        request.db_path,
        "--json",
    )

    report = _report(request)
    _write_json(layout["report"], report)
    report_sha = hashlib.sha256(layout["report"].read_bytes()).hexdigest()
    result = EvalLabResult(
        request_id=request.request_id,
        report_path=str(layout["report"]),
        report_sha256=report_sha,
        status=EvalLabStatus.COMPLETED,
    )
    _write_json(layout["result"], result)
    validated, validated_report = EvalLabAdapter().validate_result(
        request, result_path=layout["result"]
    )
    assert validated.result_digest == result.result_digest
    assert validated_report == report


def test_handoff_fails_closed_on_cli_parameter_or_config_mismatch(tmp_path: Path) -> None:
    request = _pinned_checkout(tmp_path)
    bad_parameters = request.model_dump(mode="json")
    bad_parameters["request_id"] = None
    bad_parameters["parameters"]["temperature"] = 0.7
    with pytest.raises(ValueError, match="temperature=0.0"):
        EvalLabAdapter().emit_argv(EvalLabRequest.model_validate(bad_parameters))

    bad_hash = request.model_dump(mode="json")
    bad_hash["request_id"] = None
    bad_hash["held_out"]["manifest_id"] = None
    bad_hash["held_out"]["task_definitions_sha256"] = SHA_A
    with pytest.raises(ValueError, match="tasks tree"):
        EvalLabAdapter().emit_argv(EvalLabRequest.model_validate(bad_hash))


def test_handoff_blocks_unproven_task_partition_and_wrong_git_head(tmp_path: Path) -> None:
    request = _pinned_checkout(tmp_path, held_out=False)
    handoff = EvalLabAdapter().emit_argv(request)
    assert handoff.executable is False
    assert handoff.blockers == [
        HandoffBlocker.REQUEST_PARAMETERS_NOT_CLI_BOUND,
        HandoffBlocker.TASK_PARTITION_NOT_HELD_OUT,
    ]

    head = Path(request.eval_lab_root) / ".git" / "refs" / "heads" / "main"
    head.write_text(f"{'e' * 40}\n")
    with pytest.raises(ValueError, match="Git HEAD"):
        EvalLabAdapter().emit_argv(request)


def test_forged_result_or_report_digest_is_rejected(tmp_path: Path) -> None:
    request = _request()
    report_path = tmp_path / "report.json"
    _write_json(report_path, _report(request))
    result = EvalLabResult(
        request_id=request.request_id,
        report_path=str(report_path),
        report_sha256=SHA_A,
        status=EvalLabStatus.COMPLETED,
    )
    result_path = tmp_path / "result.json"
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="report digest"):
        EvalLabAdapter().validate_result(request, result_path=result_path)

    payload = result.model_dump(mode="json")
    payload["status"] = "failed"
    payload["errors"] = ["endpoint failed"]
    with pytest.raises(ValidationError, match="result_digest"):
        EvalLabResult.model_validate(payload)


def test_contract_has_no_secret_or_api_key_fields() -> None:
    endpoint = {
        "endpoint_id": "candidate",
        "transport": "openai_compatible",
        "endpoint_url": "http://127.0.0.1:8100/v1",
        "config_sha256": SHA_A,
        "api_key": "must-not-cross-contract",
    }
    with pytest.raises(ValidationError):
        EndpointConfigIdentity.model_validate(endpoint)
    serialized = json.dumps(_request().model_dump(mode="json"))
    assert "api_key" not in serialized
    assert "secret" not in serialized


def test_endpoint_url_rejects_embedded_credentials_or_query_secret() -> None:
    for endpoint_url in (
        "http://user:password@127.0.0.1:8100/v1",
        "http://127.0.0.1:8100/v1?api_key=secret",
    ):
        with pytest.raises(ValidationError):
            EndpointConfigIdentity(
                endpoint_id="candidate",
                transport=EndpointTransport.OPENAI_COMPATIBLE,
                endpoint_url=endpoint_url,
                config_sha256=SHA_A,
            )
