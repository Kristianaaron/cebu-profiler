from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

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
from model_atlas.evaluation.glm52_candidate_eval import CandidateEvalPlan
from model_atlas.evaluation.glm52_candidate_eval_driver import (
    CandidateEvalExecutionError,
    CandidateEvalExecutor,
    EvalLabCommandResult,
    SystemdRuntimeQuiescenceVerifier,
)
from model_atlas.fit_telemetry import CanaryStep
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    LlamaCppRpcRuntimeConfig,
)
from model_atlas.runtime_canary_driver import HttpResponse, RuntimeLaunchEvidence
from model_atlas.two_node_canary_executor import RuntimeProcessIds


def _config() -> LlamaCppRpcRuntimeConfig:
    return LlamaCppRpcRuntimeConfig(
        artifact_path=Path("/artifacts/glm52-candidate.gguf"),
        artifact_sha256="a" * 64,
        llama_server_path=Path("/tools/llama-server"),
        worker_rpc_server_path=Path("/tools/ggml-rpc-server"),
        api_port=8892,
    )


def _plan(tmp_path: Path, config: LlamaCppRpcRuntimeConfig) -> CandidateEvalPlan:
    root = (tmp_path / "eval-lab").resolve()
    suite = root / "configs" / "suites" / "heldout.yaml"
    tasks_dir = root / "tasks"
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text(
        "a20da6c6b9cbf872f7c083bffe66afde40c2c8f2\n"
    )
    suite.parent.mkdir(parents=True)
    suite.write_text(
        "tasks:\n  - task_id: synthetic.math\n  - task_id: synthetic.logic\n"
    )
    tasks_dir.mkdir()
    for task_id in ("synthetic.math", "synthetic.logic"):
        (tasks_dir / f"{task_id}.yaml").write_text(
            f"id: {task_id}\n"
            "data_partition: held_out_evaluation\n"
            "execution:\n  runner: direct\n  timeout_seconds: 30\n"
        )
    held_out = FrozenHeldOutManifest(
        data_partition=DataPartition.HELD_OUT_EVALUATION,
        task_suite_id="atlas-glm52-heldout",
        task_suite_revision="c" * 40,
        task_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
        task_definitions_sha256=canonical_directory_sha256(tasks_dir),
        tracked_task_ids=["synthetic.math", "synthetic.logic"],
        corpus_sha256="d" * 64,
        tokenizer_sha256="e" * 64,
        template_sha256="f" * 64,
        evaluation_sample_ids=["heldout-1", "heldout-2"],
    )
    request = EvalLabRequest(
        candidate_artifact_path=str(config.artifact_path),
        candidate_artifact_sha256=config.artifact_sha256,
        endpoint=EndpointConfigIdentity(
            endpoint_id="candidate",
            transport=EndpointTransport.OPENAI_COMPATIBLE,
            endpoint_url=f"http://{config.api_host}:{config.api_port}/v1",
            config_sha256=config.canonical_sha256(),
        ),
        held_out=held_out,
        tasks=["synthetic.math", "synthetic.logic"],
        parameters=EvalParameters(
            seed=17, temperature=0.0, max_tokens=96, timeout_seconds=30
        ),
        eval_lab_root=str(root),
        suite_ref=str(suite),
        tasks_dir=str(tasks_dir),
        runs_root=str((tmp_path / "runs").resolve()),
        db_path=str((tmp_path / "runs.db").resolve()),
        model_id="glm52-candidate",
        model_name="glm52-candidate",
    )
    executable = root / "fake-eval-lab"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    argv = EvalLabAdapter(executable=str(executable)).emit_argv(request).argv
    return CandidateEvalPlan(
        compression_handoff_sha256="1" * 64,
        runtime_canary_handoff_sha256="2" * 64,
        runtime_canary_plan_sha256="3" * 64,
        runtime_config_sha256=config.canonical_sha256(),
        candidate_artifact_path=str(config.artifact_path),
        candidate_artifact_sha256=config.artifact_sha256,
        eval_lab_executable_sha256=executable_sha256,
        held_out_manifest_id=request.held_out.manifest_id,
        task_suite_sha256=request.held_out.task_suite_sha256,
        task_definitions_sha256=request.held_out.task_definitions_sha256,
        tokenizer_sha256=request.held_out.tokenizer_sha256,
        template_sha256=request.held_out.template_sha256,
        parameters=request.parameters,
        eval_request=request,
        argv=argv,
    )


def _launch(config: LlamaCppRpcRuntimeConfig) -> RuntimeLaunchEvidence:
    return RuntimeLaunchEvidence(
        head_pid=101,
        worker_pid=202,
        head_argv=config.head_argv(),
        worker_argv=config.worker_argv(),
        head_exe_path=str(config.llama_server_path),
        head_exe_sha256=EXPECTED_LLAMA_SERVER_SHA256,
        worker_exe_path=str(config.worker_rpc_server_path),
        worker_exe_sha256=EXPECTED_RPC_SERVER_SHA256,
    )


class _Lifecycle:
    def __init__(self, config: LlamaCppRpcRuntimeConfig) -> None:
        self.launch_evidence: RuntimeLaunchEvidence | None = _launch(config)
        self.events: list[str] = []
        self.post = self.launch_evidence
        self.stop_error = False

    def start(self, step: CanaryStep) -> RuntimeProcessIds:
        assert step.context_tokens == 4096
        self.events.extend(["start-worker", "start-head"])
        return RuntimeProcessIds(head_server_pid=101, worker_rpc_pid=202)

    def measure_post_run_worker(self, pids: RuntimeProcessIds) -> RuntimeLaunchEvidence:
        self.events.append("attest-worker")
        assert self.post is not None
        return self.post

    def stop(self) -> None:
        self.events.extend(["stop-head", "stop-worker"])
        if self.stop_error:
            raise RuntimeError("secret stop detail")


class _Http:
    def __init__(self) -> None:
        self.status = 200
        self.calls: list[tuple[str, str]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((method, url))
        if url.endswith("/models"):
            return HttpResponse(
                self.status,
                json.dumps({"data": [{"id": "glm52-candidate"}]}).encode(),
            )
        return HttpResponse(self.status, b"healthy")


def _write_run(plan: CandidateEvalPlan, task_id: str, run_id: str) -> None:
    run = Path(plan.eval_request.runs_root) / run_id
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task_id": task_id,
                "model_id": plan.eval_request.model_id,
                "random_seed": 17,
                "sampling": {"temperature": 0.0, "max_tokens": 96},
                "budgets": {"timeout_seconds": 30.0, "http_timeout_seconds": 30.0},
                "result_status": "completed",
            }
        )
    )
    (run / "result.json").write_text(
        json.dumps({"run_id": run_id, "error": None, "duration_s": 1.5})
    )
    (run / "scores.jsonl").write_text(
        json.dumps({"scorer_id": "exact", "score": 1.0}) + "\n"
    )


class _Command:
    def __init__(self, plan: CandidateEvalPlan) -> None:
        self.plan = plan
        self.returncode = 0
        self.extra_run = False
        self.calls: list[tuple[tuple[str, ...], Path, float]] = []

    def run(
        self, argv: Any, *, cwd: Path, timeout_seconds: float
    ) -> EvalLabCommandResult:
        command = tuple(argv)
        self.calls.append((command, cwd, timeout_seconds))
        if self.returncode == 0:
            rows = []
            for task, run_id in zip(
                self.plan.eval_request.tasks, ("0" * 12, "1" * 12), strict=True
            ):
                _write_run(self.plan, task, run_id)
                rows.append({"run_id": run_id, "task_id": task, "secret": "not-returned"})
            if self.extra_run:
                (Path(self.plan.eval_request.runs_root) / ("2" * 12)).mkdir()
            stdout = json.dumps(rows).encode()
        else:
            stdout = b"secret failure output"
        return EvalLabCommandResult(self.returncode, stdout)


class _Quiescence:
    def __init__(self, lifecycle: _Lifecycle) -> None:
        self.lifecycle = lifecycle
        self.fail = False

    def verify(self, pids: RuntimeProcessIds) -> None:
        self.lifecycle.events.append("verify-quiescent")
        if self.fail:
            raise RuntimeError("secret process detail")


def _executor(
    plan: CandidateEvalPlan,
    config: LlamaCppRpcRuntimeConfig,
    lifecycle: _Lifecycle,
    http: _Http,
    command: _Command,
    quiescence: _Quiescence,
) -> CandidateEvalExecutor:
    return CandidateEvalExecutor(
        config,
        lifecycle=lifecycle,
        transport=http,
        command_runner=command,
        quiescence=quiescence,
        eval_lab_cwd=Path(plan.eval_request.eval_lab_root),
    )


def test_executes_exact_plan_and_returns_verified_typed_result(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(plan)
    quiescence = _Quiescence(lifecycle)
    result = _executor(plan, config, lifecycle, http, command, quiescence).execute(plan)

    assert command.calls == [
        (plan.argv, Path(plan.eval_request.eval_lab_root), 120.0)
    ]
    assert http.calls == [
        ("GET", "http://127.0.0.1:8892/health"),
        ("GET", "http://127.0.0.1:8892/v1/models"),
    ]
    assert lifecycle.events == [
        "start-worker",
        "start-head",
        "attest-worker",
        "stop-head",
        "stop-worker",
        "verify-quiescent",
    ]
    assert result.plan_sha256 == plan.plan_sha256
    assert tuple(item.task_id for item in result.task_evidence) == tuple(
        plan.eval_request.tasks
    )
    assert result.stdout_size_bytes > 0
    assert not hasattr(result, "stdout")
    assert result.candidate_result.report.performance.tokens_per_second == 0.0


def test_nonzero_exit_is_sanitized_and_still_attested_stopped_quiesced(
    tmp_path: Path,
) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(plan)
    command.returncode = 9
    quiescence = _Quiescence(lifecycle)
    with pytest.raises(CandidateEvalExecutionError, match="exited nonzero") as caught:
        _executor(plan, config, lifecycle, http, command, quiescence).execute(plan)
    assert "secret" not in str(caught.value)
    assert lifecycle.events[-4:] == [
        "attest-worker",
        "stop-head",
        "stop-worker",
        "verify-quiescent",
    ]


def test_rejects_launch_drift_before_eval_and_still_cleans_up(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(plan)
    assert lifecycle.launch_evidence is not None
    lifecycle.launch_evidence = lifecycle.launch_evidence.model_copy(
        update={"head_argv": ("wrong",)}
    )
    lifecycle.post = lifecycle.launch_evidence
    quiescence = _Quiescence(lifecycle)
    with pytest.raises(CandidateEvalExecutionError, match="launch measurement drifted"):
        _executor(plan, config, lifecycle, http, command, quiescence).execute(plan)
    assert command.calls == []
    assert lifecycle.events[-2:] == ["stop-worker", "verify-quiescent"]


def test_rejects_post_run_identity_drift(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(plan)
    assert lifecycle.post is not None
    lifecycle.post = lifecycle.post.model_copy(update={"worker_pid": 303})
    quiescence = _Quiescence(lifecycle)
    with pytest.raises(CandidateEvalExecutionError, match="post-evaluation worker"):
        _executor(plan, config, lifecycle, http, command, quiescence).execute(plan)
    assert lifecycle.events[-2:] == ["stop-worker", "verify-quiescent"]


@pytest.mark.parametrize("stop_fails,quiescence_fails", [(True, False), (False, True)])
def test_stop_or_quiescence_uncertainty_overrides_success(
    tmp_path: Path, stop_fails: bool, quiescence_fails: bool
) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(plan)
    lifecycle.stop_error = stop_fails
    quiescence = _Quiescence(lifecycle)
    quiescence.fail = quiescence_fails
    with pytest.raises(CandidateEvalExecutionError, match="quiescence is uncertain"):
        _executor(plan, config, lifecycle, http, command, quiescence).execute(plan)


def test_fails_closed_on_extra_run_directory(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(plan)
    command.extra_run = True
    quiescence = _Quiescence(lifecycle)
    with pytest.raises(CandidateEvalExecutionError, match="discovery is ambiguous"):
        _executor(plan, config, lifecycle, http, command, quiescence).execute(plan)


def test_rejects_endpoint_config_drift_before_start(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    request = plan.eval_request.model_copy(
        update={
            "endpoint": plan.eval_request.endpoint.model_copy(
                update={"config_sha256": "f" * 64}
            )
        }
    )
    drifted = plan.model_copy(update={"eval_request": request})
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(drifted)
    quiescence = _Quiescence(lifecycle)
    with pytest.raises(CandidateEvalExecutionError, match="endpoint differs"):
        _executor(drifted, config, lifecycle, http, command, quiescence).execute(drifted)
    assert lifecycle.events == []


def test_health_failure_still_attests_and_quiesces_without_running_eval(tmp_path: Path) -> None:
    config = _config()
    plan = _plan(tmp_path, config)
    lifecycle, http, command = _Lifecycle(config), _Http(), _Command(plan)
    http.status = 503
    quiescence = _Quiescence(lifecycle)
    with pytest.raises(CandidateEvalExecutionError, match="health verification failed"):
        _executor(plan, config, lifecycle, http, command, quiescence).execute(plan)
    assert command.calls == []
    assert lifecycle.events[-4:] == [
        "attest-worker",
        "stop-head",
        "stop-worker",
        "verify-quiescent",
    ]


def test_systemd_quiescence_verifier_checks_head_then_worker_units_and_pids() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: Any) -> str:
        command = tuple(argv)
        calls.append(command)
        return "0\n" if any("MainPID" in item for item in command) else ""

    verifier = SystemdRuntimeQuiescenceVerifier(
        worker_ssh_target="10.77.0.2", runner=runner
    )
    verifier.verify(RuntimeProcessIds(head_server_pid=101, worker_rpc_pid=202))
    assert calls[0][-1] == "atlas-glm52-rpc-head"
    assert calls[1][-1] == "/proc/101"
    assert "10.77.0.2" in calls[2]
    assert calls[2][-1] == "atlas-glm52-rpc-worker"
    assert calls[3][-1] == "/proc/202"
