from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_atlas.fit_telemetry import CanaryPhase, CanaryStep
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    LlamaCppRpcRuntimeConfig,
)
from model_atlas.runtime_canary_driver import (
    CanaryRequestClient,
    HttpResponse,
    RuntimeDriverError,
    SystemdUserRuntimeLifecycle,
)


def _config() -> LlamaCppRpcRuntimeConfig:
    return LlamaCppRpcRuntimeConfig(
        artifact_path=Path("/artifacts/glm52.gguf"),
        artifact_sha256="a" * 64,
        llama_server_path=Path("/tools/llama-server"),
        worker_rpc_server_path=Path("/tools/ggml-rpc-server"),
    )


class _Runner:
    def __init__(self, config: LlamaCppRpcRuntimeConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, ...]] = []
        self.fail_head_start = False
        self.worker_pid = 202

    def __call__(self, argv: object) -> str:
        command = tuple(argv)  # type: ignore[arg-type]
        self.calls.append(command)
        worker = "10.77.0.2" in command
        if self.fail_head_start and command[:1] == ("systemd-run",):
            raise RuntimeError("no details")
        if any("MainPID" in item for item in command):
            return f"{self.worker_pid}\n" if worker else "101\n"
        if "readlink" in command:
            return (
                str(self.config.worker_rpc_server_path)
                if worker
                else str(self.config.llama_server_path)
            ) + "\n"
        if "sha256sum" in command:
            path = command[-1]
            digest = EXPECTED_RPC_SERVER_SHA256 if worker else EXPECTED_LLAMA_SERVER_SHA256
            return f"{digest}  {path}\n"
        return ""


def test_lifecycle_starts_worker_then_head_measures_pids_and_stops_head_first() -> None:
    config = _config()
    runner = _Runner(config)
    lifecycle = SystemdUserRuntimeLifecycle(
        config,
        worker_ssh_target="10.77.0.2",
        runner=runner,
        health_ready=lambda: True,
    )
    pids = lifecycle.start()
    assert pids.head_server_pid == 101
    assert pids.worker_rpc_pid == 202
    assert lifecycle.launch_evidence is not None
    assert lifecycle.launch_evidence.head_argv == config.head_argv()
    assert lifecycle.launch_evidence.worker_argv == config.worker_argv()
    starts = [item for item in runner.calls if "systemd-run" in item]
    assert "10.77.0.2" in starts[0]
    assert starts[1][:1] == ("systemd-run",)
    assert any("readlink" in item and "10.77.0.2" not in item for item in runner.calls)
    assert any("readlink" in item and "10.77.0.2" in item for item in runner.calls)
    post = lifecycle.measure_post_run_worker(pids)
    assert post.worker_pid == 202
    lifecycle.stop()
    stops = [
        item
        for item in runner.calls
        if item[-1:] == ("atlas-glm52-rpc-head",) or item[-1:] == ("atlas-glm52-rpc-worker",)
    ]
    # The final two service transitions are exactly head then worker.
    assert stops[-2][-1] == "atlas-glm52-rpc-head"
    assert stops[-1][-1] == "atlas-glm52-rpc-worker"


def test_lifecycle_fails_closed_and_rolls_back_partial_start() -> None:
    config = _config()
    runner = _Runner(config)
    runner.fail_head_start = True
    lifecycle = SystemdUserRuntimeLifecycle(
        config,
        worker_ssh_target="10.77.0.2",
        runner=runner,
        health_ready=lambda: True,
    )
    with pytest.raises(RuntimeDriverError, match="runtime service command failed"):
        lifecycle.start()
    assert any(item[-1:] == ("atlas-glm52-rpc-worker",) and "stop" in item for item in runner.calls)
    assert lifecycle.launch_evidence is None


def test_lifecycle_rejects_worker_pid_change_after_run() -> None:
    config = _config()
    runner = _Runner(config)
    lifecycle = SystemdUserRuntimeLifecycle(
        config,
        worker_ssh_target="10.77.0.2",
        runner=runner,
        health_ready=lambda: True,
    )
    pids = lifecycle.start()
    runner.worker_pid = 303
    with pytest.raises(RuntimeDriverError, match="MainPID changed"):
        lifecycle.measure_post_run_worker(pids)


class _Http:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((method, url, body))
        if url.endswith("/v1/chat/completions"):
            return HttpResponse(
                200,
                json.dumps(
                    {
                        "timings": {
                            "prompt_n": 4,
                            "prompt_ms": 2,
                            "predicted_n": 2,
                            "predicted_ms": 1,
                        }
                    }
                ).encode(),
            )
        return HttpResponse(200, b"CUDA0 RPC0")


def test_request_client_honors_repeat_warmup_seed_temperature_and_writes_no_prompt_evidence() -> (
    None
):
    transport = _Http()
    client = CanaryRequestClient(_config(), transport=transport)
    step = CanaryStep(
        step_id="throughput",
        phase=CanaryPhase.THROUGHPUT,
        context_tokens=4096,
        max_output_tokens=64,
        repeats=4,
        discard_warmup_repeats=1,
        restart_runtime=False,
    )
    observation = client.execute(step, deterministic_seed=17)
    completion_payloads = [
        body for _, url, body in transport.calls if url.endswith("/v1/chat/completions")
    ]
    assert len(completion_payloads) == 4
    assert all(
        json.loads(body or b"{}")
        == {
            "max_tokens": 64,
            "messages": [{"content": "Atlas canary.", "role": "user"}],
            "model": "glm52-mixed-gguf",
            "seed": 17,
            "stream": False,
            "temperature": 0,
        }
        for body in completion_payloads
    )
    assert observation.prompt_tps == 2000.0
    assert observation.decode_tps == 2000.0
    evidence = client.drain_evidence()
    serialized = "\n".join(item.model_dump_json() for item in evidence)
    assert "Atlas canary" not in serialized
    assert [
        item.measured_repeat for item in evidence if item.endpoint == "/v1/chat/completions"
    ] == [None, 1, 2, 3]
    assert {item.endpoint for item in evidence} >= {"/health", "/v1/models", "/metrics"}


def test_request_client_fails_when_endpoint_is_not_ready() -> None:
    class _Bad(_Http):
        def request(
            self,
            method: str,
            url: str,
            *,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
        ) -> HttpResponse:
            return HttpResponse(503, b"no")

    step = CanaryStep(
        step_id="load",
        phase=CanaryPhase.LOAD_ONLY,
        context_tokens=4096,
        max_output_tokens=0,
        repeats=1,
        discard_warmup_repeats=0,
        restart_runtime=True,
    )
    with pytest.raises(RuntimeDriverError, match="prerequisite"):
        CanaryRequestClient(_config(), transport=_Bad()).execute(step, deterministic_seed=0)
