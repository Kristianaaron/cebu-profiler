from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from model_atlas.canary_constants import WORKER_TRANSIENT_UNIT
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_RPC_SERVER_SHA256,
    PINNED_COMMIT,
    LlamaCppRpcRuntimeConfig,
)
from model_atlas.worker_rpc_lifecycle import (
    WorkerRpcLifecycleError,
    WorkerRpcSystemdLifecycle,
)


def _config() -> LlamaCppRpcRuntimeConfig:
    return LlamaCppRpcRuntimeConfig(
        artifact_path=Path("/artifacts/glm52.gguf"),
        artifact_sha256="a" * 64,
        worker_rpc_server_path=Path("/tools/ggml-rpc-server"),
    )


class _Runner:
    def __init__(self, config: LlamaCppRpcRuntimeConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, ...]] = []
        self.pid = 202
        self.digest = EXPECTED_RPC_SERVER_SHA256
        self.argv = config.worker_argv()
        self.commit = PINNED_COMMIT
        self.fail_readlink = False

    def __call__(self, argv: object) -> str:
        command: tuple[str, ...] = tuple(argv)  # type: ignore[arg-type]
        self.calls.append(command)
        if self.fail_readlink and "readlink" in command:
            raise RuntimeError("sensitive remote detail")
        if any("MainPID" in item for item in command):
            return f"{self.pid}\n"
        if "readlink" in command:
            return f"{self.config.worker_rpc_server_path}\n"
        if "sha256sum" in command:
            return f"{self.digest}  {self.config.worker_rpc_server_path}\n"
        if "cat" in command and any("/proc/" in item for item in command):
            return "\0".join(self.argv) + "\0"
        if "rev-parse" in command:
            return f"{self.commit}\n"
        return ""


def _lifecycle(runner: _Runner) -> WorkerRpcSystemdLifecycle:
    return WorkerRpcSystemdLifecycle(
        runner.config,
        worker_ssh_target="10.77.0.2",
        runner=runner,
        toolchain_root=Path("/toolchain/llama.cpp"),
    )


def test_starts_only_worker_and_returns_frozen_measured_evidence() -> None:
    runner = _Runner(_config())
    lifecycle = _lifecycle(runner)

    evidence = lifecycle.start()

    assert evidence.worker_pid == 202
    assert evidence.worker_argv == runner.config.worker_argv()
    assert evidence.worker_exe_sha256 == EXPECTED_RPC_SERVER_SHA256
    assert evidence.worker_git_commit == PINNED_COMMIT
    starts = [call for call in runner.calls if "systemd-run" in call]
    assert len(starts) == 1
    assert starts[0][-len(runner.config.worker_argv()) :] == runner.config.worker_argv()
    assert all("llama-server" not in value for call in runner.calls for value in call)
    with pytest.raises(ValidationError):
        evidence.worker_pid = 303

    lifecycle.stop()
    assert runner.calls[-1][-4:] == (
        "systemctl",
        "--user",
        "stop",
        WORKER_TRANSIENT_UNIT,
    )


def test_remeasures_exact_same_process_after_each_capture() -> None:
    runner = _Runner(_config())
    lifecycle = _lifecycle(runner)
    launch = lifecycle.start()

    assert lifecycle.remeasure_after_capture() == launch
    assert lifecycle.remeasure_after_capture() == launch
    pid_probes = [call for call in runner.calls if any("MainPID" in value for value in call)]
    assert len(pid_probes) == 3


@pytest.mark.parametrize("mutation", ["pid", "digest", "argv", "commit"])
def test_remeasurement_fails_closed_on_identity_change(mutation: str) -> None:
    runner = _Runner(_config())
    lifecycle = _lifecycle(runner)
    lifecycle.start()
    if mutation == "pid":
        runner.pid = 303
    elif mutation == "digest":
        runner.digest = "b" * 64
    elif mutation == "argv":
        runner.argv = runner.argv + ("--unexpected",)
    else:
        runner.commit = "b" * 40

    with pytest.raises(WorkerRpcLifecycleError):
        lifecycle.remeasure_after_capture()


def test_attestation_failure_is_sanitized_and_rolls_back_worker() -> None:
    runner = _Runner(_config())
    runner.fail_readlink = True
    lifecycle = _lifecycle(runner)

    with pytest.raises(WorkerRpcLifecycleError, match="worker RPC service command failed") as exc:
        lifecycle.start()

    assert "sensitive" not in str(exc.value)
    assert any(call[-1:] == (WORKER_TRANSIENT_UNIT,) and "stop" in call for call in runner.calls)
    assert lifecycle.launch_evidence is None


def test_stop_failure_clears_local_state_and_is_sanitized() -> None:
    config = _config()

    class _StopFailRunner(_Runner):
        def __call__(self, argv: object) -> str:
            command: tuple[str, ...] = tuple(argv)  # type: ignore[arg-type]
            if "stop" in command:
                raise RuntimeError("private stop output")
            return super().__call__(command)

    runner = _StopFailRunner(config)
    lifecycle = _lifecycle(runner)
    lifecycle.start()

    with pytest.raises(WorkerRpcLifecycleError, match="worker RPC service command failed") as exc:
        lifecycle.stop()
    assert "private" not in str(exc.value)
    assert lifecycle.launch_evidence is None
    with pytest.raises(WorkerRpcLifecycleError, match="not started"):
        lifecycle.remeasure_after_capture()
