"""Concrete, fail-closed I/O adapters for the two-Spark canary.

The executor owns plan ordering and evidence persistence.  This module owns
only process lifecycle and loopback requests, each through injectable protocols
so importing it (and every unit test) is side-effect free.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.fit_telemetry import CanaryStep, StepObservation
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    LlamaCppRpcRuntimeConfig,
)
from model_atlas.two_node_canary_executor import RuntimeProcessIds

__all__ = [
    "CanaryHttpEvidence",
    "CanaryRequestClient",
    "HttpResponse",
    "LoopbackHttpTransport",
    "RuntimeDriverError",
    "RuntimeLaunchEvidence",
    "SystemdUserRuntimeLifecycle",
]


class RuntimeDriverError(RuntimeError):
    """A sanitized, fail-closed external-boundary failure."""


class ArgvRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str: ...


class HealthCheck(Protocol):
    def __call__(self) -> bool: ...


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse: ...


class LoopbackHttpTransport:
    """Minimal stdlib transport.  Construct it only in the execute entry point."""

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - config is loopback
                return HttpResponse(response.status, response.read(1 << 20))
        except urllib.error.HTTPError as exc:
            return HttpResponse(exc.code, exc.read(1 << 20))
        except OSError as exc:
            raise RuntimeDriverError("loopback HTTP unavailable") from exc


class CanaryHttpEvidence(BaseModel):
    """Persistable HTTP evidence containing no request/response content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    step_id: str = Field(min_length=1)
    context_tokens: int = Field(ge=0)
    endpoint: Literal["/health", "/v1/models", "/metrics", "/v1/chat/completions"]
    status: int = Field(ge=100, le=599)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured_repeat: int | None = Field(default=None, ge=0)


class CanaryRequestClient:
    """Deterministic OpenAI-compatible loopback canary with prompt-free evidence."""

    def __init__(self, config: LlamaCppRpcRuntimeConfig, *, transport: HttpTransport) -> None:
        self._transport = transport
        self._base_url = f"http://{config.api_host}:{config.api_port}"
        self._evidence: list[CanaryHttpEvidence] = []

    def drain_evidence(self) -> tuple[CanaryHttpEvidence, ...]:
        result = tuple(self._evidence)
        self._evidence.clear()
        return result

    def health_ready(self) -> bool:
        return self._call("/health", step_id="runtime-health", context_tokens=0).status < 300

    def _call(
        self,
        endpoint: Literal["/health", "/v1/models", "/metrics", "/v1/chat/completions"],
        *,
        step_id: str,
        context_tokens: int,
        payload: bytes | None = None,
        measured_repeat: int | None = None,
    ) -> HttpResponse:
        response = self._transport.request(
            "POST" if endpoint == "/v1/chat/completions" else "GET",
            self._base_url + endpoint,
            body=payload,
            headers={"Content-Type": "application/json"} if payload is not None else None,
        )
        self._evidence.append(
            CanaryHttpEvidence(
                step_id=step_id,
                context_tokens=context_tokens,
                endpoint=endpoint,
                status=response.status,
                response_sha256=hashlib.sha256(response.body).hexdigest(),
                measured_repeat=measured_repeat,
            )
        )
        return response

    @staticmethod
    def _rates(body: bytes) -> tuple[float | None, float | None]:
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeDriverError("canary response is not JSON") from exc
        timings = value.get("timings") if isinstance(value, dict) else None
        if not isinstance(timings, dict):
            return None, None

        def rate(count_key: str, duration_key: str) -> float | None:
            count, milliseconds = timings.get(count_key), timings.get(duration_key)
            if (
                isinstance(count, (int, float))
                and isinstance(milliseconds, (int, float))
                and milliseconds > 0
            ):
                return float(count) * 1000.0 / float(milliseconds)
            return None

        return rate("prompt_n", "prompt_ms"), rate("predicted_n", "predicted_ms")

    def execute(self, step: CanaryStep, *, deterministic_seed: int) -> StepObservation:
        health = self._call("/health", step_id=step.step_id, context_tokens=step.context_tokens)
        models = self._call("/v1/models", step_id=step.step_id, context_tokens=step.context_tokens)
        metrics = self._call("/metrics", step_id=step.step_id, context_tokens=step.context_tokens)
        if any(item.status >= 300 for item in (health, models, metrics)):
            raise RuntimeDriverError("canary prerequisite endpoint failed")
        surfaces = b" ".join((health.body, models.body, metrics.body))
        devices = tuple(name for name in ("CUDA0", "RPC0") if name.encode() in surfaces)
        # The fixed synthetic content prevents user text or private prompts from
        # crossing this boundary.  It is never written to evidence or stdout.
        payload = json.dumps(
            {
                "model": "glm52-mixed-gguf",
                "messages": [{"role": "user", "content": "Atlas canary."}],
                "temperature": 0,
                "seed": deterministic_seed,
                "max_tokens": step.max_output_tokens,
                "stream": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        prompt_rates: list[float] = []
        decode_rates: list[float] = []
        for repeat in range(step.repeats):
            measured = repeat if repeat >= step.discard_warmup_repeats else None
            response = self._call(
                "/v1/chat/completions",
                step_id=step.step_id,
                context_tokens=step.context_tokens,
                payload=payload,
                measured_repeat=measured,
            )
            if response.status >= 300:
                raise RuntimeDriverError("canary completion failed")
            prompt_tps, decode_tps = self._rates(response.body)
            if measured is not None:
                if prompt_tps is not None:
                    prompt_rates.append(prompt_tps)
                if decode_tps is not None:
                    decode_rates.append(decode_tps)
        return StepObservation(
            step_id=step.step_id,
            context_tokens=step.context_tokens,
            prompt_tps=sum(prompt_rates) / len(prompt_rates) if prompt_rates else None,
            decode_tps=sum(decode_rates) / len(decode_rates) if decode_rates else None,
            observed_devices=devices,
            runtime_succeeded=True,
        )


class RuntimeLaunchEvidence(BaseModel):
    """Exact post-launch identity, measured independently on each host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    head_pid: int = Field(gt=0)
    worker_pid: int = Field(gt=0)
    head_argv: tuple[str, ...]
    worker_argv: tuple[str, ...]
    head_exe_path: str = Field(pattern=r"^/")
    head_exe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_exe_path: str = Field(pattern=r"^/")
    worker_exe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_kind: Literal["measured"] = "measured"


class SystemdUserRuntimeLifecycle:
    """Transient systemd-user worker→head lifecycle with measured process IDs.

    All process and SSH calls pass through ``runner``.  The class intentionally
    has no knowledge of maintenance stopping; an operator must establish the
    maintenance window before calling ``start``.
    """

    def __init__(
        self,
        config: LlamaCppRpcRuntimeConfig,
        *,
        worker_ssh_target: str,
        runner: ArgvRunner,
        health_ready: HealthCheck,
        worker_unit: str = "atlas-glm52-rpc-worker",
        head_unit: str = "atlas-glm52-rpc-head",
        ssh_argv: Sequence[str] = ("ssh", "-o", "BatchMode=yes"),
        max_health_attempts: int = 30,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not worker_ssh_target or not worker_unit or not head_unit:
            raise ValueError("worker target and unit names are required")
        if not ssh_argv or any(not item for item in ssh_argv) or max_health_attempts < 1:
            raise ValueError("valid SSH argv and positive health attempts are required")
        self._config, self._target, self._runner = config, worker_ssh_target, runner
        self._health_ready, self._worker_unit, self._head_unit = (
            health_ready,
            worker_unit,
            head_unit,
        )
        self._ssh_argv, self._attempts, self._sleep = tuple(ssh_argv), max_health_attempts, sleep
        self._started = False
        self._launch_evidence: RuntimeLaunchEvidence | None = None
        self._active_config: LlamaCppRpcRuntimeConfig | None = None

    @property
    def launch_evidence(self) -> RuntimeLaunchEvidence | None:
        return self._launch_evidence

    @staticmethod
    def _start_argv(unit: str, server_argv: Sequence[str]) -> tuple[str, ...]:
        return (
            "systemd-run",
            "--user",
            "--unit",
            unit,
            "--service-type=exec",
            "--property=Restart=no",
            "--property=KillMode=control-group",
            "--",
        ) + tuple(server_argv)

    def _remote(self, argv: Sequence[str]) -> tuple[str, ...]:
        return self._ssh_argv + ("--", self._target) + tuple(argv)

    def _run(self, argv: Sequence[str], *, worker: bool = False) -> str:
        try:
            return self._runner(self._remote(argv) if worker else argv)
        except Exception as exc:  # noqa: BLE001 - do not expose command output
            raise RuntimeDriverError("runtime service command failed") from exc

    def _pid(self, unit: str, *, worker: bool) -> int:
        output = self._run(
            ("systemctl", "--user", "show", "--property=MainPID", "--value", unit),
            worker=worker,
        ).strip()
        if not output.isdecimal() or int(output) <= 0:
            raise RuntimeDriverError("runtime MainPID unavailable")
        return int(output)

    def _identity(self, pid: int, *, worker: bool) -> tuple[str, str, tuple[str, ...]]:
        exe = self._run(("readlink", "-f", f"/proc/{pid}/exe"), worker=worker).strip()
        if not exe.startswith("/"):
            raise RuntimeDriverError("runtime executable path unavailable")
        fields = self._run(("sha256sum", exe), worker=worker).strip().split()
        if len(fields) < 2 or len(fields[0]) != 64 or fields[1] != exe:
            raise RuntimeDriverError("runtime executable hash unavailable")
        if any(char not in "0123456789abcdef" for char in fields[0]):
            raise RuntimeDriverError("runtime executable hash unavailable")
        raw_cmdline = self._run(("cat", f"/proc/{pid}/cmdline"), worker=worker)
        argv = tuple(part for part in raw_cmdline.split("\0") if part)
        if not argv:
            raise RuntimeDriverError("runtime command line unavailable")
        return exe, fields[0], argv

    def _measure_launch(
        self, pids: RuntimeProcessIds, config: LlamaCppRpcRuntimeConfig
    ) -> RuntimeLaunchEvidence:
        head_path, head_hash, head_argv = self._identity(pids.head_server_pid, worker=False)
        worker_path, worker_hash, worker_argv = self._identity(pids.worker_rpc_pid, worker=True)
        if (
            head_path != str(self._config.llama_server_path)
            or head_hash != EXPECTED_LLAMA_SERVER_SHA256
        ):
            raise RuntimeDriverError("head runtime identity mismatch")
        if (
            worker_path != str(self._config.worker_rpc_server_path)
            or worker_hash != EXPECTED_RPC_SERVER_SHA256
        ):
            raise RuntimeDriverError("worker runtime identity mismatch")
        if head_argv != config.head_argv() or worker_argv != config.worker_argv():
            raise RuntimeDriverError("runtime command line mismatch")
        return RuntimeLaunchEvidence(
            head_pid=pids.head_server_pid,
            worker_pid=pids.worker_rpc_pid,
            head_argv=head_argv,
            worker_argv=worker_argv,
            head_exe_path=head_path,
            head_exe_sha256=head_hash,
            worker_exe_path=worker_path,
            worker_exe_sha256=worker_hash,
        )

    def start(self, step: CanaryStep) -> RuntimeProcessIds:
        worker_started = head_started = False
        step_config = replace(self._config, context_size=step.context_tokens)
        try:
            self._run(self._start_argv(self._worker_unit, step_config.worker_argv()), worker=True)
            worker_started = True
            worker_pid = self._pid(self._worker_unit, worker=True)
            self._run(self._start_argv(self._head_unit, step_config.head_argv()))
            head_started = True
            head_pid = self._pid(self._head_unit, worker=False)
            pids = RuntimeProcessIds(head_pid, worker_pid)
            self._launch_evidence = self._measure_launch(pids, step_config)
            for attempt in range(self._attempts):
                if self._health_ready():
                    self._started = True
                    self._active_config = step_config
                    return pids
                if attempt + 1 < self._attempts:
                    self._sleep(1.0)
            raise RuntimeDriverError("loopback health did not become ready")
        except Exception as exc:  # noqa: BLE001 - cleanup each partial start
            if head_started:
                self._best_effort_stop(self._head_unit, worker=False)
            if worker_started:
                self._best_effort_stop(self._worker_unit, worker=True)
            self._launch_evidence = None
            self._active_config = None
            if isinstance(exc, RuntimeDriverError):
                raise
            raise RuntimeDriverError("transient runtime start failed") from exc

    def _best_effort_stop(self, unit: str, *, worker: bool) -> None:
        with suppress(RuntimeDriverError):
            self._run(("systemctl", "--user", "stop", unit), worker=worker)

    def stop(self) -> None:
        if not self._started:
            return
        errors: list[RuntimeDriverError] = []
        # Required order: remove head's RPC consumer before its worker provider.
        for unit, worker in ((self._head_unit, False), (self._worker_unit, True)):
            try:
                self._run(("systemctl", "--user", "stop", unit), worker=worker)
            except RuntimeDriverError as exc:
                errors.append(exc)
        self._started = False
        self._active_config = None
        if errors:
            raise RuntimeDriverError("transient runtime stop failed") from errors[0]

    def measure_post_run_worker(self, pids: RuntimeProcessIds) -> RuntimeLaunchEvidence:
        """Freshly re-check worker MainPID and executable after canary execution."""
        current_pid = self._pid(self._worker_unit, worker=True)
        current_head_pid = self._pid(self._head_unit, worker=False)
        if current_pid != pids.worker_rpc_pid or current_head_pid != pids.head_server_pid:
            raise RuntimeDriverError("runtime MainPID changed during canary")
        if self._active_config is None:
            raise RuntimeDriverError("runtime configuration identity is missing")
        evidence = self._measure_launch(pids, self._active_config)
        self._launch_evidence = evidence
        return evidence
