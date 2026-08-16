"""Fail-closed, injectable executor for the two-Spark canary plan.

This module deliberately knows only protocols, argv and evidence records.  It
does not own a systemd unit, a GPU, or an SSH session.  Production wiring uses
the small subprocess adapters below; tests inject fakes so no test can start a
runtime or touch a model artifact.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.fit_telemetry import (
    CanaryPlan,
    CanaryStep,
    FitSummary,
    StepObservation,
    StopReason,
    TelemetryCollectionError,
    TelemetryCommandSpec,
    TelemetrySample,
    TwoNodeTelemetryCollector,
    derive_fit_summary,
)
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    PINNED_COMMIT,
    LlamaCppRpcRuntimeClaim,
    LlamaCppRpcRuntimeConfig,
    LlamaCppRpcToolProbe,
    LlamaCppRpcValidationReceipt,
    LlamaCppRpcWorkerAttestation,
)
from model_atlas.schemas.evidence import EvidenceKind

_PIPE_BUF = 4096
_SHA256 = "sha256sum"

__all__ = [
    "CanaryExecutionError",
    "CanaryExecutionReceipt",
    "CanaryExecutionResult",
    "JsonlEvidenceStore",
    "RuntimeProcessIds",
    "SshWorkerHashProbe",
    "SubprocessTelemetryRunner",
    "TwoNodeCanaryExecutor",
]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CanaryExecutionError(RuntimeError):
    """Expected fail-closed executor error; its text is never persisted."""


class SubprocessTelemetryRunner:
    """Production telemetry runner: direct argv only, never a shell."""

    def __call__(self, spec: TelemetryCommandSpec) -> str:
        argv = spec.argv
        if not argv or any(not value for value in argv):
            raise CanaryExecutionError("invalid telemetry argv")
        result = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=None,
        )
        if result.returncode != 0:
            raise CanaryExecutionError("telemetry command failed")
        return result.stdout


class ArgvTextRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str: ...


class SshWorkerHashProbe:
    """Independently hash the worker RPC binary over authenticated SSH.

    The returned attestation is intentionally built from a command separate
    from the runtime receipt.  A receipt can never self-attest its worker.
    """

    def __init__(
        self,
        *,
        ssh_target: str,
        worker_host: str,
        rpc_server_path: Path,
        toolchain_root: Path,
        runner: ArgvTextRunner,
        ssh_argv: Sequence[str] = ("ssh", "-o", "BatchMode=yes"),
    ) -> None:
        if (
            not ssh_target
            or not worker_host
            or not rpc_server_path.is_absolute()
            or not toolchain_root.is_absolute()
        ):
            raise ValueError("worker identities and absolute tool paths are required")
        if not ssh_argv or any(not value for value in ssh_argv):
            raise ValueError("ssh_argv must contain non-empty values")
        self._target = ssh_target
        self._worker_host = worker_host
        self._path = rpc_server_path
        self._toolchain_root = toolchain_root
        self._runner = runner
        self._ssh_argv = tuple(ssh_argv)

    def measure(self) -> LlamaCppRpcWorkerAttestation:
        hash_argv = self._ssh_argv + ("--", self._target, _SHA256, str(self._path))
        commit_argv = self._ssh_argv + (
            "--",
            self._target,
            "git",
            "-C",
            str(self._toolchain_root),
            "rev-parse",
            "HEAD",
        )
        try:
            output = self._runner(hash_argv)
            commit = self._runner(commit_argv).strip()
        except Exception as exc:  # noqa: BLE001 - normalize a sensitive boundary
            raise CanaryExecutionError("worker RPC hash unavailable") from exc
        fields = output.strip().split()
        valid_hash = (
            len(fields) >= 2
            and len(fields[0]) == 64
            and all(char in "0123456789abcdef" for char in fields[0])
        )
        if not valid_hash or fields[1] != str(self._path):
            raise CanaryExecutionError("worker RPC hash output invalid")
        if commit != PINNED_COMMIT:
            raise CanaryExecutionError("worker llama.cpp commit mismatch")
        return LlamaCppRpcWorkerAttestation(
            host=self._worker_host,
            rpc_server_path=str(self._path),
            rpc_server_sha256=fields[0],
            commit=commit,
            evidence_kind="measured",
        )


class WorkerAttestationProbe(Protocol):
    def measure(self) -> LlamaCppRpcWorkerAttestation: ...


class RuntimeContract(Protocol):
    config: LlamaCppRpcRuntimeConfig

    def probe(
        self, worker_attestation: LlamaCppRpcWorkerAttestation | None = None
    ) -> LlamaCppRpcToolProbe: ...

    def validate_receipt(
        self,
        receipt: LlamaCppRpcValidationReceipt | None,
        *,
        independently_measured_worker: LlamaCppRpcWorkerAttestation | None,
    ) -> LlamaCppRpcRuntimeClaim: ...


@dataclass(frozen=True)
class RuntimeProcessIds:
    """Exact PIDs reported by the lifecycle that owns the launched process."""

    head_server_pid: int
    worker_rpc_pid: int

    def __post_init__(self) -> None:
        if self.head_server_pid <= 0 or self.worker_rpc_pid <= 0:
            raise ValueError("runtime PIDs must be positive")


class RuntimeLifecycle(Protocol):
    def start(self) -> RuntimeProcessIds: ...

    def stop(self) -> None: ...


class CanaryRequestClient(Protocol):
    def execute(self, step: CanaryStep, *, deterministic_seed: int) -> StepObservation: ...


class JsonlEvidenceStore:
    """Append one bounded JSON record with O_APPEND + fsync durability."""

    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError("evidence path must be absolute")
        self.path = path

    def append(self, record_type: str, payload: BaseModel) -> None:
        record = {
            "schema_version": 1,
            "record_type": record_type,
            "recorded_at": _utc_now().isoformat(),
            "payload": payload.model_dump(mode="json"),
        }
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > _PIPE_BUF:
            raise CanaryExecutionError("evidence record exceeds atomic append bound")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise CanaryExecutionError("incomplete evidence append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class CanaryExecutionReceipt(BaseModel):
    """Sanitized execution outcome; no command output, secrets, or prompts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_step_ids: tuple[str, ...]
    stop_reason: StopReason
    runtime_claim_validated: bool
    runtime_claim_reason: str
    evidence_kind: Literal[EvidenceKind.MEASURED, EvidenceKind.INFERRED]


class CanaryExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    receipt: CanaryExecutionReceipt
    summary: FitSummary


def _sample_set_id(plan: CanaryPlan, step: CanaryStep, moment: str) -> str:
    return f"{plan.canonical_sha256()[:12]}-{step.step_id}-{moment}"


def _safety_stop(
    plan: CanaryPlan,
    before: Sequence[TelemetrySample],
    after: Sequence[TelemetrySample],
    observation: StepObservation,
) -> StopReason:
    if observation.oom_detected:
        return StopReason.OOM
    for previous, current in zip(before, after, strict=True):
        if (
            current.pswpin_pages > previous.pswpin_pages
            or current.pswpout_pages > previous.pswpout_pages
        ):
            return StopReason.SWAP_ACTIVITY
        if current.mem_available_bytes < plan.stop_mem_available_below_bytes:
            return StopReason.LOW_MEM_AVAILABLE
    if "RPC0" not in observation.observed_devices:
        return StopReason.MISSING_RPC_DEVICE
    if not observation.runtime_succeeded:
        return StopReason.RUNTIME_FAILURE
    return StopReason.COMPLETED


class TwoNodeCanaryExecutor:
    """Execute an immutable plan in order and stop at the first unsafe step."""

    def __init__(
        self,
        *,
        runtime: RuntimeContract,
        worker_attestation: WorkerAttestationProbe,
        lifecycle: RuntimeLifecycle,
        requests: CanaryRequestClient,
        telemetry: TwoNodeTelemetryCollector,
        evidence: JsonlEvidenceStore,
    ) -> None:
        self._runtime = runtime
        self._worker_attestation = worker_attestation
        self._lifecycle = lifecycle
        self._requests = requests
        self._telemetry = telemetry
        self._evidence = evidence

    def _check_candidate(self, plan: CanaryPlan) -> None:
        config = self._runtime.config
        candidate = plan.candidate
        expected = (
            candidate.artifact_path == str(config.artifact_path)
            and candidate.artifact_sha256 == config.artifact_sha256
            and candidate.runtime_config_sha256 == config.canonical_sha256()
            and candidate.llama_server_sha256 == EXPECTED_LLAMA_SERVER_SHA256
            and candidate.worker_rpc_server_sha256 == EXPECTED_RPC_SERVER_SHA256
        )
        # The runtime adapter remains the authority for pinned binary hashes;
        # only source/path/config alignment belongs to the plan binding here.
        argv_matches = (
            candidate.head_argv == config.head_argv()
            and candidate.worker_argv == config.worker_argv()
        )
        if not expected or not argv_matches:
            raise CanaryExecutionError("candidate binding does not match runtime contract")

    def _runtime_receipt(
        self,
        observations: Sequence[StepObservation],
        attestation: LlamaCppRpcWorkerAttestation,
    ) -> LlamaCppRpcValidationReceipt:
        config = self._runtime.config
        has_load = any(
            item.step_id == "load-only-4k" and item.runtime_succeeded for item in observations
        )
        has_generation = any(item.runtime_succeeded for item in observations[1:])
        return LlamaCppRpcValidationReceipt(
            runtime_id="llamacpp-rpc-two-spark",
            config_sha256=config.canonical_sha256(),
            artifact_sha256=config.artifact_sha256,
            commit=PINNED_COMMIT,
            llama_server_sha256=self._runtime.probe(attestation).llama_server_sha256,
            worker_rpc_server_sha256=attestation.rpc_server_sha256,
            worker_rpc_server_path=attestation.rpc_server_path,
            worker_hash_attested=True,
            worker_host=attestation.host,
            observed_devices=tuple(
                sorted({device for item in observations for device in item.observed_devices})
            ),
            load_succeeded=has_load,
            generation_succeeded=has_generation,
            evidence_kind="measured",
        )

    def execute(self, plan: CanaryPlan) -> CanaryExecutionResult:
        """Run strictly in plan order.  Always stop a started runtime in ``finally``."""
        self._check_candidate(plan)
        attestation = self._worker_attestation.measure()
        # This fresh local hash/artifact existence check precedes any runtime claim.
        if not self._runtime.probe(attestation).available:
            raise CanaryExecutionError(
                "runtime artifact or independently measured worker is unverified"
            )

        samples: list[TelemetrySample] = []
        observations: list[StepObservation] = []
        completed: list[str] = []
        stop = StopReason.COMPLETED
        pids: RuntimeProcessIds | None = None
        try:
            for step in plan.steps:
                if pids is None or step.restart_runtime:
                    if pids is not None:
                        self._lifecycle.stop()
                    pids = self._lifecycle.start()
                try:
                    before = self._telemetry.collect(
                        sample_set_id=_sample_set_id(plan, step, "before"),
                        phase_id=step.step_id,
                        context_tokens=step.context_tokens,
                        head_server_pid=pids.head_server_pid,
                        worker_rpc_pid=pids.worker_rpc_pid,
                    )
                    for sample in before:
                        self._evidence.append("telemetry_sample", sample)
                    if any(
                        sample.mem_available_bytes < plan.stop_mem_available_below_bytes
                        for sample in before
                    ):
                        samples.extend(before)
                        stop = StopReason.LOW_MEM_AVAILABLE
                        break
                    try:
                        observation = self._requests.execute(
                            step,
                            deterministic_seed=plan.deterministic_seed,
                        )
                        identity_matches = (
                            observation.step_id == step.step_id
                            and observation.context_tokens == step.context_tokens
                        )
                        if not identity_matches:
                            raise CanaryExecutionError("request observation identity mismatch")
                    except Exception:  # noqa: BLE001 - record only typed failure evidence
                        observation = StepObservation(
                            step_id=step.step_id,
                            context_tokens=step.context_tokens,
                            observed_devices=(),
                            runtime_succeeded=False,
                        )
                    self._evidence.append("step_observation", observation)
                    after = self._telemetry.collect(
                        sample_set_id=_sample_set_id(plan, step, "after"),
                        phase_id=step.step_id,
                        context_tokens=step.context_tokens,
                        head_server_pid=pids.head_server_pid,
                        worker_rpc_pid=pids.worker_rpc_pid,
                    )
                    for sample in after:
                        self._evidence.append("telemetry_sample", sample)
                except TelemetryCollectionError:
                    stop = StopReason.MISSING_NODE
                    break
                except CanaryExecutionError:
                    stop = StopReason.RUNTIME_FAILURE
                    break
                except Exception:  # noqa: BLE001 - persist only the typed failure
                    stop = StopReason.RUNTIME_FAILURE
                    break
                samples.extend((*before, *after))
                observations.append(observation)
                stop = _safety_stop(plan, before, after, observation)
                if stop is not StopReason.COMPLETED:
                    break
                completed.append(step.step_id)
        finally:
            if pids is not None:
                self._lifecycle.stop()

        summary = derive_fit_summary(plan, samples, observations)
        runtime_receipt = self._runtime_receipt(observations, attestation)
        claim: LlamaCppRpcRuntimeClaim = self._runtime.validate_receipt(
            runtime_receipt if summary.fitted else None,
            independently_measured_worker=attestation,
        )
        receipt = CanaryExecutionReceipt(
            plan_sha256=plan.canonical_sha256(),
            completed_step_ids=tuple(completed),
            stop_reason=stop if stop is not StopReason.COMPLETED else summary.stop_reason,
            runtime_claim_validated=claim.validated,
            runtime_claim_reason=claim.reason,
            evidence_kind=(
                EvidenceKind.MEASURED if summary.both_nodes_measured else EvidenceKind.INFERRED
            ),
        )
        self._evidence.append("canary_execution_receipt", receipt)
        return CanaryExecutionResult(receipt=receipt, summary=summary)
