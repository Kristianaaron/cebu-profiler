"""Fail-closed two-node fit telemetry and non-executing canary plans.

This module describes commands but deliberately provides no subprocess runner.
Operators must inject both command execution and parsing, which keeps planning
and unit tests free of service, GPU, SSH, and network side effects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from model_atlas.schemas.evidence import EvidenceKind

TELEMETRY_SAMPLE_SCHEMA_VERSION = 1
FIT_SUMMARY_SCHEMA_VERSION = 1
CANARY_PLAN_SCHEMA_VERSION = 1
MIN_MEM_AVAILABLE_BYTES = 8 * 1024**3
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class NodeRole(StrEnum):
    HEAD = "head"
    WORKER = "worker"


class ProcessRole(StrEnum):
    SERVER = "server"
    RPC = "rpc"


class CanaryPhase(StrEnum):
    LOAD_ONLY = "load_only"
    ONE_TOKEN = "one_token"
    THROUGHPUT = "throughput"
    CONTEXT_RESTART = "context_restart"


class StopReason(StrEnum):
    COMPLETED = "completed"
    OOM = "oom"
    SWAP_ACTIVITY = "new_swap_activity"
    MISSING_RPC_DEVICE = "missing_rpc_device"
    LOW_MEM_AVAILABLE = "mem_available_below_8_gib"
    MISSING_NODE = "missing_node"
    PID_MISMATCH = "pid_mismatch"
    MISSING_FIELD = "missing_field"
    RUNTIME_FAILURE = "runtime_failure"
    NOT_RUN = "not_run"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"


class ProcessMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ProcessRole
    pid: int = Field(gt=0)
    rss_bytes: int = Field(ge=0)
    pss_bytes: int = Field(ge=0)
    private_bytes: int = Field(ge=0)
    swap_bytes: int = Field(ge=0)


class TelemetrySample(BaseModel):
    """One strict JSONL row. All counters are cumulative byte/page counters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    sample_set_id: str = Field(min_length=1)
    timestamp: datetime
    node: NodeRole
    hostname: str = Field(min_length=1)
    phase_id: str = Field(min_length=1)
    context_tokens: int = Field(ge=0)
    gpu_used_bytes: int = Field(ge=0)
    gpu_free_bytes: int = Field(ge=0)
    gpu_util_percent: Annotated[float, Field(ge=0.0, le=100.0)]
    gpu_temperature_c: float = Field(ge=0.0)
    gpu_power_watts: float = Field(ge=0.0)
    mem_available_bytes: int = Field(ge=0)
    swap_free_bytes: int = Field(ge=0)
    pswpin_pages: int = Field(ge=0)
    pswpout_pages: int = Field(ge=0)
    process: ProcessMemory
    rdma_rx_bytes: int = Field(ge=0)
    rdma_tx_bytes: int = Field(ge=0)
    disk_read_bytes: int = Field(ge=0)
    disk_write_bytes: int = Field(ge=0)
    evidence_kind: Literal[EvidenceKind.MEASURED] = EvidenceKind.MEASURED

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def process_role_matches_node(self) -> TelemetrySample:
        expected = ProcessRole.SERVER if self.node is NodeRole.HEAD else ProcessRole.RPC
        if self.process.role is not expected:
            raise ValueError(f"{self.node} sample requires {expected} process telemetry")
        return self

    def jsonl(self) -> str:
        return self.model_dump_json() + "\n"


class TelemetryCommandSpec(BaseModel):
    """An exact command description; constructing it executes nothing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: NodeRole
    argv: tuple[str, ...]
    expected_process_role: ProcessRole
    expected_pid: int = Field(gt=0)
    remote: bool

    @model_validator(mode="after")
    def command_is_nonempty_and_consistent(self) -> TelemetryCommandSpec:
        if not self.argv or any(not part for part in self.argv):
            raise ValueError("argv must contain only non-empty elements")
        if self.remote is not (self.node is NodeRole.WORKER):
            raise ValueError("only the worker telemetry command may be remote")
        expected = ProcessRole.SERVER if self.node is NodeRole.HEAD else ProcessRole.RPC
        if self.expected_process_role is not expected:
            raise ValueError("expected process role does not match node")
        return self


class CommandRunner(Protocol):
    def __call__(self, spec: TelemetryCommandSpec) -> str: ...


class SampleParser(Protocol):
    def __call__(self, payload: str) -> TelemetrySample: ...


class TelemetryCollectionError(RuntimeError):
    """Collection is incomplete; callers must stop the canary."""


def parse_sample_json(payload: str) -> TelemetrySample:
    """Strict default parser for one probe-produced JSON object."""
    return TelemetrySample.model_validate_json(payload)


class TwoNodeTelemetryCollector:
    """Collect exactly one head and one worker row through injected I/O."""

    def __init__(
        self,
        *,
        probe_argv: Sequence[str],
        worker_ssh_target: str,
        runner: CommandRunner,
        parser: SampleParser = parse_sample_json,
        ssh_argv: Sequence[str] = ("ssh", "-o", "BatchMode=yes"),
    ) -> None:
        if not probe_argv or any(not part for part in probe_argv):
            raise ValueError("probe_argv must contain only non-empty elements")
        if not worker_ssh_target:
            raise ValueError("worker_ssh_target is required")
        if not ssh_argv or any(not part for part in ssh_argv):
            raise ValueError("ssh_argv must contain only non-empty elements")
        self._probe_argv = tuple(probe_argv)
        self._worker_ssh_target = worker_ssh_target
        self._runner = runner
        self._parser = parser
        self._ssh_argv = tuple(ssh_argv)

    def command_specs(
        self,
        *,
        sample_set_id: str,
        phase_id: str,
        context_tokens: int,
        head_server_pid: int,
        worker_rpc_pid: int,
    ) -> tuple[TelemetryCommandSpec, TelemetryCommandSpec]:
        if head_server_pid <= 0 or worker_rpc_pid <= 0:
            raise ValueError("expected process PIDs must be positive")
        if not sample_set_id or not phase_id:
            raise ValueError("sample_set_id and phase_id are required")
        if context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        common = (
            "--sample-set-id",
            sample_set_id,
            "--phase-id",
            phase_id,
            "--context-tokens",
            str(context_tokens),
        )
        head_probe = (
            self._probe_argv
            + (
                "--node",
                NodeRole.HEAD,
                "--process-role",
                ProcessRole.SERVER,
                "--pid",
                str(head_server_pid),
            )
            + common
        )
        worker_probe = (
            self._probe_argv
            + (
                "--node",
                NodeRole.WORKER,
                "--process-role",
                ProcessRole.RPC,
                "--pid",
                str(worker_rpc_pid),
            )
            + common
        )
        return (
            TelemetryCommandSpec(
                node=NodeRole.HEAD,
                argv=head_probe,
                expected_process_role=ProcessRole.SERVER,
                expected_pid=head_server_pid,
                remote=False,
            ),
            TelemetryCommandSpec(
                node=NodeRole.WORKER,
                argv=self._ssh_argv + ("--", self._worker_ssh_target) + worker_probe,
                expected_process_role=ProcessRole.RPC,
                expected_pid=worker_rpc_pid,
                remote=True,
            ),
        )

    def collect(
        self,
        *,
        sample_set_id: str,
        phase_id: str,
        context_tokens: int,
        head_server_pid: int,
        worker_rpc_pid: int,
    ) -> tuple[TelemetrySample, ...]:
        samples: list[TelemetrySample] = []
        specs = self.command_specs(
            sample_set_id=sample_set_id,
            phase_id=phase_id,
            context_tokens=context_tokens,
            head_server_pid=head_server_pid,
            worker_rpc_pid=worker_rpc_pid,
        )
        for spec in specs:
            try:
                payload = self._runner(spec)
                sample = self._parser(payload)
            except Exception as exc:  # noqa: BLE001 - normalize the fail-closed boundary
                raise TelemetryCollectionError(f"{spec.node} telemetry unavailable: {exc}") from exc
            if sample.node is not spec.node:
                raise TelemetryCollectionError(
                    f"{spec.node} command returned {sample.node} telemetry"
                )
            if sample.process.role is not spec.expected_process_role:
                raise TelemetryCollectionError(f"{spec.node} process role mismatch")
            if sample.process.pid != spec.expected_pid:
                raise TelemetryCollectionError(
                    f"{spec.node} PID mismatch: expected {spec.expected_pid}, "
                    f"observed {sample.process.pid}"
                )
            if sample.sample_set_id != sample_set_id:
                raise TelemetryCollectionError(f"{spec.node} sample_set_id mismatch")
            if sample.phase_id != phase_id or sample.context_tokens != context_tokens:
                raise TelemetryCollectionError(f"{spec.node} canary phase identity mismatch")
            samples.append(sample)
        if {sample.node for sample in samples} != {NodeRole.HEAD, NodeRole.WORKER}:
            raise TelemetryCollectionError("both head and worker telemetry are required")
        if len({sample.sample_set_id for sample in samples}) != 1:
            raise TelemetryCollectionError("head and worker samples must share sample_set_id")
        return tuple(samples)


class CandidateBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    llama_server_sha256: str = Field(pattern=_SHA256_PATTERN)
    worker_rpc_server_sha256: str = Field(pattern=_SHA256_PATTERN)
    head_argv: tuple[str, ...]
    worker_argv: tuple[str, ...]
    producer_run_id: str | None = Field(default=None, min_length=1)
    producer_plan_id: str | None = Field(default=None, min_length=1)
    producer_recipe_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    producer_profile_id: str | None = Field(default=None, min_length=1)
    producer_recommendation_id: str | None = Field(default=None, min_length=1)
    producer_handoff_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def exact_argv_required(self) -> CandidateBinding:
        if not self.artifact_path.startswith("/"):
            raise ValueError("artifact_path must be absolute")
        if not self.head_argv or not self.worker_argv:
            raise ValueError("both exact runtime argv lists are required")
        if any(not item for item in self.head_argv + self.worker_argv):
            raise ValueError("runtime argv may not contain empty elements")
        producer = (
            self.producer_run_id,
            self.producer_plan_id,
            self.producer_recipe_sha256,
            self.producer_profile_id,
            self.producer_recommendation_id,
            self.producer_handoff_sha256,
        )
        if any(value is not None for value in producer) and not all(
            value is not None for value in producer
        ):
            raise ValueError("producer lineage must be supplied as one complete binding")
        return self


class CanaryStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1)
    phase: CanaryPhase
    context_tokens: int = Field(ge=0)
    parallel: Literal[1] = 1
    max_output_tokens: int = Field(ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    repeats: int = Field(ge=1)
    discard_warmup_repeats: int = Field(ge=0)
    restart_runtime: bool
    mtp_enabled: Literal[False] = False

    @model_validator(mode="after")
    def warmups_leave_a_measured_repeat(self) -> CanaryStep:
        if self.discard_warmup_repeats >= self.repeats:
            raise ValueError("at least one non-warmup repeat is required")
        return self


class CanaryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    candidate: CandidateBinding
    deterministic_seed: int = 0
    steps: tuple[CanaryStep, ...]
    stop_mem_available_below_bytes: Literal[8589934592] = 8589934592
    stop_on_oom: Literal[True] = True
    stop_on_new_swap_activity: Literal[True] = True
    require_rpc_device: Literal[True] = True
    evidence_kind: Literal[EvidenceKind.PREDICTED] = EvidenceKind.PREDICTED
    mtp_included: Literal[False] = False

    def canonical_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def build_base_canary_plan(candidate: CandidateBinding) -> CanaryPlan:
    """Return the fixed base matrix. MTP is intentionally a different plan."""
    steps = (
        CanaryStep(
            step_id="load-only-4k",
            phase=CanaryPhase.LOAD_ONLY,
            context_tokens=4096,
            max_output_tokens=0,
            repeats=1,
            discard_warmup_repeats=0,
            restart_runtime=True,
        ),
        CanaryStep(
            step_id="one-token-4k",
            phase=CanaryPhase.ONE_TOKEN,
            context_tokens=4096,
            max_output_tokens=1,
            repeats=1,
            discard_warmup_repeats=0,
            restart_runtime=False,
        ),
        CanaryStep(
            step_id="deterministic-64-4k",
            phase=CanaryPhase.THROUGHPUT,
            context_tokens=4096,
            max_output_tokens=64,
            repeats=4,
            discard_warmup_repeats=1,
            restart_runtime=False,
        ),
    ) + tuple(
        CanaryStep(
            step_id=f"restart-{context // 1024}k",
            phase=CanaryPhase.CONTEXT_RESTART,
            context_tokens=context,
            max_output_tokens=64,
            repeats=4,
            discard_warmup_repeats=1,
            restart_runtime=True,
        )
        for context in (4096, 16384, 32768, 65536)
    )
    return CanaryPlan(candidate=candidate, steps=steps)


class StepObservation(BaseModel):
    """Runtime result supplied to summary derivation; this module does not execute it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1)
    context_tokens: int = Field(ge=0)
    load_duration_seconds: float | None = Field(default=None, ge=0.0)
    prompt_tps: float | None = Field(default=None, ge=0.0)
    decode_tps: float | None = Field(default=None, ge=0.0)
    observed_devices: tuple[str, ...]
    oom_detected: bool = False
    runtime_succeeded: bool
    evidence_kind: Literal[EvidenceKind.MEASURED] = EvidenceKind.MEASURED


class NodeFitSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node: NodeRole
    minimum_mem_available_bytes: int = Field(ge=0)
    minimum_gpu_free_bytes: int = Field(ge=0)
    peak_gpu_used_bytes: int = Field(ge=0)
    peak_process_pss_bytes: int = Field(ge=0)
    peak_process_private_bytes: int = Field(ge=0)
    peak_process_swap_bytes: int = Field(ge=0)
    new_pswpin_pages: int = Field(ge=0)
    new_pswpout_pages: int = Field(ge=0)
    rdma_rx_delta_bytes: int = Field(ge=0)
    rdma_tx_delta_bytes: int = Field(ge=0)
    disk_read_delta_bytes: int = Field(ge=0)
    disk_write_delta_bytes: int = Field(ge=0)


class FitSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate: CandidateBinding
    both_nodes_measured: bool
    fitted: bool
    last_passing_context_tokens: int | None = Field(default=None, ge=0)
    load_duration_seconds: float | None = Field(default=None, ge=0.0)
    prompt_tps: float | None = Field(default=None, ge=0.0)
    decode_tps: float | None = Field(default=None, ge=0.0)
    minimum_mem_headroom_bytes: int | None = Field(default=None, ge=0)
    stop_reason: StopReason
    nodes: tuple[NodeFitSummary, ...]
    evidence_kind: EvidenceKind

    @model_validator(mode="after")
    def measured_claim_requires_both_nodes(self) -> FitSummary:
        if self.evidence_kind is EvidenceKind.MEASURED and not self.both_nodes_measured:
            raise ValueError("measured fit summary requires telemetry from both nodes")
        if self.fitted and not self.both_nodes_measured:
            raise ValueError("fit cannot pass without telemetry from both nodes")
        return self


def _delta(samples: Sequence[TelemetrySample], field: Callable[[TelemetrySample], int]) -> int:
    return max(0, field(samples[-1]) - field(samples[0]))


def _node_summary(node: NodeRole, rows: Sequence[TelemetrySample]) -> NodeFitSummary:
    ordered = sorted(rows, key=lambda sample: sample.timestamp)
    return NodeFitSummary(
        node=node,
        minimum_mem_available_bytes=min(row.mem_available_bytes for row in ordered),
        minimum_gpu_free_bytes=min(row.gpu_free_bytes for row in ordered),
        peak_gpu_used_bytes=max(row.gpu_used_bytes for row in ordered),
        peak_process_pss_bytes=max(row.process.pss_bytes for row in ordered),
        peak_process_private_bytes=max(row.process.private_bytes for row in ordered),
        peak_process_swap_bytes=max(row.process.swap_bytes for row in ordered),
        new_pswpin_pages=_delta(ordered, lambda row: row.pswpin_pages),
        new_pswpout_pages=_delta(ordered, lambda row: row.pswpout_pages),
        rdma_rx_delta_bytes=_delta(ordered, lambda row: row.rdma_rx_bytes),
        rdma_tx_delta_bytes=_delta(ordered, lambda row: row.rdma_tx_bytes),
        disk_read_delta_bytes=_delta(ordered, lambda row: row.disk_read_bytes),
        disk_write_delta_bytes=_delta(ordered, lambda row: row.disk_write_bytes),
    )


def derive_fit_summary(
    plan: CanaryPlan,
    samples: Sequence[TelemetrySample],
    observations: Sequence[StepObservation],
) -> FitSummary:
    """Derive fit evidence, stopping at the first safety failure.

    A missing node or an incomplete paired sample set is an unmeasured failure,
    never a degraded local-only result.
    """
    processed_rows: list[TelemetrySample] = []
    accepted: list[StepObservation] = []
    stop = StopReason.COMPLETED
    complete = True
    both_nodes = True

    # Evidence is consumed only in the immutable plan order.  Every executed
    # step requires exactly one observation and at least two paired snapshots
    # (before/after), each containing exactly one row from each node.
    for index, step in enumerate(plan.steps):
        if index >= len(observations):
            complete = False
            stop = StopReason.INCOMPLETE_EVIDENCE
            break
        observation = observations[index]
        if observation.step_id != step.step_id or observation.context_tokens != step.context_tokens:
            complete = False
            stop = StopReason.INCOMPLETE_EVIDENCE
            break
        step_rows = [
            sample
            for sample in samples
            if sample.phase_id == step.step_id and sample.context_tokens == step.context_tokens
        ]
        grouped: dict[str, list[TelemetrySample]] = {}
        for sample in step_rows:
            grouped.setdefault(sample.sample_set_id, []).append(sample)
        paired = len(grouped) >= 2 and all(
            len(rows) == 2 and {row.node for row in rows} == {NodeRole.HEAD, NodeRole.WORKER}
            for rows in grouped.values()
        )
        if not paired:
            complete = False
            both_nodes = False
            stop = StopReason.MISSING_NODE if step_rows else StopReason.INCOMPLETE_EVIDENCE
            break
        processed_rows.extend(step_rows)
        step_summaries = tuple(
            _node_summary(
                node,
                [sample for sample in step_rows if sample.node is node],
            )
            for node in NodeRole
        )
        if observation.oom_detected:
            stop = StopReason.OOM
        elif any(
            summary.new_pswpin_pages or summary.new_pswpout_pages for summary in step_summaries
        ):
            stop = StopReason.SWAP_ACTIVITY
        elif "RPC0" not in observation.observed_devices:
            stop = StopReason.MISSING_RPC_DEVICE
        elif any(
            summary.minimum_mem_available_bytes < plan.stop_mem_available_below_bytes
            for summary in step_summaries
        ):
            stop = StopReason.LOW_MEM_AVAILABLE
        elif not observation.runtime_succeeded:
            stop = StopReason.RUNTIME_FAILURE
        if stop is not StopReason.COMPLETED:
            complete = False
            break
        accepted.append(observation)

    if complete and len(observations) != len(plan.steps):
        complete = False
        stop = StopReason.INCOMPLETE_EVIDENCE

    rows_by_node = {
        node: [sample for sample in processed_rows if sample.node is node] for node in NodeRole
    }
    node_summaries = tuple(
        _node_summary(node, rows_by_node[node]) for node in NodeRole if rows_by_node[node]
    )
    both_nodes = both_nodes and len(node_summaries) == 2
    last = accepted[-1] if accepted else None
    load = next(
        (item.load_duration_seconds for item in accepted if item.load_duration_seconds is not None),
        None,
    )
    throughput = next(
        (
            item
            for item in reversed(accepted)
            if item.prompt_tps is not None and item.decode_tps is not None
        ),
        None,
    )
    measured = (
        both_nodes
        and bool(processed_rows)
        and all(sample.evidence_kind is EvidenceKind.MEASURED for sample in processed_rows)
    )
    return FitSummary(
        plan_sha256=plan.canonical_sha256(),
        candidate=plan.candidate,
        both_nodes_measured=both_nodes,
        fitted=measured and complete and stop is StopReason.COMPLETED,
        last_passing_context_tokens=last.context_tokens if last is not None else None,
        load_duration_seconds=load,
        prompt_tps=throughput.prompt_tps if throughput is not None else None,
        decode_tps=throughput.decode_tps if throughput is not None else None,
        minimum_mem_headroom_bytes=(
            min(summary.minimum_mem_available_bytes for summary in node_summaries)
            if both_nodes
            else None
        ),
        stop_reason=stop,
        nodes=node_summaries,
        evidence_kind=EvidenceKind.MEASURED if measured else EvidenceKind.INFERRED,
    )
