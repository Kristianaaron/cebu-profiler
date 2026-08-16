from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from model_atlas.fit_telemetry import (
    MIN_MEM_AVAILABLE_BYTES,
    CandidateBinding,
    NodeRole,
    ProcessMemory,
    ProcessRole,
    StepObservation,
    StopReason,
    TelemetryCollectionError,
    TelemetryCommandSpec,
    TelemetrySample,
    TwoNodeTelemetryCollector,
    build_base_canary_plan,
    derive_fit_summary,
)
from model_atlas.schemas.evidence import EvidenceKind

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _candidate() -> CandidateBinding:
    return CandidateBinding(
        artifact_path="/artifacts/glm52.gguf",
        artifact_sha256=SHA_A,
        runtime_config_sha256=SHA_B,
        llama_server_sha256=SHA_C,
        worker_rpc_server_sha256=SHA_D,
        head_argv=("/tools/llama-server", "--model", "/artifacts/glm52.gguf"),
        worker_argv=("/tools/ggml-rpc-server", "--device", "CUDA0"),
    )


def _sample(
    node: NodeRole,
    *,
    sample_set_id: str = "set-1",
    offset: int = 0,
    mem_available: int = 16 * 1024**3,
    pswpin: int = 10,
    pswpout: int = 20,
    pid: int | None = None,
) -> TelemetrySample:
    role = ProcessRole.SERVER if node is NodeRole.HEAD else ProcessRole.RPC
    expected_pid = (101 if node is NodeRole.HEAD else 202) if pid is None else pid
    return TelemetrySample(
        sample_set_id=sample_set_id,
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC) + timedelta(seconds=offset),
        node=node,
        hostname=f"{node}-host",
        phase_id="restart-4k",
        context_tokens=4096,
        gpu_used_bytes=50 * 1024**3,
        gpu_free_bytes=60 * 1024**3,
        gpu_util_percent=75.0,
        gpu_temperature_c=60.0,
        gpu_power_watts=80.0,
        mem_available_bytes=mem_available,
        swap_free_bytes=4 * 1024**3,
        pswpin_pages=pswpin,
        pswpout_pages=pswpout,
        process=ProcessMemory(
            role=role,
            pid=expected_pid,
            rss_bytes=10,
            pss_bytes=9,
            private_bytes=8,
            swap_bytes=0,
        ),
        rdma_rx_bytes=100 + offset,
        rdma_tx_bytes=200 + offset,
        disk_read_bytes=300 + offset,
        disk_write_bytes=400 + offset,
        evidence_kind=EvidenceKind.MEASURED,
    )


def _observation(context: int = 4096) -> StepObservation:
    return StepObservation(
        step_id=f"restart-{context}",
        context_tokens=context,
        load_duration_seconds=120.0,
        prompt_tps=100.0,
        decode_tps=12.5,
        observed_devices=("CUDA0", "RPC0"),
        runtime_succeeded=True,
    )


def test_collector_fails_closed_when_remote_is_missing() -> None:
    head = _sample(NodeRole.HEAD)

    def runner(spec: TelemetryCommandSpec) -> str:
        if spec.node is NodeRole.WORKER:
            raise TimeoutError("worker unreachable")
        return head.model_dump_json()

    collector = TwoNodeTelemetryCollector(
        probe_argv=("/opt/atlas/bin/fit-probe",),
        worker_ssh_target="10.77.0.2",
        runner=runner,
    )
    specs = collector.command_specs(
        sample_set_id="set-1",
        phase_id="restart-4k",
        context_tokens=4096,
        head_server_pid=101,
        worker_rpc_pid=202,
    )
    assert not specs[0].remote
    assert specs[1].argv[:5] == ("ssh", "-o", "BatchMode=yes", "--", "10.77.0.2")
    with pytest.raises(TelemetryCollectionError, match="worker telemetry unavailable"):
        collector.collect(
            sample_set_id="set-1",
            phase_id="restart-4k",
            context_tokens=4096,
            head_server_pid=101,
            worker_rpc_pid=202,
        )


def test_collector_rejects_pid_mismatch_and_never_falls_back_local() -> None:
    rows = {
        NodeRole.HEAD: _sample(NodeRole.HEAD).model_dump_json(),
        NodeRole.WORKER: _sample(NodeRole.WORKER, pid=999).model_dump_json(),
    }
    collector = TwoNodeTelemetryCollector(
        probe_argv=("fit-probe",),
        worker_ssh_target="10.77.0.2",
        runner=lambda spec: rows[spec.node],
    )
    with pytest.raises(TelemetryCollectionError, match="worker PID mismatch"):
        collector.collect(
            sample_set_id="set-1",
            phase_id="restart-4k",
            context_tokens=4096,
            head_server_pid=101,
            worker_rpc_pid=202,
        )


def test_strict_sample_rejects_missing_field_and_extra_field() -> None:
    payload = json.loads(_sample(NodeRole.HEAD).model_dump_json())
    del payload["rdma_rx_bytes"]
    with pytest.raises(ValidationError):
        TelemetrySample.model_validate(payload)
    payload = json.loads(_sample(NodeRole.HEAD).model_dump_json())
    payload["unknown"] = 1
    with pytest.raises(ValidationError):
        TelemetrySample.model_validate(payload)


def test_base_plan_is_bound_deterministic_and_keeps_mtp_separate() -> None:
    plan = build_base_canary_plan(_candidate())
    assert [step.step_id for step in plan.steps] == [
        "load-only-4k",
        "one-token-4k",
        "deterministic-64-4k",
        "restart-4k",
        "restart-16k",
        "restart-32k",
        "restart-64k",
    ]
    throughput = plan.steps[2]
    assert throughput.temperature == 0.0
    assert throughput.max_output_tokens == 64
    assert throughput.repeats - throughput.discard_warmup_repeats == 3
    assert all(step.parallel == 1 and not step.mtp_enabled for step in plan.steps)
    assert all(step.restart_runtime for step in plan.steps[3:])
    assert not plan.mtp_included
    assert plan.canonical_sha256() == build_base_canary_plan(_candidate()).canonical_sha256()


def test_summary_derives_two_node_fit_and_headroom() -> None:
    plan = build_base_canary_plan(_candidate())
    samples = [
        _sample(NodeRole.HEAD, sample_set_id="before", offset=0),
        _sample(NodeRole.WORKER, sample_set_id="before", offset=0),
        _sample(NodeRole.HEAD, sample_set_id="after", offset=10, mem_available=12 * 1024**3),
        _sample(NodeRole.WORKER, sample_set_id="after", offset=10, mem_available=10 * 1024**3),
    ]
    summary = derive_fit_summary(plan, samples, [_observation(4096), _observation(16384)])
    assert summary.both_nodes_measured
    assert summary.fitted
    assert summary.evidence_kind is EvidenceKind.MEASURED
    assert summary.stop_reason is StopReason.COMPLETED
    assert summary.last_passing_context_tokens == 16384
    assert summary.load_duration_seconds == 120.0
    assert summary.prompt_tps == 100.0
    assert summary.decode_tps == 12.5
    assert summary.minimum_mem_headroom_bytes == 10 * 1024**3


@pytest.mark.parametrize(
    ("samples", "observations", "reason"),
    [
        (
            [
                _sample(NodeRole.HEAD, sample_set_id="a", offset=0),
                _sample(NodeRole.WORKER, sample_set_id="a", offset=0),
                _sample(NodeRole.HEAD, sample_set_id="b", offset=1, pswpout=21),
                _sample(NodeRole.WORKER, sample_set_id="b", offset=1),
            ],
            [_observation()],
            StopReason.SWAP_ACTIVITY,
        ),
        (
            [
                _sample(NodeRole.HEAD, mem_available=MIN_MEM_AVAILABLE_BYTES - 1),
                _sample(NodeRole.WORKER),
            ],
            [_observation()],
            StopReason.LOW_MEM_AVAILABLE,
        ),
        (
            [_sample(NodeRole.HEAD), _sample(NodeRole.WORKER)],
            [_observation().model_copy(update={"oom_detected": True})],
            StopReason.OOM,
        ),
        (
            [_sample(NodeRole.HEAD), _sample(NodeRole.WORKER)],
            [_observation().model_copy(update={"observed_devices": ("CUDA0",)})],
            StopReason.MISSING_RPC_DEVICE,
        ),
    ],
)
def test_stop_thresholds(
    samples: list[TelemetrySample],
    observations: list[StepObservation],
    reason: StopReason,
) -> None:
    summary = derive_fit_summary(build_base_canary_plan(_candidate()), samples, observations)
    assert not summary.fitted
    assert summary.stop_reason is reason


def test_no_measured_claim_without_both_nodes() -> None:
    summary = derive_fit_summary(
        build_base_canary_plan(_candidate()), [_sample(NodeRole.HEAD)], [_observation()]
    )
    assert not summary.both_nodes_measured
    assert not summary.fitted
    assert summary.stop_reason is StopReason.MISSING_NODE
    assert summary.evidence_kind is not EvidenceKind.MEASURED
    assert summary.minimum_mem_headroom_bytes is None
