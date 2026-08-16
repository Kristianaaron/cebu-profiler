from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_atlas.fit_telemetry import (
    CanaryPlan,
    CandidateBinding,
    NodeRole,
    ProcessMemory,
    ProcessRole,
    StepObservation,
    TelemetryCommandSpec,
    TelemetrySample,
    TwoNodeTelemetryCollector,
    build_base_canary_plan,
)
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    LlamaCppRpcRuntimeClaim,
    LlamaCppRpcRuntimeConfig,
    LlamaCppRpcToolProbe,
    LlamaCppRpcValidationReceipt,
    LlamaCppRpcWorkerAttestation,
)
from model_atlas.schemas.evidence import EvidenceKind
from model_atlas.two_node_canary_executor import (
    JsonlEvidenceStore,
    RuntimeProcessIds,
    SshWorkerHashProbe,
    TwoNodeCanaryExecutor,
)


class _Runtime:
    config = LlamaCppRpcRuntimeConfig(
        artifact_path=Path("/artifacts/glm52.gguf"), artifact_sha256="a" * 64
    )

    def probe(self, attestation: LlamaCppRpcWorkerAttestation) -> LlamaCppRpcToolProbe:
        return LlamaCppRpcToolProbe(
            available=True,
            commit="4df29be4f4c3673f428170fda944a5b19f743bb8",
            llama_server_path="/tools/llama-server",
            llama_server_sha256="c" * 64,
            worker_rpc_server_path="/tools/ggml-rpc-server",
            worker_rpc_server_sha256=attestation.rpc_server_sha256,
            remote_worker_attested=True,
            artifact_path="/artifacts/glm52.gguf",
            artifact_sha256="a" * 64,
            artifact_verified=True,
        )

    def validate_receipt(
        self,
        receipt: LlamaCppRpcValidationReceipt | None,
        *,
        independently_measured_worker: LlamaCppRpcWorkerAttestation | None,
    ) -> LlamaCppRpcRuntimeClaim:
        return LlamaCppRpcRuntimeClaim(
            receipt is not None and independently_measured_worker is not None,
            ("llamacpp-rpc-two-spark",),
            "validated",
        )


class _Worker:
    def measure(self) -> LlamaCppRpcWorkerAttestation:
        return LlamaCppRpcWorkerAttestation(
            "169.254.200.197",
            "/tools/ggml-rpc-server",
            "d" * 64,
            "4df29be4f4c3673f428170fda944a5b19f743bb8",
        )


class _Lifecycle:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> RuntimeProcessIds:
        self.started += 1
        return RuntimeProcessIds(101, 202)

    def stop(self) -> None:
        self.stopped += 1


class _Requests:
    def __init__(self, fail_at: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at

    def execute(self, step: object, *, deterministic_seed: int) -> StepObservation:  # type: ignore[no-untyped-def]
        step_id = step.step_id
        self.calls.append(step_id)
        if self.fail_at == step_id:
            raise RuntimeError("do not serialize this detail")
        return StepObservation(
            step_id=step_id,
            context_tokens=step.context_tokens,
            observed_devices=("CUDA0", "RPC0"),
            runtime_succeeded=True,
        )


def _collector() -> TwoNodeTelemetryCollector:
    def runner(spec: TelemetryCommandSpec) -> str:
        argv = spec.argv
        phase = argv[argv.index("--phase-id") + 1]
        context = int(argv[argv.index("--context-tokens") + 1])
        sample_set = argv[argv.index("--sample-set-id") + 1]
        role = ProcessRole.SERVER if spec.node is NodeRole.HEAD else ProcessRole.RPC
        pid = 101 if spec.node is NodeRole.HEAD else 202
        return TelemetrySample(
            sample_set_id=sample_set,
            timestamp=datetime.now(UTC),
            node=spec.node,
            hostname=f"{spec.node}-host",
            phase_id=phase,
            context_tokens=context,
            gpu_used_bytes=1,
            gpu_free_bytes=2,
            gpu_util_percent=1.0,
            gpu_temperature_c=1.0,
            gpu_power_watts=1.0,
            mem_available_bytes=16 * 1024**3,
            swap_free_bytes=1,
            pswpin_pages=0,
            pswpout_pages=0,
            process=ProcessMemory(
                role=role,
                pid=pid,
                rss_bytes=1,
                pss_bytes=1,
                private_bytes=1,
                swap_bytes=0,
            ),
            rdma_rx_bytes=1,
            rdma_tx_bytes=1,
            disk_read_bytes=1,
            disk_write_bytes=1,
            evidence_kind=EvidenceKind.MEASURED,
        ).model_dump_json()

    return TwoNodeTelemetryCollector(
        probe_argv=("fit-probe",),
        worker_ssh_target="10.77.0.2",
        runner=runner,
    )


def _plan() -> CanaryPlan:
    candidate = CandidateBinding(
        artifact_path="/artifacts/glm52.gguf",
        artifact_sha256="a" * 64,
        runtime_config_sha256=_Runtime.config.canonical_sha256(),
        llama_server_sha256=EXPECTED_LLAMA_SERVER_SHA256,
        worker_rpc_server_sha256=EXPECTED_RPC_SERVER_SHA256,
        head_argv=_Runtime.config.head_argv(),
        worker_argv=_Runtime.config.worker_argv(),
    )
    return build_base_canary_plan(candidate)


def test_worker_hash_probe_uses_exact_ssh_argv() -> None:
    seen: list[tuple[str, ...]] = []
    probe = SshWorkerHashProbe(
        ssh_target="10.77.0.2",
        worker_host="169.254.200.197",
        rpc_server_path=Path("/tools/ggml-rpc-server"),
        runner=lambda argv: seen.append(tuple(argv)) or ("d" * 64 + "  rpc\n"),
    )
    attestation = probe.measure()
    assert seen == [
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            "--",
            "10.77.0.2",
            "sha256sum",
            "/tools/ggml-rpc-server",
        )
    ]
    assert attestation.host == "169.254.200.197"
    assert attestation.rpc_server_sha256 == "d" * 64


def test_executor_runs_strict_order_and_persists_paired_durable_evidence(tmp_path: Path) -> None:
    lifecycle = _Lifecycle()
    requests = _Requests()
    store = JsonlEvidenceStore(tmp_path / "canary.jsonl")
    executor = TwoNodeCanaryExecutor(
        runtime=_Runtime(),
        worker_attestation=_Worker(),
        lifecycle=lifecycle,
        requests=requests,
        telemetry=_collector(),
        evidence=store,
    )
    result = executor.execute(_plan())
    assert requests.calls == [step.step_id for step in _plan().steps]
    assert result.receipt.runtime_claim_validated
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    telemetry = [row for row in rows if row["record_type"] == "telemetry_sample"]
    assert len(telemetry) == len(_plan().steps) * 4
    assert rows[-1]["record_type"] == "canary_execution_receipt"
    assert lifecycle.stopped >= lifecycle.started


def test_executor_stops_immediately_on_runtime_failure_without_later_steps(tmp_path: Path) -> None:
    lifecycle = _Lifecycle()
    requests = _Requests(fail_at="one-token-4k")
    executor = TwoNodeCanaryExecutor(
        runtime=_Runtime(),
        worker_attestation=_Worker(),
        lifecycle=lifecycle,
        requests=requests,
        telemetry=_collector(),
        evidence=JsonlEvidenceStore(tmp_path / "canary.jsonl"),
    )
    result = executor.execute(_plan())
    assert requests.calls == ["load-only-4k", "one-token-4k"]
    assert not result.receipt.runtime_claim_validated
    assert result.receipt.completed_step_ids == ("load-only-4k",)
