from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from model_atlas.kernels import (
    KernelBackend,
    KernelBenchmarkReceipt,
    KernelBottleneck,
    KernelEvidenceCatalog,
    KernelEvidenceError,
    KernelEvidenceStatus,
    KernelExecutionPath,
    KernelHardware,
    KernelMetrics,
    KernelPhase,
    KernelProvenance,
    KernelQuery,
    KernelRepresentation,
    KernelRequirement,
    KernelRunStatus,
    KernelWorkload,
    assess_candidate_kernels,
    build_execution_manifest,
    load_catalog,
    rankability_reasons,
    summarize_catalog,
    write_catalog,
)
from model_atlas.schemas.evidence import EvidenceKind


def _receipt(
    *,
    receipt_id: str = "gb10-exl3-k3-m2",
    m: int = 2,
    execution_path: KernelExecutionPath = KernelExecutionPath.DIRECT_PACKED,
    fused: bool = True,
    materialized: bool = False,
) -> KernelBenchmarkReceipt:
    return KernelBenchmarkReceipt(
        receipt_id=receipt_id,
        generated_at=datetime(2026, 8, 30, tzinfo=UTC),
        evidence_kind=EvidenceKind.MEASURED,
        run_status=KernelRunStatus.PASSED,
        hardware=KernelHardware(
            vendor="nvidia",
            device_name="NVIDIA GB10",
            compute_capability="12.1",
            cuda_version="13.0",
            driver_version="590.48.01",
        ),
        representation=KernelRepresentation(
            format="exl3",
            abi_name="exllamav3.exl3.mcg",
            abi_version=1,
            bits_per_weight=3.0,
            codebook="mcg-3bit",
            fused_transform=fused,
            full_precision_materialized=materialized,
        ),
        workload=KernelWorkload(
            model_id="fixture/moe-v1",
            operator="experts.down",
            phase=KernelPhase.DECODE,
            m=m,
            n=6144,
            k=1024,
            tp_world_size=2,
        ),
        backend=KernelBackend(
            name="atlas-exl3-sm121",
            kernel_name="k3_decode_small_m",
            repository="https://github.com/example/runtime",
            commit="0123456789abcdef",
            execution_path=execution_path,
        ),
        metrics=KernelMetrics(
            latency_ms=0.081,
            effective_bandwidth_gbps=412.0,
            achieved_tflops=0.31,
            reconstruction_overhead_pct=18.0,
            max_abs_error=0.002,
            mean_abs_error=0.0002,
            bottleneck=KernelBottleneck.MEMORY_BANDWIDTH,
        ),
        provenance=KernelProvenance(
            source_repository="https://github.com/example/runtime",
            source_commit="0123456789abcdef",
            command=["python", "-m", "kernel_lab.exl3.benchmark"],
            seed=0,
        ),
    )


def _query(*, m: int = 2) -> KernelQuery:
    return KernelQuery(
        device_name="NVIDIA GB10",
        compute_capability="12.1",
        cuda_version="13.0",
        driver_version="590.48.01",
        representation_format="exl3",
        abi_name="exllamav3.exl3.mcg",
        abi_version=1,
        phase=KernelPhase.DECODE,
        m=m,
        n=6144,
        k=1024,
        tp_world_size=2,
    )


def test_exact_direct_packed_measurement_is_oracle_eligible() -> None:
    catalog = KernelEvidenceCatalog(receipts=[_receipt()])
    result = catalog.query(_query())
    assert result.status is KernelEvidenceStatus.MEASURED
    assert result.eligible is True
    assert result.exact_shape is True
    assert result.latency_ms == 0.081


def test_same_m_bucket_is_estimate_and_never_manifest_eligible() -> None:
    catalog = KernelEvidenceCatalog(receipts=[_receipt(m=2)])
    exact = catalog.query(_query(m=3))
    estimate = catalog.query(_query(m=3), allow_bucket_estimate=True)
    assert exact.status is KernelEvidenceStatus.UNMEASURED
    assert estimate.status is KernelEvidenceStatus.ESTIMATED
    assert estimate.eligible is False
    with pytest.raises(KernelEvidenceError, match="execution manifest blocked"):
        build_execution_manifest(
            "candidate-a",
            [KernelRequirement(tensor_or_group="experts.*.down", query=_query(m=3))],
            catalog,
        )


def test_materialized_dequant_measurement_is_compatibility_only() -> None:
    receipt = _receipt(
        execution_path=KernelExecutionPath.MATERIALIZED_DEQUANT,
        fused=False,
        materialized=True,
    )
    reasons = rankability_reasons(receipt)
    assert "execution path is not a direct kernel path" in reasons
    assert "a full-precision weight tensor was materialized" in reasons
    result = KernelEvidenceCatalog(receipts=[receipt]).query(_query())
    assert result.status is KernelEvidenceStatus.COMPATIBILITY_ONLY
    assert result.eligible is False


def test_manifest_binds_only_exact_measured_receipts() -> None:
    catalog = KernelEvidenceCatalog(receipts=[_receipt()])
    manifest = build_execution_manifest(
        "candidate-a",
        [KernelRequirement(tensor_or_group="experts.*.down", query=_query())],
        catalog,
        generated_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    assert manifest.evidence_kind == EvidenceKind.MEASURED
    assert manifest.bindings[0].receipt_id == "gb10-exl3-k3-m2"
    with pytest.raises(KernelEvidenceError, match="no kernel requirements"):
        build_execution_manifest("candidate-empty", [], catalog)


def test_candidate_runtime_objective_uses_only_measured_exact_kernel_latency() -> None:
    catalog = KernelEvidenceCatalog(receipts=[_receipt()])
    assessment = assess_candidate_kernels(
        "candidate-a",
        [KernelRequirement(tensor_or_group="experts.*.down", query=_query())],
        catalog,
    )
    assert assessment.eligible is True
    assert assessment.status is KernelEvidenceStatus.MEASURED
    assert assessment.measured_kernel_latency_ms == 0.081


def test_unseen_model_and_native_non_exl3_representation_need_no_registration() -> None:
    assert (
        KernelMetrics(latency_ms=0.1, bottleneck="future_collective_overlap_bound").bottleneck
        == "future_collective_overlap_bound"
    )
    receipt = _receipt(
        receipt_id="future-native-model-m2",
        execution_path=KernelExecutionPath.DIRECT_NATIVE,
        fused=False,
        materialized=False,
    ).model_copy(
        update={
            "representation": KernelRepresentation(
                format="future-q5",
                abi_name="vendor.future.block-q5",
                abi_version=7,
                bits_per_weight=5.25,
                fused_transform=False,
                full_precision_materialized=False,
            ),
            "workload": KernelWorkload(
                model_id="publisher/model-released-after-atlas",
                operator="novel_moe.branch_17.projection_z",
                phase="vision_encode_v2",
                m=2,
                n=6144,
                k=1024,
                tp_world_size=2,
            ),
        }
    )
    query = _query().model_copy(
        update={
            "representation_format": "future-q5",
            "abi_name": "vendor.future.block-q5",
            "abi_version": 7,
            "phase": "vision_encode_v2",
        }
    )
    result = KernelEvidenceCatalog(receipts=[receipt]).query(query)
    assert result.status is KernelEvidenceStatus.MEASURED
    assert result.eligible is True


def test_first_branch_v1_field_names_remain_import_compatible() -> None:
    payload = _receipt().model_dump(mode="json")
    representation = payload["representation"]
    representation["fused_reconstruction"] = representation.pop("fused_transform")
    representation["full_dequant_materialized"] = representation.pop(
        "full_precision_materialized"
    )
    workload = payload["workload"]
    workload["model_family"] = workload.pop("model_id")
    workload["projection"] = workload.pop("operator")
    imported = KernelBenchmarkReceipt.model_validate(payload)
    assert imported.workload.model_id == "fixture/moe-v1"
    assert imported.workload.operator == "experts.down"
    assert imported.representation.fused_transform is True


def test_runtime_m0_cpu_receipt_imports_without_becoming_speed_evidence(tmp_path: Path) -> None:
    receipt_path = tmp_path / "m0.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "cpu-oracle",
                "generated_at": "2026-08-30T18:24:38+00:00",
                "capability": {
                    "available": False,
                    "compute_capability": None,
                    "cuda_version": None,
                    "reason": "PyTorch is not installed",
                },
                "cases": [
                    {
                        "elapsed_ms": 9.3,
                        "evidence": "measured",
                        "max_abs_error": 2e-7,
                        "mean_abs_error": 4e-8,
                        "passed": True,
                        "shape": {"m": 2, "n": 128, "k": 128},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    catalog = load_catalog([receipt_path])
    summary = summarize_catalog(catalog)
    assert summary["status"] == KernelEvidenceStatus.COMPATIBILITY_ONLY.value
    assert summary["counts"][KernelEvidenceStatus.MEASURED.value] == 0
    assert summary["rows"][0]["status"] == KernelEvidenceStatus.COMPATIBILITY_ONLY.value
    assert "hardware identity is incomplete" in summary["rows"][0]["reason"]


def test_legacy_sm121_case_imports_metrics_but_requires_full_hardware_identity(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "sm121.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "sm121",
                "generated_at": "2026-08-30T20:00:00+00:00",
                "capability": {
                    "available": True,
                    "compute_capability": [12, 1],
                    "cuda_version": "13.0",
                    "reason": "native SM121 EXL3 extension is available",
                },
                "cases": [
                    {
                        "model": "zai-org/GLM-5.2",
                        "projection": "down",
                        "tensor_parallel": 2,
                        "shape": {"m": 2, "n": 6144, "k": 1024},
                        "abi": {
                            "abi_name": "exllamav3.exl3.mcg",
                            "abi_version": 1,
                            "bits": 3,
                            "codebook": "mcg",
                            "exllamav3_commit": "c5d9c657966ffeeaa9353f0cc899f18629da4a13",
                        },
                        "direct_ms": 0.08,
                        "effective_bandwidth_gbps": 400.0,
                        "effective_tflops": 0.3,
                        "reconstruction_fraction_of_naive": 0.2,
                        "max_abs_error": 0.002,
                        "mean_abs_error": 0.0002,
                        "bottleneck": "packed_memory_or_pipeline_bound",
                        "no_full_dequantized_weight_in_direct_timing": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    receipt = load_catalog([receipt_path]).receipts[0]
    assert receipt.run_status is KernelRunStatus.PASSED
    assert receipt.hardware.compute_capability == "12.1"
    assert receipt.backend.commit == "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
    assert receipt.backend.execution_path is KernelExecutionPath.DIRECT_PACKED
    assert receipt.metrics.reconstruction_overhead_pct == 20.0
    assert rankability_reasons(receipt) == [
        "hardware identity is incomplete",
        "CUDA/driver identity is incomplete",
        "legacy producer schema is compatibility-only",
    ]


def test_catalog_write_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    catalog = KernelEvidenceCatalog(
        receipts=[_receipt(receipt_id="z-receipt"), _receipt(receipt_id="a-receipt")]
    )
    first = write_catalog(tmp_path / "first.json", catalog)
    second = write_catalog(tmp_path / "second.json", catalog)
    assert first.read_bytes() == second.read_bytes()
    loaded = load_catalog([first])
    assert [receipt.receipt_id for receipt in loaded.receipts] == ["a-receipt", "z-receipt"]
