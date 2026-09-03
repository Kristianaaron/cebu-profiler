"""Receipt import, lookup, and fail-closed manifest construction."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_atlas.kernels.schema import (
    CandidateKernelAssessment,
    KernelBackend,
    KernelBenchmarkReceipt,
    KernelBinding,
    KernelBottleneck,
    KernelEvidenceStatus,
    KernelExecutionManifest,
    KernelExecutionPath,
    KernelHardware,
    KernelMetrics,
    KernelOracleResult,
    KernelPhase,
    KernelProvenance,
    KernelQuery,
    KernelRepresentation,
    KernelRequirement,
    KernelRequirementAssessment,
    KernelRunStatus,
    KernelWorkload,
)
from model_atlas.schemas.evidence import EvidenceKind

_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_DIRECT_EXECUTION_PATHS = frozenset(
    {
        KernelExecutionPath.DIRECT_NATIVE,
        KernelExecutionPath.DIRECT_PACKED,
        KernelExecutionPath.DIRECT_SPARSE,
    }
)


class KernelEvidenceError(ValueError):
    """A receipt or runtime binding is invalid or lacks required evidence."""


class KernelEvidenceCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["atlas.kernel-catalog/v1"] = "atlas.kernel-catalog/v1"
    receipts: list[KernelBenchmarkReceipt] = Field(default_factory=list)

    def normalized(self) -> KernelEvidenceCatalog:
        by_id: dict[str, KernelBenchmarkReceipt] = {}
        for receipt in self.receipts:
            prior = by_id.get(receipt.receipt_id)
            if prior is not None and prior != receipt:
                raise KernelEvidenceError(f"conflicting receipt id {receipt.receipt_id!r}")
            by_id[receipt.receipt_id] = receipt
        return KernelEvidenceCatalog(receipts=[by_id[key] for key in sorted(by_id)])

    def find_receipt(self, receipt_id: str) -> KernelBenchmarkReceipt:
        for receipt in self.receipts:
            if receipt.receipt_id == receipt_id:
                return receipt
        raise KernelEvidenceError(f"receipt {receipt_id!r} is absent from the catalog")

    def query(
        self, query: KernelQuery, *, allow_bucket_estimate: bool = False
    ) -> KernelOracleResult:
        candidates = [r for r in self.receipts if _matches_identity(r, query)]
        exact = [r for r in candidates if r.workload.m == query.m]
        eligible = [r for r in exact if not rankability_reasons(r)]
        if eligible:
            winner = min(eligible, key=lambda r: r.metrics.latency_ms or float("inf"))
            return KernelOracleResult(
                status=KernelEvidenceStatus.MEASURED,
                eligible=True,
                exact_shape=True,
                receipt_id=winner.receipt_id,
                latency_ms=winner.metrics.latency_ms,
            )
        if exact:
            receipt = min(exact, key=lambda r: r.receipt_id)
            reasons = rankability_reasons(receipt)
            status = _unrankable_status(receipt)
            return KernelOracleResult(
                status=status,
                eligible=False,
                exact_shape=True,
                receipt_id=receipt.receipt_id,
                latency_ms=receipt.metrics.latency_ms,
                reasons=reasons,
            )
        if allow_bucket_estimate:
            bucket = [
                r
                for r in candidates
                if r.workload.m_bucket == query.m_bucket and not rankability_reasons(r)
            ]
            if bucket:
                winner = min(bucket, key=lambda r: r.metrics.latency_ms or float("inf"))
                return KernelOracleResult(
                    status=KernelEvidenceStatus.ESTIMATED,
                    eligible=False,
                    exact_shape=False,
                    receipt_id=winner.receipt_id,
                    latency_ms=winner.metrics.latency_ms,
                    reasons=[
                        f"only M={winner.workload.m} was measured in tuning bucket {query.m_bucket}"
                    ],
                )
        return KernelOracleResult(
            status=KernelEvidenceStatus.UNMEASURED,
            eligible=False,
            exact_shape=False,
            reasons=["no compatible exact-shape direct-kernel measurement"],
        )


def rankability_reasons(receipt: KernelBenchmarkReceipt) -> list[str]:
    """Explain why a receipt may not influence speed rankings or manifests."""
    reasons: list[str] = []
    if receipt.evidence_kind is not EvidenceKind.MEASURED:
        reasons.append(f"evidence kind is {receipt.evidence_kind.value}, not measured")
    if receipt.run_status is not KernelRunStatus.PASSED:
        reasons.append(f"run status is {receipt.run_status.value}")
    if not receipt.hardware.compute_capability or receipt.hardware.device_name == "unknown":
        reasons.append("hardware identity is incomplete")
    if not receipt.hardware.cuda_version or not receipt.hardware.driver_version:
        reasons.append("CUDA/driver identity is incomplete")
    if receipt.backend.execution_path not in _DIRECT_EXECUTION_PATHS:
        reasons.append("execution path is not a direct kernel path")
    if (
        receipt.backend.execution_path is KernelExecutionPath.DIRECT_PACKED
        and not receipt.representation.fused_transform
    ):
        reasons.append("packed representation transform is not fused with compute")
    if receipt.representation.full_precision_materialized:
        reasons.append("a full-precision weight tensor was materialized")
    if not receipt.backend.commit:
        reasons.append("backend commit is missing")
    if receipt.provenance.producer_schema != "atlas.kernel-benchmark/v1":
        reasons.append("legacy producer schema is compatibility-only")
    if not receipt.provenance.source_commit:
        reasons.append("runtime source commit is missing")
    if receipt.metrics.latency_ms is None:
        reasons.append("latency is missing")
    return reasons


def build_execution_manifest(
    candidate_id: str,
    requirements: list[KernelRequirement],
    catalog: KernelEvidenceCatalog,
    *,
    generated_at: datetime | None = None,
) -> KernelExecutionManifest:
    """Bind every requirement or fail; estimates are intentionally ineligible."""
    if not requirements:
        raise KernelEvidenceError("execution manifest blocked: no kernel requirements")
    bindings: list[KernelBinding] = []
    failures: list[str] = []
    for requirement in requirements:
        result = catalog.query(requirement.query)
        if not result.eligible or result.receipt_id is None or result.latency_ms is None:
            failures.append(
                f"{requirement.tensor_or_group}: {result.status.value}: "
                + "; ".join(result.reasons)
            )
            continue
        receipt = catalog.find_receipt(result.receipt_id)
        bindings.append(
            KernelBinding(
                tensor_or_group=requirement.tensor_or_group,
                receipt_id=receipt.receipt_id,
                backend_name=receipt.backend.name,
                backend_commit=receipt.backend.commit,
                kernel_name=receipt.backend.kernel_name,
                latency_ms=result.latency_ms,
            )
        )
    if failures:
        raise KernelEvidenceError("execution manifest blocked: " + " | ".join(failures))
    return KernelExecutionManifest(
        candidate_id=candidate_id,
        generated_at=generated_at or datetime.now(UTC),
        bindings=bindings,
    )


def assess_candidate_kernels(
    candidate_id: str,
    requirements: list[KernelRequirement],
    catalog: KernelEvidenceCatalog,
) -> CandidateKernelAssessment:
    """Build the measured kernel-latency objective used by candidate selection.

    This deliberately reports the sum of required kernel observations, not
    model tokens/second. End-to-end decode throughput needs its own measured
    runtime receipt and must not be inferred from this component objective.
    """
    rows = [
        KernelRequirementAssessment(
            tensor_or_group=requirement.tensor_or_group,
            result=catalog.query(requirement.query),
        )
        for requirement in requirements
    ]
    eligible = bool(rows) and all(row.result.eligible for row in rows)
    if eligible:
        total = sum(row.result.latency_ms or 0.0 for row in rows)
        status = KernelEvidenceStatus.MEASURED
    else:
        total = None
        statuses = {row.result.status for row in rows}
        status = next(
            (
                candidate
                for candidate in (
                    KernelEvidenceStatus.FAILED,
                    KernelEvidenceStatus.UNSUPPORTED,
                    KernelEvidenceStatus.UNMEASURED,
                    KernelEvidenceStatus.COMPATIBILITY_ONLY,
                    KernelEvidenceStatus.ESTIMATED,
                )
                if candidate in statuses
            ),
            KernelEvidenceStatus.UNMEASURED,
        )
    return CandidateKernelAssessment(
        candidate_id=candidate_id,
        eligible=eligible,
        status=status,
        measured_kernel_latency_ms=total,
        requirements=rows,
    )


def load_catalog(paths: Sequence[str | Path]) -> KernelEvidenceCatalog:
    receipts: list[KernelBenchmarkReceipt] = []
    for raw_path in paths:
        path = Path(raw_path)
        encoded = _read_bounded(path)
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KernelEvidenceError(f"{path}: receipt is not valid JSON") from exc
        receipts.extend(_parse_payload(payload, source_sha256=hashlib.sha256(encoded).hexdigest()))
    return KernelEvidenceCatalog(receipts=receipts).normalized()


def write_catalog(path: str | Path, catalog: KernelEvidenceCatalog) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = catalog.normalized().model_dump_json(indent=2).encode() + b"\n"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    return destination


def summarize_catalog(catalog: KernelEvidenceCatalog) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts = {status.value: 0 for status in KernelEvidenceStatus}
    for receipt in catalog.normalized().receipts:
        reasons = rankability_reasons(receipt)
        status = KernelEvidenceStatus.MEASURED if not reasons else _unrankable_status(receipt)
        counts[status.value] += 1
        rows.append(
            {
                "receipt_id": receipt.receipt_id,
                "status": status.value,
                "hardware": receipt.hardware.device_name,
                "cc": receipt.hardware.compute_capability or "-",
                "representation": (
                    f"{receipt.representation.format}/"
                    f"{receipt.representation.abi_name}@{receipt.representation.abi_version}"
                ),
                "phase": receipt.workload.phase,
                "shape": f"{receipt.workload.m}x{receipt.workload.n}x{receipt.workload.k}",
                "latency_ms": receipt.metrics.latency_ms,
                "bottleneck": receipt.metrics.bottleneck,
                "reason": "; ".join(reasons),
            }
        )
    overall = KernelEvidenceStatus.UNMEASURED.value
    if rows:
        for candidate in (
            KernelEvidenceStatus.MEASURED,
            KernelEvidenceStatus.COMPATIBILITY_ONLY,
            KernelEvidenceStatus.ESTIMATED,
            KernelEvidenceStatus.FAILED,
            KernelEvidenceStatus.UNSUPPORTED,
        ):
            if counts[candidate.value]:
                overall = candidate.value
                break
    return {"status": overall, "counts": counts, "rows": rows}


def _matches_identity(receipt: KernelBenchmarkReceipt, query: KernelQuery) -> bool:
    return (
        receipt.hardware.device_name == query.device_name
        and receipt.hardware.compute_capability == query.compute_capability
        and receipt.hardware.cuda_version == query.cuda_version
        and receipt.hardware.driver_version == query.driver_version
        and receipt.representation.format == query.representation_format
        and receipt.representation.abi_name == query.abi_name
        and receipt.representation.abi_version == query.abi_version
        and receipt.workload.phase == query.phase
        and receipt.workload.n == query.n
        and receipt.workload.k == query.k
        and receipt.workload.tp_world_size == query.tp_world_size
        and receipt.workload.grouped_moe == query.grouped_moe
        and (query.backend_commit is None or receipt.backend.commit == query.backend_commit)
    )


def _unrankable_status(receipt: KernelBenchmarkReceipt) -> KernelEvidenceStatus:
    if receipt.run_status is KernelRunStatus.FAILED:
        return KernelEvidenceStatus.FAILED
    if receipt.run_status is KernelRunStatus.UNSUPPORTED:
        return KernelEvidenceStatus.UNSUPPORTED
    if receipt.evidence_kind is EvidenceKind.ESTIMATED:
        return KernelEvidenceStatus.ESTIMATED
    return KernelEvidenceStatus.COMPATIBILITY_ONLY


def _read_bounded(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise KernelEvidenceError(f"{path}: receipt must be a regular non-symlink file")
    size = path.stat().st_size
    if size > _MAX_RECEIPT_BYTES:
        raise KernelEvidenceError(f"{path}: receipt exceeds {_MAX_RECEIPT_BYTES} bytes")
    return path.read_bytes()


def _parse_payload(payload: Any, *, source_sha256: str) -> list[KernelBenchmarkReceipt]:
    if not isinstance(payload, dict):
        raise KernelEvidenceError("kernel receipt root must be an object")
    try:
        if payload.get("schema_version") == "atlas.kernel-benchmark/v1":
            return [KernelBenchmarkReceipt.model_validate(payload)]
        if payload.get("schema_version") == "atlas.kernel-catalog/v1":
            return KernelEvidenceCatalog.model_validate(payload).receipts
        if (
            payload.get("schema_version") == 1
            and {"backend", "capability", "cases"} <= payload.keys()
        ):
            return _adapt_milestone0(payload, source_sha256=source_sha256)
    except ValidationError as exc:
        raise KernelEvidenceError(f"kernel receipt validation failed: {exc}") from exc
    version = payload.get("schema_version")
    raise KernelEvidenceError(f"unsupported kernel receipt schema {version!r}")


def _adapt_milestone0(
    payload: dict[str, Any], *, source_sha256: str
) -> list[KernelBenchmarkReceipt]:
    """Import the runtime lab's v1 M0 receipt without upgrading its evidence."""
    capability = payload.get("capability")
    cases = payload.get("cases")
    if not isinstance(capability, dict) or not isinstance(cases, list):
        raise KernelEvidenceError("legacy M0 receipt capability/cases are malformed")
    backend_name = str(payload.get("backend", "unknown"))
    generated_at_raw = payload.get("generated_at")
    if not isinstance(generated_at_raw, str):
        raise KernelEvidenceError("legacy M0 receipt generated_at is missing")
    try:
        generated_at = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KernelEvidenceError("legacy M0 receipt generated_at is invalid") from exc
    receipts: list[KernelBenchmarkReceipt] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("shape"), dict):
            raise KernelEvidenceError("legacy M0 receipt case is malformed")
        shape = case["shape"]
        case_digest = hashlib.sha256(
            json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        is_cpu = backend_name == "cpu-oracle"
        available = capability.get("available") is True
        compute_capability = capability.get("compute_capability")
        if isinstance(compute_capability, list) and len(compute_capability) == 2:
            compute_capability = f"{compute_capability[0]}.{compute_capability[1]}"
        device_name = str(capability.get("device_name") or "unknown")
        direct_ms = case.get("direct_ms")
        elapsed_ms = case.get("elapsed_ms", direct_ms)
        if elapsed_ms is None:
            raise KernelEvidenceError("legacy M0 receipt case has no latency")
        raw_abi = case.get("abi")
        abi: dict[str, Any] = raw_abi if isinstance(raw_abi, dict) else {}
        direct_proof = case.get("no_full_dequantized_weight_in_direct_timing") is True
        execution_path = (
            KernelExecutionPath.CPU_REFERENCE
            if is_cpu
            else (
                KernelExecutionPath.DIRECT_PACKED
                if direct_proof
                else KernelExecutionPath.UNKNOWN
            )
        )
        bottleneck_map = {
            "mma_or_launch_bound": KernelBottleneck.MIXED,
            "exl3_reconstruction_bound": KernelBottleneck.RECONSTRUCTION,
            "packed_memory_or_pipeline_bound": KernelBottleneck.MEMORY_BANDWIDTH,
        }
        bottleneck = bottleneck_map.get(
            str(case.get("bottleneck")),
            KernelBottleneck.UNAVAILABLE if is_cpu else KernelBottleneck.UNKNOWN,
        )
        passed = case.get("passed", True) is True
        backend_commit = str(abi.get("exllamav3_commit") or "")
        receipts.append(
            KernelBenchmarkReceipt(
                receipt_id=f"legacy-m0-{index}-{case_digest}",
                generated_at=generated_at,
                evidence_kind=EvidenceKind(str(case.get("evidence", "measured"))),
                run_status=KernelRunStatus.PASSED if passed else KernelRunStatus.FAILED,
                hardware=KernelHardware(
                    vendor="nvidia" if available else "unknown",
                    device_name=device_name,
                    compute_capability=(str(compute_capability) if compute_capability else None),
                    cuda_version=(
                        str(capability["cuda_version"])
                        if capability.get("cuda_version") is not None
                        else None
                    ),
                ),
                representation=KernelRepresentation(
                    format="exl3",
                    abi_name=str(abi.get("abi_name") or "exllamav3.exl3.mcg"),
                    abi_version=int(abi.get("abi_version", 1)),
                    bits_per_weight=float(abi.get("bits", 3.0)),
                    codebook=str(abi.get("codebook") or "mcg"),
                    fused_transform=direct_proof,
                    full_precision_materialized=not direct_proof,
                ),
                workload=KernelWorkload(
                    model_id=str(case.get("model") or "milestone0-synthetic"),
                    operator=str(case.get("projection") or "synthetic"),
                    phase=KernelPhase.DECODE,
                    m=int(shape["m"]),
                    n=int(shape["n"]),
                    k=int(shape["k"]),
                    tp_world_size=int(case.get("tensor_parallel", 1)),
                ),
                backend=KernelBackend(
                    name=backend_name,
                    kernel_name=backend_name,
                    repository=(
                        "https://github.com/turboderp-org/exllamav3"
                        if backend_commit
                        else "runtime-receipt"
                    ),
                    commit=backend_commit,
                    execution_path=execution_path,
                ),
                metrics=KernelMetrics(
                    latency_ms=float(elapsed_ms),
                    effective_bandwidth_gbps=(
                        float(case["effective_bandwidth_gbps"])
                        if case.get("effective_bandwidth_gbps") is not None
                        else None
                    ),
                    achieved_tflops=(
                        float(case["effective_tflops"])
                        if case.get("effective_tflops") is not None
                        else None
                    ),
                    reconstruction_overhead_pct=(
                        100.0 * float(case["reconstruction_fraction_of_naive"])
                        if case.get("reconstruction_fraction_of_naive") is not None
                        else None
                    ),
                    max_abs_error=float(case["max_abs_error"]),
                    mean_abs_error=float(case["mean_abs_error"]),
                    bottleneck=bottleneck,
                ),
                provenance=KernelProvenance(
                    producer_schema="kernel-lab-m0/v1",
                    source_repository="runtime-receipt",
                    source_commit=str(abi.get("exllamav3_commit") or ""),
                    source_receipt_sha256=source_sha256,
                    notes=[str(capability.get("reason") or "legacy M0 adapter")],
                ),
            )
        )
    return receipts
