"""Versioned runtime-kernel evidence contracts for Cebu Profiler.

The profiler consumes receipts; it does not own CUDA kernel implementation.  A
receipt is eligible for runtime decisions only when it describes a measured,
direct execution on identified hardware at the exact requested shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from cebu_profiler.schemas.evidence import EvidenceKind


class KernelPhase(StrEnum):
    """Convenience constants, not a schema allowlist; workload phases are strings."""

    DECODE = "decode"
    PREFILL = "prefill"


class KernelRunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class KernelExecutionPath(StrEnum):
    DIRECT_NATIVE = "direct_native"
    DIRECT_PACKED = "direct_packed"
    DIRECT_SPARSE = "direct_sparse"
    MATERIALIZED_DEQUANT = "materialized_dequant"
    CPU_REFERENCE = "cpu_reference"
    UNKNOWN = "unknown"


class KernelBottleneck(StrEnum):
    """Common labels; receipt bottlenecks remain extensible strings."""

    MEMORY_BANDWIDTH = "memory_bandwidth"
    RECONSTRUCTION = "reconstruction"
    COMPUTE = "compute"
    LAUNCH = "launch"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class KernelEvidenceStatus(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    COMPATIBILITY_ONLY = "compatibility_only"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNMEASURED = "unmeasured"


class KernelHardware(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    device_name: str
    compute_capability: str | None = None
    device_uuid: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None


class KernelRepresentation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str
    abi_name: str
    abi_version: int = Field(ge=1)
    bits_per_weight: float = Field(gt=0.0)
    codebook: str | None = None
    fused_transform: bool = Field(
        validation_alias=AliasChoices("fused_transform", "fused_reconstruction")
    )
    full_precision_materialized: bool = Field(
        validation_alias=AliasChoices("full_precision_materialized", "full_dequant_materialized")
    )


class KernelWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(validation_alias=AliasChoices("model_id", "model_family"))
    operator: str = Field(validation_alias=AliasChoices("operator", "projection"))
    phase: str = Field(min_length=1)
    m: int = Field(ge=1)
    n: int = Field(ge=1)
    k: int = Field(ge=1)
    tp_world_size: int = Field(default=1, ge=1)
    grouped_moe: bool = False

    @property
    def m_bucket(self) -> str:
        return m_bucket(self.m)


class KernelBackend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kernel_name: str
    repository: str
    commit: str
    execution_path: KernelExecutionPath


class KernelMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_ms: float | None = Field(default=None, gt=0.0)
    effective_bandwidth_gbps: float | None = Field(default=None, ge=0.0)
    achieved_tflops: float | None = Field(default=None, ge=0.0)
    reconstruction_overhead_pct: float | None = Field(default=None, ge=0.0)
    max_abs_error: float | None = Field(default=None, ge=0.0)
    mean_abs_error: float | None = Field(default=None, ge=0.0)
    bottleneck: str = KernelBottleneck.UNKNOWN


class KernelProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    producer_schema: str = "cebu.kernel-benchmark/v1"
    source_repository: str
    source_commit: str
    command: list[str] = Field(default_factory=list)
    seed: int | None = None
    source_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    notes: list[str] = Field(default_factory=list)


class KernelBenchmarkReceipt(BaseModel):
    """One immutable benchmark observation at one exact workload shape."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cebu.kernel-benchmark/v1"] = "cebu.kernel-benchmark/v1"
    receipt_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
    generated_at: datetime
    evidence_kind: EvidenceKind
    run_status: KernelRunStatus
    hardware: KernelHardware
    representation: KernelRepresentation
    workload: KernelWorkload
    backend: KernelBackend
    metrics: KernelMetrics
    provenance: KernelProvenance

    @field_validator("generated_at")
    @classmethod
    def _timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _passed_receipt_has_measurement(self) -> KernelBenchmarkReceipt:
        if self.run_status is KernelRunStatus.PASSED and self.metrics.latency_ms is None:
            raise ValueError("a passed kernel receipt requires latency_ms")
        return self


class KernelQuery(BaseModel):
    """The exact kernel requirement emitted by an Cebu Profiler candidate."""

    model_config = ConfigDict(extra="forbid")

    device_name: str
    compute_capability: str
    cuda_version: str
    driver_version: str
    representation_format: str
    abi_name: str
    abi_version: int = Field(ge=1)
    phase: str = Field(min_length=1)
    m: int = Field(ge=1)
    n: int = Field(ge=1)
    k: int = Field(ge=1)
    tp_world_size: int = Field(default=1, ge=1)
    grouped_moe: bool = False
    backend_commit: str | None = None

    @property
    def m_bucket(self) -> str:
        return m_bucket(self.m)


class KernelOracleKey(BaseModel):
    """Stable index key; exact-M is retained alongside the tuning bucket."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_name: str
    compute_capability: str
    cuda_version: str
    driver_version: str
    representation_format: str
    abi_name: str
    abi_version: int
    phase: str
    m: int
    m_bucket: str
    n: int
    k: int
    tp_world_size: int
    grouped_moe: bool
    backend_commit: str


class KernelOracleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: KernelEvidenceStatus
    eligible: bool
    exact_shape: bool
    receipt_id: str | None = None
    latency_ms: float | None = None
    reasons: list[str] = Field(default_factory=list)


class KernelRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_or_group: str
    query: KernelQuery


class KernelRequirementAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_or_group: str
    result: KernelOracleResult


class CandidateKernelAssessment(BaseModel):
    """Candidate-facing runtime objective; never extrapolates model throughput."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    eligible: bool
    status: KernelEvidenceStatus
    measured_kernel_latency_ms: float | None = Field(default=None, ge=0.0)
    requirements: list[KernelRequirementAssessment]


class KernelBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_or_group: str
    receipt_id: str
    backend_name: str
    backend_commit: str
    kernel_name: str
    latency_ms: float


class KernelExecutionManifest(BaseModel):
    """Fail-closed binding from a candidate to measured runtime kernels."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cebu.kernel-execution-manifest/v1"] = (
        "cebu.kernel-execution-manifest/v1"
    )
    candidate_id: str
    generated_at: datetime
    evidence_kind: Literal[EvidenceKind.MEASURED] = EvidenceKind.MEASURED
    bindings: list[KernelBinding] = Field(min_length=1)


class KernelManifestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    requirements: list[KernelRequirement] = Field(min_length=1)


def m_bucket(value: int) -> str:
    """Small-M tuning bucket used for lookup diagnostics and future autotuning."""
    if value == 1:
        return "1"
    for upper in (4, 8, 16, 32, 64, 128):
        lower = 2 if upper == 4 else upper // 2 + 1
        if value <= upper:
            return f"{lower}-{upper}"
    return "129+"
