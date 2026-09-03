"""Cebu Profiler run schema + state machine (v2 §3C)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.schemas.evidence import EvidenceLevel
from cebu_profiler.schemas.ontology import DataPartition


class ProfilerRunStatus(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    ESTIMATING = "estimating"
    TRACING = "tracing"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED_RECOVERABLE = "failed_recoverable"
    FAILED_TERMINAL = "failed_terminal"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"


# States in which Pause/Cancel are permitted (at safe boundaries only).
_PAUSABLE = frozenset(
    {ProfilerRunStatus.TRACING, ProfilerRunStatus.VALIDATING, ProfilerRunStatus.ESTIMATING}
)


class ProfilerRunProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_layer: int = Field(default=0, ge=0)
    total_layers: int = Field(default=0, ge=0)
    stage: str | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    estimated_remaining_seconds: float | None = Field(default=None, ge=0.0)


class ProfilerRun(BaseModel):
    """A resumable, long-running Cebu Profiler analysis job (v2 §3C / §8)."""

    model_config = ConfigDict(extra="forbid")

    profiler_run_id: str
    source_model_asset_id: str
    source_checkpoint_revision: str | None = None
    calibration_suite_id: str
    data_partition: DataPartition = DataPartition.CEBU_CALIBRATION
    evidence_level: EvidenceLevel = EvidenceLevel.BASIC_SALIENCY
    status: ProfilerRunStatus = ProfilerRunStatus.DRAFT
    progress: ProfilerRunProgress | None = None
    warnings: list[str] = Field(default_factory=list)
    failure_info: dict[str, str] = Field(default_factory=dict)
    configuration_hash: str | None = None
    evidence_present: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_pausable(self) -> bool:
        return self.status in _PAUSABLE

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            ProfilerRunStatus.COMPLETED,
            ProfilerRunStatus.COMPLETED_WITH_WARNINGS,
            ProfilerRunStatus.CANCELLED,
            ProfilerRunStatus.FAILED_TERMINAL,
        }
