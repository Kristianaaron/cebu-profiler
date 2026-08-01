"""Model asset lifecycle (v2 §5)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssetType(StrEnum):
    SOURCE_CHECKPOINT = "source_checkpoint"
    RUNNABLE_LOCAL_CHECKPOINT = "runnable_local_checkpoint"
    DERIVATIVE_CHECKPOINT = "derivative_checkpoint"
    LOCAL_ENDPOINT = "local_endpoint"
    REMOTE_TEACHER_ENDPOINT = "remote_teacher_endpoint"
    STUDENT_MODEL = "student_model"
    DRAFT_MODEL = "draft_model"
    AUXILIARY_MULTIMODAL = "auxiliary_multimodal"


class ModelAsset(BaseModel):
    """An immutable model-asset record. An oversized checkpoint is still a valid
    source asset even when not directly evaluable."""

    model_config = ConfigDict(extra="forbid")

    model_asset_id: str
    display_name: str
    asset_type: AssetType
    model_family: str | None = None
    architecture: str | None = None
    checkpoint_path: str | None = None
    endpoint: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_hash: str | None = None
    quantization: str | None = None
    stored_size_bytes: int | None = Field(default=None, ge=0)
    estimated_resident_bytes: int | None = Field(default=None, ge=0)
    runtime_compatibility: bool | None = None
    atlas_compatibility: bool | None = None
    validation_status: str | None = None
    parent_model_id: str | None = None
    source_atlas_id: str | None = None
    source_experiment_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _requires_location(self) -> ModelAsset:
        if not self.checkpoint_path and not self.endpoint:
            raise ValueError("ModelAsset needs either checkpoint_path or endpoint")
        if self.asset_type == AssetType.SOURCE_CHECKPOINT and not self.checkpoint_path:
            raise ValueError("source_checkpoint requires a checkpoint_path")
        return self
