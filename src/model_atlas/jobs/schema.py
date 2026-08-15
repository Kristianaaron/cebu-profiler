"""Durable job/run schemas + state machine for the Atlas compression plane.

A :class:`Job` is the durable, resumable record of executing a *compiled
recipe*. Its directory holds the immutable plan, an append-only JSONL event
stream, per-stage content-addressed outputs, evidence, and the final manifest —
everything a UI or an agent needs to inspect a run without touching the engine's
internals.

State machine (job-level):

    PENDING -> PREPARING -> RUNNING <-> RESUMING
        -> COMPLETED | FAILED_TERMINAL | FAILED_RECOVERABLE | CANCELLED

Stage-level: PENDING -> RUNNING -> DONE | FAILED | SKIPPED

Everything that describes "what happened" lives in the journal; the manifest is
derived and stable once the run completes. Stage outputs are content-addressed:
re-running a stage with identical inputs+code produces identical hashes, which
is what makes replay/idempotency testable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.schemas.evidence import EvidenceKind

JOB_SCHEMA_VERSION = 1


class JobStatus(StrEnum):
    PENDING = "pending"
    PREPARING = "preparing"
    RUNNING = "running"
    RESUMING = "resuming"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED_TERMINAL = "failed_terminal"
    FAILED_RECOVERABLE = "failed_recoverable"
    CANCELLED = "cancelled"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class OutputRef(BaseModel):
    """Content-addressed reference to one stage output artifact."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sha256: str
    size_bytes: int = Field(default=0, ge=0)
    format: str = ""
    relpath: str = ""  # relative path inside the run dir ('' = not materialized)


class StageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    status: StageStatus = StageStatus.PENDING
    outputs: list[OutputRef] = Field(default_factory=list)
    evidence_kind: EvidenceKind = EvidenceKind.PREDICTED
    evidence_reported: str = ""  # what the backend reported (never upgraded)
    handle: str | None = None  # backend handle for resume
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""
    exit_code: int | None = None


class StageEvidence(BaseModel):
    """Typed evidence for one stage result (predicted vs measured)."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str
    evidence_kind: EvidenceKind
    declared_policy: EvidenceKind  # planner's ceiling for this stage
    gates_passed: bool = False
    measured_claim: bool = False  # evidence says MEASURED/CAUSALLY_TESTED
    note: str = ""


class RepairRecord(BaseModel):
    """One applied (or reverted) deterministic repair."""

    model_config = ConfigDict(extra="forbid")

    repair_id: str
    kind: str  # must be on the deterministic-allowlist
    target: str  # stage_id or artifact ref
    before_sha256: str = ""
    after_sha256: str = ""
    authorized_by: str = "compiler"
    applied: bool = False
    reverted: bool = False
    note: str = ""


class Job(BaseModel):
    """The immutable-ish persisted job record. Mutated ONLY via atomic replace
    of a fresh copy (never in place), so a crash mid-write cannot corrupt it."""

    model_config = ConfigDict(extra="forbid")

    job_schema_version: int = JOB_SCHEMA_VERSION
    run_id: str
    recipe_id: str
    recipe_sha256: str
    plan_id: str
    run_dir: str
    status: JobStatus = JobStatus.PENDING
    stages: dict[str, StageOutput] = Field(default_factory=dict)
    stage_order: list[str] = Field(default_factory=list)
    journal_path: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    failed_stage: str | None = None
    error: str = ""
    inputs: dict[str, object] = Field(default_factory=dict)
    source_snapshot: dict[str, object] = Field(default_factory=dict)
    source_manifest_digest: str = ""
    repair: list[RepairRecord] = Field(default_factory=list)

    def stage(self, stage_id: str) -> StageOutput:
        if stage_id not in self.stages:
            self.stages[stage_id] = StageOutput(stage_id=stage_id)
        return self.stages[stage_id]

    def is_terminal(self) -> bool:
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_WITH_WARNINGS,
            JobStatus.FAILED_TERMINAL,
            JobStatus.FAILED_RECOVERABLE,
            JobStatus.CANCELLED,
        }

    @property
    def completed_ok(self) -> bool:
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_WITH_WARNINGS,
        }
