"""Pinned, candidate-only handoff contract for the external Eval Lab.

This module deliberately does not launch Eval Lab.  It freezes the held-out
inputs, emits an argv for one reviewed Eval Lab revision, and validates the
files returned by a separately operated harness.  Candidate-only reports
cannot represent teacher-relative KLD or CKA.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVAL_LAB_REVISION = "5ee2f7cc33627b6259c0b10100d84932e676f36c"
_SHA256 = r"^[0-9a-f]{64}$"
_REVISION = r"^[0-9a-f]{40}$"


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class DataPartition(StrEnum):
    UNSET = "unset"
    CALIBRATION = "calibration"
    HELD_OUT_EVALUATION = "held_out_evaluation"


class EndpointTransport(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"


class EndpointConfigIdentity(_StrictFrozenModel):
    """Non-secret identity of an already-operated candidate endpoint."""

    endpoint_id: str = Field(min_length=1)
    transport: EndpointTransport
    config_sha256: str = Field(pattern=_SHA256)


class FrozenHeldOutManifest(_StrictFrozenModel):
    """Content-addressed evaluation inputs and explicit leakage evidence."""

    schema_version: Literal[1] = 1
    manifest_id: str | None = Field(default=None, pattern=_SHA256)
    data_partition: DataPartition
    task_suite_id: str = Field(min_length=1)
    task_suite_revision: str = Field(pattern=_REVISION)
    task_suite_sha256: str = Field(pattern=_SHA256)
    task_definitions_sha256: str = Field(pattern=_SHA256)
    tracked_task_ids: list[str] = Field(min_length=1)
    corpus_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    template_sha256: str = Field(pattern=_SHA256)
    evaluation_sample_ids: list[str] = Field(min_length=1)
    calibration_sample_ids: list[str] = Field(default_factory=list)

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"manifest_id"})

    @model_validator(mode="after")
    def _frozen_and_leak_free(self) -> FrozenHeldOutManifest:
        if self.data_partition is not DataPartition.HELD_OUT_EVALUATION:
            raise ValueError("evaluation manifest must use held_out_evaluation partition")
        if len(self.tracked_task_ids) != len(set(self.tracked_task_ids)):
            raise ValueError("tracked_task_ids must be unique")
        if len(self.evaluation_sample_ids) != len(set(self.evaluation_sample_ids)):
            raise ValueError("evaluation_sample_ids must be unique")
        if len(self.calibration_sample_ids) != len(set(self.calibration_sample_ids)):
            raise ValueError("calibration_sample_ids must be unique")
        overlap = set(self.evaluation_sample_ids) & set(self.calibration_sample_ids)
        if overlap:
            raise ValueError("held-out/calibration sample ID overlap detected")
        expected = _canonical_digest(self.identity_payload())
        if self.manifest_id is not None and self.manifest_id != expected:
            raise ValueError("held-out manifest_id does not match canonical content")
        object.__setattr__(self, "manifest_id", expected)
        return self


class EvalParameters(_StrictFrozenModel):
    seed: int = Field(ge=0)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0.0)

    @field_validator("temperature", "timeout_seconds")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("evaluation parameters must be finite")
        return value


class EvalLabRequest(_StrictFrozenModel):
    """Complete candidate evaluation request with a timestamp-free identity."""

    schema_version: Literal[1] = 1
    request_id: str | None = Field(default=None, pattern=_SHA256)
    candidate_artifact_path: str = Field(min_length=1)
    candidate_artifact_sha256: str = Field(pattern=_SHA256)
    endpoint: EndpointConfigIdentity
    held_out: FrozenHeldOutManifest
    tasks: list[str] = Field(min_length=1)
    parameters: EvalParameters

    @field_validator("candidate_artifact_path")
    @classmethod
    def _absolute_candidate_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("candidate_artifact_path must be absolute")
        return value

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"request_id"})

    @model_validator(mode="after")
    def _bind_heldout_and_identity(self) -> EvalLabRequest:
        if self.held_out.data_partition is not DataPartition.HELD_OUT_EVALUATION:
            raise ValueError("measured evaluation requires held_out_evaluation")
        if len(self.tasks) != len(set(self.tasks)):
            raise ValueError("tasks must be unique")
        untracked = set(self.tasks) - set(self.held_out.tracked_task_ids)
        if untracked:
            raise ValueError("request contains tasks not pinned by the held-out manifest")
        expected = _canonical_digest(self.identity_payload())
        if self.request_id is not None and self.request_id != expected:
            raise ValueError("request_id does not match canonical request content")
        object.__setattr__(self, "request_id", expected)
        return self


class TeacherRelativeBlocker(StrEnum):
    MISSING_BF16_TEACHER = "missing_bf16_teacher"
    FULL_LOGITS_UNAVAILABLE = "full_logits_unavailable"
    HIDDEN_ACTIVATIONS_UNAVAILABLE = "hidden_activations_unavailable"


class TaskScore(_StrictFrozenModel):
    task_id: str = Field(min_length=1)
    scores: dict[str, float] = Field(min_length=1)

    @field_validator("scores")
    @classmethod
    def _finite_scores(cls, values: dict[str, float]) -> dict[str, float]:
        if not all(name and math.isfinite(value) for name, value in values.items()):
            raise ValueError("score names must be nonempty and values finite")
        return values


class PerformanceReport(_StrictFrozenModel):
    requests: int = Field(gt=0)
    successful_requests: int = Field(ge=0)
    elapsed_seconds: float = Field(gt=0.0)
    tokens_per_second: float = Field(ge=0.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _valid_counts(self) -> PerformanceReport:
        if self.successful_requests > self.requests:
            raise ValueError("successful_requests cannot exceed requests")
        return self


class CandidateTaskReport(_StrictFrozenModel):
    """Measured candidate task/performance evidence, never teacher-relative."""

    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=_SHA256)
    data_partition: Literal[DataPartition.HELD_OUT_EVALUATION]
    evidence_kind: Literal["measured"] = "measured"
    task_scores: list[TaskScore] = Field(min_length=1)
    performance: PerformanceReport
    teacher_relative: Literal[False] = False
    token_kld: None = None
    cka: None = None
    teacher_relative_blockers: list[TeacherRelativeBlocker]

    @model_validator(mode="after")
    def _candidate_only(self) -> CandidateTaskReport:
        task_ids = [row.task_id for row in self.task_scores]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task report IDs must be unique")
        required = set(TeacherRelativeBlocker)
        if set(self.teacher_relative_blockers) != required:
            raise ValueError("candidate-only report must declare all teacher-relative blockers")
        if len(self.teacher_relative_blockers) != len(required):
            raise ValueError("teacher-relative blockers must not be duplicated")
        return self


class EvalLabStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class EvalLabResult(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    result_digest: str | None = Field(default=None, pattern=_SHA256)
    request_id: str = Field(pattern=_SHA256)
    report_path: str = Field(min_length=1)
    report_sha256: str = Field(pattern=_SHA256)
    status: EvalLabStatus
    errors: list[str] = Field(default_factory=list)

    @field_validator("report_path")
    @classmethod
    def _absolute_report_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("report_path must be absolute")
        return value

    @model_validator(mode="after")
    def _status_and_digest(self) -> EvalLabResult:
        if self.status is EvalLabStatus.COMPLETED and self.errors:
            raise ValueError("completed Eval Lab result cannot contain errors")
        if self.status is EvalLabStatus.FAILED and not self.errors:
            raise ValueError("failed Eval Lab result must contain errors")
        if any(not error for error in self.errors):
            raise ValueError("Eval Lab errors must be nonempty")
        payload = self.model_dump(mode="json", exclude={"result_digest"})
        expected = _canonical_digest(payload)
        if self.result_digest is not None and self.result_digest != expected:
            raise ValueError("result_digest does not match canonical result content")
        object.__setattr__(self, "result_digest", expected)
        return self


class EvalLabHandoff(_StrictFrozenModel):
    request_id: str = Field(pattern=_SHA256)
    eval_lab_revision: Literal["5ee2f7cc33627b6259c0b10100d84932e676f36c"] = (
        "5ee2f7cc33627b6259c0b10100d84932e676f36c"
    )
    argv: tuple[str, ...]


class EvalLabAdapter:
    """Filesystem-only handoff adapter; it never starts a command or endpoint."""

    def __init__(self, executable: str = "eval-lab") -> None:
        if not executable:
            raise ValueError("Eval Lab executable must be nonempty")
        self._executable = executable

    def emit_argv(
        self,
        request: EvalLabRequest,
        *,
        request_path: Path,
        output_dir: Path,
    ) -> EvalLabHandoff:
        if not request_path.is_absolute() or not output_dir.is_absolute():
            raise ValueError("Eval Lab handoff paths must be absolute")
        on_disk = EvalLabRequest.model_validate_json(request_path.read_text())
        if on_disk.request_id != request.request_id:
            raise ValueError("request file does not match the authorized request")
        argv = (
            self._executable,
            "run",
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--revision",
            EVAL_LAB_REVISION,
        )
        return EvalLabHandoff(request_id=request.request_id, argv=argv)  # type: ignore[arg-type]

    def validate_result(
        self, request: EvalLabRequest, *, result_path: Path
    ) -> tuple[EvalLabResult, CandidateTaskReport]:
        result = EvalLabResult.model_validate_json(result_path.read_text())
        if result.request_id != request.request_id:
            raise ValueError("Eval Lab result is bound to a different request")
        report_path = Path(result.report_path)
        if _file_sha256(report_path) != result.report_sha256:
            raise ValueError("Eval Lab report digest does not match returned bytes")
        report = CandidateTaskReport.model_validate_json(report_path.read_text())
        if report.request_id != request.request_id:
            raise ValueError("candidate report is bound to a different request")
        report_tasks = {row.task_id for row in report.task_scores}
        if report_tasks != set(request.tasks):
            raise ValueError("candidate report tasks do not match the request")
        return result, report


def eval_lab_output_layout(root: Path, request_id: str) -> dict[str, Path]:
    """Stable paths for request, candidate report, and result envelope."""

    if not root.is_absolute():
        raise ValueError("Eval Lab output root must be absolute")
    if len(request_id) != 64 or any(ch not in "0123456789abcdef" for ch in request_id):
        raise ValueError("request_id must be a lowercase SHA-256 digest")
    run_root = root / request_id
    return {
        "root": run_root,
        "request": run_root / "request.json",
        "report": run_root / "candidate-task-report.json",
        "result": run_root / "result.json",
    }
