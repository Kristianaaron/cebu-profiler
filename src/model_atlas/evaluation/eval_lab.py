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
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.networks import HttpUrl

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


def canonical_directory_sha256(root: Path) -> str:
    """Hash regular file names and bytes in a symlink-free directory tree."""

    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("hashed directory must be an absolute, non-symlink directory")
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("hashed directory must not contain symlinks")
    files = [path for path in entries if path.is_file()]
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
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
    endpoint_url: HttpUrl
    config_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _non_secret_url(self) -> EndpointConfigIdentity:
        if self.endpoint_url.username or self.endpoint_url.password:
            raise ValueError("endpoint URL must not contain credentials")
        if self.endpoint_url.query or self.endpoint_url.fragment:
            raise ValueError("endpoint URL must not contain query parameters or fragments")
        return self


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
    seed: int | None = Field(default=None, ge=0)
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
    eval_lab_root: str = Field(min_length=1)
    suite_ref: str = Field(min_length=1)
    tasks_dir: str = Field(min_length=1)
    runs_root: str = Field(min_length=1)
    db_path: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    @field_validator(
        "candidate_artifact_path",
        "eval_lab_root",
        "suite_ref",
        "tasks_dir",
        "runs_root",
        "db_path",
    )
    @classmethod
    def _absolute_paths(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("Eval Lab request paths must be absolute")
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


class HandoffBlocker(StrEnum):
    REQUEST_PARAMETERS_NOT_CLI_BOUND = "request_parameters_not_cli_bound"
    TASK_PARTITION_NOT_HELD_OUT = "task_partition_not_held_out"


class EvalLabHandoff(_StrictFrozenModel):
    request_id: str = Field(pattern=_SHA256)
    eval_lab_revision: Literal["5ee2f7cc33627b6259c0b10100d84932e676f36c"] = (
        "5ee2f7cc33627b6259c0b10100d84932e676f36c"
    )
    executable: bool
    blockers: list[HandoffBlocker] = Field(default_factory=list)
    argv: tuple[str, ...]

    @model_validator(mode="after")
    def _execution_gate(self) -> EvalLabHandoff:
        if self.executable and self.blockers:
            raise ValueError("executable handoff cannot contain blockers")
        if not self.executable and not self.blockers:
            raise ValueError("non-executable handoff requires a typed blocker")
        return self


def _read_pinned_git_head(root: Path) -> str:
    """Resolve a loose or packed Git HEAD without invoking Git."""

    git_dir = root / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise ValueError("invalid Eval Lab .git indirection")
        git_dir = (root / marker.removeprefix("gitdir: ")).resolve()
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            revision, name = line.split(" ", 1)
            if name == ref:
                return revision
    raise ValueError("Eval Lab Git HEAD ref cannot be resolved")


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _index_task_yaml(tasks_dir: Path) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for path in sorted(tasks_dir.rglob("*.yaml")):
        payload = _load_yaml_mapping(path)
        task_id = payload.get("id")
        if not isinstance(task_id, str):
            continue
        if task_id in indexed:
            raise ValueError(f"duplicate Eval Lab task id: {task_id}")
        indexed[task_id] = payload
    return indexed


def _validate_cli_effective_config(request: EvalLabRequest) -> list[HandoffBlocker]:
    """Prove settings actually applied by pinned ``eval-lab run suite``."""

    suite = _load_yaml_mapping(Path(request.suite_ref))
    raw_refs = suite.get("tasks")
    if not isinstance(raw_refs, list):
        raise ValueError("Eval Lab suite tasks must be a list")
    suite_tasks: list[str] = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, Mapping) or not isinstance(raw_ref.get("task_id"), str):
            raise ValueError("invalid task reference in Eval Lab suite")
        repetitions = raw_ref.get("repetitions") or 1
        if not isinstance(repetitions, int) or repetitions < 1:
            raise ValueError("invalid Eval Lab suite task repetitions")
        suite_tasks.extend([raw_ref["task_id"]] * repetitions)
    if suite_tasks != request.tasks:
        raise ValueError("request tasks do not exactly match pinned suite execution order")

    indexed = _index_task_yaml(Path(request.tasks_dir))
    selected: list[Mapping[str, Any]] = []
    for task_id in request.tasks:
        if task_id not in indexed:
            raise ValueError(f"pinned suite task is absent from tasks_dir: {task_id}")
        selected.append(indexed[task_id])

    # The pinned CLI supplies no sampling/seed options. DirectRunner therefore
    # applies temperature=0, max_tokens=4096, and seed=None. Timeout is read
    # from each task's execution section and a request-wide value is truthful
    # only when every selected task has the same value.
    if request.parameters.seed is not None:
        raise ValueError("pinned Eval Lab CLI cannot apply request seed")
    if request.parameters.temperature != 0.0:
        raise ValueError("pinned Eval Lab CLI applies temperature=0.0")
    if request.parameters.max_tokens != 4096:
        raise ValueError("pinned Eval Lab CLI applies max_tokens=4096")
    timeouts: set[float] = set()
    for task in selected:
        execution = task.get("execution") or {}
        if not isinstance(execution, Mapping):
            raise ValueError("Eval Lab task execution config must be a mapping")
        timeout = execution.get("timeout_seconds", 300)
        if not isinstance(timeout, (int, float)):
            raise ValueError("Eval Lab task timeout must be numeric")
        timeouts.add(float(timeout))
    if timeouts != {request.parameters.timeout_seconds}:
        raise ValueError("request timeout does not match every pinned task timeout")

    # Although pinned defaults can be checked, the CLI has no request-wide
    # seed/sampling/timeout flags and DirectRunner does not enforce the task
    # timeout around the endpoint call. Do not label this argv executable until
    # Eval Lab exposes and binds the complete requested parameter contract.
    blockers = [HandoffBlocker.REQUEST_PARAMETERS_NOT_CLI_BOUND]
    if any(task.get("data_partition", "unset") != "held_out_evaluation" for task in selected):
        blockers.append(HandoffBlocker.TASK_PARTITION_NOT_HELD_OUT)
    return blockers


class EvalLabAdapter:
    """Filesystem-only handoff adapter; it never starts a command or endpoint."""

    def __init__(self, executable: str = "eval-lab") -> None:
        if not executable:
            raise ValueError("Eval Lab executable must be nonempty")
        self._executable = executable

    def emit_argv(
        self,
        request: EvalLabRequest,
    ) -> EvalLabHandoff:
        eval_lab_root = Path(request.eval_lab_root)
        if _read_pinned_git_head(eval_lab_root) != EVAL_LAB_REVISION:
            raise ValueError("Eval Lab Git HEAD does not match pinned revision")
        suite_ref = Path(request.suite_ref)
        tasks_dir = Path(request.tasks_dir)
        root_resolved = eval_lab_root.resolve()
        if suite_ref.resolve() != suite_ref or tasks_dir.resolve() != tasks_dir:
            raise ValueError("Eval Lab input paths must not use symlinks")
        if not suite_ref.is_relative_to(root_resolved) or not tasks_dir.is_relative_to(
            root_resolved
        ):
            raise ValueError("Eval Lab suite/tasks must be inside the pinned checkout")
        if _file_sha256(suite_ref) != request.held_out.task_suite_sha256:
            raise ValueError("Eval Lab suite bytes do not match request hash")
        if canonical_directory_sha256(tasks_dir) != request.held_out.task_definitions_sha256:
            raise ValueError("Eval Lab tasks tree does not match request hash")
        blockers = _validate_cli_effective_config(request)
        argv = (
            self._executable,
            "run",
            "suite",
            request.suite_ref,
            "--model",
            request.model_id,
            "--endpoint",
            str(request.endpoint.endpoint_url),
            "--model-name",
            request.model_name,
            "--tasks-dir",
            request.tasks_dir,
            "--runs-root",
            request.runs_root,
            "--db",
            request.db_path,
            "--json",
        )
        return EvalLabHandoff(
            request_id=request.request_id,  # type: ignore[arg-type]
            executable=not blockers,
            blockers=blockers,
            argv=argv,
        )

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
