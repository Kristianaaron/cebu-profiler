"""Durable, candidate-only GLM-5.2 Eval Lab planning and evidence parsing.

This module deliberately has no subprocess, HTTP, service, or GPU boundary.
It binds a verified compression artifact and evaluation-ready runtime canary to
one pinned Eval Lab request, then turns bounded run-directory artifacts into a
candidate-only task report.  It cannot produce teacher-relative KLD or CKA.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.networks import HttpUrl

from model_atlas.evaluation.eval_lab import (
    EVAL_LAB_REVISION,
    CandidateTaskReport,
    DataPartition,
    EndpointConfigIdentity,
    EndpointTransport,
    EvalLabAdapter,
    EvalLabRequest,
    EvalParameters,
    FrozenHeldOutManifest,
    PerformanceReport,
    TaskScore,
    TeacherRelativeBlocker,
    canonical_directory_sha256,
)
from model_atlas.runtime_artifact_handoff import CompressionHandoff
from model_atlas.runtime_canary_handoff import RuntimeCanaryHandoff

_MAX_RUN_FILE_BYTES = 4 * 1024 * 1024
_SHA256 = r"^[0-9a-f]{64}$"
_RUN_ID = r"^[0-9a-f]{12}$"

GLM52_EVAL_LAB_ROOT = Path("/home/kristianaaron/tmp/eval-lab")
GLM52_EVAL_LAB_SUITE = Path("configs/suites/atlas-glm52-heldout.yaml")
GLM52_EVAL_LAB_TASKS = Path("tasks/atlas_glm52_heldout")
GLM52_EVAL_ENDPOINT = HttpUrl("http://127.0.0.1:8892/v1")
GLM52_EVAL_MODEL = "glm52-mixed-gguf"
GLM52_EVAL_SEED = 17
GLM52_EVAL_TEMPERATURE = 0.0
GLM52_EVAL_MAX_TOKENS = 96
GLM52_EVAL_TIMEOUT_SECONDS = 300.0

_GGUF_TEMPLATE_TRANSPORT_ENVELOPE = {
    "identity_kind": "gguf_embedded_chat_template_transport_v1",
    "gguf_metadata_key": "tokenizer.chat_template",
    "transport": "openai_compatible",
    "endpoint_path": "/v1/chat/completions",
    "message_envelope": {"required_fields": ["role", "content"]},
    "tools": "disabled",
}


class CandidateEvalError(RuntimeError):
    """The candidate-only plan or its persisted run evidence is invalid."""


def _canonical_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_bounded_regular(path: Path, *, limit: int = _MAX_RUN_FILE_BYTES) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise CandidateEvalError("evaluation evidence path must be absolute and symlink-free")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > limit:
            raise CandidateEvalError("evaluation evidence must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            len(encoded) != before.st_size
            or len(encoded) > limit
            or identity_before != identity_after
        ):
            raise CandidateEvalError("evaluation evidence changed during bounded read")
        return encoded
    except OSError as exc:
        raise CandidateEvalError("evaluation evidence cannot be opened safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_bounded_regular(path)).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CandidateEvalError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_symlink_free(path: Path, *, require_directory: bool = False) -> Path:
    if not path.is_absolute():
        raise CandidateEvalError("pinned evaluation paths must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise CandidateEvalError("pinned evaluation paths must not traverse symlinks")
    if require_directory and not path.is_dir():
        raise CandidateEvalError("pinned evaluation directory does not exist")
    if path.resolve() != path:
        raise CandidateEvalError("pinned evaluation paths must be canonical")
    return path


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(_read_bounded_regular(path))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CandidateEvalError(f"{label} is not valid YAML") from exc
    if not isinstance(payload, Mapping):
        raise CandidateEvalError(f"{label} must be a YAML mapping")
    return payload


def _task_index(tasks_root: Path) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    indexed: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    entries = sorted(tasks_root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise CandidateEvalError("Eval Lab task tree must not contain symlinks")
    for path in entries:
        if path.is_dir():
            continue
        if not path.is_file():
            raise CandidateEvalError("Eval Lab task tree may contain only regular files")
        if path.name != "task.yaml":
            continue
        payload = _load_yaml(path, "Eval Lab task definition")
        task_id = payload.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise CandidateEvalError("Eval Lab task definition lacks an id")
        if task_id in indexed:
            raise CandidateEvalError(f"duplicate Eval Lab task id: {task_id}")
        indexed[task_id] = (path, payload)
    return indexed


def _suite_task_order(suite: Mapping[str, Any]) -> tuple[str, ...]:
    references = suite.get("tasks")
    if not isinstance(references, list) or not references:
        raise CandidateEvalError("Eval Lab suite must contain tasks")
    ordered: list[str] = []
    for reference in references:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("task_id"), str):
            raise CandidateEvalError("Eval Lab suite contains an invalid task reference")
        repetitions = reference.get("repetitions", 1)
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions != 1:
            raise CandidateEvalError("GLM held-out suite requires one execution per unique task")
        ordered.append(reference["task_id"])
    if len(ordered) != len(set(ordered)):
        raise CandidateEvalError("GLM held-out suite task ids must be unique")
    return tuple(ordered)


def _selected_corpus_sha256(
    tasks_root: Path,
    ordered: tuple[str, ...],
    indexed: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> str:
    """Hash ordered held-out payload files, excluding task configuration YAML.

    The task-tree hash binds configuration and all files.  This second identity
    separately binds the bytes presented to the model (prompts, attachments,
    and fixtures) and their task-relative names in suite execution order.
    """

    digest = hashlib.sha256()
    payload_files = 0
    for task_id in ordered:
        task_path, _ = indexed[task_id]
        task_root = task_path.parent
        files = sorted(path for path in task_root.rglob("*") if path.is_file())
        for path in files:
            if path == task_path:
                continue
            if path.is_symlink():
                raise CandidateEvalError("Eval Lab corpus must not contain symlinks")
            relative = path.relative_to(tasks_root).as_posix().encode("utf-8")
            task_bytes = task_id.encode("utf-8")
            digest.update(len(task_bytes).to_bytes(8, "big"))
            digest.update(task_bytes)
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(_sha256(path)))
            payload_files += 1
    if payload_files == 0:
        raise CandidateEvalError("Eval Lab held-out corpus contains no payload files")
    return digest.hexdigest()


def gguf_embedded_template_identity(candidate_artifact_sha256: str) -> str:
    """Bind template semantics without pretending the template is a sidecar.

    The compression handoff's artifact digest covers every GGUF byte, including
    ``tokenizer.chat_template``.  Hashing that digest with Atlas's fixed
    OpenAI-compatible message envelope therefore identifies both the embedded
    template and the transport semantics used to invoke it.  This is not a
    claim that the raw template text was independently extracted.
    """

    artifact_sha256 = _require_sha256(candidate_artifact_sha256, "candidate artifact SHA")
    return _canonical_digest(
        {
            "candidate_artifact_sha256": artifact_sha256,
            "transport_envelope": _GGUF_TEMPLATE_TRANSPORT_ENVELOPE,
        }
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateEvalError(f"{label} must be numeric")
    measured = float(value)
    if not math.isfinite(measured):
        raise CandidateEvalError(f"{label} must be finite")
    return measured


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CandidateEvalPlan(_Frozen):
    """Content-addressed candidate evaluation authority with no side effects."""

    schema_version: Literal[1] = 1
    plan_sha256: str | None = Field(default=None, pattern=_SHA256)
    compression_handoff_sha256: str = Field(pattern=_SHA256)
    runtime_canary_handoff_sha256: str = Field(pattern=_SHA256)
    runtime_canary_plan_sha256: str = Field(pattern=_SHA256)
    runtime_config_sha256: str = Field(pattern=_SHA256)
    candidate_artifact_path: str = Field(pattern=r"^/")
    candidate_artifact_sha256: str = Field(pattern=_SHA256)
    eval_lab_revision: Literal["a20da6c6b9cbf872f7c083bffe66afde40c2c8f2"] = (
        "a20da6c6b9cbf872f7c083bffe66afde40c2c8f2"
    )
    eval_lab_executable_sha256: str = Field(pattern=_SHA256)
    held_out_manifest_id: str = Field(pattern=_SHA256)
    task_suite_sha256: str = Field(pattern=_SHA256)
    task_definitions_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    template_sha256: str = Field(pattern=_SHA256)
    parameters: EvalParameters
    eval_request: EvalLabRequest
    argv: tuple[str, ...] = Field(min_length=1)

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"plan_sha256"})

    @model_validator(mode="after")
    def _bind_content(self) -> CandidateEvalPlan:
        if self.eval_request.request_id is None or self.eval_request.held_out.manifest_id is None:
            raise ValueError("candidate evaluation request identities are incomplete")
        if self.held_out_manifest_id != self.eval_request.held_out.manifest_id:
            raise ValueError("candidate evaluation held-out manifest differs from request")
        held_out = self.eval_request.held_out
        if (
            self.task_suite_sha256 != held_out.task_suite_sha256
            or self.task_definitions_sha256 != held_out.task_definitions_sha256
            or self.tokenizer_sha256 != held_out.tokenizer_sha256
            or self.template_sha256 != held_out.template_sha256
            or self.parameters != self.eval_request.parameters
        ):
            raise ValueError("candidate evaluation inputs differ from the frozen request")
        if (
            self.candidate_artifact_path != self.eval_request.candidate_artifact_path
            or self.candidate_artifact_sha256 != self.eval_request.candidate_artifact_sha256
        ):
            raise ValueError("candidate evaluation artifact differs from request")
        if self.runtime_config_sha256 != self.eval_request.endpoint.config_sha256:
            raise ValueError("candidate endpoint config differs from runtime canary")
        if not self.argv or any(not item for item in self.argv):
            raise ValueError("candidate evaluation argv contains an empty item")
        try:
            executable = Path(self.argv[0])
            if (
                not executable.is_absolute()
                or _sha256(executable) != self.eval_lab_executable_sha256
            ):
                raise ValueError("candidate evaluation executable identity changed")
            emitted = EvalLabAdapter(executable=str(executable)).emit_argv(self.eval_request)
        except CandidateEvalError as exc:
            raise ValueError("candidate evaluation executable identity changed") from exc
        if not emitted.executable or emitted.blockers or emitted.argv != self.argv:
            raise ValueError("candidate evaluation argv differs from the pinned request")
        expected = _canonical_digest(self.identity_payload())
        if self.plan_sha256 is not None and self.plan_sha256 != expected:
            raise ValueError("candidate evaluation plan digest differs from canonical content")
        object.__setattr__(self, "plan_sha256", expected)
        return self


class CandidateEvalTaskEvidence(_Frozen):
    """Hashes and locations for one actual Eval Lab direct-task run."""

    task_id: str = Field(min_length=1)
    run_id: str = Field(pattern=_RUN_ID)
    manifest_path: str = Field(pattern=r"^/")
    manifest_sha256: str = Field(pattern=_SHA256)
    result_path: str = Field(pattern=r"^/")
    result_sha256: str = Field(pattern=_SHA256)
    scores_path: str = Field(pattern=r"^/")
    scores_sha256: str = Field(pattern=_SHA256)


class CandidateEvalResult(_Frozen):
    """Content-addressed candidate-only report derived from actual run files."""

    schema_version: Literal[1] = 1
    result_sha256: str | None = Field(default=None, pattern=_SHA256)
    plan_sha256: str = Field(pattern=_SHA256)
    task_evidence: tuple[CandidateEvalTaskEvidence, ...] = Field(min_length=1)
    report: CandidateTaskReport

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"result_sha256"})

    @model_validator(mode="after")
    def _bind_content(self) -> CandidateEvalResult:
        task_ids = tuple(item.task_id for item in self.task_evidence)
        report_ids = tuple(item.task_id for item in self.report.task_scores)
        if task_ids != report_ids:
            raise ValueError("candidate task evidence and report order differ")
        if len({item.run_id for item in self.task_evidence}) != len(self.task_evidence):
            raise ValueError("candidate task run IDs must be unique")
        if self.report.performance.requests != len(self.task_evidence):
            raise ValueError("candidate performance request count differs from task evidence")
        expected = _canonical_digest(self.identity_payload())
        if self.result_sha256 is not None and self.result_sha256 != expected:
            raise ValueError("candidate evaluation result digest differs from canonical content")
        object.__setattr__(self, "result_sha256", expected)
        return self


def build_candidate_eval_plan(
    *,
    compression_handoff: CompressionHandoff,
    runtime_canary_handoff: RuntimeCanaryHandoff,
    eval_request: EvalLabRequest,
    adapter: EvalLabAdapter | None = None,
) -> CandidateEvalPlan:
    """Bind a pinned Eval Lab request to verified producer and canary lineage."""

    if runtime_canary_handoff.handoff_sha256 is None:
        raise CandidateEvalError("runtime canary handoff digest is incomplete")
    if not runtime_canary_handoff.validated_for_evaluation:
        raise CandidateEvalError("runtime canary is not evaluation-ready")
    candidate = runtime_canary_handoff.plan.candidate
    expected_lineage = (
        candidate.artifact_path == compression_handoff.artifact_path
        and candidate.artifact_sha256 == compression_handoff.artifact_sha256
        and candidate.producer_run_id == compression_handoff.producer_run_id
        and candidate.producer_plan_id == compression_handoff.producer_plan_id
        and candidate.producer_recipe_sha256 == compression_handoff.producer_recipe_sha256
        and candidate.producer_profile_id == compression_handoff.producer_profile_id
        and candidate.producer_recommendation_id == compression_handoff.producer_recommendation_id
        and candidate.producer_handoff_sha256 == compression_handoff.handoff_sha256
    )
    if not expected_lineage:
        raise CandidateEvalError("runtime canary lineage differs from compression handoff")
    if eval_request.endpoint.config_sha256 != candidate.runtime_config_sha256:
        raise CandidateEvalError("Eval Lab endpoint config differs from runtime canary")
    if (
        eval_request.candidate_artifact_path != compression_handoff.artifact_path
        or eval_request.candidate_artifact_sha256 != compression_handoff.artifact_sha256
    ):
        raise CandidateEvalError("Eval Lab request artifact differs from compression handoff")
    emitted = (adapter or EvalLabAdapter()).emit_argv(eval_request)
    if not emitted.executable or emitted.blockers:
        raise CandidateEvalError("Eval Lab request is not executable under the pinned contract")
    if emitted.request_id != eval_request.request_id:
        raise CandidateEvalError("Eval Lab argv is bound to a different request")
    assert eval_request.held_out.manifest_id is not None
    return CandidateEvalPlan(
        compression_handoff_sha256=compression_handoff.handoff_sha256,
        runtime_canary_handoff_sha256=runtime_canary_handoff.handoff_sha256,
        runtime_canary_plan_sha256=runtime_canary_handoff.plan.canonical_sha256(),
        runtime_config_sha256=candidate.runtime_config_sha256,
        candidate_artifact_path=compression_handoff.artifact_path,
        candidate_artifact_sha256=compression_handoff.artifact_sha256,
        eval_lab_executable_sha256=_sha256(Path(emitted.argv[0])),
        held_out_manifest_id=eval_request.held_out.manifest_id,
        task_suite_sha256=eval_request.held_out.task_suite_sha256,
        task_definitions_sha256=eval_request.held_out.task_definitions_sha256,
        tokenizer_sha256=eval_request.held_out.tokenizer_sha256,
        template_sha256=eval_request.held_out.template_sha256,
        parameters=eval_request.parameters,
        eval_request=eval_request,
        argv=emitted.argv,
    )


def build_glm52_candidate_eval_plan(
    *,
    compression_handoff: CompressionHandoff,
    runtime_canary_handoff: RuntimeCanaryHandoff,
    eval_output_root: Path,
    verified_tokenizer_sha256: str,
) -> CandidateEvalPlan:
    """Build the production-default, filesystem-only GLM candidate plan.

    ``eval_output_root`` is an explicit operation boundary.  This function
    reads and hashes pinned inputs but creates no directories and runs no
    process; the later operator owns endpoint startup and Eval Lab execution.
    """

    tokenizer_sha256 = _require_sha256(verified_tokenizer_sha256, "verified tokenizer SHA")
    eval_lab_root = _require_symlink_free(GLM52_EVAL_LAB_ROOT, require_directory=True)
    suite_path = eval_lab_root / GLM52_EVAL_LAB_SUITE
    tasks_root = _require_symlink_free(eval_lab_root / GLM52_EVAL_LAB_TASKS, require_directory=True)
    executable = eval_lab_root / ".venv" / "bin" / "eval-lab"
    _require_symlink_free(executable)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise CandidateEvalError("pinned Eval Lab executable is absent or not executable")
    suite = _load_yaml(suite_path, "Eval Lab suite")
    suite_id = suite.get("id")
    if suite_id != "suite.atlas-glm52-heldout.001":
        raise CandidateEvalError("Eval Lab suite id differs from the pinned GLM suite")
    ordered = _suite_task_order(suite)
    indexed = _task_index(tasks_root)
    missing = set(ordered) - set(indexed)
    if missing:
        raise CandidateEvalError("Eval Lab suite references missing held-out tasks")
    task_tree_sha256 = canonical_directory_sha256(tasks_root)
    corpus_sha256 = _selected_corpus_sha256(tasks_root, ordered, indexed)

    operation_root = _require_symlink_free(eval_output_root, require_directory=True)
    request = EvalLabRequest(
        candidate_artifact_path=compression_handoff.artifact_path,
        candidate_artifact_sha256=compression_handoff.artifact_sha256,
        endpoint=EndpointConfigIdentity(
            endpoint_id="glm52-mixed-gguf-loopback-8892",
            transport=EndpointTransport.OPENAI_COMPATIBLE,
            endpoint_url=GLM52_EVAL_ENDPOINT,
            config_sha256=runtime_canary_handoff.plan.candidate.runtime_config_sha256,
        ),
        held_out=FrozenHeldOutManifest(
            data_partition=DataPartition.HELD_OUT_EVALUATION,
            task_suite_id=str(suite_id),
            task_suite_revision=EVAL_LAB_REVISION,
            task_suite_sha256=_sha256(suite_path),
            task_definitions_sha256=task_tree_sha256,
            tracked_task_ids=list(ordered),
            corpus_sha256=corpus_sha256,
            tokenizer_sha256=tokenizer_sha256,
            template_sha256=gguf_embedded_template_identity(compression_handoff.artifact_sha256),
            evaluation_sample_ids=list(ordered),
        ),
        tasks=list(ordered),
        parameters=EvalParameters(
            seed=GLM52_EVAL_SEED,
            temperature=GLM52_EVAL_TEMPERATURE,
            max_tokens=GLM52_EVAL_MAX_TOKENS,
            timeout_seconds=GLM52_EVAL_TIMEOUT_SECONDS,
        ),
        eval_lab_root=str(eval_lab_root),
        suite_ref=str(suite_path),
        tasks_dir=str(tasks_root),
        runs_root=str(operation_root / "runs"),
        db_path=str(operation_root / "runstore.db"),
        model_id=GLM52_EVAL_MODEL,
        model_name=GLM52_EVAL_MODEL,
    )
    return build_candidate_eval_plan(
        compression_handoff=compression_handoff,
        runtime_canary_handoff=runtime_canary_handoff,
        eval_request=request,
        adapter=EvalLabAdapter(executable=str(executable)),
    )


def build_task_evidence(task_id: str, run_dir: Path) -> CandidateEvalTaskEvidence:
    """Create strict path/digest evidence for a completed Eval Lab run directory."""

    if not run_dir.is_absolute() or run_dir.is_symlink() or run_dir.name == "":
        raise CandidateEvalError("Eval Lab run directory must be absolute and symlink-free")
    run = run_dir.resolve(strict=True)
    if run != run_dir:
        raise CandidateEvalError("Eval Lab run directory must not traverse symlinks")
    manifest = run / "manifest.json"
    result = run / "result.json"
    scores = run / "scores.jsonl"
    manifest_payload = _load_object(manifest, "run manifest")
    run_id = manifest_payload.get("run_id")
    if not isinstance(run_id, str):
        raise CandidateEvalError("run manifest lacks a run_id")
    return CandidateEvalTaskEvidence(
        task_id=task_id,
        run_id=run_id,
        manifest_path=str(manifest),
        manifest_sha256=_sha256(manifest),
        result_path=str(result),
        result_sha256=_sha256(result),
        scores_path=str(scores),
        scores_sha256=_sha256(scores),
    )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(_read_bounded_regular(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateEvalError(f"{label} is not JSON") from exc
    if not isinstance(payload, dict):
        raise CandidateEvalError(f"{label} must be a JSON object")
    return payload


def _task_scores(path: Path) -> dict[str, float]:
    encoded = _read_bounded_regular(path)
    rows = encoded.splitlines()
    if not rows:
        raise CandidateEvalError("Eval Lab score evidence is empty")
    scores: dict[str, float] = {}
    for line in rows:
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CandidateEvalError("Eval Lab score evidence is invalid JSONL") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("scorer_id"), str):
            raise CandidateEvalError("Eval Lab score evidence schema is invalid")
        scorer = payload["scorer_id"]
        if not scorer or scorer in scores:
            raise CandidateEvalError("Eval Lab score identifiers must be unique")
        scores[scorer] = _finite_number(payload.get("score"), "Eval Lab score")
    return scores


def _validate_task_paths(plan: CandidateEvalPlan, item: CandidateEvalTaskEvidence) -> None:
    root = Path(plan.eval_request.runs_root)
    if root.is_symlink() or root.resolve() != root:
        raise CandidateEvalError("Eval Lab runs root must not traverse symlinks")
    expected = root / item.run_id
    paths = {
        Path(item.manifest_path): expected / "manifest.json",
        Path(item.result_path): expected / "result.json",
        Path(item.scores_path): expected / "scores.jsonl",
    }
    for actual, required in paths.items():
        if actual != required:
            raise CandidateEvalError("Eval Lab task evidence path disagrees with its run identity")


def parse_candidate_eval_runs(
    plan: CandidateEvalPlan,
    task_evidence: tuple[CandidateEvalTaskEvidence, ...],
) -> CandidateEvalResult:
    """Verify actual Eval Lab run artifacts and derive a candidate-only report."""

    if plan.plan_sha256 is None or plan.eval_request.request_id is None:
        raise CandidateEvalError("candidate evaluation plan identities are incomplete")
    expected_tasks = tuple(plan.eval_request.tasks)
    if tuple(item.task_id for item in task_evidence) != expected_tasks:
        raise CandidateEvalError("Eval Lab task evidence differs from the frozen request")
    task_scores: list[TaskScore] = []
    durations: list[float] = []
    for item in task_evidence:
        _validate_task_paths(plan, item)
        manifest_path = Path(item.manifest_path)
        result_path = Path(item.result_path)
        scores_path = Path(item.scores_path)
        if (
            _sha256(manifest_path) != item.manifest_sha256
            or _sha256(result_path) != item.result_sha256
            or _sha256(scores_path) != item.scores_sha256
        ):
            raise CandidateEvalError("Eval Lab task evidence bytes drifted")
        manifest = _load_object(manifest_path, "run manifest")
        result = _load_object(result_path, "run result")
        if (
            manifest.get("run_id") != item.run_id
            or manifest.get("task_id") != item.task_id
            or manifest.get("model_id") != plan.eval_request.model_id
            or manifest.get("random_seed") != plan.eval_request.parameters.seed
            or manifest.get("sampling")
            != {
                "temperature": plan.eval_request.parameters.temperature,
                "max_tokens": plan.eval_request.parameters.max_tokens,
            }
            or manifest.get("result_status") != "completed"
        ):
            raise CandidateEvalError("Eval Lab run manifest differs from the frozen request")
        budgets = manifest.get("budgets")
        if not isinstance(budgets, dict) or (
            _finite_number(budgets.get("timeout_seconds"), "task timeout")
            != plan.eval_request.parameters.timeout_seconds
            or _finite_number(budgets.get("http_timeout_seconds"), "HTTP timeout")
            != plan.eval_request.parameters.timeout_seconds
        ):
            raise CandidateEvalError(
                "Eval Lab run timeout evidence differs from the frozen request"
            )
        if result.get("run_id") != item.run_id or result.get("error") is not None:
            raise CandidateEvalError("Eval Lab run result is incomplete")
        duration = _finite_number(result.get("duration_s"), "run duration")
        if duration <= 0:
            raise CandidateEvalError("Eval Lab run duration must be positive")
        durations.append(duration)
        task_scores.append(TaskScore(task_id=item.task_id, scores=_task_scores(scores_path)))
    ordered = sorted(durations)
    p50 = ordered[(len(ordered) - 1) // 2] * 1000.0
    p95 = ordered[math.ceil(len(ordered) * 0.95) - 1] * 1000.0
    report = CandidateTaskReport(
        request_id=plan.eval_request.request_id,
        data_partition=DataPartition.HELD_OUT_EVALUATION,
        task_scores=task_scores,
        performance=PerformanceReport(
            requests=len(task_evidence),
            successful_requests=len(task_evidence),
            elapsed_seconds=sum(durations),
            tokens_per_second=0.0,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
        ),
        teacher_relative_blockers=list(TeacherRelativeBlocker),
    )
    return CandidateEvalResult(
        plan_sha256=plan.plan_sha256,
        task_evidence=task_evidence,
        report=report,
    )


__all__ = [
    "CandidateEvalError",
    "CandidateEvalPlan",
    "CandidateEvalResult",
    "CandidateEvalTaskEvidence",
    "build_candidate_eval_plan",
    "build_glm52_candidate_eval_plan",
    "build_task_evidence",
    "gguf_embedded_template_identity",
    "parse_candidate_eval_runs",
]
