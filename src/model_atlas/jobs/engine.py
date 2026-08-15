"""Durable job engine: state machine, atomic journals, crash-safe resume.

The :class:`JobEngine` turns a :class:`CompiledRecipe` into a *run* on disk and
executes its stages via the backend registry. Durability rules:

* every transition is an append-only JSONL event *then* an atomic job.json
  replace (write-ahead ordering: event first, snapshot second);
* stage outputs are staged, hashed, and atomically promoted content-addressed;
* the source checkpoint is snapshotted before the run and assert-read-only
  every stage boundary (immutable_source=true);
* provenance never upgrades silently: a stage's recorded `evidence_kind` is the
  intersection of its declared policy ceiling and what the backend reported —
  if a backend reports MEASURED but the planner policy is PREDICTED, the run
  records PREDICTED and emits an explicit non-escalation event;
* cancellation/failure land in explicit terminal states; a crashed engine can
  resume from the last durable journal point (idempotent re-execution of the
  in-flight stage).

Fail-closed: a stage whose backend is unavailable raises ``BackendUnavailable``
and the job goes ``FAILED_TERMINAL`` (missing dependency = nothing simulated).
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_atlas.backend.contract import BackendUnavailable
from model_atlas.backend.registry import BackendRegistry
from model_atlas.checkpoint.validators import RegisterValidator
from model_atlas.jobs.artifacts import (
    ContentAddressedStore,
    SourceIntegrityError,
    StageStager,
    acquire_file_lock,
    assert_source_readonly,
    atomic_write_json,
    atomic_write_text,
    release_file_lock,
    sha256_file,
    source_manifest,
    source_manifest_digest,
    source_snapshot,
)
from model_atlas.jobs.schema import Job, JobStatus, RepairRecord, StageOutput, StageStatus
from model_atlas.recipe.compiler import CompiledRecipe, canonical_json
from model_atlas.recipe.schema import RecipeStage, StageEffectClass, ValidationGate
from model_atlas.repair.gate import RepairGate, RepairProposal
from model_atlas.repair.gate import sha256_hex as _repair_sha256
from model_atlas.schemas.evidence import EvidenceKind

_EVIDENCE_RANK = {  # higher = closer to measured
    EvidenceKind.INFERRED: 0,
    EvidenceKind.PREDICTED: 1,
    EvidenceKind.ESTIMATED: 2,
    EvidenceKind.MEASURED: 3,
    EvidenceKind.CAUSALLY_TESTED: 4,
}


def _suppressed_evidence(policy: EvidenceKind, reported: EvidenceKind) -> EvidenceKind:
    """Provenance non-escalation: result is min(policy, reported)."""
    return policy if _EVIDENCE_RANK[reported] > _EVIDENCE_RANK[policy] else reported


CheckpointGateValidator = Callable[[Any, Any, Any, dict[str, object]], tuple[bool, str]]


def _gate_validator_for(
    backend_id: str, kind: str, stager: StageStager
) -> CheckpointGateValidator | None:
    """Resolve a REGISTERED backend-independent checkpoint validator for the
    integrity/format/checkpoint gate kinds. Returns None when no validator is
    wired for this backend+kind (the run fails closed — an algorithm adapter is
    never falsely marked available just because a validator exists elsewhere)."""
    from model_atlas.checkpoint.validators import get_checkpoint_validator

    registered = get_checkpoint_validator(backend_id, kind)
    if registered is None:
        return None
    return _wrap_registered(stager, registered)


def _wrap_registered(stager: StageStager, registered: RegisterValidator) -> CheckpointGateValidator:
    """Wrap a registered registry validator (backend_id, staged_dir, format) ->
    CheckpointValidationResult into the engine gate-callable signature."""

    def _run(
        gate: object,
        recipe_stage: RecipeStage,
        _stager: StageStager,
        staged_refs: dict[str, object],
    ) -> tuple[bool, str]:
        result = registered(recipe_stage.backend.backend_id, stager.staging, "")
        return result.ok, result.detail

    return _run


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _StageRef:
    """Minimal staged-output reference (name + full digest) used to hand staged
    files to backend validators before publication."""

    def __init__(self, path: Path, sha256: str) -> None:
        self.path = path
        self.sha256 = sha256
        self.name = path.name

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256, "path": str(self.path)}


def _looks_like_derivative(name: str) -> bool:
    """Heuristic: a staged file that plausibly carries real weight/serialization
    bytes (a non-evidence derivative). Compression stages must produce one of
    these to be marked DONE."""
    low = name.lower()
    if low.endswith((".bin", ".safetensors", ".index.json", ".npy", ".pt", ".npz")):
        return True
    return "weight" in low or "derivative" in low or "shard" in low


def _source_content_hashes(source_path: str) -> dict[str, str]:
    """Complete recursive relative-path -> sha256 manifest of the source (no
    first-eight cap). Raises if the source is missing. Note: a global file
    digest is *not* returned here; use :func:`source_manifest_digest` for the
    whole-source canonical digest."""
    m = source_manifest(source_path)
    if m.get("type") == "missing":
        raise RuntimeError(f"source {source_path} is missing (cannot content-hash)")
    raw = m.get("files", {})
    files: dict[str, object] = raw if isinstance(raw, dict) else {}
    out: dict[str, str] = {}
    for k, v in files.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


class JobJournal:
    """Append-only JSONL event journal for a run (agent/UI readable)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, object]) -> None:
        line = canonical_json({"ts": _now(), **event}) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def tail(self, n: int = 20) -> list[dict[str, object]]:
        return self.read()[-n:]


class JobEngine:
    def __init__(
        self, compiled: CompiledRecipe, registry: BackendRegistry, work_root: Path
    ) -> None:
        self.compiled = compiled
        self.registry = registry
        self.work_root = Path(work_root)
        self.run_dir = self.work_root / "runs" / compiled.run_id({})  # default identity
        self.store = ContentAddressedStore(self.run_dir)
        self.journal = JobJournal(self.run_dir / "events.jsonl")
        self.plan_path = self.run_dir / "plan.json"
        self.job_path = self.run_dir / "job.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.repro_path = self.run_dir / "reproduce.sh"

    # ------------------------------------------------------------- run dir
    def _bind_run(self, inputs: dict[str, object]) -> None:
        """Bind this engine's run dir to the ACTUAL input identity, so runs with
        different inputs land in different, deterministic directories."""
        run_id = self.compiled.run_id(inputs)
        self._run_id = run_id
        self.run_dir = self.work_root / "runs" / run_id
        self.store = ContentAddressedStore(self.run_dir)
        self.journal = JobJournal(self.run_dir / "events.jsonl")
        self.plan_path = self.run_dir / "plan.json"
        self.job_path = self.run_dir / "job.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.repro_path = self.run_dir / "reproduce.sh"

    def _init_run_dir(self, inputs: dict[str, object]) -> Job:
        self._bind_run(inputs)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal.append({"event": "run.initialized", "run_id": self.compiled.run_id(inputs)})
        src = self.compiled.recipe.source.checkpoint_path
        if not src:
            snap: dict[str, object] = {"type": "missing"}
        else:
            snap = source_snapshot(src)
        job = Job(
            run_id=self.compiled.run_id(inputs),
            recipe_id=self.compiled.recipe_id,
            recipe_sha256=self.compiled.recipe_sha256,
            plan_id=self.compiled.plan_id,
            run_dir=str(self.run_dir),
            journal_path=str(self.journal.path),
            inputs=inputs,
            source_snapshot=snap,
            source_manifest_digest=source_manifest_digest(snap),
            stages={},
            stage_order=[s.id for s in self.compiled.recipe.stages],
        )
        return job

    def _load_job(self) -> Job | None:
        if not self.job_path.exists():
            return None
        d = json.loads(self.job_path.read_text(encoding="utf-8"))
        return Job.model_validate(d)

    def stage(self, stage_id: str) -> StageOutput:
        """Return the StageOutput record for a stage (creating a fresh pending
        record in the loaded job if absent)."""
        job = self._load_job()
        if job is None:
            job = Job(
                run_id=self.compiled.run_id({}),
                recipe_id=self.compiled.recipe_id,
                recipe_sha256=self.compiled.recipe_sha256,
                plan_id=self.compiled.plan_id,
                run_dir=str(self.run_dir),
                stages={},
                stage_order=[s.id for s in self.compiled.recipe.stages],
            )
        return job.stage(stage_id)

    def _save(self, job: Job) -> None:
        """Persist the job snapshot ATOMICALLY. Callers must append the journal
        event BEFORE calling ``_save`` where the event describes the change (so
        the journal is write-ahead of the state it records). This method only
        captures the canonical snapshot; ordering is enforced by the callers."""
        atomic_write_json(self.job_path, job.model_dump(mode="json"))

    def _transition(self, job: Job, event: dict[str, object]) -> None:
        """Write-ahead transition: journal event (durably fsynced) THEN the
        atomic job.json snapshot. A crash between the two leaves the journal
        having recorded the intent; resume replays from the journal point."""
        self.journal.append(event)
        self._save(job)

    # ------------------------------------------------------------- execute
    def run(self, inputs: dict[str, object] | None = None) -> Job:
        """Execute (or resume) the compiled recipe. Idempotent at stage level.

        The run dir is bound to the concrete input identity for THIS call
        (never a placeholder); calling with different inputs lands in a
        different deterministic run dir."""
        inputs = inputs or {}
        self._bind_run(inputs)
        acquired = acquire_file_lock(self.run_dir / "run.lock")
        if not acquired:
            raise RuntimeError(f"run {self.compiled.run_id(inputs)} is locked by another engine")
        try:
            return self._run_safe(inputs)
        finally:
            release_file_lock(self.run_dir / "run.lock")

    def _run_safe(self, inputs: dict[str, object]) -> Job:
        """Run boundary: SourceIntegrityError from ANY pre-stage/final check is
        journaled (write-ahead) and atomically saved FAILED_TERMINAL — never
        recoverable."""
        try:
            return self._run_locked(inputs)
        except SourceIntegrityError as exc:
            job = self._load_job() or Job(
                run_id=self.compiled.run_id(inputs),
                recipe_id=self.compiled.recipe_id,
                recipe_sha256=self.compiled.recipe_sha256,
                plan_id=self.compiled.plan_id,
                run_dir=str(self.run_dir),
            )
            job.status = JobStatus.FAILED_TERMINAL
            job.error = str(exc)
            # write-ahead: terminal event, then atomic snapshot
            self.journal.append(
                {
                    "event": "run.terminal",
                    "status": "failed_terminal",
                    "reason": "source_integrity",
                    "detail": str(exc),
                }
            )
            self._save(job)
            return job

    def _run_locked(self, inputs: dict[str, object]) -> Job:
        existing = self._load_job()
        if existing is not None:
            if existing.status is JobStatus.RUNNING or existing.status is JobStatus.RESUMING:
                # crashed mid-stage: recover from journal point
                self.journal.append({"event": "state.recovered", "from": existing.status.value})
            self.journal.append({"event": "state.resume.begin"})
            return self._resume_locked(existing, inputs)
        job = self._init_run_dir(inputs)
        # Persist the plan as a versioned compiled-plan artifact (recipe + exact
        # pins + CANONICAL inputs) so reproduce.sh/`job start --plan` reproduce
        # the exact NONEMPTY run id.
        from model_atlas.recipes import CompiledPlanArtifact

        artifact = CompiledPlanArtifact.from_compiled(
            self.compiled, inputs=inputs, registry=self.registry
        )
        artifact.verify()
        atomic_write_json(self.plan_path, artifact.to_plain_dict())
        self._write_reproduce(job)
        # write-ahead: journal run.created BEFORE the job snapshot
        self.journal.append({"event": "run.created", "stages": job.stage_order})
        self._save(job)
        return self._resume_locked(job, inputs)

    def _write_reproduce(self, job: Job) -> None:
        # The plan artifact (written by compile-recipe / the engine plan.json)
        # must carry canonical inputs so `reproduce.sh` reproduces the NONEMPTY
        # run id. We encode the exact inputs into a tiny JSON sidecar the CLI
        # --plan path can hand to `job start`.
        import json as _json

        plan_artifact_path = self.run_dir / "plan.json"
        inputs_path = self.run_dir / "canonical-inputs.json"
        atomic_write_json(inputs_path, job.inputs)
        cmd = (
            "model-atlas job start --plan "
            + shlex.quote(str(plan_artifact_path))
            + " --out "
            + shlex.quote(str(self.work_root))
        )
        text = (
            "#!/usr/bin/env bash\n"
            "# exact reproduction of run "
            + job.run_id
            + "\n"
            + "# recipe sha256: "
            + self.compiled.recipe_sha256
            + "\n"
            + "# canonical inputs: "
            + _json.dumps(job.inputs, sort_keys=True)
            + "\n"
            + "cd "
            + shlex.quote(str(Path.cwd()))
            + "\n"
            + cmd
            + "\n"
        )
        atomic_write_text(self.repro_path, text)

    # ------------------------------------------------------------- resume
    def resume(self, inputs: dict[str, object] | None = None) -> Job:
        """Resume, binding the EXPLICIT input identity BEFORE loading the job so
        the engine is always pointed at the correct run dir (a fresh engine has
        no state yet — nonempty inputs must address their own run)."""
        concrete = inputs if inputs is not None else {}
        if inputs is None:
            # no inputs: discover the persisted identity first
            probe_job = self._load_job()
            if probe_job is None:
                raise RuntimeError("cannot resume: job.json missing (no run started)")
            concrete = probe_job.inputs
        resumable_run_id = self.compiled.run_id(concrete)
        self._bind_run(concrete)  # explicit bind BEFORE loading
        job = self._load_job()
        if job is None:
            raise RuntimeError(
                f"cannot resume {resumable_run_id}: job.json missing at {self.run_dir}"
            )
        if resumable_run_id != job.run_id:
            raise RuntimeError(
                f"resume identity mismatch: inputs {concrete} recompute run_id "
                f"{resumable_run_id} but persisted job.run_id at {self.run_dir} is "
                f"{job.run_id}; refusing to resume a different run"
            )
        if self.run_dir != Path(job.run_dir):
            raise RuntimeError(
                f"resume run_dir mismatch: bound {self.run_dir} != persisted {job.run_dir}"
            )
        if job.status is JobStatus.RUNNING or job.status is JobStatus.RESUMING:
            # crashed mid-stage: recover from journal point
            self.journal.append({"event": "state.recovered", "from": job.status.value})
        acquired = acquire_file_lock(self.run_dir / "run.lock")
        if not acquired:
            raise RuntimeError(f"run {job.run_id} is locked by another engine")
        try:
            self.journal.append({"event": "state.resume.begin"})
            return self._resume_into_final(job, concrete)
        finally:
            release_file_lock(self.run_dir / "run.lock")

    def _resume_into_final(self, job: Job, inputs: dict[str, object]) -> Job:
        """Resume boundary: a SourceIntegrityError from ANY pre-stage check
        (before re-staging) is journaled + atomically saved FAILED_TERMINAL."""
        try:
            return self._resume_locked(job, inputs)
        except SourceIntegrityError as exc:
            job2 = self._load_job() or job
            job2.status = JobStatus.FAILED_TERMINAL
            job2.error = str(exc)
            self.journal.append(
                {
                    "event": "run.terminal",
                    "status": "failed_terminal",
                    "reason": "source_integrity",
                    "detail": str(exc),
                }
            )
            self._save(job2)
            return job2

    def _resume_locked(self, job: Job, inputs: dict[str, object]) -> Job:
        # P1: verify every already-DONE stage's published outputs are still
        # content-valid before trusting ANY resume (a stage whose blob was
        # lost/corrupted is NOT trusted as done) — this runs even when the run
        # would otherwise be terminal, so a completed run's DONE outputs are
        # still integrity-checked on every resume attempt.
        for stage_id in job.stage_order:
            so = job.stage(stage_id)
            if so.status is StageStatus.DONE:
                ok, detail = self._verify_done_outputs(so)
                if not ok:
                    raise RuntimeError(
                        f"resume refused: stage {stage_id} was DONE but its outputs "
                        f"failed verification ({detail})"
                    )
        # Crashes mid-stage leave the job RUNNING/RESUMING; a recoverable stage
        # exception leaves FAILED_RECOVERABLE; both resume. Terminal states and a
        # hard success refuse re-entry (deterministic, no double execution).
        if job.status in {
            JobStatus.FAILED_TERMINAL,
            JobStatus.CANCELLED,
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_WITH_WARNINGS,
        }:
            self.journal.append({"event": "resume.refused", "status": job.status.value})
            return job
        job.status = JobStatus.RUNNING
        job.failed_stage = None
        job.error = ""
        # write-ahead: journal the transition BEFORE the snapshot
        self.journal.append({"event": "run.resumed"})
        self._save(job)
        for stage_id in job.stage_order:
            stage = job.stage(stage_id)
            if stage.status in (StageStatus.DONE, StageStatus.SKIPPED):
                if stage.status is StageStatus.DONE:
                    self._journal_stage_start(job, stage_id, resumed=True)
                continue
            # a FAILED stage is re-executed from scratch on resume
            if stage.status is StageStatus.FAILED:
                stage.status = StageStatus.PENDING
                stage.message = ""
                stage.outputs = []
                stage.exit_code = None
                stage.finished_at = None
                self.journal.append({"event": "stage.retry", "stage": stage_id})
            self._assert_source_immutable(job)
            self._execute_stage(job, stage_id, inputs)
            if stage.status is StageStatus.FAILED:
                job.failed_stage = stage_id
                if stage.message.startswith("[fail-closed]") or stage.message.startswith(
                    "[source-integrity-terminal]"
                ):
                    job.status = JobStatus.FAILED_TERMINAL
                else:
                    job.status = JobStatus.FAILED_RECOVERABLE
                job.error = stage.message
                detail = stage.message or "stage failed"
                # write-ahead: terminal event first, then the snapshot
                self.journal.append(
                    {
                        "event": "run.terminal",
                        "status": job.status.value,
                        "stage": stage_id,
                        "detail": detail,
                    }
                )
                self._save(job)
                return job
        all_ok = all(job.stage(s).status is StageStatus.DONE for s in job.stage_order)
        if all_ok:
            job.status = JobStatus.COMPLETED
        else:
            job.status = JobStatus.COMPLETED_WITH_WARNINGS
        self._assert_source_immutable(job)
        self._write_manifest(job)
        # write-ahead: completed event before final snapshot
        self.journal.append({"event": "run.completed", "status": job.status.value})
        self._save(job)
        return job

    def _journal_stage_start(self, job: Job, stage_id: str, resumed: bool) -> None:
        self.journal.append(
            {
                "event": "stage." + ("resumed" if resumed else "start"),
                "stage": stage_id,
            }
        )

    def _assert_source_immutable(self, job: Job) -> None:
        """Verify source integrity (snapshot equality + declared path-bound
        hashes or canonical manifest digest + whole manifest digest). Any
        mismatch raises SourceIntegrityError — the run boundary catches it and
        journals/saves FAILED_TERMINAL (never recoverable)."""
        src = self.compiled.recipe.source.checkpoint_path
        if not src:
            return
        try:
            # (1) stat+hash manifest equality against the run-start snapshot
            if self.compiled.recipe.constraints.immutable_source:
                assert_source_readonly(job.source_snapshot, src)
            m = source_manifest(src)
            if m.get("type") == "missing":
                raise SourceIntegrityError("source is missing (cannot verify integrity)")
            raw_files_obj = m.get("files", {})
            raw_files: dict[str, object] = raw_files_obj if isinstance(raw_files_obj, dict) else {}
            files: dict[str, str] = {
                k: v for k, v in raw_files.items() if isinstance(k, str) and isinstance(v, str)
            }
            # (2) DECLARED source hashes: path-bound dict OR canonical manifest
            #     digest.
            declared_sha = self.compiled.recipe.source.sha256
            declared_digest = self.compiled.recipe.source.manifest_digest
            if isinstance(declared_sha, dict):
                declared_map: dict[str, str] = {
                    k: v
                    for k, v in declared_sha.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
                # EXACT path-set equality: the declared mapping must contain
                # exactly the complete recursive measured path set (no missing,
                # no extra paths), plus per-path hash equality.
                declared_paths = set(declared_map)
                measured_paths = set(files)
                if declared_paths != measured_paths:
                    raise SourceIntegrityError(
                        "executable source sha256 path set must equal the complete "
                        f"recursive measured path set: declared {len(declared_paths)} "
                        f"paths ({sorted(declared_paths - measured_paths)} extra, "
                        f"{sorted(measured_paths - declared_paths)} missing) — or "
                        "provide the canonical full manifest_digest instead"
                    )
                for rel, digest in declared_map.items():
                    if files[rel] != digest:
                        raise SourceIntegrityError(
                            f"declared path {rel!r} hash {digest[:16]}… != measured "
                            f"{files[rel][:16]}…"
                        )
            elif declared_sha:
                raise SourceIntegrityError(
                    "SourceIdentity.sha256 must be a path-bound dict "
                    "{relative_path -> sha256}, not a flat list"
                )
            # (3) canonical manifest digest must match the declared one AND the
            #     run-start snapshot's.
            job_digest = source_manifest_digest(m)
            if declared_digest and declared_digest != job_digest:
                raise SourceIntegrityError(
                    f"declared manifest digest {declared_digest[:16]}… != recomputed "
                    f"{job_digest[:16]}…"
                )
            if job.source_manifest_digest and job.source_manifest_digest != job_digest:
                raise SourceIntegrityError(
                    f"run-start manifest digest {job.source_manifest_digest[:16]}… != "
                    f"recomputed {job_digest[:16]}…"
                )
        except SourceIntegrityError:
            raise
        except RuntimeError as exc:
            raise SourceIntegrityError(str(exc)) from exc

    def _execute_stage(self, job: Job, stage: str, inputs: dict[str, object]) -> None:
        recipe_stage = self._stage_by_id(stage)
        so = job.stage(stage)
        so.status = StageStatus.RUNNING
        so.started_at = datetime.now(UTC)
        so.message = ""
        job.updated_at = datetime.now(UTC)
        # P3: require_available=false is DRY-RUN-ONLY — the stage may compile for
        # planning, but execution is explicitly non-executable.
        if not recipe_stage.backend.require_available:
            so.status = StageStatus.FAILED
            so.message = (
                "[fail-closed] stage is dry-run-only (require_available=false); "
                "it compiles for planning but may never execute"
            )
            so.exit_code = 1
            self.journal.append(
                {
                    "event": "stage.failed_closed",
                    "stage": stage,
                    "detail": so.message,
                }
            )
            so.finished_at = datetime.now(UTC)
            job.updated_at = datetime.now(UTC)
            self._save(job)
            return
        # write-ahead: stage.start event BEFORE the RUNNING snapshot
        self.journal.append(
            {
                "event": "stage.start",
                "stage": stage,
                "backend": recipe_stage.backend.backend_id,
            }
        )
        self._save(job)
        try:
            # Isolated staging: the adapter writes into the stage's private
            # scratch; nothing is visible to the run until verified + published.
            stager = StageStager(self.run_dir, stage)
            context: dict[str, object] = {
                "workdir": str(self.run_dir),
                "staging_dir": str(stager.staging),
                "output_sink": str(stager.staging),
                "parameters": dict(recipe_stage.parameters),
                "stage_id": stage,
                "run_id": job.run_id,
                "inputs": inputs,
            }
            handle = self.registry.prepare_stage(recipe_stage.backend.backend_id, context)
            so.handle = handle
            result = self.registry.execute_stage(recipe_stage.backend.backend_id, context, handle)

            # -- expected_outputs gate: every declared expected output must exist
            #    in staging before anything is considered a stage result --
            produced_names = {p.name for p in stager.staging.iterdir() if p.is_file()}
            for exp in recipe_stage.expected_outputs:
                if exp not in produced_names:
                    raise RuntimeError(
                        f"stage {stage} did not produce expected output {exp!r} "
                        f"(found {sorted(produced_names)})"
                    )

            # -- derivative agreement: a compression stage must be served by a
            #    real derivative producer AND the record + adapter must agree AND
            #    the stage must declare TYPED, NON-EMPTY expected checkpoint
            #    outputs (safetensors/index/tensor artifacts) — never filename
            #    heuristics. The declared formats drive the structural validator.
            if recipe_stage.effect_class in {
                StageEffectClass.QUANTIZATION,
                StageEffectClass.REFINEMENT,
                StageEffectClass.RESIDUAL,
                StageEffectClass.CONDITIONING,
            }:
                rec = self._backend_record(recipe_stage.backend.backend_id)
                adapter = self.registry.adapter_for(recipe_stage.backend.backend_id)
                adapter_derivative = (
                    getattr(adapter, "produces_derivative", False) if adapter else False
                )
                record_derivative = (
                    bool(getattr(rec, "produces_derivative", False)) if rec else False
                )
                if not (adapter_derivative and record_derivative):
                    raise RuntimeError(
                        f"stage {stage} is a compression stage but record/adapter "
                        f"derivative agreement failed: adapter_derivative="
                        f"{adapter_derivative}, record_derivative={record_derivative}"
                    )
                # typed, non-empty expected checkpoint outputs are REQUIRED
                if not recipe_stage.expected_outputs:
                    raise RuntimeError(
                        f"stage {stage} is a compression stage but declares no typed "
                        "expected checkpoint outputs (safetensors/index/tensor); "
                        "failing before any CAS publish"
                    )
                for exp in recipe_stage.expected_outputs:
                    if exp not in produced_names:
                        raise RuntimeError(f"stage {stage} compression output {exp!r} not produced")
                # integrity/format/checkpoint gate is REQUIRED for the declared
                # format (real structural validation, never name heuristics).
                gate_kinds = {g.kind for g in recipe_stage.validation_gates if g.required}
                if not (gate_kinds & {"integrity", "format", "checkpoint"}):
                    raise RuntimeError(
                        f"stage {stage} is a compression stage but declares no "
                        "integrity/format/checkpoint validation gate; fail closed"
                    )

            # ══ pre-publish ordering ══
            # (1) source integrity check happens BEFORE any CAS publication
            self._assert_source_immutable(job)
            # (2) every required gate kind runs against STAGING (all kinds:
            #     validator, eq_control, identity_control, integrity/format), and
            #     unknown/unwired kinds are rejected.
            self._run_validation_gates_staging(job, stage, stager)
            # (3) only now publish: content-address + full-digest verify + commit
            refs = stager.commit(self.store)
            # publish evidence + mark recorded evidence
            evidence_ref = self.store.put_json(
                f"{stage}.evidence.json",
                {
                    "stage": stage,
                    "backend": recipe_stage.backend.backend_id,
                    "result": result,
                    "recorded_kind": recipe_stage.evidence_policy.value,
                    "expected_outputs": list(recipe_stage.expected_outputs),
                },
            )
            refs = [evidence_ref, *refs]
            so.outputs = refs
            policy = recipe_stage.evidence_policy
            for ref in refs:
                self.journal.append(
                    {
                        "event": "stage.output",
                        "stage": stage,
                        "name": ref.name,
                        "sha256": ref.sha256,
                    }
                )
            so.evidence_kind = policy
            so.evidence_reported = "in-process deterministic computation (probe/estimate)"
            self._stage_evidence(job, stage, reported=EvidenceKind.PREDICTED, policy=policy)
            so.status = StageStatus.DONE
            so.finished_at = datetime.now(UTC)
            so.exit_code = 0
        except SourceIntegrityError as exc:
            # source integrity failures are ALWAYS terminal (never recoverable)
            so.status = StageStatus.FAILED
            so.message = "[source-integrity-terminal] " + str(exc)
            so.exit_code = 2
            self.journal.append(
                {"event": "stage.source_integrity_failed", "stage": stage, "detail": str(exc)}
            )
        except BackendUnavailable as exc:
            so.status = StageStatus.FAILED
            so.message = "[fail-closed] " + str(exc)
            so.exit_code = 1
            self.journal.append(
                {"event": "stage.failed_closed", "stage": stage, "detail": str(exc)}
            )
        except Exception as exc:  # noqa: BLE001
            so.status = StageStatus.FAILED
            so.message = str(exc)
            so.exit_code = 1
            self.journal.append({"event": "stage.failed", "stage": stage, "detail": str(exc)})
        finally:
            so.finished_at = so.finished_at or datetime.now(UTC)
            job.updated_at = datetime.now(UTC)
            self._save(job)

    def _backend_record(self, backend_id: str) -> object | None:
        try:
            return self.registry.requires(backend_id)
        except Exception:  # noqa: BLE001
            return None

    def _run_validation_gates_staging(self, job: Job, stage: str, stager: StageStager) -> None:
        """Run EVERY required gate kind against the STAGED (not yet published)
        stage output. Supported kinds: validator (backend adapter validator),
        eq_control (numerical-equivalence), identity_control (identity/no-op
        control), integrity/format/checkpoint (real structural validator). A
        declared gate of an unknown or not-execution-wired kind fails closed —
        it is never skipped and never demoted."""
        recipe_stage = self._stage_by_id(stage)
        known_kinds = {
            "validator",
            "eq_control",
            "identity_control",
            "integrity",
            "format",
            "checkpoint",
        }
        for gate in recipe_stage.validation_gates:
            if not gate.required:
                continue
            if gate.kind == "dry_run":
                continue  # dry_run is a pre-flight check, not a DONE gate
            if gate.kind not in known_kinds:
                raise RuntimeError(
                    f"stage {stage} declares UNKNOWN validation gate kind "
                    f"{gate.kind!r}; not execution-wired — fail closed"
                )
            outcome = self._execute_staging_gate(gate, recipe_stage, stager)
            if outcome is not True:
                raise RuntimeError(
                    f"stage {stage} staging gate {gate.gate_id!r} ({gate.kind}) FAILED: {outcome}"
                )

    def _execute_staging_gate(
        self,
        gate: ValidationGate,
        recipe_stage: RecipeStage,
        stager: StageStager,
    ) -> bool | str:
        """Execute one declared gate against staging. Returns True on pass, an
        error string on fail/unwired (fail closed)."""
        staged_refs: dict[str, object] = {
            p.name: _StageRef(p, sha256_file(p)) for p in stager.staging.iterdir() if p.is_file()
        }
        if gate.kind == "validator":
            result = self.registry.validate_stage(recipe_stage.backend.backend_id, {}, staged_refs)
            if result.get("status") == "unvalidated" or not result.get("validated"):
                return f"backend validator returned {result}"
            return True
        if gate.kind in {"eq_control", "identity_control", "integrity", "format", "checkpoint"}:
            # these are deterministic validators over the STAGED bytes; they are
            # only "wired" when a real validator implementation is registered for
            # the backend, otherwise the run fails closed (never invents a pass).
            validator = _gate_validator_for(recipe_stage.backend.backend_id, gate.kind, stager)
            if validator is None:
                return (
                    f"{gate.kind} gate not execution-wired for backend "
                    f"{recipe_stage.backend.backend_id!r} — requires a real "
                    f"validator; fail closed"
                )
            ok, detail = validator(gate, recipe_stage, stager, staged_refs)
            return True if ok else (detail or f"{gate.kind} gate failed")
        return f"unknown validation gate kind {gate.kind!r}"

    def _verify_done_outputs(self, so: StageOutput) -> tuple[bool, str]:
        """Verify every published output of a DONE stage is intact (full-digest,
        content-addressed). Returns (ok, detail)."""
        if not so.outputs:
            return False, "no outputs published for a DONE stage"
        for ref in so.outputs:
            if not self.store.verify(ref):
                return False, f"output {ref.name} failed content verification"
        return True, ""

    def _stage_by_id(self, stage_id: str) -> RecipeStage:
        for s in self.compiled.recipe.stages:
            if s.id == stage_id:
                return s
        raise KeyError(f"compiled recipe has no stage {stage_id!r}")

    def _stage_evidence(
        self,
        job: Job,
        stage: str,
        *,
        reported: EvidenceKind,
        policy: EvidenceKind,
    ) -> None:
        suppressed = _suppressed_evidence(policy, reported)
        if suppressed is not reported:
            self.journal.append(
                {
                    "event": "evidence.non_escalation",
                    "stage": stage,
                    "reported": reported.value,
                    "recorded": suppressed.value,
                    "policy_ceiling": policy.value,
                }
            )
        job.stage(stage).evidence_kind = suppressed

    # ------------------------------------------------------------- controls
    def cancel(self, reason: str = "operator cancel") -> Job:
        job = self._load_job()
        if job is None:
            raise RuntimeError("no run to cancel")
        if job.is_terminal():
            return job
        job.status = JobStatus.CANCELLED
        job.error = reason
        # write-ahead: cancel event before snapshot
        self.journal.append({"event": "run.cancelled", "reason": reason})
        self._save(job)
        return job

    # ------------------------------------------------------------- repair
    def apply_repair(self, proposal: RepairProposal, stage_id: str, target_output: str) -> Job:
        """Public: acquire the run lock, delegate to the private locked
        implementation, release the lock. Serializes against other repair
        transactions on the same run (no lost updates)."""
        acquired = acquire_file_lock(self.run_dir / "run.lock")
        if not acquired:
            raise RuntimeError(f"run {self.run_dir.name} is locked by another engine")
        try:
            return self._apply_repair_locked(proposal, stage_id, target_output)
        finally:
            release_file_lock(self.run_dir / "run.lock")

    def _apply_repair_locked(
        self,
        proposal: RepairProposal,
        stage_id: str,
        target_output: str,
        *,
        repair_gate: RepairGate | None = None,
    ) -> Job:
        """Apply a deterministic repair inside the DURABLE job transaction
        (caller holds the run lock). The gate persists restore+new CAS refs,
        the produced blob is re-read from the CAS (full-digest verified), and
        the target StageOutput ref is atomically updated with a content-derived
        (collision-free) repair_id, journaled + save + manifest. No caller
        bytes are retained."""
        gate = repair_gate or RepairGate()
        job = self._load_job()
        if job is None:
            raise RuntimeError("no run to repair")
        so = job.stages.get(stage_id)
        if so is None or not any(o.name == target_output for o in so.outputs):
            raise RuntimeError(f"no stage {stage_id!r} has output ref named {target_output!r}")
        orig_ref = next(r for r in so.outputs if r.name == target_output)
        compiled = gate.validate_proposal(proposal).compiled
        if compiled is None:
            raise RuntimeError("repair proposal not compilable")
        ok, res, produced = gate.apply(compiled, cas=self.store, target_ref=orig_ref)
        if not ok or produced is None or res.compiled is None:
            raise RuntimeError(f"repair apply failed: {res.errors}")
        # persist the repaired blob and atomically update the target ref
        new_ref = self.store.put_bytes(orig_ref.name, produced)
        gate.publish_apply(res.compiled, target=so, new_ref=new_ref)
        # CONTENT-DERIVED, collision-free repair id = hash(canonical full
        # proposal params + registered transform identity/version + stage +
        # output + BEFORE and AFTER full hash digests). A target may only have
        # ONE active (applied, not-reverted) repair; and the same digest id must
        # be UNIQUE across ALL repair history (apply + rollback) so rollback can
        # look up the exact unique record.
        repair_id = _repair_sha256(
            canonical_json(
                {
                    "kind": proposal.kind,
                    "transform_identity": f"{proposal.kind}@{res.compiled.transform_version}",
                    "params": dict(proposal.params),
                    "stage": stage_id,
                    "target": target_output,
                    "before_sha256": res.compiled.restore_key,
                    "after_sha256": res.compiled.new_key,
                }
            ).encode("utf-8")
        )
        if any(
            r.applied and r.stage_id == stage_id and r.target == target_output for r in job.repair
        ):
            raise RuntimeError(
                f"target {stage_id}:{target_output} already has an applied repair; "
                "refusing a second (collision-free record lookup)"
            )
        if any(r.repair_id == repair_id for r in job.repair):
            raise RuntimeError(
                f"repair {repair_id!r} already exists in repair history (collision-free)"
            )
        record = RepairRecord(
            repair_id=repair_id,
            kind=proposal.kind,
            transform_version=res.compiled.transform_version,
            target=target_output,
            stage_id=stage_id,
            before_sha256=res.compiled.restore_key,
            after_sha256=res.compiled.new_key,
            restore_ref=res.compiled.restore_key,
            new_ref=new_ref.sha256,
            applied=True,
        )
        job.repair.append(record)
        # write-ahead: repair event BEFORE the snapshot (durable transaction)
        self.journal.append(
            {
                "event": "repair.applied",
                "stage": stage_id,
                "target": target_output,
                "kind": proposal.kind,
                "repair_id": repair_id,
                "restore_ref": record.restore_ref,
                "new_ref": record.new_ref,
            }
        )
        self._save(job)
        # REGENERATE the manifest inside the same transaction (recoverably
        # derived from the just-saved job record)
        self._write_manifest(job)
        return job

    def rollback_repair(self, repair_id: str) -> Job:
        """Public: acquire the run lock, delegate to the private locked
        implementation, release the lock (serialized, no lost updates)."""
        acquired = acquire_file_lock(self.run_dir / "run.lock")
        if not acquired:
            raise RuntimeError(f"run {self.run_dir.name} is locked by another engine")
        try:
            return self._rollback_repair_locked(repair_id)
        finally:
            release_file_lock(self.run_dir / "run.lock")

    def _rollback_repair_locked(self, repair_id: str) -> Job:
        """Roll back an applied repair by loading the recorded restore CAS ref
        (caller holds the run lock), verifying its full digest, and atomically
        restoring the target StageOutput ref + record state. No caller-retained
        bytes: everything is read from the CAS and the persisted RepairRecord."""
        job = self._load_job()
        if job is None:
            raise RuntimeError("no run to roll back")
        matches = [r for r in job.repair if r.repair_id == repair_id]
        if len(matches) != 1:
            raise RuntimeError(
                f"repair {repair_id!r}: expected exactly one unique record, found {len(matches)}"
            )
        rec = matches[0]
        if not rec.restore_ref:
            raise RuntimeError(f"repair {repair_id!r} has no restore ref")
        # find the target StageOutput via the recorded stage_id (collision-free)
        if not rec.stage_id:
            raise RuntimeError(f"repair {repair_id!r} has no stage_id")
        so = job.stages.get(rec.stage_id)
        if so is None or not any(o.name == rec.target for o in so.outputs):
            raise RuntimeError(
                f"repair {repair_id!r}: no stage {rec.stage_id!r} has output {rec.target!r}"
            )
        # load + verify the recorded restore ref from the CAS
        restore_bytes = self.store.read_from_key(rec.restore_ref)
        from model_atlas.repair import sha256_hex as rsh

        if rsh(restore_bytes) != rec.before_sha256:
            raise RuntimeError(
                f"rollback refused: restore ref digest mismatch for {rec.restore_ref[:16]}…"
            )
        restore_ref = self.store.put_bytes(rec.target, restore_bytes)
        # atomically restore the target ref + record state
        existing = [i for i, r in enumerate(so.outputs) if r.name == rec.target]
        if existing:
            so.outputs[existing[0]] = restore_ref
        else:
            so.outputs.append(restore_ref)
        rec.applied = False
        rec.reverted = True
        rec.note = f"rolled back to {rec.before_sha256[:16]}…"
        # write-ahead: rollback event before the snapshot
        self.journal.append(
            {"event": "repair.rolled_back", "repair_id": repair_id, "target": rec.target}
        )
        self._save(job)
        self._write_manifest(job)
        return job

    # ------------------------------------------------------------- manifest
    def _write_manifest(self, job: Job) -> None:
        mode = "json"
        manifest = {
            "schema_version": 1,
            "run_id": job.run_id,
            "recipe_id": job.recipe_id,
            "recipe_sha256": job.recipe_sha256,
            "plan_id": job.plan_id,
            "status": job.status.value,
            "sources": self.compiled.recipe.source.model_dump(mode=mode),
            "constraints": self.compiled.recipe.constraints.model_dump(mode=mode),
            "stages": {
                sid: {
                    "status": so.status.value,
                    "outputs": [r.model_dump(mode=mode) for r in so.outputs],
                    "evidence_kind": so.evidence_kind.value,
                    "evidence_reported": so.evidence_reported,
                    "message": so.message,
                    "exit_code": so.exit_code,
                }
                for sid, so in job.stages.items()
            },
            "repairs": [r.model_dump(mode=mode) for r in job.repair],
            "readiness": self._readiness(job),
            "backend_status_snapshot": dict(self.compiled.backend_status_snapshot),
            "reproduce": str(self.repro_path),
        }
        atomic_write_json(self.manifest_path, manifest)
        self.journal.append({"event": "run.manifest_written"})

    def _readiness(self, job: Job) -> dict[str, object]:
        return {
            "all_stages_done": job.completed_ok,
            "runtime_benchmarked": False,  # two-Spark runtime profiling is a separate gate
            "published": False,
        }

    def inspect(self, job_id: str | None = None) -> dict[str, object]:
        """Agent-readable run records (manifest + events + stage evidence)."""
        job = self._load_job()
        if job is None:
            return {"error": "no job", "run_dir": str(self.run_dir)}
        manifest = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "run_id": job.run_id,
            "status": job.status.value,
            "manifest": manifest,
            "events": self.journal.read(),
            "stage_order": job.stage_order,
            "stages": {sid: so.model_dump(mode="json") for sid, so in job.stages.items()},
        }

    # ------------------------------------------------------------- plumbing
    def validate_outputs(self, stage_id: str) -> dict[str, object]:
        job = self._load_job()
        if job is None:
            raise RuntimeError("no run to validate")
        so = job.stage(stage_id)
        recipe_stage = self._stage_by_id(stage_id)
        outputs: dict[str, object] = {r.name: r for r in so.outputs}
        result = self.registry.validate_stage(recipe_stage.backend.backend_id, {}, outputs)
        integrity = [self.store.verify(r) for r in so.outputs]
        ok = all(integrity) and bool(result.get("validated"))
        return {
            "stage": stage_id,
            "validated": ok,
            "integrity": integrity,
            "backend_validation": result,
        }
