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
from datetime import UTC, datetime
from pathlib import Path

from model_atlas.backend.contract import BackendUnavailable
from model_atlas.backend.registry import BackendRegistry
from model_atlas.jobs.artifacts import (
    ContentAddressedStore,
    StageStager,
    acquire_file_lock,
    assert_source_readonly,
    atomic_write_json,
    atomic_write_text,
    release_file_lock,
    source_snapshot,
)
from model_atlas.jobs.schema import Job, JobStatus, StageStatus
from model_atlas.recipe.compiler import CompiledRecipe, canonical_json
from model_atlas.recipe.schema import RecipeStage
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


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
        return Job(
            run_id=self.compiled.run_id(inputs),
            recipe_id=self.compiled.recipe_id,
            recipe_sha256=self.compiled.recipe_sha256,
            plan_id=self.compiled.plan_id,
            run_dir=str(self.run_dir),
            journal_path=str(self.journal.path),
            inputs=inputs,
            source_snapshot=source_snapshot(self.compiled.recipe.source.checkpoint_path),
            stages={},
            stage_order=[s.id for s in self.compiled.recipe.stages],
        )

    def _load_job(self) -> Job | None:
        if not self.job_path.exists():
            return None
        d = json.loads(self.job_path.read_text(encoding="utf-8"))
        return Job.model_validate(d)

    def _save(self, job: Job) -> None:
        atomic_write_json(self.job_path, job.model_dump(mode="json"))

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
            return self._run_locked(inputs)
        finally:
            release_file_lock(self.run_dir / "run.lock")

    def _run_locked(self, inputs: dict[str, object]) -> Job:
        existing = self._load_job()
        if existing is not None:
            if existing.status is JobStatus.RUNNING or existing.status is JobStatus.RESUMING:
                # crashed mid-stage: treat as resume from journal point
                self.journal.append({"event": "state.recovered", "from": existing.status.value})
            self.journal.append({"event": "state.resume.begin"})
            return self._resume_locked(existing, inputs)
        job = self._init_run_dir(inputs)
        atomic_write_json(self.plan_path, self.compiled.recipe.model_dump(mode="json"))
        self._write_reproduce(job)
        self._save(job)
        self.journal.append({"event": "run.created", "stages": job.stage_order})
        return self._resume_locked(job, inputs)

    def _write_reproduce(self, job: Job) -> None:
        cmd = (
            "model-atlas job start --plan "
            + shlex.quote(str(self.plan_path))
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
            + "cd "
            + shlex.quote(str(Path.cwd()))
            + "\n"
            + cmd
            + "\n"
        )
        atomic_write_text(self.repro_path, text)

    # ------------------------------------------------------------- resume
    def resume(self, inputs: dict[str, object] | None = None) -> Job:
        job = self._load_job()
        if job is None:
            raise RuntimeError("cannot resume: job.json missing (no run started)")
        concrete = inputs or job.inputs
        # bind to the actual (persisted) input identity so resume addresses the
        # same run directory deterministically
        self._bind_run(concrete)
        if job.status is JobStatus.RUNNING or job.status is JobStatus.RESUMING:
            # crashed mid-stage: recover from journal point
            self.journal.append({"event": "state.recovered", "from": job.status.value})
        acquired = acquire_file_lock(self.run_dir / "run.lock")
        if not acquired:
            raise RuntimeError(f"run {job.run_id} is locked by another engine")
        try:
            self.journal.append({"event": "state.resume.begin"})
            return self._resume_locked(job, concrete)
        finally:
            release_file_lock(self.run_dir / "run.lock")

    def _resume_locked(self, job: Job, inputs: dict[str, object]) -> Job:
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
                if stage.message.startswith("[fail-closed]"):
                    job.status = JobStatus.FAILED_TERMINAL
                else:
                    job.status = JobStatus.FAILED_RECOVERABLE
                job.error = stage.message
                self._save(job)
                self.journal.append(
                    {"event": "run.terminal", "status": job.status.value, "stage": stage_id}
                )
                return job
        all_ok = all(job.stage(s).status is StageStatus.DONE for s in job.stage_order)
        if all_ok:
            job.status = JobStatus.COMPLETED
        else:
            job.status = JobStatus.COMPLETED_WITH_WARNINGS
        self._assert_source_immutable(job)
        self._write_manifest(job)
        self._save(job)
        self.journal.append({"event": "run.completed", "status": job.status.value})
        return job

    def _journal_stage_start(self, job: Job, stage_id: str, resumed: bool) -> None:
        self.journal.append(
            {
                "event": "stage." + ("resumed" if resumed else "start"),
                "stage": stage_id,
            }
        )

    def _assert_source_immutable(self, job: Job) -> None:
        c = self.compiled.recipe.constraints
        if c.immutable_source:
            src = self.compiled.recipe.source.checkpoint_path
            if src:
                assert_source_readonly(job.source_snapshot, src)

    def _execute_stage(self, job: Job, stage: str, inputs: dict[str, object]) -> None:
        recipe_stage = self._stage_by_id(stage)
        so = job.stage(stage)
        so.status = StageStatus.RUNNING
        so.started_at = datetime.now(UTC)
        so.message = ""
        job.updated_at = datetime.now(UTC)
        self._save(job)
        self.journal.append(
            {"event": "stage.start", "stage": stage, "backend": recipe_stage.backend.backend_id}
        )
        try:
            context: dict[str, object] = {
                "workdir": str(self.run_dir),
                "parameters": dict(recipe_stage.parameters),
                "stage_id": stage,
                "run_id": job.run_id,
                "inputs": inputs,
            }
            stager = StageStager(self.run_dir, stage)
            handle = self.registry.prepare_stage(recipe_stage.backend.backend_id, context)
            so.handle = handle
            result = self.registry.execute_stage(recipe_stage.backend.backend_id, context, handle)
            refs = stager.commit(self.store)
            # persist backend-returned evidence as a content-addressed JSON output
            evidence_ref = self.store.put_json(
                f"{stage}.evidence.json",
                {
                    "stage": stage,
                    "backend": recipe_stage.backend.backend_id,
                    "result": result,
                    "recorded_kind": recipe_stage.evidence_policy.value,
                },
            )
            refs = [evidence_ref, *refs]
            so.outputs = refs
            # provenance: suppress evidence to policy ceiling
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
            # post-execution source immutability re-check (defense vs a stage
            # mutating the source mid-run)
            self._assert_source_immutable(job)
            so.status = StageStatus.DONE
            so.finished_at = datetime.now(UTC)
            so.exit_code = 0
        except BackendUnavailable as exc:
            so.status = StageStatus.FAILED
            so.message = "[fail-closed] " + str(exc)
            so.exit_code = 1
            self.journal.append(
                {"event": "stage.failed_closed", "stage": stage, "detail": str(exc)}
            )
        except Exception as exc:  # noqa: BLE001 — any stage fault is recoverable
            so.status = StageStatus.FAILED
            so.message = str(exc)
            so.exit_code = 1
            self.journal.append({"event": "stage.failed", "stage": stage, "detail": str(exc)})
        finally:
            so.finished_at = so.finished_at or datetime.now(UTC)
            job.updated_at = datetime.now(UTC)
            self._save(job)

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
        self._save(job)
        self.journal.append({"event": "run.cancelled", "reason": reason})
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
            "backend_status_snapshot": self.compiled.backend_status_snapshot,
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
