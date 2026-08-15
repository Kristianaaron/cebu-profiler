"""Control-plane facade + service-layer API for the Atlas compression plane.

The control plane is the single entry point a UI or another agent uses to
inspect capabilities, compile/dry-run a recipe, start/status/resume/validate a
run, and inspect lineage. It is deliberately dependency-light (no framework) so
it can be embedded in a future FastAPI service, a CLI, or a notebook.

Exposed operations (user-requestable):

* ``capabilities()`` — backend registry + declared capabilities + availability
* ``compile(recipe)`` / ``dry_run(recipe)`` — deterministic compiled plan
* ``start(recipe|plan, inputs)`` — durable job run
* ``status(run_id)`` — job/event/stage state
* ``resume(run_id)`` — crash-safe resume
* ``validate(run_id, stage)`` — stage output validation
* ``cancel(run_id)``
* ``lineage(recipe)`` — recipe/plan/run ids + hashes (reproducibility)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from model_atlas.backend.registry import BackendRegistry, build_default_registry
from model_atlas.jobs.engine import JobEngine
from model_atlas.recipe.compiler import CapabilityRegistryLike, CompiledRecipe, RecipeCompiler
from model_atlas.recipe.schema import CompressionRecipe
from model_atlas.recipes.artifact import CompiledPlanArtifact


class ControlPlane:
    def __init__(
        self,
        registry: BackendRegistry | None = None,
        work_root: str | Path = "controlplane_runs",
    ) -> None:
        self.registry = registry or build_default_registry()
        self.work_root = Path(work_root)
        self.compiler = RecipeCompiler(cast(CapabilityRegistryLike, self.registry))

    # ------------------------------------------------------------ capability
    def capabilities(self) -> dict[str, object]:
        cap = self.registry.to_dict()
        cap["controls"] = {
            "compiler": "recipe-compiler-v1",
            "default-policy": "no_pruning=true (fidelity-first)",
            "hybrid-policy": "rejected unless a backend declares the exact combination",
        }
        return cap

    # --------------------------------------------------------------- compile
    def compile_recipe(self, recipe: CompressionRecipe) -> CompiledRecipe:
        return self.compiler.compile(recipe)

    def dry_run(self, recipe: CompressionRecipe) -> dict[str, object]:
        issues, rid, sha = self.compiler.validate(recipe)
        errors = [i for i in issues if i.severity == "error"]
        return {
            "recipe_id": rid,
            "recipe_sha256": sha,
            "compiles": not errors,
            "issues": [i.to_dict() for i in issues],
            "backend_snapshot": {
                s.backend.backend_id: self.registry.backend_status_value(s.backend.backend_id)
                for s in recipe.stages
            },
        }

    # ------------------------------------------------------------------ jobs
    def start(
        self,
        recipe: CompressionRecipe,
        inputs: dict[str, object] | None = None,
        verify_artifact: CompiledPlanArtifact | None = None,
    ) -> JobEngine:
        # When a compiled-plan artifact is supplied we verify its pinned backend
        # metadata against the LIVE registry BEFORE executing anything — the
        # artifact, not a silent recompile, is the authoritative execution
        # source.
        if verify_artifact is not None:
            verify_artifact.verify_pins_against(self.registry)
        compiled = self.compile_recipe(recipe)
        return self.start_compiled(compiled, inputs)

    def start_compiled(
        self, compiled: CompiledRecipe, inputs: dict[str, object] | None = None
    ) -> JobEngine:
        engine = JobEngine(compiled, self.registry, self.work_root)
        engine.run(inputs or {})
        return engine

    def engine_for(self, run_id: str) -> JobEngine:
        """Rebuild an engine bound to an existing run dir (for status/resume).

        Exact identity validation: the run dir path is derived deterministically
        from the persisted plan + inputs (no glob semantics). All four of
        ``run_id``, persisted job.run_id, recomputed run_id, and dir name must
        agree or this fails closed.
        """
        from model_atlas.jobs.schema import Job

        # EXACT path (no glob): derive run_dir from the recipe ids + inputs in
        # the persisted job, then verify the file truly lives there.
        expected_dir = self.work_root / "runs" / run_id
        job_path = expected_dir / "job.json"
        if not job_path.exists():
            raise KeyError(f"no run dir for {run_id!r} at {expected_dir}")
        from model_atlas.recipes import CompiledPlanArtifact

        job = Job.model_validate_json(job_path.read_text(encoding="utf-8"))
        if job.run_id != run_id:
            raise RuntimeError(
                f"run identity mismatch: persisted job.run_id={job.run_id} != requested {run_id}"
            )
        artifact = CompiledPlanArtifact.model_validate_json(
            (expected_dir / "plan.json").read_text(encoding="utf-8")
        )
        artifact.verify()
        # executable resume is only permitted when the artifact's recorded pins
        # still match the LIVE registry (backend id/version/adapter identity/
        # status/capability hash)
        artifact.verify_pins_against(self.registry)
        plan = artifact.recipe
        compiled = self.compile_recipe(plan)
        if compiled.plan_id != job.plan_id:
            raise RuntimeError(
                f"run {run_id} plan replay mismatch: {compiled.plan_id} != {job.plan_id}"
            )
        recomputed = self.compiler.compile(plan).run_id(job.inputs)
        if recomputed != run_id:
            raise RuntimeError(
                f"run {run_id} input-identity mismatch: recomputed {recomputed} != {run_id} "
                "(persisted inputs do not reproduce this run_id)"
            )
        engine = JobEngine(compiled, self.registry, self.work_root)
        # bind the engine to the EXACT persisted run identity
        engine._bind_run(job.inputs)
        if engine.run_dir != expected_dir:
            raise RuntimeError("engine run_dir failed to match the persisted run dir")
        engine._run_id = run_id  # pin the verified identity
        return engine

    def status(self, run_id: str) -> dict[str, object]:
        return self.engine_for(run_id).inspect()

    def resume(self, run_id: str) -> dict[str, object]:
        engine = self.engine_for(run_id)
        engine.resume()
        return engine.inspect()

    def validate(self, run_id: str, stage_id: str) -> dict[str, object]:
        return self.engine_for(run_id).validate_outputs(stage_id)

    def cancel(self, run_id: str, reason: str = "operator cancel") -> dict[str, object]:
        engine = self.engine_for(run_id)
        job = engine.cancel(reason)
        return {"run_id": job.run_id, "status": job.status.value}

    # --------------------------------------------------------------- lineage
    def lineage(self, recipe: CompressionRecipe) -> dict[str, object]:
        """Lineage/ids are content-derived and available even for a recipe that
        fails closed on compile (unavailable backends). Issues are reported;
        they never block lineage inspection. No reproduce_command is emitted for
        an uncompilable recipe (the CLI cannot run it); for a compilable recipe
        the command is the functional --plan start path."""
        issues, recipe_id, recipe_hash = self.compiler.validate(recipe)
        fatal = [i for i in issues if i.severity == "error"]
        reproduce = ""
        if not fatal:
            reproduce = "model-atlas job start --plan <path-to-immutable-plan.json> --out " + str(
                self.work_root
            )
        return {
            "recipe_id": recipe_id,
            "recipe_hash": recipe_hash,
            "plan_id": recipe_id,
            "run_id": self.compiler.compile(recipe).run_id({}) if not fatal else None,
            "compiles": not fatal,
            "issues": [i.to_dict() for i in issues],
            "stages": [s.id for s in recipe.stages],
            "backend_snapshot": {
                s.backend.backend_id: self.registry.backend_status_value(s.backend.backend_id)
                for s in recipe.stages
            },
            "source": recipe.source.source_id,
            "calibration": recipe.calibration.calibration_id,
            "reproduce_command": reproduce,
        }
