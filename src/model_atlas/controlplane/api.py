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

from model_atlas.backend.registry import BackendRegistry, build_default_registry
from model_atlas.jobs.engine import JobEngine
from model_atlas.recipe.compiler import CompiledRecipe, RecipeCompiler
from model_atlas.recipe.schema import CompressionRecipe


class ControlPlane:
    def __init__(
        self,
        registry: BackendRegistry | None = None,
        work_root: str | Path = "controlplane_runs",
    ) -> None:
        self.registry = registry or build_default_registry()
        self.work_root = Path(work_root)
        self.compiler = RecipeCompiler(self.registry)

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
        self, recipe: CompressionRecipe, inputs: dict[str, object] | None = None
    ) -> JobEngine:
        compiled = self.compile_recipe(recipe)
        return self.start_compiled(compiled, inputs)

    def start_compiled(
        self, compiled: CompiledRecipe, inputs: dict[str, object] | None = None
    ) -> JobEngine:
        engine = JobEngine(compiled, self.registry, self.work_root)
        engine.run(inputs or {})
        return engine

    def engine_for(self, run_id: str) -> JobEngine:
        """Rebuild an engine bound to an existing run dir (for status/resume)."""
        dirs = sorted((self.work_root / "runs").glob(run_id))
        if not dirs:
            raise KeyError(f"no run dir for {run_id!r}")
        run_dir = dirs[0]
        job_path = run_dir / "job.json"
        from model_atlas.jobs.schema import Job

        job = Job.model_validate_json(job_path.read_text(encoding="utf-8"))
        plan = CompressionRecipe.model_validate_json((run_dir / "plan.json").read_text())
        compiled = self.compile_recipe(plan)
        # ensure the recomputed ids match the persisted run (determinism check)
        if compiled.plan_id != job.plan_id:
            raise RuntimeError(
                f"run {run_id} plan replay mismatch: {compiled.plan_id} != {job.plan_id}"
            )
        engine = JobEngine(compiled, self.registry, self.work_root)
        # bind the engine to the persisted run's actual input identity
        engine._bind_run(job.inputs)
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
        they never block lineage inspection."""
        issues, recipe_id, recipe_hash = self.compiler.validate(recipe)
        fatal = [i for i in issues if i.severity == "error"]
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
            "reproduce_command": (
                f"model-atlas job start --recipe-id {recipe_id} --out "
                f"{self.work_root}  # sha256 {recipe_hash[:12]}"
            ),
        }
