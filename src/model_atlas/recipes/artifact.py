"""Versioned immutable compiled-plan artifact.

The compiled-plan artifact is the single canonical, machine-readable output of
``compile-recipe`` and the single verified input of ``job start --plan`` and
``reproduce.sh``. It is versioned and deeply immutable:

  * ``schema_version`` — artifact schema version (this file);
  * ``recipe`` — the canonical authored recipe (CompressionRecipe);
  * ``recipe_sha256`` / ``recipe_id`` / ``plan_id`` — content-addresses;
  * ``resolved_pins`` — stage -> exact resolved backend version (+ status);
  * ``inputs`` — canonical job inputs (the run-id inputs);
  * ``run_id`` — deterministic from recipe + inputs;
  * ``reproduce_command`` — the exact CLI start command.

Writing is ``model-atlas compile-recipe --out plan.json``. Starting verifies that
the artifact's ids/hash are internally consistent before the engine reads it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.recipe.compiler import CompiledRecipe, canonical_json
from model_atlas.recipe.schema import CompressionRecipe

PLAN_ARTIFACT_SCHEMA = 1


class CompiledPlanArtifact(BaseModel):
    """One versioned, immutable compiled-plan artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = PLAN_ARTIFACT_SCHEMA
    recipe: CompressionRecipe
    recipe_sha256: str
    recipe_id: str
    plan_id: str
    resolved_pins: dict[str, dict[str, str]] = Field(default_factory=dict)
    backend_status_snapshot: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, object] = Field(default_factory=dict)
    run_id: str
    reproduce_command: str = ""

    @classmethod
    def from_compiled(
        cls,
        compiled: CompiledRecipe,
        inputs: dict[str, object] | None = None,
    ) -> CompiledPlanArtifact:
        """Build a versioned immutable artifact from a compiled plan + inputs."""
        inputs = inputs or {}
        run_id = compiled.run_id(inputs)
        return cls(
            recipe=compiled.recipe,
            recipe_sha256=compiled.recipe_sha256,
            recipe_id=compiled.recipe_id,
            plan_id=compiled.plan_id,
            resolved_pins=_resolve_pins(compiled),
            backend_status_snapshot=dict(compiled.backend_status_snapshot),
            inputs=inputs,
            run_id=run_id,
            reproduce_command=(
                f"model-atlas job start --plan <this-file> --out <work-root> # run_id {run_id}"
            ),
        )

    def canonical_payload(self) -> str:
        """Deterministic canonical serialization of everything that matters."""
        return canonical_json(self.model_dump(exclude={"reproduce_command"}))

    def verify(self) -> None:
        """Self-consistency: ids/hash must match the embedded recipe and the
        deterministic run_id from recipe+inputs. Raises on any mismatch."""
        sha = self.recipe_sha256
        if not sha:
            raise ValueError("compiled-plan artifact has no recipe_sha256")
        from model_atlas.recipe.compiler import _compute_recipe_sha

        if _compute_recipe_sha(self.recipe) != sha:
            raise ValueError(
                "compiled-plan artifact is corrupt: embedded recipe does not match "
                "its declared recipe_sha256"
            )
        rid = "recipe-" + sha[:24]
        if rid != self.recipe_id or rid != self.plan_id:
            raise ValueError(
                f"compiled-plan artifact ids inconsistent: recipe_id {self.recipe_id} "
                f"plan_id {self.plan_id} vs recomputed {rid}"
            )
        expected_run = _run_id_from_payload(sha, self.plan_id, self.inputs)
        if self.run_id != expected_run:
            raise ValueError(
                f"compiled-plan artifact run_id {self.run_id} != recomputed "
                f"{expected_run} from inputs {self.inputs}"
            )


def _resolve_pins(compiled: CompiledRecipe) -> dict[str, dict[str, str]]:
    """Stage -> {backend_id, resolved_version (exact)}. A stage without an
    exact resolved version is recorded as dry-run-only (non-executable)."""
    pins: dict[str, dict[str, str]] = {}
    for s in compiled.recipe.stages:
        pins[s.id] = {
            "backend_id": s.backend.backend_id,
            "pinned_version": s.backend.version,
            "resolved_version": compiled.backend_status_snapshot.get(s.id, "unresolved"),
        }
    return pins


def _run_id_from_payload(recipe_sha256: str, plan_id: str, inputs: dict[str, object]) -> str:
    import hashlib

    payload = canonical_json(
        {"plan_id": plan_id, "recipe_sha256": recipe_sha256, "job_inputs": inputs}
    )
    return "run-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
