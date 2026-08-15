"""Typed service/API facade for the Atlas recommendation workflow.

Facade operations (used by the local GUI and agents):
  * list_profiles() / import_profile(path) — discover + import completed
    Atlas profiles (JSON), keyed by stable profile_id.
  * recommend(profile_id, target) — deterministic versioned recommendation.
  * preview_recipe(recipe_draft, ...) / compile_recipe(recipe) — compile an
    EDITABLE recipe draft (recipe compile/verify, fail-closed on errors).
  * start(recipe) — start ONLY a verified executable plan (verifies pins
    against the live registry before starting).
  * job_status / job_events / job_validate / job_lineage / job_output — expose
    progress, events, validation, lineage, content-addressed outputs.

Authorization: recommendations/compiles are deterministic and versioned;
agent-readable explanations never authorize anything. Repair application and
approval are NOT exposed here — agents cannot silently mutate/approve repairs
(those live behind the repair gate + engine transaction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_atlas.backend.registry import BackendRegistry, build_default_registry
from model_atlas.controlplane.api import ControlPlane
from model_atlas.jobs.engine import JobEngine
from model_atlas.recipe.compiler import CompiledRecipe
from model_atlas.recipe.schema import (
    CompressionRecipe,
    RecipeConstraints,
    RecipeStage,
    StageEffectClass,
)
from model_atlas.recipes import CompiledPlanArtifact
from model_atlas.recipes.builtin import glm52_no_pruning_recipe
from model_atlas.recommend.policy import (
    AtlasProfile,
    Recommendation,
    RecommendationPolicy,
    RecTarget,
)

# Policy-versioned method -> canonical no-pruning recipe stage ids. A selected
# recommendation method contributes exactly these stages to the draft recipe;
# dependent stages are pulled in transitively by format closure.
_SELECTION_STAGES = {
    "teacher-identity": ["t1-identity"],
    "calibration": ["t2-calibration"],
    "sensitivity": ["t3-sensitivity"],
    "bit-allocation": ["t6-bit-allocation"],
    "kv-optimization": ["t12-kv"],
    "exl3-primary": ["t7-exl3"],
    "nvfp4-substitute": ["t10-nvfp4"],
    "modelopt-nvfp4": ["t10-nvfp4"],
    "llm-compressor": ["t11-tail"],
}

_STAGE_ORDER = [
    "t1-identity",
    "t2-calibration",
    "t3-sensitivity",
    "t4-representation",
    "t5-conditioning",
    "t6-bit-allocation",
    "t7-exl3",
    "t8-refinement",
    "t9-residual",
    "t10-nvfp4",
    "t11-tail",
    "t12-kv",
    "t13-runtime",
    "t14-eval",
]


class RecommendationService:
    """Facade over profiles + policy + control plane (versioned policy v1)."""

    def __init__(
        self,
        registry: BackendRegistry | None = None,
        profile_root: str | Path = "profiles",
        work_root: str | Path = "controlplane_runs",
    ) -> None:
        self.registry = registry or build_default_registry()
        self.policy = RecommendationPolicy(self.registry)
        self.plane = ControlPlane(registry=self.registry, work_root=work_root)
        self.profile_root = Path(profile_root)
        self.profile_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- profiles
    def list_profiles(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for f in sorted(self.profile_root.rglob("*.json")):
            try:
                prof = self.import_profile(f)
            except Exception:  # noqa: BLE001 — skip unparseable profiles
                continue
            out.append({"profile_id": prof.profile_id_of(), "path": str(f)})
        return out

    def import_profile(self, path: str | Path) -> AtlasProfile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return _profile_from_dict(data)

    def save_profile(self, profile: AtlasProfile, path: str | Path | None = None) -> str:
        p = (
            Path(path)
            if path is not None
            else self.profile_root / f"{profile.profile_id_of()}.json"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_profile_to_dict(profile), indent=2, sort_keys=True))
        return str(p)

    # --------------------------------------------------------- recommendation
    def recommend(
        self,
        profile: AtlasProfile | str,
        target: RecTarget | None = None,
        *,
        memory_target_gib: float | None = None,
        allow_pruning: bool = False,
    ) -> Recommendation:
        prof = self._resolve_profile(profile)
        return self.policy.recommend(
            prof,
            target or RecTarget(),
            memory_target_gib=memory_target_gib,
            allow_pruning=allow_pruning,
        )

    # ----------------------------------------------------------- recipe
    def preview_recipe(self, recipe: CompressionRecipe) -> dict[str, Any]:
        """Dry-run compile of an editable recipe draft; never starts."""
        issues, rid, sha = self.plane.compiler.validate(recipe)
        return {
            "recipe_id": rid,
            "recipe_sha256": sha,
            "compiles": not [i for i in issues if i.severity == "error"],
            "issues": [i.to_dict() for i in issues],
        }

    def compile_recipe(self, recipe: CompressionRecipe) -> CompiledRecipe:
        """Compile (immutable) an editable recipe; fails closed on errors."""
        return self.plane.compile_recipe(recipe)

    # ------------------------------------------------------- draft builder
    def _selected_recipe(
        self,
        selected: list[str] | None = None,
        *,
        none_is_all: bool = False,
    ) -> CompressionRecipe:
        """Build a deterministic no-pruning recipe draft from a subset of the
        policy's recommended methods.

        ``selected`` names recommendation methods (policy method ids). Stages
        are taken from the canonical builtin GLM-5.2 recipe and closed over
        transitive ``requires_formats`` dependencies, then reordered by the
        canonical stage order. ``None`` selects all stages unless
        ``none_is_all`` is False (then the builtin's full no-pruning recipe is
        used). Compression stages are stripped of backend pins that would make
        the draft non-compilable (unavailable/derivative-only/pinned-version
        gates): keeping them would make the *editable draft* fail closed before
        any diff/preview could be shown. Execution is still gated: a real
        ``/api/start`` serves a selection only after this draft compiles
        cleanly AND the resulting verified plan's live pins pass.
        """
        base = glm52_no_pruning_recipe()
        stages_by_id = {s.id: s for s in base.stages}
        if selected is None or none_is_all:
            stages = list(base.stages)
        else:
            wanted: set[str] = set()
            for method in selected:
                wanted.update(_SELECTION_STAGES.get(method, ()))
            wanted = self._format_closure(stages_by_id, wanted)
            stages = [stages_by_id[sid] for sid in _STAGE_ORDER if sid in wanted]
        recipe = base.model_copy(deep=True)
        recipe.stages = stages
        recipe.constraints = RecipeConstraints(
            no_pruning=True,
            allow_pruning_capability=False,
            preserve_non_expert_backbone=True,
            immutable_source=True,
            allow_hybrid_precision=False,
            max_resident_gib=115.0,
            derived_format="safetensors",
        )
        # strip unavailable/derivative-only exact-pin gates so the EDITABLE
        # draft previews; execution remains fail-closed behind a verified plan.
        stripped = []
        for s in recipe.stages:
            if s.effect_class in {
                StageEffectClass.CONDITIONING,
                StageEffectClass.QUANTIZATION,
                StageEffectClass.RESIDUAL,
                StageEffectClass.REFINEMENT,
            } or s.backend.backend_id in {"exl3", "modelopt_nvfp4", "llm_compressor"}:
                s2 = s.model_copy(deep=True)
                s2.backend = s.backend.model_copy(update={"require_available": False})
                stripped.append(s2)
            else:
                stripped.append(s)
        recipe.stages = stripped
        return recipe

    @staticmethod
    def _format_closure(
        stages_by_id: dict[str, RecipeStage],
        wanted: set[str],
    ) -> set[str]:
        """Transitively add the producers of every format a wanted stage
        requires, over the requested stage set only."""
        by_format: dict[str, set[str]] = {}
        for sid, s in stages_by_id.items():
            for f in s.produces_format:
                by_format.setdefault(f, set()).add(sid)
        frontier = list(wanted)
        while frontier:
            sid = frontier.pop()
            for f in stages_by_id[sid].requires_formats:
                for prod in by_format.get(f, ()):
                    if prod not in wanted:
                        wanted.add(prod)
                        frontier.append(prod)
        return wanted

    def recipe_preview(
        self, selected: list[str] | None = None
    ) -> dict[str, Any]:
        """Deterministic preview-from-selection: build the draft, run the
        non-fatal compile/dry-run, and report diff / compile blockers /
        readiness / verified plan (if it compiles). Never mutates."""
        recipe = self._selected_recipe(selected)
        issues, rid, sha = self.plane.compiler.validate(recipe)
        errors = [i for i in issues if i.severity == "error"]
        compiles = not errors
        plan: dict[str, Any] | None = None
        if compiles:
            plan = self._verified_plan_dict(recipe)
        return {
            "selected_methods": list(selected or _SELECTION_STAGES.keys()),
            "recipe_id": rid,
            "recipe_sha256": sha,
            "compiles": compiles,
            "stages": [s.model_dump(mode="json") for s in recipe.stages],
            "issues": [i.to_dict() for i in issues],
            "compile_blockers": [i.to_dict() for i in errors],
            "readiness": {
                "verified_plan": plan is not None,
                "pins_pass": bool(plan and plan.get("pins_pass")),
                "executable": bool(plan and plan.get("pins_pass")),
            },
            "plan": plan,
            "diff": self._recipe_diff(recipe),
        }

    def _verified_plan_dict(self, recipe: CompressionRecipe) -> dict[str, Any]:
        """Compile the given recipe and produce a verified-plan summary:
        plan_id + whether the compiled artifact's live pins pass. The full
        pinned artifact is NOT returned to the client (no embedded recipe); the
        GUI only needs plan_id + readiness + reproduce_command."""
        try:
            compiled = self.plane.compile_recipe(recipe)
        except Exception as exc:  # noqa: BLE001
            return {"pins_pass": False, "error": str(exc)}
        try:
            artifact = CompiledPlanArtifact.from_compiled(
                compiled, inputs={}, registry=self.registry
            )
            artifact.verify()
            artifact.verify_pins_against(self.registry)
        except Exception as exc:  # noqa: BLE001
            return {
                "plan_id": compiled.plan_id,
                "pins_pass": False,
                "error": str(exc),
            }
        return {
            "plan_id": compiled.plan_id,
            "pins_pass": True,
            "reproduce_command": artifact.reproduce_command,
        }

    @staticmethod
    def _recipe_diff(recipe: CompressionRecipe) -> dict[str, Any]:
        """Policy-informative diff vs the canonical builtin no-pruning recipe:
        which canonical stages are present vs omitted, and the constrained
        memory ceiling. Nothing here is a mutation."""
        base = glm52_no_pruning_recipe()
        keep = {s.id for s in recipe.stages}
        return {
            "against": base.name,
            "enabled_stages": [s.id for s in recipe.stages],
            "omitted_stages": [s.id for s in base.stages if s.id not in keep],
            "no_pruning": recipe.constraints.no_pruning,
            "max_resident_gib": recipe.constraints.max_resident_gib,
        }

    # ------------------------------------------------------------ execution
    def start(
        self, recipe: CompressionRecipe, inputs: dict[str, object] | None = None
    ) -> JobEngine:
        """Start ONLY a verified executable plan: compile then verify pins
        against the LIVE registry before starting; fail closed otherwise."""
        compiled = self.compile_recipe(recipe)
        artifact = CompiledPlanArtifact.from_compiled(
            compiled, inputs=inputs or {}, registry=self.registry
        )
        artifact.verify()
        artifact.verify_pins_against(self.registry)
        engine = JobEngine(compiled, self.registry, self.plane.work_root)
        engine.run(inputs or {})
        return engine

    def job_status(self, run_id: str) -> dict[str, Any]:
        return dict(self.plane.status(run_id))

    def job_events(self, run_id: str) -> list[dict[str, Any]]:
        events = self.plane.status(run_id).get("events", [])
        return list(events) if isinstance(events, list) else []

    def job_validate(self, run_id: str, stage_id: str) -> dict[str, Any]:
        return dict(self.plane.validate(run_id, stage_id))

    def job_lineage(self, recipe: CompressionRecipe) -> dict[str, Any]:
        return dict(self.plane.lineage(recipe))

    def job_output(
        self,
        run_id: str,
        *,
        stage_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any] | bytes:
        """Content-addressed outputs of a run.

        With no ``name`` returns metadata for every published output (optionally
        filtered to a single ``stage_id``). With a ``name`` returns the blob's
        raw bytes — the stage filter narrows a name collision across stages.
        Fails closed (raises) when the run or output does not exist.
        """
        engine = self.plane.engine_for(run_id)
        inspect = engine.inspect()
        stages_raw = inspect.get("stages", {})
        stages: dict[str, Any] = stages_raw if isinstance(stages_raw, dict) else {}
        refs: list[tuple[str, dict[str, Any]]] = []
        for sid, so in stages.items():
            so_dict = so.model_dump(mode="json") if hasattr(so, "model_dump") else so
            stage_refs = (
                [r.model_dump(mode="json") for r in so.outputs]
                if hasattr(so, "outputs")
                else so_dict.get("outputs", [])
                if isinstance(so_dict, dict)
                else []
            )
            if stage_id is not None and sid != stage_id:
                continue
            for r in stage_refs:
                if isinstance(r, dict):
                    refs.append((sid, r))
        if name is not None:
            hits = [(sid, r) for sid, r in refs if r.get("name") == name]
            if not hits:
                raise KeyError(
                    f"run {run_id!r} has no output named {name!r}"
                    + (f" on stage {stage_id!r}" if stage_id else "")
                )
            sha = str(hits[0][1].get("sha256", ""))
            return engine.store.read_from_key(sha)
        return {
            "run_id": run_id,
            "outputs": [
                {
                    "stage": sid,
                    "name": r.get("name"),
                    "sha256": r.get("sha256"),
                    "size_bytes": r.get("size_bytes", 0),
                    "format": r.get("format", ""),
                    "relpath": r.get("relpath", ""),
                }
                for sid, r in refs
            ],
        }

    # ------------------------------------------------------------- helpers
    def _resolve_profile(self, profile: AtlasProfile | str) -> AtlasProfile:
        if isinstance(profile, AtlasProfile):
            return profile
        for f in self.profile_root.rglob("*.json"):
            try:
                p = self.import_profile(f)
            except Exception:  # noqa: BLE001
                continue
            if self._profile_key_matches(p, profile):
                return p
        raise KeyError(f"profile {profile!r} not found under {self.profile_root}")

    @staticmethod
    def _profile_key_matches(p: AtlasProfile, key: str) -> bool:
        """Stable lookup keys: authoritative content id, the file's declared
        (optional, non-authoritative) profile_id, or the model name. Empty or
        'unknown'-only matches never short-circuit a real profile."""
        if p.profile_id_of() == key:
            return True
        if p.profile_id and p.profile_id != "imported" and p.profile_id == key:
            return True
        return bool(p.model and p.model != "unknown" and p.model == key)


def _profile_to_dict(profile: AtlasProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id_of(),
        "model": profile.model,
        "seed": profile.seed,
        "hardware_model_arch": profile.hardware_model_arch,
        "routing_consistency_passed": profile.routing_consistency_passed,
        "evidence": {
            k: {"kind": v.kind, "present": v.present, "coverage": v.coverage}
            for k, v in profile.evidence.items()
        },
        "notes": profile.notes,
    }


def _profile_from_dict(data: dict[str, Any]) -> AtlasProfile:
    return AtlasProfile.from_dict(data)
