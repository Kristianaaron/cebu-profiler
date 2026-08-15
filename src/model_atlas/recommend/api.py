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
from model_atlas.recipe.schema import CompressionRecipe
from model_atlas.recipes import CompiledPlanArtifact
from model_atlas.recommend.policy import (
    AtlasProfile,
    Recommendation,
    RecommendationPolicy,
    RecTarget,
)


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
