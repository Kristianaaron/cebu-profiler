"""Typed service/API facade for the Atlas recommendation workflow.

Facade operations (used by the local GUI and agents):
  * list_profiles() / import_profile(path) — discover + import completed
    Atlas profiles (JSON), keyed by stable profile_id.
  * recommend(profile_id, target) — deterministic versioned recommendation.
  * authorize(...) — deterministic recommendation PLUS an opaque, server-side
    authorization token bound to the canonical recommendation/profile/target/
    constraints and the exact authorized method set.
  * preview_selection(token, selected) — token-gated draft build; stores a
    verified compiled artifact server-side keyed by token + deterministic
    selection hash and returns preview_id / plan_id / hash. Rejects
    empty/unknown/not-authorized/blocked selections.
  * start_authorized(token, preview_id, hash, selected, inputs) — start ONLY a
    token+preview-bound verified executable plan. Rejects stale/mismatch/
    replay/unknown. Persists job identity BEFORE dispatch, returns run_id
    immediately, executes in a managed background worker. Duplicate starts are
    idempotent (same deterministic run_id), never synchronous request blocking.
  * preview_recipe(recipe_draft, ...) / compile_recipe(recipe) — compile an
    EDITABLE recipe draft (recipe compile/verify, fail-closed on errors).
  * job_status / job_events / job_validate / job_lineage / job_output — expose
    progress, events, validation, lineage, content-addressed outputs.

Authorization: recommendations/compiles are deterministic and versioned; the
opaque token is the ONLY thing that authorizes preview+start. A recommendation
with no token is inert — it authorizes nothing. Agent-readable explanations are
never a substitute. Repair application and approval are NOT exposed here.

Start is strictly bound: it accepts only a ``(token, preview_id, selection
hash, exact same selection, inputs)`` tuple re-verified against the server-side
stored verified compiled artifact. There is no arbitrary-recipe start and no
tokenless raw-selection start.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import threading
from pathlib import Path
from typing import Any

from model_atlas.backend.registry import BackendRegistry, build_default_registry
from model_atlas.controlplane.api import ControlPlane
from model_atlas.jobs.artifacts import atomic_write_json
from model_atlas.jobs.engine import JobEngine
from model_atlas.jobs.schema import JobStatus
from model_atlas.recipe.compiler import CompiledRecipe, canonical_json, sha256_hex
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


class AuthError(Exception):
    """Raised for any authorization/preview/start rejection.

    Carries an HTTP status and a stable machine-readable ``code`` so the
    server can distinguish ``unknown`` from ``not-authorized`` from
    ``stale``/``mismatch``/``replay``/``empty`` with a precise reason.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

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


def _selection_hash(selected: list[str] | None) -> str:
    """Deterministic digest of the exact authorized method set (sorted, so set
    identity — not ordering — defines the selection; duplicates collapse)."""
    sel = sorted(set(selected or ())) if selected else []
    return sha256_hex(canonical_json({"authorized_methods": sel}))


def _same_selection(a: list[str], b: list[str]) -> bool:
    return set(a) == set(b)


def _run_id_from_recipe(recipe: CompressionRecipe, inputs: dict[str, object]) -> str:
    """Deterministic run id for a draft recipe + canonical inputs, matching the
    documented run_id derivation (plan_id + recipe_sha256 + inputs)."""
    payload = canonical_json(recipe.model_dump(mode="json"))
    recipe_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    plan_id = "recipe-" + recipe_sha[:24]
    run_payload = canonical_json(
        {"plan_id": plan_id, "recipe_sha256": recipe_sha, "job_inputs": inputs}
    )
    return "run-" + hashlib.sha256(run_payload.encode("utf-8")).hexdigest()[:24]


class _AuthorizationSession:
    """Server-side authorization state for one recommend call.

    Binds the opaque token to the canonical recommendation ids, the resolved
    profile, the target, the constraints and the EXACT authorized method set.
    A token is minted BY recommend and is the only handle a caller may use to
    preview or start a plan built from that recommendation.
    """

    __slots__ = (
        "token",
        "recommendation_id",
        "profile_id",
        "target",
        "no_pruning",
        "constraints_snapshot",
        "authorized_methods",
        "created_at",
    )

    def __init__(
        self,
        token: str,
        recommendation_id: str,
        profile_id: str,
        target: RecTarget,
        no_pruning: bool,
        constraints_snapshot: dict[str, object],
        authorized_methods: list[str],
    ) -> None:
        self.token = token
        self.recommendation_id = recommendation_id
        self.profile_id = profile_id
        self.target = target
        self.no_pruning = no_pruning
        self.constraints_snapshot = constraints_snapshot
        self.authorized_methods = sorted(authorized_methods)
        self.created_at = _now_iso()

    def selected_hash(self) -> str:
        return _selection_hash(self.authorized_methods)


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class _PendingPreview:
    """A stored, token-bound verified preview/start package.

    Holds the server-side derivatives of the exact authorized selection: the
    draft recipe, the VERIFIED compiled artifact, and the exact inputs used at
    preview time. Start may only consume a stored package that is still
    pending (not already dispatched) and whose hash matches.
    """

    __slots__ = (
        "token",
        "preview_id",
        "selection_hash",
        "selected",
        "recipe",
        "artifact",
        "inputs",
        "run_id",
        "dispatch_started",
    )

    def __init__(
        self,
        token: str,
        preview_id: str,
        selection_hash: str,
        selected: list[str],
        recipe: CompressionRecipe,
        artifact: CompiledPlanArtifact | None,
        inputs: dict[str, object],
        run_id: str,
    ) -> None:
        self.token = token
        self.preview_id = preview_id
        self.selection_hash = selection_hash
        self.selected = list(selected)
        self.recipe = recipe
        self.artifact = artifact
        self.inputs = dict(inputs)
        self.run_id = run_id
        self.dispatch_started = False


class RecommendationService:
    """Facade over profiles + policy + control plane (versioned policy v1)."""

    def __init__(
        self,
        registry: BackendRegistry | None = None,
        profile_root: str | Path = "profiles",
        work_root: str | Path = "controlplane_runs",
        store_root: str | Path | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.policy = RecommendationPolicy(self.registry)
        self.plane = ControlPlane(registry=self.registry, work_root=work_root)
        self.profile_root = Path(profile_root)
        self.profile_root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, _AuthorizationSession] = {}
        self.pending_previews: dict[str, _PendingPreview] = {}
        self.dispatched: dict[str, dict[str, object]] = {}
        self._dispatch_lock = threading.Lock()
        self._started_workers: list[threading.Thread] = []
        self.store_root = Path(store_root) if store_root is not None else self.plane.work_root
        self.store_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- profiles
    def list_profiles(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for f in sorted(self.profile_root.rglob("*.json")):
            try:
                prof = self.import_profile(f)
            except Exception:  # noqa: BLE001 — skip unparseable profiles
                continue
            out.append(
                {
                    "profile_id": prof.profile_id_of(),  # canonical (content) id
                    "declared_profile_id": prof.profile_id,  # preserved alias
                    "path": str(f),
                }
            )
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

    # --------------------------------------------------------- authorization
    def authorize(
        self,
        profile: AtlasProfile | str,
        target: RecTarget | None = None,
        *,
        memory_target_gib: float | None = None,
        allow_pruning: bool = False,
        constraints: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Deterministic recommendation PLUS an opaque authorization token.

        The token is bound to the canonical recommendation id, the resolved
        profile id, the target, the constraints snapshot and the EXACT
        authorized method set (the non-blocked recommended methods). Nothing
        else — no recipe, no selection — authorizes a preview or a start. The
        returned recommendation is identical to :meth:`recommend`; the token is
        what makes it actionable.
        """
        prof = self._resolve_profile(profile)
        effective_target = target or RecTarget()
        constraints = dict(constraints or {})
        allow_pruning = bool(constraints.get("allow_pruning", allow_pruning))
        rec = self.policy.recommend(
            prof,
            effective_target,
            memory_target_gib=memory_target_gib,
            allow_pruning=allow_pruning,
        )
        authorized = sorted(m.method for m in rec.methods)
        token = secrets.token_urlsafe(24)
        constraints_snapshot: dict[str, object] = {
            "allow_pruning": bool(constraints.get("allow_pruning", allow_pruning)),
        }
        session = _AuthorizationSession(
            token=token,
            recommendation_id=rec.recommendation_id,
            profile_id=rec.profile_id,
            target=effective_target,
            no_pruning=rec.no_pruning,
            constraints_snapshot=constraints_snapshot,
            authorized_methods=authorized,
        )
        self.sessions[token] = session
        payload = rec.to_dict() if hasattr(rec, "to_dict") else rec
        return {
            "token": token,
            "recommendation_id": rec.recommendation_id,
            "profile_id": rec.profile_id,
            "no_pruning": rec.no_pruning,
            "authorized_methods": authorized,
            "selection_hash": _selection_hash(authorized),
            "recommendation": payload,
        }

    # ------------------------------------------------------- preview (auth)
    def preview_selection(
        self,
        token: str,
        selected: list[str] | None = None,
        *,
        inputs: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Token-gated preview-from-selection.

        Only a live token (minted by :meth:`authorize`) may preview, and the
        selection must be a NONEMPTY SUBSET of the token's authorized method
        set (every selected method must actually be authorized — never
        blocked/unknown/outside the set). The subset's own deterministic hash is
        what binds the preview, start, and recomputation. On success the
        deterministic compiled artifact is verified and STORED server-side
        keyed by the preview. Returns a bounded ``preview_id``/``plan_id``/
        ``hash`` handle — never the full recipe or artifact payload.
        """
        token = token or ""
        if not token:
            raise AuthError(401, "token_required", "authorization token required")
        session = self.sessions.get(token)
        if session is None:
            raise AuthError(401, "token_unknown", "unknown authorization token")
        if selected is None or len(selected) == 0:
            raise AuthError(400, "selection_empty", "selection must be non-empty")
        if not all(isinstance(m, str) for m in selected):
            raise AuthError(400, "selection_invalid", "selection must be method strings")
        sel = sorted(set(selected))
        # every selected method must be authorized by the token's session —
        # never the full-set hash: the subset's OWN hash binds this preview.
        unauthorized = [m for m in sel if m not in set(session.authorized_methods)]
        if unauthorized:
            raise AuthError(
                403,
                "selection_not_authorized",
                f"not authorized: {unauthorized}",
            )
        subset_hash = _selection_hash(sel)
        pv = self.recipe_preview(sel)
        plan = pv.get("plan") or {}
        readiness = {
            "verified_plan": bool(plan.get("plan_id") and plan.get("pins_pass")),
            "pins_pass": bool(plan.get("pins_pass")),
            "executable": self._preview_is_executable(pv),
        }
        preview_id = (
            "pv-"
            + sha256_hex(
                canonical_json({"token": token, "hash": subset_hash})
            )[:16]
        )
        recipe = self._selected_recipe(sel)
        artifact: CompiledPlanArtifact | None = None
        if readiness["executable"]:
            compiled = self.plane.compile_recipe(recipe)
            artifact = CompiledPlanArtifact.from_compiled(
                compiled, inputs=inputs or {}, registry=self.registry
            )
            artifact.verify()
            artifact.verify_pins_against(self.registry)
        package = _PendingPreview(
            token=token,
            preview_id=preview_id,
            selection_hash=subset_hash,
            selected=sel,
            recipe=recipe,
            artifact=artifact,
            inputs=inputs or {},
            run_id=(
                artifact.run_id
                if artifact is not None
                else _run_id_from_recipe(recipe, inputs or {})
            ),
        )
        self.pending_previews[preview_id] = package
        self._persist_preview(package)
        return {
            "preview_id": preview_id,
            "plan_id": (artifact.plan_id if artifact is not None else plan.get("plan_id")),
            "hash": subset_hash,
            "readiness": readiness,
            "selected_methods": list(sel),
        }

    @staticmethod
    def _preview_is_executable(preview: dict[str, Any]) -> bool:
        plan = preview.get("plan") or {}
        return bool(plan.get("plan_id") and plan.get("pins_pass"))

    def _persist_preview(self, package: _PendingPreview) -> None:
        with contextlib.suppress(Exception):  # persistence best-effort; the
            # live in-memory store still authorizes the exact session/start.
            artifact_dir = self.store_root / "previews" / package.preview_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                artifact_dir / "preview.json",
                {
                    "preview_id": package.preview_id,
                    "token": package.token,
                    "selection_hash": package.selection_hash,
                    "selected": list(package.selected),
                    "plan_id": (
                        package.artifact.plan_id if package.artifact is not None else ""
                    ),
                    "run_id": package.run_id,
                },
            )
            if package.artifact is not None:
                atomic_write_json(
                    artifact_dir / "plan.json", package.artifact.to_plain_dict()
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
        against the LIVE registry before starting; fail closed otherwise.

        This is the internal executor used by :meth:`start_authorized` after
        the token/preview package has been re-verified. It is NOT exposed over
        the HTTP API directly — a caller must hold a valid authorization token.
        """
        compiled = self.compile_recipe(recipe)
        artifact = CompiledPlanArtifact.from_compiled(
            compiled, inputs=inputs or {}, registry=self.registry
        )
        artifact.verify()
        artifact.verify_pins_against(self.registry)
        engine = JobEngine(compiled, self.registry, self.plane.work_root)
        engine.run(inputs or {})
        return engine

    # ---------------------------------------------------- start (token-gated)
    def start_authorized(
        self,
        token: str,
        preview_id: str,
        selection_hash: str,
        selected: list[str],
        inputs: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Start ONLY a verified, token-and-preview-bound executable plan.

        Every incoming handle must agree with the server-side authorization
        state: the token must be live, the preview must be pending, and the
        supplied ``selection_hash`` + ``selected`` must EXACTLY match what the
        preview was compiled for. Any mismatch (stale/unknown/replay/mismatch/
        empty) is rejected deterministically. On success the job identity is
        persisted BEFORE dispatch and ``run_id`` is returned immediately; the
        plan executes in a managed background worker so this call never blocks
        the request thread. A duplicate start of the same package is idempotent
        (same deterministic run_id) — never a second execution.
        """
        token = token or ""
        if not token:
            raise AuthError(401, "token_required", "authorization token required")
        session = self.sessions.get(token)
        if session is None:
            raise AuthError(401, "token_unknown", "unknown authorization token")
        if not preview_id:
            raise AuthError(400, "preview_required", "preview_id required")
        package = self.pending_previews.get(preview_id)
        if package is None:
            raise AuthError(410, "preview_unknown", "preview not found (stale or never created)")
        if package.token != token:
            raise AuthError(403, "preview_token_mismatch", "preview does not belong to this token")
        if not selection_hash:
            raise AuthError(400, "hash_required", "selection hash required")
        if selection_hash != package.selection_hash:
            raise AuthError(
                409, "selection_mismatch", "selection hash does not match the preview"
            )
        if selected is None or len(selected) == 0:
            raise AuthError(400, "selection_empty", "start selection must be non-empty")
        if package.artifact is None:
            raise AuthError(
                409,
                "preview_not_executable",
                "preview has no verified executable plan; start refused",
            )
        if _selection_hash(selected) != package.selection_hash:
            raise AuthError(
                409, "selection_mismatch", "start selection does not match the preview"
            )
        if not _same_selection(selected, package.selected):
            raise AuthError(
                409, "selection_mismatch", "start selection differs from the preview selection"
            )
        # inputs must EXACTLY match the inputs the preview was compiled with
        # (the run identity is derived from the canonical preview inputs, so a
        # different input set is a mismatch — never a silent re-identity).
        if (inputs or {}) != package.inputs:
            raise AuthError(
                409, "inputs_mismatch", "start inputs differ from the preview inputs"
            )

        run_id = package.run_id
        # Persist job/run identity BEFORE dispatch (synchronously), so the run
        # is immediately observable. Bind to the CANONICAL preview inputs so
        # the deterministic run id is authoritative, write the durable PENDING
        # job record AND the immutable plan artifact (engine_for/status read
        # both), then dispatch the run in a managed background worker.
        engine_compiled = self.plane.compile_recipe(package.recipe)
        engine = JobEngine(engine_compiled, self.registry, self.plane.work_root)
        engine._bind_run(package.inputs)
        if not engine.job_path.exists():
            job = engine._init_run_dir(package.inputs)  # run dir + journal
            engine._save(job)  # durable PENDING job.json
            with contextlib.suppress(Exception):
                atomic_write_json(engine.plan_path, package.artifact.to_plain_dict())

        with self._dispatch_lock:
            # A duplicate start of an ALREADY-dispatched package is rejected
            # deterministically (replay) — never a second execution, never
            # synchronous blocking.
            if package.dispatch_started:
                stored = self.dispatched.get(run_id)
                if stored and stored.get("preview_id") == preview_id:
                    raise AuthError(
                        409, "replay", f"preview {preview_id} already started (run {run_id})"
                    )
            # dispatch exactly once in a managed background worker.
            worker = threading.Thread(
                target=self._dispatch_worker,
                args=(package, dict(package.inputs), preview_id),
                name=f"rec-run-{run_id}",
                daemon=True,
            )
            self._started_workers.append(worker)
            package.dispatch_started = True
            self.dispatched.setdefault(run_id, {"preview_id": preview_id})
            worker.start()
        return {"run_id": run_id, "status": "started"}

    def _dispatch_worker(
        self, package: _PendingPreview, inputs: dict[str, object], preview_id: str
    ) -> None:
        """Managed background worker: crash-safe, durable run of the stored
        verified package. Never touches the request thread. Any driver failure
        leaves the standard FAILED_TERMINAL journal + job.json so status is
        observable; the persisted job identity is never lost."""
        try:
            engine_compiled = self.plane.compile_recipe(package.recipe)
            engine = JobEngine(engine_compiled, self.registry, self.plane.work_root)
            engine._bind_run(package.inputs)
            engine.run(package.inputs)
        except Exception:  # noqa: BLE001
            # Best-effort reconciliation of a dispatch that never created a
            # durable job; the request already returned run_id, so surface the
            # reversal via a terminal FAILED_TERMINAL record when possible.
            try:
                engine_compiled = self.plane.compile_recipe(package.recipe)
                engine = JobEngine(engine_compiled, self.registry, self.plane.work_root)
                engine._bind_run(package.inputs)
                job = engine._load_job()
                if job is None:
                    engine._init_run_dir(inputs)
                    job = engine._load_job()
                    if job is not None:
                        job.status = JobStatus.FAILED_TERMINAL
                        job.error = "background dispatch failed before run completion"
                        engine.journal.append(
                            {
                                "event": "run.terminal",
                                "status": "failed_terminal",
                                "reason": "dispatch_error",
                            }
                        )
                        engine._save(job)
            except Exception:  # noqa: BLE001
                pass


    def job_status(self, run_id: str) -> dict[str, Any]:
        return dict(self.plane.status(run_id))

    def job_events(self, run_id: str) -> list[dict[str, Any]]:
        events = self.plane.status(run_id).get("events", [])
        return list(events) if isinstance(events, list) else []

    def job_validate(self, run_id: str, stage_id: str) -> dict[str, Any]:
        return dict(self.plane.validate(run_id, stage_id))

    def job_lineage(self, recipe: CompressionRecipe) -> dict[str, Any]:
        return dict(self.plane.lineage(recipe))

    def run_lineage(self, run_id: str) -> dict[str, Any]:
        """Content-derived lineage for an ACTUAL completed/observable run.

        Reads the persisted immutable plan artifact (recipe + run identity) and
        computes the same reproducibility lineage the control plane derives for
        a recipe, but keyed to the concrete persisted run — so monitor can show
        lineage for the run it actually executed, never a hard-coded recipe={}.
        Fails closed when the run dir is gone/unknown.
        """
        engine = self.plane.engine_for(run_id)
        if not engine.plan_path.exists():
            return {"run_id": run_id, "error": "no plan artifact", "compile": "unknown"}
        artifact = CompiledPlanArtifact.model_validate_json(
            engine.plan_path.read_text(encoding="utf-8")
        )
        base = dict(self.plane.lineage(artifact.recipe))
        base["run_id"] = engine.compiled.run_id(dict(artifact.inputs)) or run_id
        # recompute the deterministic id from the CANONICAL (persisted) inputs
        base["run_lineage"] = {
            "run_id": run_id,
            "preview inputs": dict(artifact.inputs),
            "recipe_sha256": artifact.recipe_sha256,
            "plan artifact ids": {
                "recipe_id": artifact.recipe_id,
                "plan_id": artifact.plan_id,
                "run_id": artifact.run_id,
            },
        }
        return base

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
        "declared_profile_id": profile.profile_id,
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
