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

import hashlib
import json
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from model_atlas.backend.registry import BackendRegistry, build_default_registry
from model_atlas.controlplane.api import ControlPlane
from model_atlas.jobs.artifacts import atomic_write_json
from model_atlas.jobs.engine import JobEngine
from model_atlas.jobs.schema import JobStatus
from model_atlas.recipe.compiler import CompiledRecipe, canonical_json, sha256_hex
from model_atlas.recipe.schema import (
    CalibrationIdentity,
    CompressionRecipe,
    RecipeConstraints,
    RecipeStage,
    SourceIdentity,
    StageEffectClass,
)
from model_atlas.recipes import CompiledPlanArtifact
from model_atlas.recipes.builtin import glm52_no_pruning_recipe, llamacpp_gguf_mixed_recipe
from model_atlas.recommend.policy import (
    METHOD_CATALOG,
    METHOD_CATALOG_VERSION,
    RECOMMENDATION_POLICY_VERSION,
    AtlasProfile,
    CompressionIntent,
    MethodFamily,
    ProfileExecutionBinding,
    Recommendation,
    RecommendationPolicy,
    RecTarget,
    method_catalog_digest,
    method_spec,
    required_families,
)

# Lifetime of an opaque authorization token. After this the token is EXPIRED
# and every preview/start re-validates it (plus canonical profile + policy)
# against the CURRENT on-disk profile, so a stale authorization can never run.
AUTH_TOKEN_TTL_SECONDS = 3600


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
        "intent",
        "no_pruning",
        "constraints_snapshot",
        "authorized_methods",
        "created_at",
        "expires_at",
        "profile_source_path",
        "profile_fingerprint",
        "profile_bytes_hash",
        "profile_model_arch",
        "execution_binding",
        "revoked",
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
        intent: CompressionIntent = CompressionIntent.QUANTIZE_ONLY,
        ttl_seconds: float = AUTH_TOKEN_TTL_SECONDS,
    ) -> None:
        self.token = token
        self.recommendation_id = recommendation_id
        self.profile_id = profile_id
        self.target = target
        self.intent = intent
        self.no_pruning = no_pruning
        self.constraints_snapshot = constraints_snapshot
        self.authorized_methods = sorted(authorized_methods)
        self.created_at = _now_iso()
        self.expires_at = _iso_from_epoch(time.time() + ttl_seconds)
        self.profile_source_path: str = ""
        self.profile_fingerprint: str = ""
        self.profile_bytes_hash: str = ""
        self.profile_model_arch: str = ""
        self.execution_binding: ProfileExecutionBinding | None = None
        self.revoked = False

    def selected_hash(self) -> str:
        return _selection_hash(self.authorized_methods)


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _iso_from_epoch(epoch: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _epoch_from_iso(iso: str) -> float:
    from datetime import UTC, datetime

    try:
        return datetime.fromisoformat(iso).astimezone(UTC).timestamp()
    except (ValueError, TypeError):
        return float("inf")


def _is_expired(session: _AuthorizationSession) -> bool:
    return _epoch_from_iso(session.expires_at) <= time.time()


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
        "authorization_digest",
        "selected",
        "recipe",
        "artifact",
        "inputs",
        "run_id",
        "plan_id",
        "recipe_sha256",
        "target_snapshot",
        "constraints_snapshot",
        "authorization_snapshot",
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
        authorization_digest: str = "",
    ) -> None:
        self.token = token
        self.preview_id = preview_id
        self.selection_hash = selection_hash
        self.authorization_digest = authorization_digest or selection_hash
        self.selected = list(selected)
        self.recipe = recipe
        self.artifact = artifact
        self.inputs = dict(inputs)
        self.run_id = run_id
        self.plan_id = artifact.plan_id if artifact is not None else ""
        self.recipe_sha256 = artifact.recipe_sha256 if artifact is not None else ""
        self.target_snapshot: dict[str, object] = {}
        self.constraints_snapshot: dict[str, object] = {}
        self.authorization_snapshot: dict[str, object] = {}
        self.dispatch_started = False


class RecommendationService:
    """Facade over profiles + policy + control plane (versioned policy v1)."""

    def __init__(
        self,
        registry: BackendRegistry | None = None,
        profile_root: str | Path = "profiles",
        work_root: str | Path = "controlplane_runs",
        store_root: str | Path | None = None,
        *,
        token_ttl_seconds: float = AUTH_TOKEN_TTL_SECONDS,
        supervised_executor: bool = True,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.policy = RecommendationPolicy(self.registry)
        self.plane = ControlPlane(registry=self.registry, work_root=work_root)
        self.profile_root = Path(profile_root)
        self.profile_root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, _AuthorizationSession] = {}
        self.pending_previews: dict[str, _PendingPreview] = {}
        self.token_ttl_seconds = token_ttl_seconds
        self._dispatch_lock = threading.Lock()
        self._started_workers: list[threading.Thread] = []
        self.store_root = Path(store_root) if store_root is not None else self.plane.work_root
        self.store_root.mkdir(parents=True, exist_ok=True)
        # GLOBAL run_id reservation/idempotency across tokens: run_id ->
        # resolved info. Populated from the durable registry at construction so
        # a restarted service still refuses to double-dispatch the same run.
        self.dispatched: dict[str, dict[str, object]] = {}
        self._load_dispatch_registry()
        self._load_persisted_previews()
        # The executor is created lazily on the first queued run. It is daemonized
        # because durability/reconciliation, not interpreter shutdown ordering,
        # is the crash-safety boundary. Explicit server/context shutdown still
        # drains it during a normal lifecycle.
        self.supervised_executor = supervised_executor
        self._shutdown_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._idle = threading.Condition(self._dispatch_lock)
        self._pending_runs: list[_PendingPreview] = []
        self._reconstruct_dispatch_registry()
        self._reconcile_startup()
        if self._pending_runs and self.supervised_executor:
            with self._dispatch_lock:
                self._ensure_worker_locked()

    def __enter__(self) -> RecommendationService:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

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

    def _bind_profile(self, session: _AuthorizationSession, prof: AtlasProfile) -> None:
        """Bind a token to the FILE-backed canonical profile it authorized, so
        a later preview/start can detect on-disk DRIFT (profile edits) and
        refuse stale authorizations."""
        from model_atlas.jobs.artifacts import sha256_file

        session.profile_fingerprint = prof.profile_id_of()
        session.profile_model_arch = prof.hardware_model_arch
        session.execution_binding = prof.execution
        src: Path | None = None
        for f in sorted(self.profile_root.rglob("*.json")):
            try:
                if self.import_profile(f).profile_id_of() == prof.profile_id_of():
                    src = f
                    break
            except Exception:  # noqa: BLE001
                continue
        session.profile_source_path = str(src) if src is not None else ""
        if src is None:
            return
        try:
            session.profile_bytes_hash = sha256_file(src)
        except Exception:  # noqa: BLE001
            session.profile_bytes_hash = ""

    # --------------------------------------------------------- recommendation
    def recommend(
        self,
        profile: AtlasProfile | str,
        target: RecTarget | None = None,
        *,
        memory_target_gib: float | None = None,
        allow_pruning: bool = False,
        intent: CompressionIntent = CompressionIntent.QUANTIZE_ONLY,
    ) -> Recommendation:
        prof = self._resolve_profile(profile)
        return self.policy.recommend(
            prof,
            target or RecTarget(),
            memory_target_gib=memory_target_gib,
            allow_pruning=allow_pruning,
            intent=intent,
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
        intent: CompressionIntent = CompressionIntent.QUANTIZE_ONLY,
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
        intent = CompressionIntent(intent)
        raw_allow_pruning = constraints.get("allow_pruning", allow_pruning)
        if not isinstance(raw_allow_pruning, bool):
            raise AuthError(
                400,
                "constraint_invalid",
                "allow_pruning must be a JSON boolean",
            )
        allow_pruning = raw_allow_pruning
        rec = self.policy.recommend(
            prof,
            effective_target,
            memory_target_gib=memory_target_gib,
            allow_pruning=allow_pruning,
            intent=intent,
        )
        # Bind the session to the policy's canonical target, including an
        # explicit memory_target_gib override.
        effective_target = rec.target
        authorized = sorted(m.method for m in rec.methods)
        token = secrets.token_urlsafe(24)
        constraints_snapshot: dict[str, object] = {
            "allow_pruning": allow_pruning,
            "intent": intent.value,
        }
        session = _AuthorizationSession(
            token=token,
            recommendation_id=rec.recommendation_id,
            profile_id=rec.profile_id,
            target=effective_target,
            intent=intent,
            no_pruning=rec.no_pruning,
            constraints_snapshot=constraints_snapshot,
            authorized_methods=authorized,
            ttl_seconds=self.token_ttl_seconds,
        )
        self._bind_profile(session, prof)
        self.sessions[token] = session
        payload = rec.to_dict() if hasattr(rec, "to_dict") else rec
        return {
            "token": token,
            "recommendation_id": rec.recommendation_id,
            "profile_id": rec.profile_id,
            "no_pruning": rec.no_pruning,
            "intent": intent.value,
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
        with self._dispatch_lock:
            self._recheck_session(session)
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
        recipe = self._selected_recipe(sel, session=session)
        pv = self._recipe_preview_for(recipe, sel)
        plan = pv.get("plan") or {}
        intent_satisfied, intent_blockers, actual_families = self._intent_effect_gate(
            session.intent, recipe
        )
        readiness = {
            "verified_plan": bool(plan.get("plan_id") and plan.get("pins_pass")),
            "pins_pass": bool(plan.get("pins_pass")),
            "intent_satisfied": intent_satisfied,
            "executable": self._preview_is_executable(pv) and intent_satisfied,
        }
        artifact: CompiledPlanArtifact | None = None
        if readiness["executable"]:
            compiled = self.plane.compile_recipe(recipe)
            artifact = CompiledPlanArtifact.from_compiled(
                compiled, inputs=inputs or {}, registry=self.registry
            )
            artifact.verify()
            artifact.verify_pins_against(self.registry)
            if artifact.recipe_sha256 != pv.get("recipe_sha256"):
                raise AuthError(
                    500,
                    "recipe_identity_mismatch",
                    "compiled artifact recipe identity differs from preview",
                )
        target_snapshot: dict[str, object] = {
            "model_arch": session.target.hardware_model_arch,
            "compute": session.target.compute_arch,
            "topology": session.target.topology,
            "runtime": session.target.runtime_backend,
            "memory_gib": session.target.memory_target_gib,
        }
        constraints_snapshot = dict(session.constraints_snapshot or {})
        authorization_snapshot = self._authorization_snapshot(session)
        recipe_sha256 = str(pv.get("recipe_sha256") or "")
        plan_id = artifact.plan_id if artifact is not None else str(plan.get("plan_id") or "")
        authorization_digest = self._preview_authorization_digest(
            token=token,
            selected=sel,
            selection_hash=subset_hash,
            inputs=inputs or {},
            target_snapshot=target_snapshot,
            constraints_snapshot=constraints_snapshot,
            authorization_snapshot=authorization_snapshot,
            recipe_sha256=recipe_sha256,
            plan_id=plan_id,
            artifact=artifact,
        )
        preview_id = "pv-" + authorization_digest[:24]
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
            authorization_digest=authorization_digest,
        )
        package.recipe_sha256 = recipe_sha256
        package.target_snapshot = target_snapshot
        package.constraints_snapshot = constraints_snapshot
        package.authorization_snapshot = authorization_snapshot
        # IMMUTABLE preview: once created, a preview_id may NEVER be repointed to
        # a different recipe/identity. Overwriting is refused outright.
        if preview_id in self.pending_previews:
            raise AuthError(
                409,
                "preview_conflict",
                "preview already bound to this token+selection; overwrite refused",
            )
        # Persistence is the publication boundary. A failed write leaves neither
        # a visible in-memory preview nor a partial final directory.
        self._persist_preview(package)
        self.pending_previews[preview_id] = package
        return {
            "preview_id": preview_id,
            "plan_id": (artifact.plan_id if artifact is not None else plan.get("plan_id")),
            "hash": authorization_digest,
            "readiness": readiness,
            "intent": session.intent.value,
            "actual_families": [family.value for family in actual_families],
            "intent_blockers": intent_blockers,
            "selected_methods": list(sel),
            "run_id": package.run_id,
            "recipe_id": pv.get("recipe_id"),
            "recipe_sha256": pv.get("recipe_sha256"),
        }

    @staticmethod
    def _authorization_snapshot(session: _AuthorizationSession) -> dict[str, object]:
        binding = session.execution_binding
        return {
            "recommendation_id": session.recommendation_id,
            "policy_version": RECOMMENDATION_POLICY_VERSION,
            "method_catalog_version": METHOD_CATALOG_VERSION,
            "method_catalog_sha256": method_catalog_digest(),
            "profile_id": session.profile_id,
            "profile_fingerprint": session.profile_fingerprint,
            "profile_bytes_sha256": session.profile_bytes_hash,
            "profile_model_arch": session.profile_model_arch,
            "intent": session.intent.value,
            "no_pruning": session.no_pruning,
            "authorized_methods": list(session.authorized_methods),
            "execution_binding_sha256": (
                sha256_hex(canonical_json(binding.to_dict())) if binding is not None else ""
            ),
        }

    @staticmethod
    def _preview_authorization_digest(
        *,
        token: str,
        selected: list[str],
        selection_hash: str,
        inputs: dict[str, object],
        target_snapshot: dict[str, object],
        constraints_snapshot: dict[str, object],
        authorization_snapshot: dict[str, object],
        recipe_sha256: str,
        plan_id: str,
        artifact: CompiledPlanArtifact | None,
    ) -> str:
        artifact_identity = (
            sha256_hex(canonical_json(artifact.to_plain_dict())) if artifact is not None else ""
        )
        return sha256_hex(
            canonical_json(
                {
                    "token": token,
                    "selected": selected,
                    "selection_hash": selection_hash,
                    "inputs": inputs,
                    "target": target_snapshot,
                    "constraints": constraints_snapshot,
                    "authorization": authorization_snapshot,
                    "recipe_sha256": recipe_sha256,
                    "plan_id": plan_id,
                    "artifact_identity": artifact_identity,
                }
            )
        )

    @staticmethod
    def _preview_is_executable(preview: dict[str, Any]) -> bool:
        plan = preview.get("plan") or {}
        return bool(plan.get("plan_id") and plan.get("pins_pass"))

    @staticmethod
    def _intent_effect_gate(
        intent: CompressionIntent,
        recipe: CompressionRecipe,
    ) -> tuple[bool, list[dict[str, str]], tuple[MethodFamily, ...]]:
        """Authorize intent from actual recipe effects, never method labels."""
        effect_families: set[MethodFamily] = set()
        for stage in recipe.stages:
            if stage.effect_class == StageEffectClass.QUANTIZATION:
                effect_families.add(MethodFamily.QUANTIZATION)
            elif stage.effect_class == StageEffectClass.PRUNING:
                effect_families.add(MethodFamily.PRUNING)
        actual = tuple(sorted(effect_families, key=lambda family: family.value))
        required = set(required_families(intent))
        missing = sorted(required - effect_families, key=lambda family: family.value)
        forbidden: list[MethodFamily] = []
        if intent == CompressionIntent.QUANTIZE_ONLY and MethodFamily.PRUNING in effect_families:
            forbidden.append(MethodFamily.PRUNING)
        if intent == CompressionIntent.PRUNE_ONLY and MethodFamily.QUANTIZATION in effect_families:
            forbidden.append(MethodFamily.QUANTIZATION)
        blockers = [
            {
                "code": "intent_missing_effect",
                "message": f"{intent.value} requires compiled {family.value} effect",
            }
            for family in missing
        ]
        blockers.extend(
            {
                "code": "intent_forbidden_effect",
                "message": f"{intent.value} forbids compiled {family.value} effect",
            }
            for family in forbidden
        )
        if intent == CompressionIntent.CUSTOM:
            blockers.append(
                {
                    "code": "custom_effect_contract_required",
                    "message": "custom execution requires an explicit declared effect set",
                }
            )
        return not blockers, blockers, actual

    def _persist_preview(self, package: _PendingPreview) -> None:
        """Atomically persist the verified preview package (preview.json +
        plan.json) to the store root under an atomic temp-then-replace, so the
        preview is durable BEFORE it is ever served. Persistence is NOT
        best-effort here — start depends on the same on-disk identity."""

        preview_root = self.store_root / "previews"
        preview_root.mkdir(parents=True, exist_ok=True)
        artifact_dir = preview_root / package.preview_id
        if artifact_dir.exists():
            raise AuthError(409, "preview_conflict", "durable preview already exists")
        staging = preview_root / f".{package.preview_id}.staging-{secrets.token_hex(8)}"
        staging.mkdir(parents=False, exist_ok=False)
        preview_data: dict[str, object] = {
            "preview_id": package.preview_id,
            "token": package.token,
            "selection_hash": package.selection_hash,
            "authorization_digest": package.authorization_digest,
            "selected": list(package.selected),
            "plan_id": package.plan_id,
            "run_id": package.run_id,
            "recipe_sha256": package.recipe_sha256,
            "target_snapshot": dict(package.target_snapshot),
            "constraints_snapshot": dict(package.constraints_snapshot),
            "authorization_snapshot": dict(package.authorization_snapshot),
        }
        try:
            atomic_write_json(staging / "preview.json", preview_data)
            if package.artifact is not None:
                atomic_write_json(staging / "plan.json", package.artifact.to_plain_dict())
            staging.rename(artifact_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

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
        session: _AuthorizationSession | None = None,
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

        When ``session`` is supplied, the recipe's constraints/target are bound
        to THAT session's canonical target + constraints (never a hardcoded
        ceiling), so the preview reflects exactly what was authorized.
        """
        selected_specs = []
        try:
            for method in selected or ():
                selected_specs.append(method_spec(method))
        except KeyError as exc:
            raise AuthError(
                400,
                "method_unknown",
                f"unknown or unclassified compression method: {exc.args[0]}",
            ) from exc
        binding = session.execution_binding if session is not None else None
        if (
            session is not None
            and session.profile_model_arch
            and session.profile_model_arch != session.target.hardware_model_arch
        ):
            raise AuthError(
                409,
                "profile_target_mismatch",
                "profile model architecture differs from authorization target",
            )
        templates = {spec.recipe_template for spec in selected_specs}
        if "llamacpp_gguf_mixed" in templates:
            if len(templates) != 1 or len(selected_specs) != 1:
                raise AuthError(
                    409,
                    "recipe_composition_conflict",
                    "llama.cpp mixed GGUF must be selected as the sole derivative recipe",
                )
            if session is None or binding is None:
                raise AuthError(
                    409,
                    "profile_execution_identity_missing",
                    "GGUF derivative requires source, calibration, and tokenizer identity",
                )
            if session.intent != CompressionIntent.QUANTIZE_ONLY or not session.no_pruning:
                raise AuthError(
                    409,
                    "recipe_composition_conflict",
                    "llama.cpp mixed GGUF is authorized only for quantize-only/no-pruning",
                )
            source = SourceIdentity(
                source_id=binding.source_id,
                checkpoint_path=binding.checkpoint_path,
                checkpoint_revision=binding.checkpoint_revision,
                sha256=dict(binding.source_sha256),
                manifest_digest=binding.source_manifest_digest,
            )
            recipe = llamacpp_gguf_mixed_recipe(source)
            recipe.calibration = CalibrationIdentity(
                calibration_id=binding.calibration_id,
                corpus_name=binding.corpus_name,
                seed=binding.calibration_seed,
                partition=binding.calibration_partition,
                corpus_records_path=binding.corpus_records_path,
                tokenizer_sha256=binding.tokenizer_hash,
            )
            recipe.hardware = recipe.hardware.model_copy(
                update={
                    "model_arch": session.target.hardware_model_arch,
                    "compute_arch": session.target.compute_arch,
                    "topology": session.target.topology,
                    "runtime_backend": "none",
                }
            )
            recipe.constraints = recipe.constraints.model_copy(
                update={"max_resident_gib": session.target.memory_target_gib}
            )
            return recipe

        base = glm52_no_pruning_recipe()
        if binding is not None:
            base.source = SourceIdentity(
                source_id=binding.source_id,
                checkpoint_path=binding.checkpoint_path,
                checkpoint_revision=binding.checkpoint_revision,
                sha256=dict(binding.source_sha256),
                manifest_digest=binding.source_manifest_digest,
            )
            base.calibration = CalibrationIdentity(
                calibration_id=binding.calibration_id,
                corpus_name=binding.corpus_name,
                seed=binding.calibration_seed,
                partition=binding.calibration_partition,
                corpus_records_path=binding.corpus_records_path,
                tokenizer_sha256=binding.tokenizer_hash,
            )
        if session is not None:
            base.hardware = base.hardware.model_copy(
                update={
                    "model_arch": session.target.hardware_model_arch,
                    "compute_arch": session.target.compute_arch,
                    "topology": session.target.topology,
                    "runtime_backend": session.target.runtime_backend,
                }
            )
        stages_by_id = {s.id: s for s in base.stages}
        if selected is None or none_is_all:
            stages = list(base.stages)
        else:
            wanted: set[str] = set()
            for method in selected:
                try:
                    wanted.update(method_spec(method).recipe_stage_ids)
                except KeyError as exc:
                    raise AuthError(
                        400,
                        "method_unknown",
                        f"unknown or unclassified compression method: {method}",
                    ) from exc
            wanted = self._format_closure(stages_by_id, wanted)
            stages = [stages_by_id[sid] for sid in _STAGE_ORDER if sid in wanted]
        derivative_effects = {
            StageEffectClass.CONDITIONING,
            StageEffectClass.QUANTIZATION,
            StageEffectClass.REFINEMENT,
            StageEffectClass.RESIDUAL,
            StageEffectClass.REPAIR,
            StageEffectClass.PRUNING,
        }
        if session is not None and binding is None and any(
            stage.effect_class in derivative_effects for stage in stages
        ):
            raise AuthError(
                409,
                "profile_execution_identity_missing",
                "derivative recipe requires source, calibration, and tokenizer identity",
            )
        recipe = base.model_copy(deep=True)
        recipe.stages = stages
        memory_gib = session.target.memory_target_gib if session is not None else 115.0
        recipe.constraints = RecipeConstraints(
            no_pruning=session.no_pruning if session is not None else True,
            allow_pruning_capability=(
                not session.no_pruning if session is not None else False
            ),
            preserve_non_expert_backbone=True,
            immutable_source=True,
            allow_hybrid_precision=(
                session is not None and session.intent == CompressionIntent.HYBRID
            ),
            max_resident_gib=memory_gib,
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

    def recipe_preview(self, selected: list[str] | None = None) -> dict[str, Any]:
        """Deterministic preview-from-selection: build the draft, run the
        non-fatal compile/dry-run, and report diff / compile blockers /
        readiness / verified plan (if it compiles). Never mutates."""
        recipe = self._selected_recipe(selected)
        return self._recipe_preview_for(recipe, selected)

    def _recipe_preview_for(
        self,
        recipe: CompressionRecipe,
        selected: list[str] | None,
    ) -> dict[str, Any]:
        """Validate and summarize the exact recipe object used for artifacts."""
        issues, rid, sha = self.plane.compiler.validate(recipe)
        errors = [i for i in issues if i.severity == "error"]
        compiles = not errors
        plan: dict[str, Any] | None = None
        if compiles:
            plan = self._verified_plan_dict(recipe)
        return {
            "selected_methods": list(
                selected or [spec.method for spec in METHOD_CATALOG]
            ),
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
        *,
        plan_id: str = "",
        recipe_sha256: str = "",
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
        with self._dispatch_lock:
            self._recheck_session(session)
        if not preview_id:
            raise AuthError(400, "preview_required", "preview_id required")
        package = self.pending_previews.get(preview_id)
        if package is None:
            raise AuthError(410, "preview_unknown", "preview not found (stale or never created)")
        if package.token != token:
            raise AuthError(403, "preview_token_mismatch", "preview does not belong to this token")
        if not selection_hash:
            raise AuthError(400, "hash_required", "selection hash required")
        if selection_hash != package.authorization_digest:
            raise AuthError(409, "preview_mismatch", "authorization digest does not match preview")
        if selected is None or len(selected) == 0:
            raise AuthError(400, "selection_empty", "start selection must be non-empty")
        if package.artifact is None:
            raise AuthError(
                409,
                "preview_not_executable",
                "preview has no verified executable plan; start refused",
            )
        if _selection_hash(selected) != package.selection_hash:
            raise AuthError(409, "selection_mismatch", "start selection does not match the preview")
        if not _same_selection(selected, package.selected):
            raise AuthError(
                409, "selection_mismatch", "start selection differs from the preview selection"
            )
        # inputs must EXACTLY match the inputs the preview was compiled with
        # (the run identity is derived from the canonical preview inputs, so a
        # different input set is a mismatch — never a silent re-identity).
        if (inputs or {}) != package.inputs:
            raise AuthError(409, "inputs_mismatch", "start inputs differ from the preview inputs")
        # Both immutable handles are mandatory, never optional compatibility
        # fields. Omitting either cannot authorize execution.
        if not plan_id:
            raise AuthError(400, "plan_required", "plan_id required")
        if not recipe_sha256:
            raise AuthError(400, "recipe_required", "recipe_sha256 required")
        if plan_id != package.plan_id:
            raise AuthError(409, "plan_mismatch", "plan_id does not match the preview's plan")
        if recipe_sha256 != package.recipe_sha256:
            raise AuthError(409, "recipe_mismatch", "recipe digest does not match the preview")

        run_id = package.run_id
        with self._dispatch_lock:
            # Starts are rejected once shutdown() has begun: the supervised
            # executor is drained+dropped, so a late reservation could never be
            # driven. Fail closed rather than leak an undriven reserved run.
            if self._shutdown_event.is_set():
                raise AuthError(
                    503, "service_shutting_down", "service is shutting down; start refused"
                )
            self._recheck_session(session)
            current_snapshot = self._authorization_snapshot(session)
            if package.authorization_snapshot != current_snapshot:
                raise AuthError(
                    409,
                    "authorization_snapshot_mismatch",
                    "preview authorization lineage differs from the live session",
                )
            intent_satisfied, intent_blockers, _ = self._intent_effect_gate(
                session.intent, package.recipe
            )
            if not intent_satisfied:
                details = "; ".join(
                    blocker["message"] for blocker in intent_blockers
                )
                raise AuthError(
                    409,
                    "intent_effect_mismatch",
                    f"compiled recipe effects do not satisfy intent: {details}",
                )
            expected_authorization_digest = self._preview_authorization_digest(
                token=token,
                selected=package.selected,
                selection_hash=package.selection_hash,
                inputs=package.inputs,
                target_snapshot=package.target_snapshot,
                constraints_snapshot=package.constraints_snapshot,
                authorization_snapshot=current_snapshot,
                recipe_sha256=package.recipe_sha256,
                plan_id=package.plan_id,
                artifact=package.artifact,
            )
            if expected_authorization_digest != package.authorization_digest:
                raise AuthError(
                    409,
                    "authorization_digest_mismatch",
                    "preview authorization digest failed canonical revalidation",
                )
            # Under the dispatch lock, RE-VERIFY the stored artifact + its live
            # pins and compare fresh identities before dispatch (fail closed on
            # any drift — a recipe whose pins no longer hold can never start).
            self._reverify_under_lock(package)
            # GLOBAL run_id reservation/idempotency across tokens: if this exact
            # deterministic run was ALREADY reserved by ANY earlier token/preview,
            # we never dispatch it a second time — replay, not a new run.
            if run_id in self.dispatched:
                raise AuthError(409, "replay", f"run {run_id} already reserved by another start")
            if package.dispatch_started:
                raise AuthError(
                    409, "replay", f"preview {preview_id} already started (run {run_id})"
                )
            # Persist job identity (durable PENDING job.json) + immutable plan
            # atomically under the lock BEFORE returning run_id, so the run is
            # immediately observable and a crash can never lose a reserved run.
            created_run = self._persist_run(package, run_id)
            self.dispatched[run_id] = {"preview_id": preview_id, "token": token}
            try:
                self._save_dispatch_registry()
            except Exception:
                self.dispatched.pop(run_id, None)
                if created_run:
                    self._remove_created_run(package)
                raise
            package.dispatch_started = True
            if self.supervised_executor:
                self._pending_runs.append(package)
                self._ensure_worker_locked()
                self._idle.notify()
            else:
                # non-supervised fallback: dedicated non-daemon worker (joined
                # on shutdown so a normal lifecycle never drops a run).
                worker = threading.Thread(
                    target=self._exec_one,
                    args=(package,),
                    name=f"rec-run-{run_id}",
                    daemon=False,
                )
                self._started_workers.append(worker)
                worker.start()
        return {"run_id": run_id, "status": "started"}

    def _persist_run(self, package: _PendingPreview, run_id: str) -> bool:
        """Service-side authority for job+plan persistence: ATOMIC job.json +
        plan.json writes BEFORE dispatch. No suppression here — a persistence
        failure aborts the start (never starts a job that is not on disk). The
        plan is written first, then the pending job; if either fails the partial
        run dir is removed so a caller never observes a half-persisted run."""
        compiled = self.plane.compile_recipe(package.recipe)
        expected_run_dir = self.plane.work_root / "runs" / run_id
        created_run = not expected_run_dir.exists()
        engine = JobEngine(compiled, self.registry, self.plane.work_root)
        engine._bind_run(package.inputs)
        run_dir = Path(engine.run_dir)
        try:
            # durable plan artifact FIRST (content-addressed recipe + pins +
            # inputs); engine_for/status read both job.json and plan.json. The
            # artifact is non-None here (re-verified before dispatch).
            if package.artifact is None:
                raise RuntimeError("preview has no verified executable plan")
            if not created_run:
                existing = self.plane.engine_for(run_id)
                existing_artifact = CompiledPlanArtifact.model_validate_json(
                    existing.plan_path.read_text(encoding="utf-8")
                )
                existing_identity = (
                    existing_artifact.run_id,
                    existing_artifact.plan_id,
                    existing_artifact.recipe_sha256,
                    canonical_json(dict(existing_artifact.inputs)),
                )
                package_identity = (
                    package.artifact.run_id,
                    package.artifact.plan_id,
                    package.artifact.recipe_sha256,
                    canonical_json(dict(package.artifact.inputs)),
                )
                if existing_identity != package_identity:
                    raise AuthError(409, "run_conflict", "existing run plan differs")
                if existing._load_job() is None:
                    raise AuthError(409, "run_conflict", "existing run has no durable job")
                return False
            atomic_write_json(engine.plan_path, package.artifact.to_plain_dict())
            if not engine.job_path.exists():
                job = engine._init_run_dir(package.inputs)  # run dir + journal
                engine._save(job)  # durable PENDING job.json (atomic)
            else:
                existing_job = engine._load_job()
                if existing_job is None:
                    existing_job = engine._init_run_dir(package.inputs)
                    engine._save(existing_job)
        except Exception:
            # abort cleanly: no reserved run, no half-persisted run dir
            if created_run and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return created_run

    def _remove_created_run(self, package: _PendingPreview) -> None:
        """Rollback only a run directory created by this failed reservation."""
        compiled = self.plane.compile_recipe(package.recipe)
        engine = JobEngine(compiled, self.registry, self.plane.work_root)
        engine._bind_run(package.inputs)
        shutil.rmtree(engine.run_dir, ignore_errors=True)

    def _reverify_under_lock(self, package: _PendingPreview) -> None:
        """Under the dispatch lock, reverify the stored verified artifact and its
        live pins against the current registry, and compare fresh recipe/plan
        identities. Any drift fails closed (start refused)."""
        if package.artifact is None:
            raise AuthError(
                409, "preview_not_executable", "preview has no verified executable plan"
            )
        try:
            package.artifact.verify()
            package.artifact.verify_pins_against(self.registry)
        except Exception as exc:  # noqa: BLE001
            raise AuthError(
                409, "pins_drift", f"artifact/live pins drifted since preview: {exc}"
            ) from exc
        compiled = self.plane.compile_recipe(package.recipe)
        if compiled.plan_id != package.plan_id:
            raise AuthError(409, "plan_drift", "recipe no longer compiles to the preview plan_id")

    def _executor_loop(self) -> None:
        """Supervised non-daemon executor: drains the dispatch queue. Runs
        outside the request thread; waits on the idle condition and the
        shutdown event. A queued run is executed crash-safely by the durable
        engine; any driver exception terminalizes the persisted job."""
        while True:
            package: _PendingPreview | None = None
            with self._idle:
                if self._shutdown_event.is_set() and not self._pending_runs:
                    return
                if not self._pending_runs:
                    if not threading.main_thread().is_alive():
                        # main thread finished and nothing queued: allow the
                        # process to exit cleanly (the service was never given a
                        # shutdown() call, e.g. an embedding test harness).
                        return
                    self._idle.wait(1.0)
                    continue
                package = self._pending_runs.pop(0)
            if package is not None:
                self._exec_one(package)

    def _ensure_worker_locked(self) -> None:
        """Start the lazy supervised worker while the dispatch lock is held."""
        if self._shutdown_event.is_set():
            raise AuthError(503, "service_shutting_down", "service is shutting down")
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._executor_loop,
            name="atlas-recommend-executor",
            daemon=True,
        )
        self._worker.start()

    def _exec_one(self, package: _PendingPreview) -> None:
        try:
            compiled = self.plane.compile_recipe(package.recipe)
            engine = JobEngine(compiled, self.registry, self.plane.work_root)
            engine._bind_run(package.inputs)
            engine.run(package.inputs)
        except Exception:  # noqa: BLE001
            # Driver failure ALWAYS terminalizes a persisted nonterminal job —
            # never a silent drop of a reserved run.
            self._terminalize_run(package)

    def _terminalize_run(self, package: _PendingPreview) -> None:
        try:
            compiled = self.plane.compile_recipe(package.recipe)
            engine = JobEngine(compiled, self.registry, self.plane.work_root)
            engine._bind_run(package.inputs)
            job = engine._load_job()
            if job is None:
                job = engine._init_run_dir(package.inputs)
                engine._save(job)
            if job is None:
                raise RuntimeError("no durable job to terminalize")
            if job.is_terminal():
                return
            job.status = JobStatus.FAILED_TERMINAL
            job.error = "executor driver failed before run completion"
            engine.journal.append(
                {
                    "event": "run.terminal",
                    "status": "failed_terminal",
                    "reason": "driver_failure",
                    "detail": job.error,
                }
            )
            engine._save(job)
        except Exception:  # noqa: BLE001
            pass

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """Graceful shutdown: stop accepting new dispatch and JOIN the
        supervised non-daemon executor so in-flight/pending reserved runs reach
        a terminal state (no silent drop of a reserved run)."""
        self._shutdown_event.set()
        with self._idle:
            self._idle.notify_all()
        worker = self._worker
        thread: threading.Thread | None = None
        if worker is not None and worker.is_alive():
            thread = worker
            if worker is not threading.current_thread():
                worker.join(timeout=(timeout or 30.0) if wait else 0.0)
        if wait:
            # join any dedicated non-daemon workers still running (non-supervised)
            for w in list(self._started_workers):
                if w.is_alive() and w is not thread:
                    w.join(timeout=(timeout or 30.0))

    # startup: reconcile any pending/dispatched runs left from a previous
    # process (crash restart). Reserved runs that are still nonterminal and
    # unpicked are re-queued; terminal ones are ignored.
    def _load_persisted_previews(self) -> None:
        """Reload server-side verified preview packages that were persisted to
        the store root by a previous process. This lets a restarted service
        resume START that references a preview_id that outlived the process
        (in-memory pending_previews is not the source of continuity). Fails
        closed on any malformed/verification-failing entry (skipped)."""
        from model_atlas.recipes import CompiledPlanArtifact

        preview_root = self.store_root / "previews"
        if not preview_root.exists():
            return
        for pdir in sorted(preview_root.iterdir()):
            pv_path = pdir / "preview.json"
            plan_path = pdir / "plan.json"
            if not (pv_path.exists() and plan_path.exists()):
                continue
            try:
                meta = json.loads(pv_path.read_text(encoding="utf-8"))
                preview_id = str(meta.get("preview_id", ""))
                if not preview_id:
                    continue
                selection_hash = str(meta.get("selection_hash", ""))
                authorization_digest = str(meta.get("authorization_digest", ""))
                if not authorization_digest:
                    continue
                selected = list(meta.get("selected") or [])
                token = str(meta.get("token", ""))
                artifact = CompiledPlanArtifact.model_validate_json(
                    plan_path.read_text(encoding="utf-8")
                )
                artifact.verify()
                artifact.verify_pins_against(self.registry)
            except Exception:  # noqa: BLE001 — fail closed (skip bad entries)
                continue
            run_id = str(meta.get("run_id", "")) or artifact.run_id
            pkg = _PendingPreview(
                token=token,
                preview_id=preview_id,
                selection_hash=selection_hash,
                selected=selected,
                recipe=artifact.recipe,
                artifact=artifact if artifact is not None else None,
                inputs=dict(artifact.inputs),
                run_id=run_id,
                authorization_digest=authorization_digest,
            )
            pkg.plan_id = str(meta.get("plan_id", "")) or artifact.plan_id
            pkg.recipe_sha256 = str(meta.get("recipe_sha256", ""))
            pkg.target_snapshot = dict(meta.get("target_snapshot") or {})
            pkg.constraints_snapshot = dict(meta.get("constraints_snapshot") or {})
            pkg.authorization_snapshot = dict(meta.get("authorization_snapshot") or {})
            if preview_id not in self.pending_previews:
                self.pending_previews[preview_id] = pkg

    def _reconcile_startup(self) -> None:
        for run_id in list(self.dispatched):
            engine = self._engine_for_run(run_id)
            if engine is None:
                continue
            job = engine._load_job()
            if job is None or job.is_terminal():
                continue
            # re-enqueue a reserved-but-not-driven run
            pkg = self._package_for_run(run_id)
            if pkg is not None and pkg not in self._pending_runs:
                self._pending_runs.append(pkg)

    def _load_dispatch_registry(self) -> None:
        """Load the durable GLOBAL run reservation registry so a restarted
        service never double-dispatches an already-reserved run."""
        reg = self.store_root / "dispatch-registry.json"
        data: list[dict[str, object]] = []
        try:
            if reg.exists():
                loaded = json.loads(reg.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    data = loaded
        except Exception:  # noqa: BLE001
            data = []
        for entry in data:
            if isinstance(entry, dict):
                rid = entry.get("run_id")
                if isinstance(rid, str):
                    self.dispatched[rid] = {
                        "preview_id": entry.get("preview_id", ""),
                        "token": entry.get("token", ""),
                    }

    def _reconstruct_dispatch_registry(self) -> None:
        """Recover reservations from verified durable preview+run artifacts.

        This closes the crash window between durable run creation and registry
        promotion: a restart treats every valid persisted run as reserved.
        """
        changed = False
        for package in self.pending_previews.values():
            run_id = package.run_id
            if not run_id or run_id in self.dispatched:
                continue
            try:
                engine = self.plane.engine_for(run_id)
                if not (engine.plan_path.exists() and engine.job_path.exists()):
                    continue
                artifact = CompiledPlanArtifact.model_validate_json(
                    engine.plan_path.read_text(encoding="utf-8")
                )
                artifact.verify()
                if artifact.run_id != run_id or artifact.plan_id != package.plan_id:
                    continue
            except Exception:  # noqa: BLE001
                continue
            self.dispatched[run_id] = {
                "preview_id": package.preview_id,
                "token": package.token,
            }
            changed = True
        if changed:
            self._save_dispatch_registry()

    def _save_dispatch_registry(self) -> None:
        entries = [
            {
                "run_id": rid,
                "preview_id": str(info.get("preview_id", "")),
                "token": str(info.get("token", "")),
            }
            for rid, info in self.dispatched.items()
        ]
        atomic_write_json(self.store_root / "dispatch-registry.json", entries)

    def _package_for_run(self, run_id: str) -> _PendingPreview | None:
        info = self.dispatched.get(run_id)
        if not info:
            return None
        pid = str(info.get("preview_id", ""))
        return self.pending_previews.get(pid)

    def _engine_for_run(self, run_id: str) -> JobEngine | None:
        try:
            expected = self.plane.work_root / "runs" / run_id
            if not (expected / "job.json").exists():
                return None
            return self.plane.engine_for(run_id)
        except Exception:  # noqa: BLE001
            return None

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
    def _recheck_session(self, session: _AuthorizationSession) -> None:
        """Re-validate a live token's authorization against the CANONICAL source
        of truth at every preview/start:

          * expired tokens and revoked tokens are rejected (token_unknown).
          * the on-disk (or in-memory) canonical profile is re-resolved and its
            identity + byte content re-compared: if the profile has DRIFTED since
            the token was minted, the token no longer authorizes anything
            (token_stale) — a stale profile authorization can never run.
          * the canonical policy is re-run against the resolved profile + the
            session's exact target/constraints; if the authorized method set (or
            its hash) no longer matches, the token is stale (token_stale).
        """
        if session.revoked:
            raise AuthError(401, "token_unknown", "authorization token revoked")
        if _is_expired(session):
            raise AuthError(401, "token_expired", "authorization token expired")
        # A session minted by the real authorize() path always carries a profile
        # binding. Directly-seeded legacy sessions (tests) have no binding and
        # opt out of the profile/policy recheck — they still honor expiry/revoke.
        if not session.profile_fingerprint:
            return
        # Resolve the canonical profile through the token's bound source path
        # (authoritative) so an external profile edit that changes the content
        # id is detected as DRIFT (token_stale), not an unresolvable id.
        try:
            if session.profile_source_path and Path(session.profile_source_path).exists():
                prof = self.import_profile(session.profile_source_path)
            else:
                prof = self._resolve_profile(session.profile_id)
        except (KeyError, OSError, ValueError):
            raise AuthError(
                401, "token_stale", "profile changed since authorization; re-authorize"
            ) from None
        if prof.profile_id_of() != session.profile_fingerprint:
            raise AuthError(401, "token_stale", "profile changed since authorization; re-authorize")
        if session.profile_source_path:
            from model_atlas.jobs.artifacts import sha256_file

            try:
                if sha256_file(Path(session.profile_source_path)) != session.profile_bytes_hash:
                    raise AuthError(
                        401,
                        "token_stale",
                        "profile bytes changed since authorization; re-authorize",
                    )
            except OSError:
                raise AuthError(
                    401, "token_stale", "profile file unavailable since authorization"
                ) from None
        # re-run the canonical policy against the resolved profile + session
        # target/constraints; a drift in the recommended set invalidates.
        rec = self.policy.recommend(
            prof,
            session.target,
            memory_target_gib=session.target.memory_target_gib,
            allow_pruning=bool((session.constraints_snapshot or {}).get("allow_pruning", False)),
            intent=session.intent,
        )
        current = sorted(m.method for m in rec.methods)
        if (
            rec.recommendation_id != session.recommendation_id
            or rec.no_pruning != session.no_pruning
            or rec.intent != session.intent
            or _selection_hash(current) != session.selected_hash()
        ):
            raise AuthError(401, "token_stale", "policy changed since authorization; re-authorize")

    def revoke_token(self, token: str) -> bool:
        """Revoke a token so every preview/start using it is rejected (401)."""
        session = self.sessions.get(token)
        if session is None:
            return False
        session.revoked = True
        return True

    def expire_tokens(self) -> int:
        """Expire every currently-live token (for tests / operator override)."""
        from datetime import UTC, datetime

        now = datetime.now(UTC).timestamp()
        n = 0
        for s in list(self.sessions.values()):
            if not s.revoked and not _is_expired(s):
                s.expires_at = _iso_from_epoch(now - 1)
                n += 1
        return n

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
    payload: dict[str, Any] = {
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
    if profile.execution is not None:
        payload["execution"] = profile.execution.to_dict()
    return payload


def _profile_from_dict(data: dict[str, Any]) -> AtlasProfile:
    return AtlasProfile.from_dict(data)
