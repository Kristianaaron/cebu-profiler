"""Versioned immutable compiled-plan artifact.

The compiled-plan artifact is the single canonical, machine-readable output of
``compile-recipe`` and the single VERIFIED input of ``job start --plan`` and
``reproduce.sh``. It is versioned and deeply immutable:

  * ``schema_version`` — artifact schema version (this file);
  * ``recipe`` — the canonical authored recipe (CompressionRecipe);
  * ``recipe_sha256`` / ``recipe_id`` / ``plan_id`` — content-addresses;
  * ``resolved_pins`` — stage -> snapshot of the ACTUAL selected backend:
    backend_id, exact resolved version, ADAPTER IDENTITY, lifecycle status and
    a capability hash over its declared capabilities/derivative/resource limits;
  * ``inputs`` — canonical job inputs (the run-id inputs);
  * ``run_id`` — deterministic from recipe + inputs;
  * ``reproduce_command`` — the exact CLI start command.

Writing is ``model-atlas compile-recipe --out plan.json``. Starting verifies the
artifact ids/hash AND compares the pinned adapter identity + capability hash
against the live registry before the engine may execute — never silently
discarding pin metadata and recompiling only the recipe.
"""

from __future__ import annotations

import hashlib
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.recipe.compiler import CompiledRecipe, canonical_json
from model_atlas.recipe.schema import CompressionRecipe

PLAN_ARTIFACT_SCHEMA = 3


class CompiledPlanArtifact(BaseModel):
    """One versioned, immutable compiled-plan artifact.

    ``resolved_pins`` and ``backend_status_snapshot`` are exposed as frozen
    (immutable) mappings — the ``model_validate`` path wraps passed dicts with
    MappingProxyType so no caller can mutate a loaded artifact's pins/status.
    """

    _ = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = PLAN_ARTIFACT_SCHEMA
    # PRIVATE canonical immutable recipe payload (JSON string) — NEVER a mutable
    # CompressionRecipe. `recipe` property reconstructs a fresh copy on every
    # access, so no caller can reach a mutable embedded recipe.
    recipe_payload_json: str = Field(default="", repr=False, exclude=True)
    recipe_sha256: str
    recipe_id: str
    plan_id: str
    resolved_pins: dict[str, dict[str, str]] = Field(default_factory=dict)
    backend_status_snapshot: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, object] = Field(default_factory=dict)
    run_id: str
    reproduce_command: str = ""
    canonical_identity: str = Field(default="", repr=False, exclude=True)

    __slice_hooks__ = object()

    def __init__(self, **data: object) -> None:
        """Freeze the mutable mappings + canonical identity; store the recipe
        ONLY as its serialized mutable payload. Public `recipe` access returns a
        defensively reconstructed copy — mutation of any returned copy can never
        mutate the artifact or change identity/verify()."""
        raw_recipe = data.get("recipe")
        if isinstance(raw_recipe, CompressionRecipe):
            data["recipe_payload_json"] = raw_recipe.model_dump_json()
        elif isinstance(raw_recipe, str):
            data["recipe_payload_json"] = raw_recipe
        elif isinstance(raw_recipe, dict):
            # rehydration from the persisted plan artifact (recipe as a model
            # dict) — serialize it back to the canonical JSON payload.
            data["recipe_payload_json"] = canonical_json(raw_recipe)
        data.pop("recipe", None)
        serializable: dict[str, object] = {}
        for k, v in data.items():
            if k in {"reproduce_command", "canonical_identity"}:
                continue
            if k == "recipe":
                serializable[k] = data["recipe_payload_json"]  # canonical JSON string
            elif isinstance(v, BaseModel):
                serializable[k] = v.model_dump(mode="json")
            elif isinstance(v, dict):
                serializable[k] = {
                    kk: (vv.model_dump(mode="json") if isinstance(vv, BaseModel) else vv)
                    for kk, vv in v.items()
                }
            else:
                serializable[k] = v
        data["canonical_identity"] = canonical_json(serializable)
        super().__init__(**data)
        object.__setattr__(
            self,
            "resolved_pins",
            MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.resolved_pins.items()}),
        )
        object.__setattr__(
            self, "backend_status_snapshot", MappingProxyType(dict(self.backend_status_snapshot))
        )
        object.__setattr__(self, "inputs", _deep_freeze(self.inputs))

    @property
    def recipe(self) -> CompressionRecipe:
        """DEFENSIVE reconstruction on EVERY access: a fresh CompressionRecipe
        is parsed from the private canonical payload. Mutating the returned copy
        never affects the artifact, its identity, or verify() (which re-parses
        the same payload)."""
        return CompressionRecipe.model_validate_json(self.recipe_payload_json)

    @property
    def recipe_payload(self) -> str:
        """Private canonical immutable recipe JSON (never exposed as mutable)."""
        return self.recipe_payload_json

    def verify(self) -> None:
        """Self-consistency: ids/hash must match the embedded recipe (parsed
        fresh from the private payload) and the deterministic run_id from
        recipe+inputs. Raises on any mismatch. Any previous mutation attempt on
        a returned `recipe` copy cannot affect this."""
        sha = self.recipe_sha256
        if not sha:
            raise ValueError("compiled-plan artifact has no recipe_sha256")
        from model_atlas.recipe.compiler import _compute_recipe_sha

        current = self.recipe  # freshly reconstructed
        if _compute_recipe_sha(current) != sha:
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
        unfrozen = _deep_unfreeze(self.inputs)
        expected_run = _run_id_from_payload(
            sha,
            self.plan_id,
            unfrozen if isinstance(unfrozen, dict) else {"inputs": unfrozen},
        )
        if self.run_id != expected_run:
            raise ValueError(
                f"compiled-plan artifact run_id {self.run_id} != recomputed "
                f"{expected_run} from inputs {unfrozen}"
            )

    def to_plain_dict(self) -> dict[str, object]:
        """Plain JSON-serializable dict (recursively converts frozen
        MappingProxy/list wrappers back to plain dict/list) for persistence
        without exposing mutable views."""
        return {
            "schema_version": self.schema_version,
            "recipe": self.recipe.model_dump(mode="json"),
            "recipe_sha256": self.recipe_sha256,
            "recipe_id": self.recipe_id,
            "plan_id": self.plan_id,
            "resolved_pins": {k: dict(v) for k, v in self.resolved_pins.items()},
            "backend_status_snapshot": dict(self.backend_status_snapshot),
            "inputs": _deep_unfreeze(self.inputs),
            "run_id": self.run_id,
            "reproduce_command": self.reproduce_command,
            "canonical_identity": self.canonical_identity,
        }

    def frozen_pins(self) -> MappingProxyType[str, MappingProxyType[str, str]]:
        """Read-only view of resolved_pins (nested frozen)."""
        return self.resolved_pins  # type: ignore[return-value]

    @property
    def canonical_payload(self) -> str:
        """Canonical immutable payload captured at construction — mutating any
        nested member of this artifact does NOT change identity."""
        return self.canonical_identity

    @classmethod
    def from_compiled(
        cls,
        compiled: CompiledRecipe,
        inputs: dict[str, object] | None = None,
        registry: object | None = None,
    ) -> CompiledPlanArtifact:
        """Build a versioned immutable artifact from a compiled plan + inputs.
        When a registry is provided the per-stage pins snapshot the actual
        backend's adapter identity + capability hash for later live verification."""
        inputs = inputs or {}
        run_id = compiled.run_id(inputs)
        pins = _snapshot_pins(compiled, registry=registry)
        return cls(
            recipe=compiled.recipe,
            recipe_sha256=compiled.recipe_sha256,
            recipe_id=compiled.recipe_id,
            plan_id=compiled.plan_id,
            resolved_pins=pins,
            backend_status_snapshot=dict(compiled.backend_status_snapshot),
            inputs=inputs,
            run_id=run_id,
            reproduce_command=(
                f"model-atlas job start --plan <this-file> --out <work-root> # run_id {run_id}"
            ),
        )

    def verify_pins_against(self, registry: object) -> None:
        """Compare every pinned stage's backend id, exact version, adapter
        identity, lifecycle status and capability hash against the LIVE
        registry. Every field is MANDATORY — a missing field fails verification
        (an incomplete pin record can never be executed)."""
        from model_atlas.backend.registry import BackendRegistry

        if not isinstance(registry, BackendRegistry):
            raise ValueError("verify_pins_against requires a BackendRegistry")
        required_fields = (
            "backend_id",
            "pinned_version",
            "resolved_version",
            "status",
            "adapter_identity",
            "capability_hash",
            "execution_identity_sha256",
        )
        # EXACT stage-id pin set: every recipe stage must have exactly one pin
        # record (no missing, no extra).
        expected_stages = {s.id for s in self.recipe.stages}
        pinned_stages = set(self.resolved_pins)
        if pinned_stages != expected_stages:
            raise ValueError(
                f"artifact pins must cover EXACTLY the recipe stages: expected "
                f"{sorted(expected_stages)} but pins cover {sorted(pinned_stages)}"
            )
        for stage_id, pin in self.resolved_pins.items():
            missing = [f for f in required_fields if not pin.get(f)]
            if missing:
                raise ValueError(f"artifact pin {stage_id}: missing mandatory field(s) {missing}")
            backend_id = pin["backend_id"]
            rec = registry.get(backend_id)
            if rec is None:
                raise ValueError(
                    f"artifact pin {stage_id}: backend {backend_id!r} no longer registered"
                )
            if pin["resolved_version"] != rec.version:
                raise ValueError(
                    f"artifact pin {stage_id}: resolved version {pin['resolved_version']} "
                    f"!= live {rec.version}"
                )
            if pin["adapter_identity"] != _adapter_identity(rec):
                raise ValueError(
                    f"artifact pin {stage_id}: adapter identity {pin['adapter_identity']} "
                    f"!= live {_adapter_identity(rec)}"
                )
            if pin["status"] != rec.status.value:
                raise ValueError(
                    f"artifact pin {stage_id}: status {pin['status']} != live {rec.status.value}"
                )
            if pin["capability_hash"] != _capability_hash(rec):
                raise ValueError(
                    f"artifact pin {stage_id}: capability hash {pin['capability_hash']} "
                    f"!= live {_capability_hash(rec)}"
                )
            stage = next(s for s in self.recipe.stages if s.id == stage_id)
            if stage.backend.require_available:
                available, version, evidence = rec.probe_fresh()
                if not available:
                    raise ValueError(
                        f"artifact pin {stage_id}: backend is no longer available: {evidence}"
                    )
                if version is not None and version != rec.version:
                    raise ValueError(
                        f"artifact pin {stage_id}: fresh version {version} != live {rec.version}"
                    )
            live_execution_identity = _execution_identity_hash(rec)
            if pin["execution_identity_sha256"] != live_execution_identity:
                raise ValueError(
                    f"artifact pin {stage_id}: execution identity "
                    f"{pin['execution_identity_sha256']} != live {live_execution_identity}"
                )


def _adapter_identity(record: object) -> str:
    adapter = getattr(record, "adapter", None) if record is not None else None
    if adapter is None:
        return "none-adapter"
    return f"{adapter.backend_id}::{adapter.__class__.__name__}"


def _capability_hash(record: object) -> str:
    """Canonical hash over the backend's declared capabilities + derivative flag
    + resource limits (the execution-relevant contract)."""
    caps = tuple(sorted(getattr(record, "declared_capabilities", ())))
    derivative = bool(getattr(record, "produces_derivative", False))
    lim = getattr(record, "resource_limits", None)
    lim_dict = (
        None
        if lim is None
        else {
            "host": getattr(lim, "host_gb", 0.0),
            "scratch": getattr(lim, "scratch_gb", 0.0),
            "workers": getattr(lim, "workers", 1),
        }
    )
    payload = canonical_json(
        {"capabilities": list(caps), "produces_derivative": derivative, "limits": lim_dict}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _execution_identity_hash(record: object) -> str:
    """Hash a freshly measured, execution-affecting backend/tool identity."""
    fresh = getattr(record, "fresh_execution_identity", None)
    if not callable(fresh):
        raise ValueError("backend record cannot provide an execution identity")
    identity = fresh()
    if not identity:
        raise ValueError("backend execution identity must not be empty")
    payload = canonical_json(dict(identity))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _snapshot_pins(
    compiled: CompiledRecipe, registry: object | None = None
) -> dict[str, dict[str, str]]:
    """Snapshot, per stage, the ACTUAL selected backend's resolved id/version,
    adapter identity, lifecycle status and capability hash."""
    pins: dict[str, dict[str, str]] = {}
    for s in compiled.recipe.stages:
        pin: dict[str, str] = {
            "backend_id": s.backend.backend_id,
            "pinned_version": s.backend.version,
            "resolved_version": "unresolved",
            "status": "unresolved",
            "adapter_identity": "unresolved-adapter",
            "capability_hash": "unresolved-capability",
            "execution_identity_sha256": "unresolved-execution-identity",
        }
        if registry is not None:
            rec = getattr(registry, "get", lambda *_: None)(s.backend.backend_id)
            if rec is not None:
                pin["resolved_version"] = getattr(rec, "version", "unresolved") or "unresolved"
                st = getattr(rec, "status", None)
                pin["status"] = (
                    st.value if st is not None and hasattr(st, "value") else str(st or "unresolved")
                )
                pin["adapter_identity"] = _adapter_identity(rec)
                pin["capability_hash"] = _capability_hash(rec)
                pin["execution_identity_sha256"] = _execution_identity_hash(rec)
        pins[s.id] = pin
    return pins


def _deep_freeze(value: object) -> object:
    """Recursively freeze nested dicts/lists so NO public mutation can reach
    artifact identity (MappingProxy for dicts, tuple for lists, recursion for
    nested values)."""
    from types import MappingProxyType

    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: object) -> object:
    """Recursively convert frozen wrappers back to plain dict/list for
    serialization."""
    from types import MappingProxyType

    if isinstance(value, (MappingProxyType, dict)):
        return {k: _deep_unfreeze(v) for k, v in dict(value).items()}
    if isinstance(value, (tuple, list)):
        return [_deep_unfreeze(v) for v in value]
    return value


def _run_id_from_payload(recipe_sha256: str, plan_id: str, inputs: dict[str, object]) -> str:
    payload = canonical_json(
        {"plan_id": plan_id, "recipe_sha256": recipe_sha256, "job_inputs": inputs}
    )
    return "run-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
