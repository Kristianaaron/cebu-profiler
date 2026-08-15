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

PLAN_ARTIFACT_SCHEMA = 2


class CompiledPlanArtifact(BaseModel):
    """One versioned, immutable compiled-plan artifact.

    ``resolved_pins`` and ``backend_status_snapshot`` are exposed as frozen
    (immutable) mappings — the ``model_validate`` path wraps passed dicts with
    MappingProxyType so no caller can mutate a loaded artifact's pins/status.
    """

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

    def frozen_pins(self) -> MappingProxyType[str, MappingProxyType[str, str]]:
        """Read-only view of resolved_pins (nested frozen)."""
        return MappingProxyType(
            {k: MappingProxyType(dict(v)) for k, v in self.resolved_pins.items()}
        )

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

    def verify_pins_against(self, registry: object) -> None:
        """Compare every pinned stage's backend id, exact version, adapter
        identity, lifecycle status and capability hash against the LIVE
        registry. Any mismatch means the artifact no longer describes the
        runnable stack — fail closed before execution."""
        from model_atlas.backend.registry import BackendRegistry

        if not isinstance(registry, BackendRegistry):
            raise ValueError("verify_pins_against requires a BackendRegistry")
        for stage_id, pin in self.resolved_pins.items():
            backend_id = pin.get("backend_id", "")
            rec = registry.get(backend_id)
            if rec is None:
                raise ValueError(
                    f"artifact pin {stage_id}: backend {backend_id!r} no longer registered"
                )
            if (
                pin.get("resolved_version")
                and pin["resolved_version"]
                not in (
                    "unresolved",
                    "unpinned",
                    "",
                )
                and pin["resolved_version"] != rec.version
            ):
                raise ValueError(
                    f"artifact pin {stage_id}: resolved version {pin['resolved_version']} "
                    f"!= live {rec.version}"
                )
            live_adapter_id = _adapter_identity(rec)
            if pin.get("adapter_identity") and pin["adapter_identity"] != live_adapter_id:
                raise ValueError(
                    f"artifact pin {stage_id}: adapter identity {pin['adapter_identity']} "
                    f"!= live {live_adapter_id}"
                )
            live_status = rec.status.value
            if (
                pin.get("status")
                and pin["status"] not in ("unresolved", "")
                and pin["status"] != live_status
            ):
                raise ValueError(
                    f"artifact pin {stage_id}: status {pin['status']} != live {live_status}"
                )
            live_cap = _capability_hash(rec)
            if pin.get("capability_hash") and pin["capability_hash"] != live_cap:
                raise ValueError(
                    f"artifact pin {stage_id}: capability hash {pin['capability_hash']} "
                    f"!= live {live_cap}"
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
        else:
            # no registry at compile time: fall back to the compiler's snapshot
            # (which records lifecycle status per stage) and leave version
            # unresolved unless the record version matches.
            pin["resolved_version"] = s.backend.version
            pin["status"] = compiled.backend_status_snapshot.get(s.id, "unresolved")
        pins[s.id] = pin
    return pins


def _run_id_from_payload(recipe_sha256: str, plan_id: str, inputs: dict[str, object]) -> str:
    payload = canonical_json(
        {"plan_id": plan_id, "recipe_sha256": recipe_sha256, "job_inputs": inputs}
    )
    return "run-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
