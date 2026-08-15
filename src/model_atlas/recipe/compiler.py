"""Deterministic recipe compiler: ordering, compatibility, no-pruning policy.

The compiler turns an authored :class:`CompressionRecipe` into an **immutable
compiled plan** — every decision (ordering, compatibility, backend pin
resolution) is made once and frozen into the plan, so execution is
deterministic and re-verifiable.

Responsibilities:

1. Stable ``recipe_id`` / ``run_id`` from canonical inputs (nothing random).
2. Topological/ordering validation — a stage may only consume formats that an
   earlier stage (or an available backend capability) produces.
3. Hybrid-precision rejection — EXL3 + NVFP4 + FP8 in one recipe is only
   accepted when some backend *explicitly declares* support for that exact
   combination.
4. ``no_pruning`` enforcement, transitively — pruning stages are rejected under
   no_pruning; under an opt-in pruning-capability recipe the compiler still
   verifies the capability is declared.
5. Fail-closed missing backends — an unavailable dependency is a compile error,
   never a silent pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from model_atlas.recipe.schema import CompressionRecipe, RecipeStatus, StageEffectClass


@runtime_checkable
class BackendRecordLike(Protocol):
    """What the compiler may observe about a registered backend (contract subset)."""

    status: RecipeStatus
    version: str


@runtime_checkable
class CapabilityRegistryLike(Protocol):
    """The compile-time capability surface the compiler consumes, so
    ``recipe`` never imports ``backend.registry`` (no import cycle), and any
    capability provider is pluggable."""

    def get(self, backend_id: str) -> BackendRecordLike | None: ...
    def is_backend_available(self, backend_id: str) -> bool: ...
    def declares_capability(self, capability: str) -> bool: ...
    def declares_hybrid(self, formats: set[str]) -> bool: ...
    def backend_status_value(self, backend_id: str) -> str: ...


def canonical_json(obj: object) -> str:
    """Deterministic JSON encoding with sorted keys (the canonical form)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(digest_str: str) -> str:
    return hashlib.sha256(digest_str.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompileIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    stage_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "stage_id": self.stage_id,
        }


@dataclass(frozen=True)
class CompiledRecipe:
    """Immutable compiled plan. Nothing here may change after compilation."""

    recipe: CompressionRecipe
    recipe_id: str
    recipe_sha256: str
    plan_id: str  # == recipe_id (the compiled plan is the canonical recipe)
    resolved_backends: dict[str, str] = field(default_factory=dict)  # stage_id -> backend_id
    issues: tuple[CompileIssue, ...] = ()  # non-fatal warnings recorded
    backend_status_snapshot: dict[str, str] = field(default_factory=dict)
    compiled_by: str = "model-atlas"

    def run_id(self, job_inputs: dict[str, object]) -> str:
        """Stable run id derived from the compiled plan + concrete job inputs."""
        payload = canonical_json(
            {
                "plan_id": self.plan_id,
                "recipe_sha256": self.recipe_sha256,
                "job_inputs": job_inputs,
            }
        )
        return "run-" + sha256_hex(payload)[:24]

    def to_dict(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_sha256": self.recipe_sha256,
            "plan_id": self.plan_id,
            "resolved_backends": self.resolved_backends,
            "issues": [i.to_dict() for i in self.issues],
            "backend_status_snapshot": self.backend_status_snapshot,
            "compiled_by": self.compiled_by,
            "name": self.recipe.name,
        }


class RecipeCompileError(ValueError):
    """Raised when a recipe cannot be compiled (fail-closed)."""

    def __init__(self, message: str, issues: list[CompileIssue] | None = None) -> None:
        super().__init__(message)
        self.issues = issues or []


def _compute_recipe_sha(recipe: CompressionRecipe) -> str:
    """Hash over the canonical serialization of everything that defines the run.

    Excludes ``recipe_id`` (it is about to be derived), and ``created_at``/name
    are included so two differently-named recipes never collide. Stage order is
    part of the input (the recipe is authored ordered).
    """
    payload = recipe.model_dump(exclude={"recipe_id", "created_at"})
    return sha256_hex(canonical_json(payload))


def recipe_id_of(recipe: CompressionRecipe) -> str:
    """Stable recipe id: first 12 hex of the canonical content hash."""
    return "recipe-" + _compute_recipe_sha(recipe)[:24]


def compute_recipe_id(recipe: CompressionRecipe) -> str:
    return recipe_id_of(recipe)


class RecipeCompiler:
    """Stateless (except the injected registry) deterministic compiler."""

    def __init__(self, registry: CapabilityRegistryLike) -> None:
        self._registry = registry

    # ------------------------------------------------------------------ API
    def validate(self, recipe: CompressionRecipe) -> tuple[list[CompileIssue], str, str]:
        """Return (issues, recipe_id, recipe_sha). Never raises."""
        issues, rid, sha = self._compile(recipe, strict=False)
        return issues, rid, sha

    def compile(self, recipe: CompressionRecipe) -> CompiledRecipe:
        """Fail-closed: raise RecipeCompileError on any error issue."""
        issues, rid, sha = self._compile(recipe, strict=True)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise RecipeCompileError("recipe failed to compile; " + _summarize(errors), errors)
        resolved = {s.id: s.backend.backend_id for s in recipe.stages}
        status = {
            s.id: self._registry.backend_status_value(s.backend.backend_id) for s in recipe.stages
        }
        return CompiledRecipe(
            recipe=recipe,
            recipe_id=rid,
            recipe_sha256=sha,
            plan_id=rid,
            resolved_backends=resolved,
            issues=tuple(i for i in issues if i.severity == "warning"),
            backend_status_snapshot=status,
        )

    # ------------------------------------------------------------- internals
    def _compile(
        self, recipe: CompressionRecipe, *, strict: bool
    ) -> tuple[list[CompileIssue], str, str]:
        """Core compile pass. strict=True -> error issues raised at end."""
        sha = _compute_recipe_sha(recipe)
        rid = "recipe-" + sha[:24]
        issues: list[CompileIssue] = []

        issues += self._check_no_pruning(recipe)
        issues += self._check_ordering(recipe)
        issues += self._check_hybrid(recipe)
        issues += self._check_backends(recipe)
        return issues, rid, sha

    def _check_no_pruning(self, recipe: CompressionRecipe) -> list[CompileIssue]:
        issues: list[CompileIssue] = []
        pruning = [s for s in recipe.stages if s.effect_class is StageEffectClass.PRUNING]

        if recipe.constraints.allow_pruning_capability:
            # Opt-in capability recipe: require the capability to be actually
            # declared on the registry, AND the specific backend serving each
            # pruning stage to declare it (else a non-pruning backend could
            # masquerade as a pruning step).
            declared = self._registry.declares_capability("pruning")
            if not pruning:
                issues.append(
                    CompileIssue(
                        "warning",
                        "pruning_capability_unused",
                        "allow_pruning_capability=true but no pruning stage present",
                    )
                )
            if not declared:
                issues.append(
                    CompileIssue(
                        "error",
                        "pruning_capability_not_registered",
                        "allow_pruning_capability=true but no backend declares the "
                        "pruning capability; pruning may not run",
                    )
                )
            for s in pruning:
                rec = self._registry.get(s.backend.backend_id)
                rec_caps = getattr(rec, "declared_capabilities", ()) if rec is not None else ()
                if "pruning" not in rec_caps:
                    issues.append(
                        CompileIssue(
                            "error",
                            "pruning_stage_backend_not_capable",
                            f"pruning stage served by backend {s.backend.backend_id!r} "
                            "which does not declare the pruning capability",
                            s.id,
                        )
                    )
            return issues

        if recipe.constraints.no_pruning:
            for s in pruning:
                issues.append(
                    CompileIssue(
                        "error",
                        "no_pruning_violation",
                        "no_pruning=true forbids pruning stage; it would be removed by "
                        "the canonical fidelity-first policy",
                        s.id,
                    )
                )
            # transitive: any stage requiring a pruning-produced format is also illegal
            produced_pruning_fmt = {f for s in pruning for f in s.produces_format}
            for s in recipe.stages:
                overlap = set(s.requires_formats) & produced_pruning_fmt
                if overlap:
                    issues.append(
                        CompileIssue(
                            "error",
                            "no_pruning_violation_transitive",
                            f"stage consumes a pruning-produced format {sorted(overlap)}",
                            s.id,
                        )
                    )
        return issues

    def _check_ordering(self, recipe: CompressionRecipe) -> list[CompileIssue]:
        issues: list[CompileIssue] = []
        produced: set[str] = set()
        ids_seen: set[str] = set()
        for s in recipe.stages:
            if s.id in ids_seen:
                issues.append(
                    CompileIssue("error", "duplicate_stage_id", f"duplicate stage {s.id}", s.id)
                )
            ids_seen.add(s.id)
            missing = set(s.requires_formats) - produced
            if missing:
                issues.append(
                    CompileIssue(
                        "error",
                        "missing_dependency",
                        f"requires formats {sorted(missing)} not produced by an earlier stage",
                        s.id,
                    )
                )
            produced.update(s.produces_format)
        return issues

    def _check_hybrid(self, recipe: CompressionRecipe) -> list[CompileIssue]:
        issues: list[CompileIssue] = []
        # Only real serialized precision/serialization formats count as a
        # "precision mix". Maps, profiles, and intermediate plan artifacts never
        # make a recipe hybrid — a hybrid claim requires the recipe to actually
        # write multiple on-disk precision encodings of weights.
        _PRECISION_FORMATS = frozenset(
            {
                "exl3",
                "modelopt_nvfp4",
                "nvfp4",
                "fp8_e4m3",
                "int4",
                "int8",
                "bf16",
                "fp16",
                "aqlm",
                "mxfp4",
                "compressed-tensors",
                "fp8",
            }
        )
        quant_fmts: set[str] = set()
        for s in recipe.stages:
            for f in s.produces_format:
                if f in _PRECISION_FORMATS:
                    quant_fmts.add(f)
        if len(quant_fmts) < 2:
            return issues  # single-precision (or none) — no hybrid surface

        combos = ",".join(sorted(quant_fmts))
        # The ONLY way an unsupported composition compiles is an explicit
        # capability declaration by the selected backend/runtime for that exact
        # combination. An author flag must NEVER demote unsupported hybrid to a
        # warning — execution would still fail at run time, but a warning could
        # be dismissed and an unsupported recipe promoted. Both paths are errors
        # unless a runtime/backend explicitly declares support.
        declared = self._registry.declares_hybrid(quant_fmts)
        if not declared:
            message = (
                f"recipe mixes precision formats {{{combos}}} but no backend "
                "explicitly declares support for that exact combination. "
                "(EXL3+NVFP4+FP8 is rejected unless a runtime/backend declares "
                "it.) allow_hybrid_precision alone never authorizes an "
                "unsupported composition."
            )
            if recipe.constraints.allow_hybrid_precision:
                message += (
                    " allow_hybrid_precision=true was set, but that only records "
                    "author intent — it does not substitute for a declared "
                    "runtime/backend capability, so compile still fails closed."
                )
            issues.append(
                CompileIssue(
                    "error",
                    "unsupported_hybrid_precision",
                    message,
                )
            )
        return issues

    def _check_backends(self, recipe: CompressionRecipe) -> list[CompileIssue]:
        issues: list[CompileIssue] = []
        for s in recipe.stages:
            pin = s.backend
            record = self._registry.get(pin.backend_id)
            if record is None:
                issues.append(
                    CompileIssue(
                        "error",
                        "backend_missing",
                        f"backend {pin.backend_id!r} is not registered; nothing can run it",
                        s.id,
                    )
                )
                continue
            if pin.require_available and not self._registry.is_backend_available(pin.backend_id):
                issues.append(
                    CompileIssue(
                        "error",
                        "backend_unavailable",
                        f"backend {pin.backend_id!r} is registered but "
                        f"unavailable ({record.status.value}); fail closed — "
                        "compile cannot proceed without a runtime dependency",
                        s.id,
                    )
                )
            if pin.version and pin.version != "unpinned" and record.version != pin.version:
                issues.append(
                    CompileIssue(
                        "warning",
                        "backend_version_mismatch",
                        f"stage pins backend version {pin.version!r} but registry has "
                        f"{record.version!r}",
                        s.id,
                    )
                )
        return issues


def _summarize(issues: list[CompileIssue]) -> str:
    return "; ".join(f"[{i.code}] {i.message}" for i in issues[:8])
