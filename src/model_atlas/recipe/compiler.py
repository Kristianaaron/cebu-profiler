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
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from model_atlas.recipe.schema import CompressionRecipe, RecipeStatus, StageEffectClass

# The compiled plan must be DEEPLY immutable: its embedded recipe tree is frozen
# so no caller can mutate a plan field (or a nested stage/constraint) after
# compilation. Fresh immutable copies are made on each compile to avoid freezing
# the caller's original (which may still be authored/re-edited).
_IMMUTABLE_RECIPE_CFG = {"frozen": True, "extra": "forbid"}


@runtime_checkable
class BackendRecordLike(Protocol):
    """What the compiler may observe about a registered backend (contract subset)."""

    status: RecipeStatus
    version: str
    formats: tuple[str, ...]
    supported_formats: tuple[str, ...]
    architectures: tuple[str, ...]
    compute_archs: tuple[str, ...]
    topologies: tuple[str, ...]
    runtime_compat: tuple[str, ...]
    produces_derivative: bool
    fail_closed: bool
    declared_capabilities: tuple[str, ...]
    parameters: tuple[ParameterSpecLike, ...]
    resource_limits: ResourceLimitsLike | None


@runtime_checkable
class ParameterSpecLike(Protocol):
    """Parameter-schema subset the compiler validates against."""

    name: str
    type: str
    required: bool
    enum: tuple[str, ...]
    minimum: float | None
    maximum: float | None
    default: str | None

    def validate(self, value: str) -> list[str]: ...


@runtime_checkable
class ResourceLimitsLike(Protocol):
    """Bounded resource envelope a backend may serve."""

    max_host_gb: float
    max_scratch_gb: float
    max_workers: int


@runtime_checkable
class CapabilityRegistryLike(Protocol):
    """The compile-time capability surface the compiler consumes, so
    ``recipe`` never imports ``backend.registry`` (no import cycle), and any
    capability provider is pluggable."""

    def get(self, backend_id: str) -> BackendRecordLike | None: ...
    def is_backend_available(self, backend_id: str) -> bool: ...
    def declares_capability(self, capability: str) -> bool: ...
    def backend_declares_hybrid(self, backend_id: str, formats: set[str]) -> bool: ...
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
    """Immutable compiled plan. Nothing here may change after compilation.

    ``resolved_backends`` and ``backend_status_snapshot`` are exposed as
    read-only Mappings — callers cannot mutate the compiled plan's resolved
    pins/status snapshot (mutating a copied dict never affects the plan).
    """

    _recipe_payload: str = field(repr=False, compare=False)
    recipe_id: str
    recipe_sha256: str
    plan_id: str
    resolved_backends: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    issues: tuple[CompileIssue, ...] = ()
    backend_status_snapshot: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    compiled_by: str = "model-atlas"

    @property
    def recipe(self) -> CompressionRecipe:
        """Fresh, value-semantics copy reconstructed from the frozen payload."""
        return CompressionRecipe.model_validate_json(self._recipe_payload)

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
            _recipe_payload=canonical_json(recipe.model_dump(mode="json")),
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
        # Build the full format-consumption DAG RETAINING every producer edge:
        # produced_by maps a format to the list of ALL stages producing it (a
        # format produced by more than one stage keeps every producer), so
        # pruning taint flows downstream along every producer relationship.
        produced_by: dict[str, list[str]] = {}
        for s in recipe.stages:
            for f in s.produces_format:
                produced_by.setdefault(f, []).append(s.id)
        requires_of: dict[str, set[str]] = {s.id: set() for s in recipe.stages}
        for s in recipe.stages:
            for f in s.requires_formats:
                for producer in produced_by.get(f, []):
                    if producer != s.id:
                        requires_of[s.id].add(producer)
        # downstream reachability from each pruning stage over ALL edges
        tainted: set[str] = set()
        stack: list[str] = [s.id for s in pruning]
        while stack:
            node = stack.pop()
            if node in tainted:
                continue
            tainted.add(node)
            for other in recipe.stages:
                if node in requires_of[other.id]:
                    stack.append(other.id)

        if recipe.constraints.allow_pruning_capability:
            # Opt-in capability recipe: require the capability actually declared
            # AND each pruning-stage backend to declare it.
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
            # transitive taint: any stage reachable from a pruning stage is illegal
            for stage in recipe.stages:
                if stage.id in tainted and stage.effect_class is not StageEffectClass.PRUNING:
                    issues.append(
                        CompileIssue(
                            "error",
                            "no_pruning_violation_transitive",
                            "stage transitively depends on a pruning-produced format "
                            "(full DAG reachability)",
                            stage.id,
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
        # make a recipe hybrid — a hybrid claim requires multiple on-disk
        # precision encodings of weights.
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
        # capability declared by a SELECTED, available, version-RESOLVED backend
        # that ALSO actually PRODUCES one of the precision formats in the set.
        # A profiling/eval/analysis backend that never emits a weight format can
        # never authorize a hybrid.
        declared = False
        declaring_producers: list[str] = []
        producer_ids = {
            stage_id.id
            for stage_id in recipe.stages
            if any(f in _PRECISION_FORMATS for f in stage_id.produces_format)
        }
        for stage_id in recipe.stages:
            if stage_id.id not in producer_ids:
                continue  # only an actual format-producing stage counts
            pin = stage_id.backend
            rec = self._registry.get(pin.backend_id)
            if rec is None:
                continue
            if not self._registry.is_backend_available(pin.backend_id):
                continue
            if not pin.version or pin.version == "unpinned" or rec.version != pin.version:
                continue  # must be exact resolved version
            if self._registry.backend_declares_hybrid(pin.backend_id, quant_fmts):
                declared = True
                declaring_producers.append(stage_id.id)

        if not declared or not declaring_producers:
            message = (
                f"recipe mixes precision formats {{{combos}}} but no SELECTED, "
                "available, version-resolved FORMAT-PRODUCING backend explicitly "
                "declares support for that exact combination. (EXL3+NVFP4+FP8 is "
                "rejected unless a runtime/backend declares it; a profiling/eval "
                "stage that never emits a weight format cannot authorize it.) "
                "allow_hybrid_precision alone never authorizes an unsupported "
                "composition, nor does a declaration on an unrelated, unavailable, "
                "or non-producing record."
            )
            issues.append(CompileIssue("error", "unsupported_hybrid_precision", message))
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

            # -- availability (fail closed unless explicitly dry-run-only) --
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

            # -- minimum lifecycle status (e.g. stage requires validated) --
            if _STATUS_RANK.get(record.status, 99) < _STATUS_RANK.get(pin.minimum_status, 0):
                issues.append(
                    CompileIssue(
                        "error",
                        "backend_status_below_minimum",
                        f"backend {pin.backend_id!r} status {record.status.value} is "
                        f"below stage minimum {pin.minimum_status.value}",
                        s.id,
                    )
                )

            # -- version pin: only an EXACT resolved version is executable.
            #    "unpinned" = authoring draft that can compile for planning but
            #    is dry-run-only (non-executable); an exact pinned version must
            #    match the record's resolved version to execute.
            if not pin.version or pin.version == "unpinned":
                issues.append(
                    CompileIssue(
                        "error",
                        "backend_version_unpinned",
                        f"stage pins backend version {pin.version or '<none>'!r}; "
                        "executable stages require an exact resolved version (unpinned "
                        "is dry-run-only/non-executable)",
                        s.id,
                    )
                )
            elif record.version != pin.version:
                issues.append(
                    CompileIssue(
                        "error",
                        "backend_version_mismatch",
                        f"stage pins backend version {pin.version!r} but the selected "
                        f"backend is {record.version!r}; exact pins required for "
                        "reproducible execution",
                        s.id,
                    )
                )

            # -- format compatibility: only REAL serialization/weight formats a
            #    stage claims to produce must be declared by the backend. Plan
            #    artifacts (research maps/profiles/plans, manifests) exchange
            #    JSON internally and are not backend-format contracts.
            supported = set(record.formats) | set(record.supported_formats)
            produced = set()
            for _f in s.produces_format:
                if _f not in _PLAN_ARTIFACTS and _f not in {"manifest.json", "jsonl-events"}:
                    produced.add(_f)
            for f in produced:
                if f not in supported:
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_format_mismatch",
                            f"stage produces format {f!r} but backend "
                            f"{pin.backend_id!r} does not declare it",
                            s.id,
                        )
                    )

            # -- parameter schema --
            allowed = {p.name for p in record.parameters}
            for p in record.parameters:
                if p.required and p.name not in s.parameters:
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_required_param_missing",
                            f"stage missing required parameter {p.name!r} for "
                            f"backend {pin.backend_id!r}",
                            s.id,
                        )
                    )
            for k in s.parameters:
                if k not in allowed:
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_param_unknown",
                            f"stage parameter {k!r} not declared by backend {pin.backend_id!r}",
                            s.id,
                        )
                    )
            for param in record.parameters:
                if param.name in s.parameters:
                    for err in param.validate(s.parameters[param.name]):
                        issues.append(CompileIssue("error", "backend_param_invalid", err, s.id))
            # -- resource bounds: a stage may not exceed its backend's declared
            #    capabilities (bounded-resource contract) --
            if record.resource_limits is not None:
                lim = record.resource_limits
                if s.resources.max_host_gb > lim.max_host_gb:
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_resource_exceeded",
                            f"stage host_gb {s.resources.max_host_gb} exceeds backend "
                            f"limit {lim.max_host_gb}",
                            s.id,
                        )
                    )
                if s.resources.max_workers > lim.max_workers:
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_resource_exceeded",
                            f"stage workers {s.resources.max_workers} exceeds backend "
                            f"limit {lim.max_workers}",
                            s.id,
                        )
                    )

            # -- architecture/runtime compatibility: only stages producing REAL
            #    serialization/weight formats must be compatible with the target
            #    MODEL arch, GPU compute arch, topology, and runtime — four
            #    SEPARATE axes (glm-5.2 ≠ gb10-sm121 ≠ 2x-spark ≠ vllm-modelopt).
            real_formats = [f for f in s.produces_format if f not in _PLAN_ARTIFACTS]
            if real_formats:
                hw = recipe.hardware
                # backend declares per-family architectures ("glm-5.2","k3",…);
                # "any" is a documentation marker only used when the backend is
                # genuinely family-agnostic.
                if hw.model_arch not in record.architectures and "any" not in record.architectures:
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_arch_incompatible",
                            f"backend {pin.backend_id!r} not compatible with model "
                            f"arch {hw.model_arch!r} (declared {record.architectures})",
                            s.id,
                        )
                    )
                # topology axis (node layout) declared separately on the record
                if hw.topology not in record.topologies and "any" not in record.topologies:
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_topology_incompatible",
                            f"backend {pin.backend_id!r} not compatible with topology "
                            f"{hw.topology!r} (declared {record.topologies})",
                            s.id,
                        )
                    )
                # GPU compute-arch axis (compute capability), e.g. gb10-sm121
                if (
                    hw.compute_arch not in record.compute_archs
                    and "any" not in record.compute_archs
                ):
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_compute_arch_incompatible",
                            f"backend {pin.backend_id!r} not compatible with compute "
                            f"arch {hw.compute_arch!r} (declared {record.compute_archs})",
                            s.id,
                        )
                    )
                if (
                    hw.runtime_backend not in record.runtime_compat
                    and "any" not in record.runtime_compat
                ):
                    issues.append(
                        CompileIssue(
                            "error",
                            "backend_runtime_incompatible",
                            f"backend {pin.backend_id!r} not compatible with runtime "
                            f"{hw.runtime_backend!r} (declared {record.runtime_compat})",
                            s.id,
                        )
                    )

            # -- P0 derivative gate: a compression stage must be served by a
            #    backend that produces a REAL derivative; a probe-only producer
            #    can never make a compression stage succeed at compile time --
            if (
                s.effect_class
                in {
                    StageEffectClass.QUANTIZATION,
                    StageEffectClass.REFINEMENT,
                    StageEffectClass.RESIDUAL,
                    StageEffectClass.CONDITIONING,
                }
                and not record.produces_derivative
            ):
                issues.append(
                    CompileIssue(
                        "error",
                        "backend_not_derivative_producer",
                        f"stage {s.id} is a compression stage but backend "
                        f"{pin.backend_id!r} is probe/analysis-only "
                        "(produces_derivative=False); no real derivative can result",
                        s.id,
                    )
                )
        return issues


_STATUS_RANK = {
    RecipeStatus.UNAVAILABLE: 0,
    RecipeStatus.DISCOVERED: 1,
    RecipeStatus.EXPERIMENTAL: 2,
    RecipeStatus.VALIDATED: 3,
    RecipeStatus.RECOMMENDED: 4,
}

# Non-serialized plan artifacts: analysis maps, profiles, plans, and manifests
# exchanged as JSON. They are NOT backend format contracts (a backend producing
# an "exl3" or "modelopt_nvfp4" serialization is what the format gate checks).
_PLAN_ARTIFACTS = frozenset(
    {
        "manifest.json",
        "jsonl-events",
        "corpus-profile",
        "sensitivity-map",
        "representation-map",
        "bit-allocation",
        "kv-plan",
        "routing-trace",
        "keep-map",
        "runtime-profile",
        "eval-results",
        "conditioned-weights",
    }
)


def _summarize(issues: list[CompileIssue]) -> str:
    return "; ".join(f"[{i.code}] {i.message}" for i in issues[:8])
