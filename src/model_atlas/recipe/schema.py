"""Canonical, versioned, serializable compression-recipe schema.

A ``CompressionRecipe`` is the deterministic, immutable description of *how* a
derivative will be produced from a source model: which plugin backend performs
which stage, in which order, under which constraints, pinned to which versions,
seeds, expected formats, and validation gates.

The recipe is a **plan**, never an execution log. Executing it produces a
:class:`~model_atlas.jobs.schema.Job` and stage outputs referenced by
content-addressed hashes; nothing in the recipe tree ever mutates.

Contract (fidelity-first, GLM-5.2 canonical product rule):

* ``no_pruning=true`` is the default policy. No stage whose effect class is
  ``pruning`` (or which transitively depends on one) may appear, unless an
  explicit, separately-registered pruning capability recipe is compiled —
  pruning-capability stages are always opt-in and never auto-injected.
* Evidence kinds may only move *down* the ladder (predicted -> measured), never
  up; the compiler records that provenance policy here as
  :attr:`RecipeStage.evidence_policy`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from model_atlas.schemas.evidence import EvidenceKind

RECIPE_SCHEMA_VERSION = 1


class StageEffectClass(StrEnum):
    """What kind of model change a stage performs.

    Used by the compiler to enforce no-pruning transitivity and ordering
    rules without assuming any particular method name.
    """

    IDENTITY = "identity"  # teacher/reference identity, source hashing
    PROFILING = "profiling"  # corpus / calibration / memory / runtime profiling
    SENSITIVITY = "sensitivity"  # semantic/activation/Hessian/routing/spectral analysis
    REPRESENTATION = "representation"  # shared cross-expert representation analysis
    CONDITIONING = "conditioning"  # distribution/outlier conditioning
    ALLOCATION = "allocation"  # global bit-allocation (GEMQ/MixQuant-informed)
    QUANTIZATION = "quantization"  # EXL3 primary quant, NVFP4/FP8/BF16 substitution
    REFINEMENT = "refinement"  # ReQuant-style fixed-budget refinement
    RESIDUAL = "residual"  # selective low-rank/sparse residual correction
    KV = "kv"  # KV-cache optimization
    RUNTIME = "runtime"  # real two-Spark memory/runtime profiling
    EVALUATION = "evaluation"  # Eval Lab / Pareto evaluation
    REPAIR = "repair"  # healing/distillation — deterministic repair, not pruning
    PRUNING = "pruning"  # TENP/FlexMoE — a separately-registered opt-in capability


class RecipeStatus(StrEnum):
    """Backend/method lifecycle status (discovered -> validated)."""

    UNAVAILABLE = "unavailable"
    DISCOVERED = "discovered"
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    RECOMMENDED = "recommended"


class PublishPolicy(StrEnum):
    PRIVATE = "private"
    DRAFT = "draft"
    SHARED = "shared"
    RECOMMENDED = "recommended"


class FormatExpectation(BaseModel):
    """Expected input/output format of a stage, declared so the compiler can
    check a backend really accepts it (interface-agnostic formats)."""

    model_config = ConfigDict(extra="forbid")

    format: str  # e.g. "safetensors", "exl3", "modelopt_nvfp4", "fp8_e4m3", "jsonl-events"
    optional: bool = False
    note: str = ""


class ResourceBounds(BaseModel):
    """Bounded resource declarations for a stage (never unlimited)."""

    model_config = ConfigDict(extra="forbid")

    max_host_gb: float = Field(default=0.0, ge=0.0)
    max_scratch_gb: float = Field(default=0.0, ge=0.0)
    max_workers: int = Field(default=1, ge=1, le=256)
    max_wall_seconds: float = Field(default=0.0, ge=0.0)  # 0 = unbounded (advisory)
    note: str = ""


class ValidationGate(BaseModel):
    """A named validation gate a stage output must pass before promotion."""

    model_config = ConfigDict(extra="forbid")

    gate_id: str
    kind: str = "validator"  # validator | dry_run | eq_control | identity_control
    required: bool = True
    params: dict[str, str] = Field(default_factory=dict)


class StageBackendPin(BaseModel):
    """Backend/version pin for a stage (the registry entry may move; this pins
    the exact compatible interface the stage was compiled against)."""

    model_config = ConfigDict(extra="forbid")

    backend_id: str
    version: str  # backend_version pin; "" or "unpinned" = not pinned at authoring time
    minimum_status: RecipeStatus = RecipeStatus.DISCOVERED
    # FAIL-CLOSED by default. Setting False makes the stage dry-run-only: it can
    # compile for inspection/planning but can never EXECUTE in a job.
    require_available: bool = True


class RecipeStage(BaseModel):
    """One ordered stage of a compression recipe.

    ``id`` is author-assigned and unique within a recipe; ``effect_class``,
    ``produces_format``, and ``requires_formats`` give the compiler everything it
    needs for deterministic compatibility checks (it never peeks inside plugin
    implementations).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    effect_class: StageEffectClass
    backend: StageBackendPin
    parameters: dict[str, str] = Field(default_factory=dict)
    produces_format: list[str] = Field(default_factory=list)
    requires_formats: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)  # artifact names/urbs
    seed: int | None = None
    validation_gates: list[ValidationGate] = Field(default_factory=list)
    resources: ResourceBounds = Field(default_factory=ResourceBounds)
    publish_policy: PublishPolicy = PublishPolicy.DRAFT
    evidence_policy: EvidenceKind = EvidenceKind.PREDICTED
    notes: str = ""


class SourceIdentity(BaseModel):
    """Immutable source/checkpoint identity (path + hashes).

    ``sha256`` is a path-bound dict {relative_path -> sha256} (exact
    membership + per-path hash equality enforced at run time), OR one may
    declare ``manifest_digest`` — the canonical digest of the whole recursive
    source manifest. At least one of the two must be provided for a
    mutable-verifiable source; both are verified when present.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    checkpoint_path: str
    checkpoint_revision: str | None = None
    sha256: dict[str, str] = Field(default_factory=dict)
    manifest_digest: str = ""
    params_estimate: int = Field(default=0, ge=0)  # estimated parameters; 0 = unmeasured


class CalibrationIdentity(BaseModel):
    """Corpus/calibration identity + balance (AGENTS invariant 6)."""

    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    corpus_name: str
    seed: int = 0
    partition: str = "atlas_calibration"
    capability_labels: list[str] = Field(default_factory=list)
    corpus_records_path: str | None = None
    tokenizer_sha256: str = Field(default="", pattern=r"^$|^[0-9a-f]{64}$")
    note: str = ""


class HardwareEnvelope(BaseModel):
    """Software/hardware envelope for the run (what the plan targets).

    ``model_arch`` is the MODEL family (e.g. glm-5.2, k3); ``compute_arch`` is
    the GPU/CPU compute capability (e.g. gb10-sm121); ``topology`` is the node
    layout (e.g. 2x-spark); ``runtime_backend`` is the serving runtime (e.g.
    vllm-modelopt). These are SEPARATE axes — never compare glm-5.2 to
    gb10-sm121, or vllm-modelopt to sm121.
    """

    model_config = ConfigDict(extra="forbid")

    node_count: int = Field(default=2, ge=1)
    model_arch: str = "glm-5.2"
    compute_arch: str = "gb10-sm121"  # GPU/CPU compute capability
    topology: str = "2x-spark"
    per_node_host_gib: float = Field(default=120.0, gt=0.0)
    per_node_gpu_gib: float | None = None
    interconnect: str = "connectx-7"
    runtime_backend: str = "vllm-modelopt"
    note: str = ""


class RecipeConstraints(BaseModel):
    """Constraints that gate the whole recipe (compiled transitively)."""

    model_config = ConfigDict(extra="forbid")

    no_pruning: bool = True  # CANONICAL DEFAULT: fidelity-first, no pruning
    allow_pruning_capability: bool = False  # opt-in TENP/FlexMoE capability
    preserve_non_expert_backbone: bool = True  # attention/MLA/roMers/norms/embeds/head
    immutable_source: bool = True
    allow_hybrid_precision: bool = False  # EXL3+NVFP4+FP8 hybrid needs explicit support
    max_resident_gib: float = Field(default=0.0, ge=0.0)  # 0 = not bounded
    derived_format: str = "safetensors"


class PublishRule(BaseModel):
    """Publish policy: what must hold before a compiled plan/derivative can be
    published (promoted to a shared/recommended tier)."""

    model_config = ConfigDict(extra="forbid")

    require_all_stages_validated: bool = True
    require_no_pruning: bool = False  # publishing a pruning run is gated by capability
    evidence_kind_min: EvidenceKind = EvidenceKind.MEASURED
    require_runtime_benchmarked: bool = True  # measured two-Spark profiling required
    require_repair_or_validated: bool = True


class CompressionRecipe(BaseModel):
    """The canonical, immutable recipe document.

    ``recipe_id`` is computed by the :mod:`recipe.compiler` from canonical
    content (the same constructor inputs always produce the same id). It is
    ``None`` in transit until the recipe is passed through the compiler.
    """

    model_config = ConfigDict(extra="forbid")

    recipe_schema_version: int = RECIPE_SCHEMA_VERSION
    name: str
    description: str = ""
    source: SourceIdentity
    calibration: CalibrationIdentity
    hardware: HardwareEnvelope = Field(default_factory=HardwareEnvelope)
    constraints: RecipeConstraints = Field(default_factory=RecipeConstraints)
    stages: list[RecipeStage] = Field(default_factory=list)
    publish: PublishRule = Field(default_factory=PublishRule)
    backend_pins: dict[str, str] = Field(default_factory=dict)  # backend_id -> version
    # Authoring metadata. NOTE: created_at is NOT part of the canonical content
    # that derives recipe_id/recipe_sha256 (the id is content-addressed, not
    # time-addressed), so two authoring times never change a recipe's identity.
    # recipe_id is None while the recipe is being authored; it is assigned by
    # the compiler (recipe_id_of) and is never read back as authoritative from
    # this authoring structure at compile time.
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recipe_id: str | None = None

    @field_validator("stages")
    @classmethod
    def _unique_stage_ids(cls, v: list[RecipeStage]) -> list[RecipeStage]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate stage ids in recipe: {ids}")
        return v

    @field_validator("recipe_schema_version")
    @classmethod
    def _version(cls, v: int, info: ValidationInfo) -> int:
        if v != RECIPE_SCHEMA_VERSION:
            raise ValueError(
                f"recipe schema version {v} unsupported (current {RECIPE_SCHEMA_VERSION})"
            )
        return v

    @model_validator(mode="after")
    def _pruning_policy_consistent(self) -> CompressionRecipe:
        # The compiler is the single enforcement point for no_pruning (it can
        # trace transitive consumption). The schema only rejects the self-
        # contradictory combination (no_pruning + allow_pruning_capability flag).
        if not self.constraints.no_pruning:
            return self
        if self.constraints.allow_pruning_capability:
            raise ValueError("no_pruning=true conflicts with allow_pruning_capability=true")
        return self
