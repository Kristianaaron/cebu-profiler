"""Deterministic versioned recommendation policy for Atlas compression.

Consumes an Atlas profile (evidence + coverage), a hardware envelope, a memory
target, and the backend capability registry, and emits RANKED methods/stages
with reasons, evidence references, expected memory/quality direction,
confidence, blockers, protected sensitive regions, an immutable no-pruning
default, and STABLE recommendation/recipe ids.

Rules (deterministic, no model inference):
  * A method/stage that requires a backend that is unavailable or not
    derivative-producing is BLOCKED (never recommended for execution).
  * A stage whose evidence is missing entirely DECREASES confidence and can
    BLOCK the decision (blocked for recommendation when it lacks even
    ESTIMATED evidence for the default policy).
  * Duplicate recommendations for the same (profile, hardware, memory-target,
    policy-version) produce the SAME recommendation id.
  * `no_pruning` defaults True and can only be disabled via an explicit
    allow_pruning flag on the policy (guarded by a separately registered
    pruning-capability backend).
  * Expected memory/quality directions are DECLARED (never invented metrics);
    confidence is computed ONLY from available evidence + backend status.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from model_atlas.backend.registry import BackendRegistry
from model_atlas.recipe.compiler import canonical_json
from model_atlas.recipe.schema import RecipeStatus, StageEffectClass
from model_atlas.schemas.evidence import EvidenceKind


class RecConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


RECOMMENDATION_POLICY_VERSION = "policy-v2-catalog"
METHOD_CATALOG_VERSION = 1


class CompressionIntent(StrEnum):
    QUANTIZE_ONLY = "quantize_only"
    PRUNE_ONLY = "prune_only"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class MethodFamily(StrEnum):
    ANALYSIS = "analysis"
    CONDITIONING = "conditioning"
    ALLOCATION = "allocation"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    REFINEMENT = "refinement"
    RESIDUAL = "residual"
    RECOVERY = "recovery"
    KV = "kv"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class MethodSpec:
    """Single fail-closed authority for a selectable Atlas method."""

    method: str
    priority: int
    family: MethodFamily
    backend_id: str
    evidence_stages: tuple[str, ...]
    recipe_stage_ids: tuple[str, ...]
    effect_classes: tuple[StageEffectClass, ...]
    compatible_intents: tuple[CompressionIntent, ...]
    memory_direction: str
    routing_dependent: bool = False
    planning_only: bool = False
    provenance_ids: tuple[str, ...] = ()

    def identity_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "priority": self.priority,
            "family": self.family.value,
            "backend_id": self.backend_id,
            "evidence_stages": list(self.evidence_stages),
            "recipe_stage_ids": list(self.recipe_stage_ids),
            "effect_classes": [effect.value for effect in self.effect_classes],
            "compatible_intents": [intent.value for intent in self.compatible_intents],
            "memory_direction": self.memory_direction,
            "routing_dependent": self.routing_dependent,
            "planning_only": self.planning_only,
            "provenance_ids": list(self.provenance_ids),
        }


@dataclass(frozen=True)
class StageEvidence:
    """Minimal typed evidence summary pulled from an Atlas profile.

    ``kind`` is always one of the valid ``EvidenceKind`` values. Any evidence
    item that cannot be parsed into a valid, present claim (missing fields,
    unknown/arbitrary kind, malformed coverage) is represented deterministically
    as ``kind="unknown"``, ``present=False`` — never silently estimated. When a
    claim is rejected, ``detail`` records the reason so a run can be audited
    (and so profile identity reflects the malformed input rather than hiding it).
    """

    stage_id: str
    # from EvidenceKind; "unknown" for rejected items
    kind: str
    present: bool = True
    coverage: float | None = None  # calibration coverage (0..1)
    detail: str = ""  # reason a claim was rejected, else ""

    @classmethod
    def from_dict(cls, key: str, v: Any) -> StageEvidence:
        """Deterministically parse ONE evidence item from a real Atlas run.

        Fail-closed semantics — a malformed item never fabricates support:
          * a plain string is a valid kind only if it names a real EvidenceKind;
            any other string is ``unknown``/absent (never ``estimated``).
          * a dict with a missing ``kind`` or ``present`` is ``unknown``/absent.
          * a ``kind`` that is not a valid EvidenceKind is ``unknown``, never
            promoted to ``estimated``.
          * ``coverage`` must be a numeric 0..1; a bool or out-of-range value is
            rejected and the item is treated as absent.
          * any other value type is refused (no invented claim).
        Rejected items set ``detail`` to the reason.
        """
        if isinstance(v, str):
            if v not in EvidenceKind._value2member_map_:
                return cls(
                    stage_id=key, kind="unknown", present=False,
                    detail=f"unknown kind {v!r}",
                )
            return cls(stage_id=key, kind=v, present=True)
        if isinstance(v, dict):
            kind = v.get("kind")
            present = v.get("present")
            if kind is None or present is None:
                return cls(
                    stage_id=key, kind="unknown", present=False,
                    detail="missing kind/present",
                )
            if kind not in EvidenceKind._value2member_map_:
                return cls(
                    stage_id=key, kind="unknown", present=False,
                    detail=f"unknown kind {kind!r}",
                )
            coverage: float | None = None
            detail = ""
            cov = v.get("coverage")
            if cov is not None:
                if isinstance(cov, bool) or not isinstance(cov, (int, float)):
                    detail = f"invalid coverage {cov!r}"
                    coverage = None
                elif not (0.0 <= float(cov) <= 1.0):
                    detail = f"coverage out of range {cov!r}"
                    coverage = None
                else:
                    coverage = float(cov)
            if detail:
                # malformed coverage: fail closed — the item carries no usable
                # coverage, so it must not count as supporting evidence.
                return cls(
                    stage_id=key, kind="unknown", present=False, coverage=None, detail=detail
                )
            return cls(stage_id=key, kind=kind, present=bool(present), coverage=coverage)
        # non-string/non-dict evidence value: refuse to guess — treat as
        # absent rather than invent a claim (unknown stays unknown).
        return cls(
            stage_id=key, kind="unknown", present=False,
            detail=f"unsupported evidence value {v!r}",
        )


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProfileExecutionBinding:
    """Immutable model/calibration identity required for derivative execution."""

    source_id: str
    checkpoint_path: str
    checkpoint_revision: str | None
    source_manifest_digest: str
    source_sha256: tuple[tuple[str, str], ...]
    calibration_id: str
    corpus_name: str
    calibration_seed: int
    calibration_partition: str
    corpus_records_path: str | None
    tokenizer_hash: str

    def __post_init__(self) -> None:
        required = {
            "source_id": self.source_id,
            "checkpoint_path": self.checkpoint_path,
            "calibration_id": self.calibration_id,
            "corpus_name": self.corpus_name,
            "calibration_partition": self.calibration_partition,
            "tokenizer_hash": self.tokenizer_hash,
        }
        if any(not value for value in required.values()):
            raise ValueError("profile execution identity fields must be nonempty")
        if not self.source_manifest_digest and not self.source_sha256:
            raise ValueError("profile execution identity requires source hashes")
        digests = [self.tokenizer_hash]
        if self.source_manifest_digest:
            digests.append(self.source_manifest_digest)
        digests.extend(digest for _, digest in self.source_sha256)
        if any(not _SHA256_RE.fullmatch(digest) for digest in digests):
            raise ValueError("profile execution digests must be lowercase SHA-256")
        paths = [path for path, _ in self.source_sha256]
        if len(paths) != len(set(paths)) or any(not path for path in paths):
            raise ValueError("profile source hash paths must be nonempty and unique")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileExecutionBinding:
        hashes = data.get("source_sha256") or {}
        if not isinstance(hashes, dict):
            raise ValueError("source_sha256 must be a path-to-digest object")
        return cls(
            source_id=str(data.get("source_id", "")),
            checkpoint_path=str(data.get("checkpoint_path", "")),
            checkpoint_revision=(
                str(data["checkpoint_revision"])
                if data.get("checkpoint_revision") is not None
                else None
            ),
            source_manifest_digest=str(data.get("source_manifest_digest", "")),
            source_sha256=tuple(
                sorted((str(path), str(digest)) for path, digest in hashes.items())
            ),
            calibration_id=str(data.get("calibration_id", "")),
            corpus_name=str(data.get("corpus_name", "")),
            calibration_seed=int(data.get("calibration_seed", 0)),
            calibration_partition=str(data.get("calibration_partition", "atlas_calibration")),
            corpus_records_path=(
                str(data["corpus_records_path"])
                if data.get("corpus_records_path") is not None
                else None
            ),
            tokenizer_hash=str(data.get("tokenizer_hash", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_revision": self.checkpoint_revision,
            "source_manifest_digest": self.source_manifest_digest,
            "source_sha256": dict(self.source_sha256),
            "calibration_id": self.calibration_id,
            "corpus_name": self.corpus_name,
            "calibration_seed": self.calibration_seed,
            "calibration_partition": self.calibration_partition,
            "corpus_records_path": self.corpus_records_path,
            "tokenizer_hash": self.tokenizer_hash,
        }


@dataclass(frozen=True)
class AtlasProfile:
    """A completed Atlas profile (evidence + coverage) with a stable identity."""

    profile_id: str
    model: str
    seed: int = 0
    evidence: dict[str, StageEvidence] = field(default_factory=dict)
    routing_consistency_passed: bool | None = None
    hardware_model_arch: str = "glm-5.2"
    execution: ProfileExecutionBinding | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtlasProfile:
        """Build a profile from a real exported Atlas run dict.

        Evidence values may be a V3 ``dict`` (``{"kind": ..., "present": ...,
        "coverage": ...}``) OR a plain string kind; both are parsed into
        ``StageEvidence``. Evidence keys are normalized through the alias map.
        """
        evidence: dict[str, StageEvidence] = {}
        for k, v in (data.get("evidence") or {}).items():
            st = StageEvidence.from_dict(k, v)
            canonical = canonical_stage(k)
            if canonical in evidence and st.present:
                prev = evidence[canonical]
                if _SRC_PRECEDENCE.get(st.kind, 5) <= _SRC_PRECEDENCE.get(prev.kind, 5):
                    continue  # keep the strongest observed claim
            evidence[canonical] = st
        execution_data = data.get("execution")
        execution = (
            ProfileExecutionBinding.from_dict(execution_data)
            if isinstance(execution_data, dict)
            else None
        )
        return cls(
            profile_id=str(
                data.get("declared_profile_id")
                or data.get("declared_id")
                or data.get("profile_id", "imported")
            ),
            model=str(data.get("model", data.get("source_model", "unknown"))),
            seed=int(data.get("seed", 0)),
            evidence=evidence,
            routing_consistency_passed=_as_bool(data.get("routing_consistency_passed")),
            hardware_model_arch=str(
                data.get(
                    "hardware_model_arch",
                    (data.get("hardware") or {}).get("model_arch", "glm-5.2"),
                )
            ),
            execution=execution,
            notes=str(data.get("notes", "")),
        )

    def profile_id_of(self) -> str:
        payload = canonical_json(
            {
                "model": self.model,
                "hardware_model_arch": self.hardware_model_arch,
                "notes": self.notes,
                "declared": self.profile_id,
                "routing_consistency": self.routing_consistency_passed,
                "execution": self.execution.to_dict() if self.execution else None,
                "stages": sorted(
                    (
                        k,
                        v.kind,
                        v.present,
                        v.coverage,
                        v.detail,
                    )
                    for k, v in self.evidence.items()
                ),
            }
        )
        return "profile-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class RecTarget:
    hardware_model_arch: str = "glm-5.2"
    compute_arch: str = "gb10-sm121"
    topology: str = "2x-spark"
    runtime_backend: str = "vllm-modelopt"
    memory_target_gib: float = 115.0


@dataclass(frozen=True)
class RecBlock:
    code: str
    message: str
    stage_id: str | None = None


@dataclass(frozen=True)
class MethodRecommendation:
    method: str
    rank: int
    reason: str
    evidence_refs: list[str] = field(default_factory=list)
    expected_memory_direction: str = "down"  # declared, never measured
    expected_quality_direction: str = "down"  # declared
    confidence: RecConfidence = RecConfidence.MEDIUM
    confidence_text: str = ""
    blockers: list[RecBlock] = field(default_factory=list)
    status: RecipeStatus = RecipeStatus.DISCOVERED
    backend_id: str = ""
    requires_stages: list[str] = field(default_factory=list)
    protected_regions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Recommendation:
    recommendation_id: str
    policy_version: str
    profile_id: str
    target: RecTarget
    no_pruning: bool = True
    methods: tuple[MethodRecommendation, ...] = ()
    blocked_methods: tuple[MethodRecommendation, ...] = ()
    confidence: RecConfidence = RecConfidence.MEDIUM
    summary: str = ""
    generated_by: str = "model-atlas-recommendation-policy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "policy_version": self.policy_version,
            "profile_id": self.profile_id,
            "target": {
                "hardware_model_arch": self.target.hardware_model_arch,
                "compute_arch": self.target.compute_arch,
                "topology": self.target.topology,
                "runtime_backend": self.target.runtime_backend,
                "memory_target_gib": self.target.memory_target_gib,
            },
            "no_pruning": self.no_pruning,
            "methods": [_m(m) for m in self.methods],
            "blocked_methods": [_m(m) for m in self.blocked_methods],
            "confidence": self.confidence.value,
            "summary": self.summary,
            "generated_by": self.generated_by,
        }


def _m(m: MethodRecommendation) -> dict[str, Any]:
    return {
        "method": m.method,
        "rank": m.rank,
        "reason": m.reason,
        "evidence_refs": list(m.evidence_refs),
        "expected_memory_direction": m.expected_memory_direction,
        "expected_quality_direction": m.expected_quality_direction,
        "confidence": m.confidence.value,
        "confidence_text": m.confidence_text,
        "blockers": [b.__dict__ for b in m.blockers],
        "status": m.status.value,
        "backend_id": m.backend_id,
        "requires_stages": list(m.requires_stages),
        "protected_regions": list(m.protected_regions),
    }


_ALL_INTENTS = tuple(CompressionIntent)
_QUANT_INTENTS = (
    CompressionIntent.QUANTIZE_ONLY,
    CompressionIntent.HYBRID,
    CompressionIntent.CUSTOM,
)

METHOD_CATALOG: tuple[MethodSpec, ...] = (
    MethodSpec(
        "teacher-identity", 1, MethodFamily.ANALYSIS, "atlas_analysis_v3",
        ("identity",), ("t1-identity",), (StageEffectClass.IDENTITY,),
        _ALL_INTENTS, "same", planning_only=True,
    ),
    MethodSpec(
        "calibration", 2, MethodFamily.ANALYSIS, "atlas_analysis_v3",
        ("corpus_semantic",), ("t2-calibration",), (StageEffectClass.PROFILING,),
        _ALL_INTENTS, "same", planning_only=True,
    ),
    MethodSpec(
        "sensitivity", 3, MethodFamily.ANALYSIS, "atlas_analysis_v3",
        ("spectral", "shared_structure", "routing_consistency"),
        ("t3-sensitivity",), (StageEffectClass.SENSITIVITY,), _ALL_INTENTS,
        "down", planning_only=True,
    ),
    MethodSpec(
        "bit-allocation", 4, MethodFamily.ALLOCATION, "atlas_analysis_v3",
        ("global_bit_budget",), ("t6-bit-allocation",),
        (StageEffectClass.ALLOCATION,), _ALL_INTENTS, "down", planning_only=True,
        provenance_ids=("GEMQ", "MixQuant"),
    ),
    MethodSpec(
        "nvfp4-substitute", 5, MethodFamily.QUANTIZATION, "modelopt_nvfp4",
        ("nvfp4_suitability",), ("t10-nvfp4",),
        (StageEffectClass.QUANTIZATION,), _QUANT_INTENTS, "down",
        routing_dependent=True, provenance_ids=("NVIDIA-ModelOpt-NVFP4",),
    ),
    MethodSpec(
        "kv-optimization", 6, MethodFamily.KV, "atlas_analysis_v3",
        ("kv_budget",), ("t12-kv",), (StageEffectClass.KV,), _ALL_INTENTS,
        "down", planning_only=True,
    ),
    MethodSpec(
        "exl3-primary", 7, MethodFamily.QUANTIZATION, "exl3",
        ("global_bit_budget",), ("t7-exl3",),
        (StageEffectClass.QUANTIZATION,), _QUANT_INTENTS, "down",
        routing_dependent=True, provenance_ids=("EXL3",),
    ),
    MethodSpec(
        "llm-compressor", 8, MethodFamily.QUANTIZATION, "llm_compressor",
        ("global_bit_budget",), ("t11-tail",),
        (StageEffectClass.QUANTIZATION,), _QUANT_INTENTS, "down",
        routing_dependent=True, provenance_ids=("LLM-Compressor",),
    ),
    MethodSpec(
        "modelopt-nvfp4", 9, MethodFamily.QUANTIZATION, "modelopt_nvfp4",
        ("nvfp4_suitability",), ("t10-nvfp4",),
        (StageEffectClass.QUANTIZATION,), _QUANT_INTENTS, "down",
        routing_dependent=True, provenance_ids=("NVIDIA-ModelOpt-NVFP4",),
    ),
)

def validate_method_catalog(specs: tuple[MethodSpec, ...]) -> None:
    ids = [spec.method for spec in specs]
    priorities = [spec.priority for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate method IDs in METHOD_CATALOG")
    if len(priorities) != len(set(priorities)) or any(value < 1 for value in priorities):
        raise ValueError("method priorities must be unique positive integers")
    for spec in specs:
        if not (
            spec.method
            and spec.backend_id
            and spec.evidence_stages
            and spec.recipe_stage_ids
            and spec.effect_classes
            and spec.compatible_intents
        ):
            raise ValueError(f"incomplete MethodSpec identity: {spec.method!r}")
        effects = set(spec.effect_classes)
        is_pruning_family = spec.family == MethodFamily.PRUNING
        has_pruning_effect = StageEffectClass.PRUNING in effects
        if is_pruning_family != has_pruning_effect:
            raise ValueError(f"pruning family/effect mismatch: {spec.method}")
        if is_pruning_family and CompressionIntent.QUANTIZE_ONLY in spec.compatible_intents:
            raise ValueError(f"pruning MethodSpec permits quantize-only: {spec.method}")
        if (
            spec.family == MethodFamily.QUANTIZATION
            and StageEffectClass.QUANTIZATION not in effects
        ):
            raise ValueError(f"quantization MethodSpec lacks quantization effect: {spec.method}")
        artifact_mutating_effects = {
            StageEffectClass.CONDITIONING,
            StageEffectClass.PRUNING,
            StageEffectClass.QUANTIZATION,
            StageEffectClass.REFINEMENT,
            StageEffectClass.REPAIR,
            StageEffectClass.RESIDUAL,
        }
        if spec.planning_only and effects & artifact_mutating_effects:
            raise ValueError(f"planning-only MethodSpec claims derivative effect: {spec.method}")


validate_method_catalog(METHOD_CATALOG)
_METHOD_SPECS = {spec.method: spec for spec in METHOD_CATALOG}


def method_spec(method: str) -> MethodSpec:
    """Resolve an explicit catalog entry; unknown methods never gain a family."""
    try:
        return _METHOD_SPECS[method]
    except KeyError as exc:
        raise KeyError(f"unknown or unclassified compression method: {method}") from exc


def method_catalog_digest(specs: tuple[MethodSpec, ...] = METHOD_CATALOG) -> str:
    validate_method_catalog(specs)
    payload = canonical_json(
        {
            "catalog_version": METHOD_CATALOG_VERSION,
            "methods": [
                spec.identity_dict()
                for spec in sorted(specs, key=lambda item: item.method)
            ],
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_METHOD_STAGES = {
    spec.method: list(spec.evidence_stages) for spec in METHOD_CATALOG
}
_ANALYSIS_METHODS = {
    spec.method for spec in METHOD_CATALOG if spec.planning_only
}

# The compression methods that operate on router-indexed expert tensors. Their
# saliency/suitability evidence is only trustworthy if the routing-consistency
# identity gate PASSED; a failed OR UNKNOWN (never established) gate makes every
# router/expert index suspect, so these methods are BLOCKED (typed blocker), not
# merely confidence-downgraded. Analysis/planning methods run in-repo and are
# not blocked on this gate.
_ROUTING_DEPENDENT = frozenset(
    spec.method for spec in METHOD_CATALOG if spec.routing_dependent
)

# Evidence keys produced by the real V3 pipeline → canonical policy stage names.
# V3 run evidence (run.evidence: stage -> kind) names the NVFP4 suitability key
# "nvfp4"; the policy consumes it as "nvfp4_suitability". Unknown keys are never
# silently dropped — they fall through unchanged (unknown stays unknown).
_EVIDENCE_ALIASES = {
    "nvfp4": "nvfp4_suitability",
}

_PRUNE_CAP = "pruning"
# versions that mean "not pinned to a real release" — a blocker for execution.
_UNPINNED_VERSIONS = frozenset({"", "n/a", "unpinned", "needs-pin", "none"})

# Strength of an evidence source: measured/causally_tested > estimated >
# predicted > inferred. Used to keep the strongest observed claim when an
# alias and the canonical key both address the same stage.
_SRC_PRECEDENCE = {
    EvidenceKind.CAUSALLY_TESTED.value: 0,
    EvidenceKind.MEASURED.value: 0,
    EvidenceKind.ESTIMATED.value: 1,
    EvidenceKind.PREDICTED.value: 2,
    EvidenceKind.INFERRED.value: 3,
}


# Declared qualitative tiers used ONLY for RANKING recommended methods — never
# as invented "fit" metrics. Evidence coverage and the memory target are
# partitioned into coarse, explicitly declared bands; ordering then prefers
# higher-confidence, better-covered methods, and (under TIGHT memory pressure)
# methods that actively reduce memory. A stable method-id rank breaks all ties.
_COVERAGE_HIGH_BAND = 0.7
_COVERAGE_ADEQUATE_BAND = 0.4
_MEM_TIGHT_GIB = 96.0
_MEM_RELAXED_GIB = 144.0


def _coverage_band(coverage: float) -> int:
    """Declared qualitative partition of mean evidence coverage (0..1): high /
    adequate / low. Used only for ranking order, not to fabricate metrics."""
    if coverage >= _COVERAGE_HIGH_BAND:
        return 0
    if coverage >= _COVERAGE_ADEQUATE_BAND:
        return 1
    return 2


def _memory_pressure(memory_target_gib: float) -> str:
    """Declared qualitative pressure from the memory target: tight / standard /
    relaxed. Used only to bias ranking under tight budgets."""
    if memory_target_gib <= _MEM_TIGHT_GIB:
        return "tight"
    if memory_target_gib >= _MEM_RELAXED_GIB:
        return "relaxed"
    return "standard"


def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(v)


def _is_pinned(version: str) -> bool:
    return version.strip().lower() not in _UNPINNED_VERSIONS


def canonical_stage(key: str) -> str:
    """Map a produced evidence key (e.g. V3's ``nvfp4``) to the canonical
    policy stage name. Unknown keys are returned unchanged."""
    return _EVIDENCE_ALIASES.get(key, key)


def _stage_keys(canonical: str) -> frozenset[str]:
    """All evidence keys (canonical + aliases) that satisfy a policy stage."""
    keys = {canonical}
    keys.update(a for a, c in _EVIDENCE_ALIASES.items() if c == canonical)
    return frozenset(keys)


class RecommendationPolicy:
    """Versioned deterministic recommendation policy."""

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------- API
    def recommend(
        self,
        profile: AtlasProfile,
        target: RecTarget,
        *,
        allow_pruning: bool = False,
        memory_target_gib: float | None = None,
    ) -> Recommendation:
        """Deterministic ranking. Missing evidence reduces confidence/blocks;
        never invents metrics."""
        if memory_target_gib is not None:
            target = RecTarget(
                hardware_model_arch=target.hardware_model_arch,
                compute_arch=target.compute_arch,
                topology=target.topology,
                runtime_backend=target.runtime_backend,
                memory_target_gib=memory_target_gib,
            )
        no_pruning = not self._pruning_permitted(allow_pruning)  # no_pruning defaults True
        methods: list[MethodRecommendation] = []
        blocked: list[MethodRecommendation] = []
        for method, stage_ids in _METHOD_STAGES.items():
            rec = self._score(profile, target, method, stage_ids, no_pruning)
            (blocked if rec.blockers else methods).append(rec)
        methods.sort(key=lambda r: self._ordering_sort_key(r, profile, target))
        blocked.sort(key=lambda r: r.method)
        # overall confidence = min over non-blocked recommended methods (plus
        # profile coverage), INSUFFICIENT if any critical stage evidence missing
        conf = self._overall_confidence(methods, profile)
        rid = self._recommendation_id(profile, target, no_pruning)
        return Recommendation(
            recommendation_id=rid,
            policy_version=RECOMMENDATION_POLICY_VERSION,
            profile_id=profile.profile_id_of(),
            target=target,
            no_pruning=no_pruning,
            methods=tuple(methods),
            blocked_methods=tuple(blocked),
            confidence=conf,
            summary=_summarize(conf, no_pruning),
        )

    # --------------------------------------------------------- scoring
    def _score(
        self,
        profile: AtlasProfile,
        target: RecTarget,
        method: str,
        stage_ids: list[str],
        no_pruning: bool,
    ) -> MethodRecommendation:
        backend = self._backend_for(method)
        blockers: list[RecBlock] = []
        # Backend resolution: an UNREGISTERED backend id blocks as backend_missing.
        if backend is None:
            if method not in _ANALYSIS_METHODS:
                blockers.append(RecBlock("backend_missing", f"no backend registered for {method}"))
            backend = {
                "backend_id": method,
                "available": False,
                "derivative": False,
                "status": "missing",
                "pinned": False,
            }
        # backend availability: only COMPRESSION methods require the backend to
        # be available + derivative-producing (analysis/planning run in-repo)
        if method not in _ANALYSIS_METHODS:
            if not backend["available"]:
                blockers.append(
                    RecBlock("backend_unavailable", f"{backend['backend_id']} unavailable")
                )
            if not backend["derivative"]:
                blockers.append(
                    RecBlock(
                        "not_derivative_producer",
                        f"{backend['backend_id']} probe-only; no derivative",
                    )
                )
            if not backend.get("pinned", True):
                blockers.append(
                    RecBlock(
                        "backend_unpinned",
                        f"{backend['backend_id']} not version-pinned; cannot execute",
                    )
                )
        # router-dependent compression methods additionally require the
        # routing-consistency identity gate to have PASSED. A failed OR unknown
        # (never established) gate means router/expert indices may be stale, so
        # the method is BLOCKED with a typed blocker — evidence danger, not just
        # a confidence nuance. Analysis/planning methods are not gated here.
        if profile.routing_consistency_passed is not True and method in _ROUTING_DEPENDENT:
            blockers.append(
                RecBlock(
                    "routing_consistency_failed",
                    "routing-consistency not PASSED; router-indexed expert "
                    "evidence cannot be trusted",
                )
            )
        # evidence: missing evidence for any required stage -> LOW/INSUFFICIENT
        # confidence, and if the default policy requires it, BLOCKED.
        missing: list[str] = []
        for canonical in stage_ids:
            ev = self._evidence_for(profile, canonical)
            if ev is None or not ev.present:
                missing.append(canonical)
        evidence_refs: list[str] = []
        for canonical in stage_ids:
            if self._evidence_for(profile, canonical) is not None:
                evidence_refs.append(canonical)
        protected = ["attention", "mla", "norms", "embedding", "lm_head", "router"]
        if missing:
            rec = MethodRecommendation(
                method=method,
                rank=99,
                reason="missing evidence for required stage(s); confidence "
                "INSUFFICIENT and decision blocked until re-profiled",
                evidence_refs=evidence_refs,
                confidence=RecConfidence.INSUFFICIENT,
                confidence_text=f"missing: {sorted(missing)}",
                blockers=[
                    *blockers,
                    RecBlock("missing_evidence", f"missing evidence: {sorted(missing)}"),
                ],
                status=RecipeStatus.DISCOVERED,
                backend_id=backend["backend_id"] if backend else "",
                requires_stages=stage_ids,
                protected_regions=protected,
            )
            return rec
        rank = self._rank_for(method)
        reason, conf, ctext = self._reason(method, profile, backend)
        _status_raw = backend["status"] if backend else RecipeStatus.DISCOVERED.value
        try:
            _status = RecipeStatus(_status_raw)
        except ValueError:
            _status = RecipeStatus.DISCOVERED
        return MethodRecommendation(
            method=method,
            rank=rank,
            reason=reason,
            evidence_refs=evidence_refs,
            expected_memory_direction=self._mem_dir(method),
            expected_quality_direction="down",
            confidence=conf,
            confidence_text=ctext,
            blockers=blockers,
            status=_status,
            backend_id=backend["backend_id"] if backend else method,
            requires_stages=stage_ids,
            protected_regions=protected,
        )

    # --------------------------------------------------------- helpers
    def _ordering_sort_key(
        self, rec: MethodRecommendation, profile: AtlasProfile, target: RecTarget
    ) -> tuple[Any, ...]:
        """Deterministic, declared-qualitative ordering key for RECOMMENDED
        (non-blocked) methods.

        Uses ONLY declared qualitative pressure:
          * evidence coverage band (high > adequate > low),
          * the method's own confidence (HIGH > MEDIUM > LOW) — the policy's
            declared measure of evidence strength,
          * memory direction bias only under tight pressure (memory-reducing
            methods rank ahead; only a coarse qualitative tie-break),
          * the policy's stable method-id rank as the final definite tie-break.

        No invented/estimated per-method fit metric enters the decision; these
        are coarse declared tiers over already-computed policy evidence and an
        explicit user memory target.
        """
        pressure = _memory_pressure(target.memory_target_gib)
        band = _coverage_band(self._coverage(profile))
        conf = _C_RANK[rec.confidence]
        if pressure == "tight":
            # under tight memory, memory-reducing methods rank FIRST; coverage
            # and the stable method-id rank still break ties.
            mem = 0 if self._mem_dir(rec.method) == "down" else 1
            return (band, mem, conf, self._rank_for(rec.method))
        return (band, conf, self._rank_for(rec.method))

    def _backend_for(self, method: str) -> dict[str, Any] | None:
        bid = method_spec(method).backend_id
        rec = self._registry.get(bid)
        if rec is None:
            return None
        return {
            "backend_id": bid,
            "available": rec.is_available(self._registry),
            "derivative": rec.produces_derivative,
            "status": rec.status.value,
            "version": rec.version,
            "pinned": _is_pinned(rec.version),
        }

    def _evidence_for(self, profile: AtlasProfile, canonical: str) -> StageEvidence | None:
        """Resolve a policy stage's evidence, honoring aliases and giving the
        default the broadest legal key set (identity stage is optional)."""
        if canonical == "identity":
            return profile.evidence.get("identity") or profile.evidence.get("teacher_identity")
        for key in _stage_keys(canonical):
            if key in profile.evidence:
                return profile.evidence[key]
        return None

    def _pruning_permitted(self, allow_pruning: bool) -> bool:
        """Pruning stays forbidden unless the caller EXPLICITLY requests it AND
        a separately-registered, available, version-pinned, derivative-producing
        pruning-capable backend exists. No verified backend -> never permitted."""
        if not allow_pruning:
            return False
        for rec in self._registry.names():
            record = self._registry.get(rec)
            if record is None:
                continue
            if _PRUNE_CAP not in record.declared_capabilities:
                continue
            if not record.produces_derivative:
                continue
            if not _is_pinned(record.version):
                continue
            if record.is_available(self._registry):
                return True
        return False

    def _rank_for(self, method: str) -> int:
        return method_spec(method).priority

    def _mem_dir(self, method: str) -> str:
        return method_spec(method).memory_direction

    def _reason(
        self,
        method: str,
        profile: AtlasProfile,
        backend: dict[str, Any] | None,
    ) -> tuple[str, RecConfidence, str]:
        evidence_count = len(profile.evidence)
        # routing_consistency is a hard gate: if it FAILED, any recommendation
        # relying on router-indexed evidence is INSUFFICIENT (indices could be
        # stale), regardless of per-stage kind.
        if profile.routing_consistency_passed is False:
            return (
                f"{method}: routing-consistency FAILED; router-dependent evidence "
                "cannot be trusted",
                RecConfidence.INSUFFICIENT,
                "routing_consistency_passed=false",
            )
        coverage = self._coverage(profile)
        if backend and backend["status"] in {"validated", "recommended"}:
            return (
                f"{method}: evidence-backed + backend {backend['backend_id']} "
                f"validated/recommended (coverage {coverage:.2f})",
                RecConfidence.HIGH,
                f"validated backend; coverage {coverage:.2f}",
            )
        if evidence_count >= 5 and coverage >= 0.5:
            return (
                f"{method}: rich profile evidence ({evidence_count} stages, "
                f"coverage {coverage:.2f})",
                RecConfidence.MEDIUM,
                f"{evidence_count} evidence stages; coverage {coverage:.2f}",
            )
        return (
            f"{method}: limited evidence ({evidence_count} stages, coverage {coverage:.2f})",
            RecConfidence.LOW,
            f"only {evidence_count} stages; coverage {coverage:.2f}",
        )

    def _overall_confidence(
        self, methods: list[MethodRecommendation], profile: AtlasProfile
    ) -> RecConfidence:
        if not methods:
            return RecConfidence.INSUFFICIENT
        if profile.routing_consistency_passed is False:
            return RecConfidence.INSUFFICIENT
        worst = min((m.confidence for m in methods), key=lambda c: _C_RANK[c])
        if worst in (RecConfidence.HIGH, RecConfidence.MEDIUM):
            if len(profile.evidence) < 4 or self._coverage(profile) < 0.4:
                return RecConfidence.LOW
            return worst
        return worst

    def _coverage(self, profile: AtlasProfile) -> float:
        """Mean calibration coverage over present evidence stages (0 when none;
        never invents coverage)."""
        covs = [
            e.coverage for e in profile.evidence.values() if e.present and e.coverage is not None
        ]
        if not covs:
            return 0.0
        return sum(covs) / len(covs)

    def _recommendation_id(self, profile: AtlasProfile, target: RecTarget, no_pruning: bool) -> str:
        payload = canonical_json(
            {
                "policy": RECOMMENDATION_POLICY_VERSION,
                "method_catalog": method_catalog_digest(),
                "profile": profile.profile_id_of(),
                "target": {
                    "model_arch": target.hardware_model_arch,
                    "compute": target.compute_arch,
                    "topology": target.topology,
                    "runtime": target.runtime_backend,
                    "memory_gib": target.memory_target_gib,
                },
                "no_pruning": no_pruning,
            }
        )
        return "rec-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


_C_RANK = {
    RecConfidence.HIGH: 0,
    RecConfidence.MEDIUM: 1,
    RecConfidence.LOW: 2,
    RecConfidence.INSUFFICIENT: 3,
}


def _summarize(conf: RecConfidence, no_pruning: bool) -> str:
    return (
        f"Deterministic policy recommendation. no_pruning={no_pruning}; overall "
        f"confidence {conf.value}. Missing evidence reduces confidence and blocks "
        "decisions; predictions are never reported as measured."
    )
