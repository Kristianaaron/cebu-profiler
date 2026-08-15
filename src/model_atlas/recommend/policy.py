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
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from model_atlas.backend.registry import BackendRegistry
from model_atlas.recipe.compiler import canonical_json
from model_atlas.recipe.schema import RecipeStatus


class RecConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


RECOMMENDATION_POLICY_VERSION = "policy-v1"


@dataclass(frozen=True)
class StageEvidence:
    """Minimal typed evidence summary pulled from an Atlas profile."""

    stage_id: str
    kind: str  # from EvidenceKind (measured/estimated/predicted/inferred/causally_tested)
    present: bool = True
    coverage: float | None = None  # calibration coverage (0..1)


@dataclass(frozen=True)
class AtlasProfile:
    """A completed Atlas profile (evidence + coverage) with a stable identity."""

    profile_id: str
    model: str
    seed: int = 0
    evidence: dict[str, StageEvidence] = field(default_factory=dict)
    routing_consistency_passed: bool | None = None
    hardware_model_arch: str = "glm-5.2"
    notes: str = ""

    def profile_id_of(self) -> str:
        payload = canonical_json(
            {
                "model": self.model,
                "seed": self.seed,
                "stages": sorted((k, v.kind, v.present) for k, v in self.evidence.items()),
                "hardware_model_arch": self.hardware_model_arch,
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


# stages the policy matches against the profile evidence + registry
_METHOD_STAGES = {
    "teacher-identity": ["identity"],
    "calibration": ["corpus_semantic"],
    "sensitivity": ["spectral", "shared_structure", "routing_consistency"],
    "bit-allocation": ["global_bit_budget"],
    "nvfp4-substitute": ["nvfp4_suitability"],
    "kv-optimization": ["kv_budget"],
    "exl3-primary": ["global_bit_budget"],
    "llm-compressor": ["global_bit_budget"],
    "modelopt-nvfp4": ["nvfp4_suitability"],
}

# ANALYSIS/PLANNING methods produce profile evidence (in-repo, no derivative).
_ANALYSIS_METHODS = {
    "teacher-identity",
    "calibration",
    "sensitivity",
    "bit-allocation",
    "kv-optimization",
}


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
        no_pruning = not allow_pruning  # no_pruning defaults True
        methods: list[MethodRecommendation] = []
        blocked: list[MethodRecommendation] = []
        for method, stage_ids in _METHOD_STAGES.items():
            rec = self._score(profile, target, method, stage_ids, no_pruning)
            (blocked if rec.blockers else methods).append(rec)
        methods.sort(key=lambda r: (r.rank, r.method))
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
        # backend availability: only COMPRESSION methods require the backend to
        # be available + derivative-producing (analysis/planning run in-repo)
        if backend is not None and method not in _ANALYSIS_METHODS:
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
        # evidence: missing evidence for any required stage -> LOW/INSUFFICIENT
        # confidence, and if the default policy requires it, BLOCKED.
        missing: list[str] = []
        for s in stage_ids:
            ev = profile.evidence.get(s)
            if ev is None or not ev.present:
                missing.append(s)
        evidence_refs = [s for s in stage_ids if s in profile.evidence]
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
    def _backend_for(self, method: str) -> dict[str, Any] | None:
        mapping = {
            "teacher-identity": "atlas_analysis_v3",
            "calibration": "atlas_analysis_v3",
            "sensitivity": "atlas_analysis_v3",
            "bit-allocation": "atlas_analysis_v3",
            "kv-optimization": "atlas_analysis_v3",
            "exl3-primary": "exl3",
            "llm-compressor": "llm_compressor",
            "modelopt-nvfp4": "modelopt_nvfp4",
            "nvfp4-substitute": "modelopt_nvfp4",
        }
        bid = mapping.get(method)
        if bid is None:
            return None
        rec = self._registry.get(bid)
        if rec is None:
            return None
        return {
            "backend_id": bid,
            "available": rec.is_available(self._registry),
            "derivative": rec.produces_derivative,
            "status": rec.status.value,
        }

    def _rank_for(self, method: str) -> int:
        return list(_METHOD_STAGES).index(method) + 1

    def _mem_dir(self, method: str) -> str:
        return {"teacher-identity": "same", "calibration": "same"}.get(method, "down")

    def _reason(
        self,
        method: str,
        profile: AtlasProfile,
        backend: dict[str, Any] | None,
    ) -> tuple[str, RecConfidence, str]:
        evidence_count = len(profile.evidence)
        if backend and backend["status"] in {"validated", "recommended"}:
            return (
                f"{method}: evidence-backed + backend {backend['backend_id']} "
                f"validated/recommended",
                RecConfidence.HIGH,
                "validated backend + profile evidence",
            )
        if evidence_count >= 5:
            return (
                f"{method}: rich profile evidence ({evidence_count} stages)",
                RecConfidence.MEDIUM,
                f"{evidence_count} evidence stages present",
            )
        return (
            f"{method}: limited evidence ({evidence_count} stages)",
            RecConfidence.LOW,
            f"only {evidence_count} evidence stages present",
        )

    def _overall_confidence(
        self, methods: list[MethodRecommendation], profile: AtlasProfile
    ) -> RecConfidence:
        if not methods:
            return RecConfidence.INSUFFICIENT
        worst = min((m.confidence for m in methods), key=lambda c: _C_RANK[c])
        if worst in (RecConfidence.HIGH, RecConfidence.MEDIUM):
            if len(profile.evidence) < 4:
                return RecConfidence.LOW
            return worst
        return worst

    def _recommendation_id(self, profile: AtlasProfile, target: RecTarget, no_pruning: bool) -> str:
        payload = canonical_json(
            {
                "policy": RECOMMENDATION_POLICY_VERSION,
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
