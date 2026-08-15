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
from model_atlas.schemas.evidence import EvidenceKind


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtlasProfile:
        """Build a profile from a real exported Atlas run dict.

        Evidence values may be a V3 ``dict`` (``{"kind": ..., "present": ...,
        "coverage": ...}``) OR a plain string kind; both are parsed into
        ``StageEvidence``. Evidence keys are normalized through the alias map.
        """
        evidence: dict[str, StageEvidence] = {}
        for k, v in (data.get("evidence") or {}).items():
            st = cls._parse_evidence(k, v)
            canonical = canonical_stage(k)
            if canonical in evidence and st.present:
                prev = evidence[canonical]
                if _SRC_PRECEDENCE.get(st.kind, 5) <= _SRC_PRECEDENCE.get(prev.kind, 5):
                    continue  # keep the strongest observed claim
            evidence[canonical] = st
        return cls(
            profile_id=str(data.get("profile_id", "imported")),
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
            notes=str(data.get("notes", "")),
        )

    @staticmethod
    def _parse_evidence(key: str, v: Any) -> StageEvidence:
        present = True
        coverage: float | None = None
        kind = "estimated"
        if isinstance(v, str):
            kind = v
        elif isinstance(v, dict):
            kind = str(v.get("kind", "estimated"))
            present = bool(v.get("present", True))
            cov = v.get("coverage")
            if isinstance(cov, (int, float)) and not isinstance(cov, bool):
                coverage = float(cov)
        else:
            # non-string/non-dict evidence value: refuse to guess — treat as
            # absent rather than invent a claim (unknown stays unknown).
            return StageEvidence(stage_id=key, kind=kind, present=False, coverage=None)
        return StageEvidence(
            stage_id=key,
            kind=kind,
            present=present,
            coverage=coverage,
        )

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

# method -> registered backend id (all are registered; an absent registration
# surfaces as backend_missing instead of silently recommending).
_BACKEND_ALIASES = {
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
    def _backend_for(self, method: str) -> dict[str, Any] | None:
        bid = _BACKEND_ALIASES.get(method)
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
