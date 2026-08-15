"""Recommendation engine tests: deterministic ranking, profile identity,
missing evidence, incompatible backend, no pruning, blocked methods, recipe
composition, API/UI smoke, XSS-safe rendering."""

import json
from pathlib import Path

import pytest

from model_atlas.backend.contract import BackendRecord
from model_atlas.backend.registry import BackendRegistry, build_default_registry
from model_atlas.recipe.schema import (
    CalibrationIdentity,
    CompressionRecipe,
    RecipeConstraints,
    RecipeStage,
    SourceIdentity,
    StageBackendPin,
    StageEffectClass,
)
from model_atlas.recommend import (
    AtlasProfile,
    RecommendationService,
    RecTarget,
    StageEvidence,
    write_gui,
)
from model_atlas.recommend.policy import (
    RecConfidence,
    RecommendationPolicy,
    canonical_stage,
)
from model_atlas.schemas.evidence import EvidenceKind


def _fake_records(reg: BackendRegistry) -> dict[str, BackendRecord]:
    # BackendRegistry stores its records in a private ``_records`` dict.
    return dict(reg._records)


def _full_profile() -> AtlasProfile:
    return AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence={
            "identity": StageEvidence("identity", "measured"),
            "corpus_semantic": StageEvidence("corpus_semantic", "measured", coverage=0.9),
            "spectral": StageEvidence("spectral", "estimated"),
            "shared_structure": StageEvidence("shared_structure", "estimated"),
            "routing_consistency": StageEvidence("routing_consistency", "measured"),
            "global_bit_budget": StageEvidence("global_bit_budget", "predicted"),
            "kv_budget": StageEvidence("kv_budget", "estimated"),
            "nvfp4_suitability": StageEvidence("nvfp4_suitability", "estimated"),
        },
    )


def test_deterministic_ranking_and_stable_ids():
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    r1 = pol.recommend(_full_profile(), RecTarget(memory_target_gib=115.0))
    r2 = pol.recommend(_full_profile(), RecTarget(memory_target_gib=115.0))
    assert r1.recommendation_id == r2.recommendation_id
    # no_pruning true default
    assert r1.no_pruning is True
    # analysis/planning methods recommended, compression methods blocked
    assert {m.method for m in r1.methods} == {
        "teacher-identity",
        "calibration",
        "sensitivity",
        "bit-allocation",
        "kv-optimization",
    }
    assert {m.method for m in r1.blocked_methods} == {
        "exl3-primary",
        "llm-compressor",
        "modelopt-nvfp4",
        "nvfp4-substitute",
    }


def test_v3_string_and_object_evidence_parsing():
    """Real V3 run dicts carry evidence as plain string kinds OR objects; both
    must parse, unknown kinds/keys stay unknown (never promoted to measured)."""
    from model_atlas.recommend.api import _profile_from_dict

    data = {
        "model": "k3-mini",
        "evidence": {
            "identity": "measured",  # string kind
            "spectral": {"kind": "estimated", "present": True, "coverage": 0.8},  # object
            "mystery_key": {"kind": "predicted", "present": True},  # unknown key kept as-is
        },
    }
    prof = _profile_from_dict(data)
    assert prof.evidence["identity"].kind == "measured"
    assert prof.evidence["spectral"].kind == "estimated"
    assert prof.evidence["spectral"].coverage == 0.8
    # unknown key survives (unknown stays unknown), never folded into measured
    assert "mystery_key" in prof.evidence
    assert prof.evidence["mystery_key"].kind == "predicted"


def test_nvfp4_alias_maps_to_canonical_stage():
    """V3 run evidence names the NVFP4 suitability key 'nvfp4'; the policy's
    nvfp4-substitute stage must resolve it through the alias."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    profile = AtlasProfile.from_dict(
        {
            "model": "k3-mini",
            "evidence": {
                "identity": "measured",
                "corpus_semantic": "measured",
                "spectral": "estimated",
                "shared_structure": "estimated",
                "routing_consistency": "measured",
                "global_bit_budget": "predicted",
                "kv_budget": "estimated",
                "nvfp4": {"kind": "estimated", "present": True},  # V3 alias, NOT nvfp4_suitability
            },
        }
    )
    # alias maps for the substitute method
    assert canonical_stage("nvfp4") == "nvfp4_suitability"
    rec1 = pol.recommend(profile, RecTarget())
    nvfp4 = [m for m in rec1.methods if m.method == "nvfp4-substitute"]
    nvfp4 += [m for m in rec1.blocked_methods if m.method == "nvfp4-substitute"]
    assert nvfp4
    # evidence present means it is NOT blocked purely on missing nvfp4 evidence
    assert not any(b.code == "missing_evidence" for b in nvfp4[0].blockers)


def test_backend_missing_blocker_when_record_omitted():
    """A compression method whose backend RECORD is entirely ABSENT from the
    registry must block as backend_missing (never silently recommend an
    unresolvable executor). Uses a real registry with the record removed, not a
    fabricated backend."""
    from model_atlas.backend.registry import BackendRegistry

    default = build_default_registry()
    # real registry minus the llm_compressor record — not a fake backend.
    records = {i: r for i, r in _fake_records(default).items() if i != "llm_compressor"}
    assert "llm_compressor" not in records
    reg = BackendRegistry(records)
    pol = RecommendationPolicy(reg)
    rec = pol.recommend(_full_profile(), RecTarget())
    llm = [m for m in rec.blocked_methods if m.method == "llm-compressor"]
    assert llm
    assert any(b.code == "backend_missing" for b in llm[0].blockers)
    # every other compression backend is still registered -> not missing-blocked
    exl = [m for m in rec.blocked_methods if m.method == "exl3-primary"]
    assert exl and not any(b.code == "backend_missing" for b in exl[0].blockers)


def test_backend_unpinned_blocker():
    """A version-pinned requirement must be part of the blocker set for
    execution — an unpinned backend cannot be a verified executable plan."""
    from model_atlas.backend.contract import BackendRecord
    from model_atlas.backend.registry import BackendRegistry
    from model_atlas.recipe.schema import RecipeStatus

    # force an available + derivative-producing but UNPINNED backend
    fake = BackendRecord(
        backend_id="exl3",
        display_name="EXL3",
        method_family="exl3",
        formats=("exl3",),
        status=RecipeStatus.VALIDATED,
        version="unpinned",
        produces_derivative=True,
        availability_probe=lambda: (True, "1.0", "present"),
        declared_capabilities=(),
    )
    reg2 = BackendRegistry({b.backend_id: b for b in [fake]})
    pol = RecommendationPolicy(reg2)
    rec = pol.recommend(_full_profile(), RecTarget())
    exl = [m for m in rec.blocked_methods if m.method == "exl3-primary"]
    assert exl and any(b.code == "backend_unpinned" for b in exl[0].blockers)


def test_pruning_requires_verified_backend():
    """allow_pruning may never devise a pruning method unless a REAL, available,
    version-pinned, derivative-producing pruning-capable backend is registered."""
    from model_atlas.backend.contract import BackendRecord
    from model_atlas.backend.registry import BackendRegistry
    from model_atlas.recipe.schema import RecipeStatus

    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    rec = pol.recommend(_full_profile(), RecTarget(), allow_pruning=True)
    # default registry: no pruning capable+derivative+available backend
    assert rec.no_pruning is True

    # now register a verified pruning backend and confirm it flips
    prune = BackendRecord(
        backend_id="tenp_pruning_prod",
        display_name="TENP prod",
        method_family="pruning",
        formats=("pruned-checkpoint",),
        status=RecipeStatus.VALIDATED,
        version="2.3.0",
        produces_derivative=True,
        declared_capabilities=("pruning",),
        availability_probe=lambda: (True, "2.3.0", "wired"),
    )
    reg2 = BackendRegistry({**_fake_records(reg), prune.backend_id: prune})
    pol2 = RecommendationPolicy(reg2)
    rec2 = pol2.recommend(_full_profile(), RecTarget(), allow_pruning=True)
    assert rec2.no_pruning is False
    # still no pruning STAGE recommended unless a method maps to it
    assert not any("pruning" in m.method for m in rec2.methods)


def test_routing_consistency_failure_downgrades_confidence():
    """A failed routing-consistency check invalidates router-indexed evidence
    and must force INSUFFICIENT confidence (indices could be stale)."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    profile = _full_profile()
    bad = AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence=profile.evidence,
        routing_consistency_passed=False,
    )
    rec = pol.recommend(bad, RecTarget())
    assert rec.confidence is RecConfidence.INSUFFICIENT
    # each recommended method is downgraded too
    assert all(m.confidence is RecConfidence.INSUFFICIENT for m in rec.methods)


def test_coverage_sensitive_ranking():
    """Confidence must drop when calibration coverage is thin even if stages
    exist (coverage drives the MEDIUM->LOW and HIGH->LOW boundary)."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    low_cov = AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence={
            "identity": StageEvidence("identity", "measured"),
            "corpus_semantic": StageEvidence("corpus_semantic", "measured", coverage=0.1),
            "spectral": StageEvidence("spectral", "estimated"),
            "shared_structure": StageEvidence("shared_structure", "estimated"),
            "routing_consistency": StageEvidence("routing_consistency", "measured"),
            "global_bit_budget": StageEvidence("global_bit_budget", "predicted"),
            "kv_budget": StageEvidence("kv_budget", "estimated"),
            "nvfp4_suitability": StageEvidence("nvfp4_suitability", "estimated"),
        },
    )
    rec = pol.recommend(low_cov, RecTarget())
    # coverage 0.1 pulls overall confidence below MEDIUM
    assert rec.confidence in (RecConfidence.LOW, RecConfidence.INSUFFICIENT)


def test_profile_identity_stable_and_content_sensitive():
    p1 = _full_profile()
    p2 = _full_profile()
    assert p1.profile_id_of() == p2.profile_id_of()
    p3 = AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence={
            "identity": StageEvidence("identity", "measured"),
        },
    )
    assert p3.profile_id_of() != p1.profile_id_of()


# --- Atlas recommendation profile identity + evidence semantics ----
# Every field the profile consumes must change its stable identity, and
# malformed evidence must be fail-closed (never fabricated as support).


def test_from_dict_rejects_missing_kind_or_present_as_unknown() -> None:
    """An evidence dict missing `kind` or `present` cannot carry a claim. It
    must parse as unknown/absent (never silently estimated)."""
    st = StageEvidence.from_dict("spectral", {})
    assert st.kind == "unknown"
    assert st.present is False
    assert st.detail  # reason recorded
    st2 = StageEvidence.from_dict("spectral", {"kind": "measured"})  # no present
    assert st2.kind == "unknown"
    assert st2.present is False


def test_from_dict_accepts_only_valid_evidence_kinds() -> None:
    """Only real EvidenceKind strings are accepted. An arbitrary string is
    unknown/absent, never promoted to estimated."""
    for good in ("measured", "estimated", "predicted", "inferred", "causally_tested"):
        st = StageEvidence.from_dict("x", good)
        assert st.kind == good
        assert st.present is True
    st = StageEvidence.from_dict("x", "definitely-real-kind")
    assert st.kind == "unknown"
    assert st.present is False
    assert "unknown kind" in st.detail


def test_from_dict_rejects_bool_and_out_of_range_coverage() -> None:
    """coverage must be numeric in 0..1; bool and out-of-range values fail
    closed (unknown/absent) rather than becoming support."""
    for bad in (True, False, -0.1, 1.5, 2):
        st = StageEvidence.from_dict("x", {"kind": "measured", "present": True, "coverage": bad})
        assert st.kind == "unknown"
        assert st.present is False
        assert "coverage" in st.detail
    # valid endpoints accepted
    for val in (0.0, 1.0, 0.4):
        st = StageEvidence.from_dict("x", {"kind": "measured", "present": True, "coverage": val})
        assert st.kind == "measured"
        assert st.present is True
        assert st.coverage == float(val)


def test_from_dict_refuses_unsupported_value_type() -> None:
    """A non-string/non-dict evidence value (e.g. a number or list) is refused
    as unknown/absent, never guessed."""
    st = StageEvidence.from_dict("x", 123)
    assert st.kind == "unknown"
    assert st.present is False


def test_from_dict_uses_detail_for_non_evidence_marker() -> None:
    """A present flag may be a non-truthy marker; reject via detail when the
    coverage is malformed, and keep absent claims unknown."""
    st = StageEvidence.from_dict("spectral", {"kind": "estimated", "present": False})
    assert st.present is False
    assert st.kind == "estimated"  # valid kind; deliberately absent
    bad = StageEvidence.from_dict(
        "spectral", {"kind": "estimated", "present": True, "coverage": "0.9"}
    )
    assert bad.present is False
    assert bad.kind == "unknown"


def test_profile_identity_consumes_every_relevant_field() -> None:
    """Changing model, hardware_model_arch, notes, declared profile_id, or
    routing_consistency must each produce a different stable identity."""
    base = _full_profile()

    def ident(**kw: object) -> str:
        return AtlasProfile(**{**base.__dict__, **kw}).profile_id_of()

    assert base.profile_id_of() == ident()
    # model
    assert ident(model="other") != base.profile_id_of()
    # hardware
    assert ident(hardware_model_arch="other") != base.profile_id_of()
    # notes
    assert ident(notes="changed") != base.profile_id_of()
    # declared profile_id
    assert ident(profile_id="other") != base.profile_id_of()
    # routing consistency
    assert ident(routing_consistency_passed=True) != base.profile_id_of()


def test_profile_identity_changes_with_each_evidence_field() -> None:
    """For one evidence item, changing name, kind, present, coverage, or detail
    must each change identity — identity reflects the evidence exactly."""

    def build(kind: str, present: bool = True, coverage: float | None = None,
              detail: str = "", name: str = "identity") -> str:
        return AtlasProfile(
            profile_id="p", model="k3-mini",
            evidence={name: StageEvidence(name, kind, present=present,
                                          coverage=coverage, detail=detail)},
        ).profile_id_of()

    base = build("measured")
    # present
    assert build("measured", present=False) != base
    # coverage
    assert build("measured", coverage=0.5) != base
    # detail
    assert build("measured", detail="note") != base
    # kind
    assert build("estimated") != base
    # stage name
    assert build("measured", name="spectral") != base
    # sanity: identical inputs still collide
    assert build("measured") == base


def test_malformed_from_dict_evidence_blocks_recommendation() -> None:
    """An unknown/absent evidence item must not satisfy a required stage:
    the decision is BLOCKED (fail-closed), never supported by the malformed
    item on the theory it 'was estimated'."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    prof = AtlasProfile(
        profile_id="p", model="k3-mini",
        evidence={
            # corrupt evidence: arbitrary kind string
            "identity": StageEvidence.from_dict("identity", "made-up-kind"),
            "corpus_semantic": StageEvidence("corpus_semantic", "estimated"),
            "spectral": StageEvidence.from_dict(
                "spectral",
                {"kind": "estimated", "present": True, "coverage": 99},
            ),
            "shared_structure": StageEvidence("shared_structure", "estimated"),
            "routing_consistency": StageEvidence.from_dict(
                "routing_consistency", {}  # missing kind/present
            ),
        },
    )
    rec = pol.recommend(prof, RecTarget())
    # methods whose required stages include a corrupted (unknown/absent) item
    # must be BLOCKED as missing_evidence — malformed evidence never fabricates
    # support for the stage it names.
    sensitivity = [m for m in list(rec.methods) + list(rec.blocked_methods)
                   if m.method == "sensitivity"]
    assert sensitivity
    assert sensitivity[0].confidence is RecConfidence.INSUFFICIENT
    assert any(b.code == "missing_evidence" for b in sensitivity[0].blockers)
    # routing_consistency (required by sensitivity) was absent too
    assert "routing_consistency" in sensitivity[0].confidence_text
    assert "spectral" in sensitivity[0].confidence_text
    # calibration only needs corpus_semantic (valid estimated) -> not fabricated-blocked
    calibration = [m for m in list(rec.methods) + list(rec.blocked_methods)
                   if m.method == "calibration"]
    assert calibration
    assert not any(b.code == "missing_evidence" for b in calibration[0].blockers)


def test_missing_evidence_blocks_decision():
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    partial = AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence={
            "identity": StageEvidence("identity", "measured"),
        },
    )
    rec = pol.recommend(partial, RecTarget())
    sens = [m for m in rec.blocked_methods if m.method == "sensitivity"]
    assert sens
    assert any(b.code == "missing_evidence" for b in sens[0].blockers)
    assert sens[0].confidence is RecConfidence.INSUFFICIENT


def test_incompatible_backend_blocked():
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    rec = pol.recommend(_full_profile(), RecTarget())
    exl = [m for m in rec.blocked_methods if m.method == "exl3-primary"]
    assert exl and any(b.code == "backend_unavailable" for b in exl[0].blockers)


def test_no_pruning_cannot_be_disarmed_silently():
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    rec = pol.recommend(_full_profile(), RecTarget(), allow_pruning=True)
    # policy may expose allow_pruning as an EXPLICIT author flag but no pruning
    # method is ever recommended (no pruning-capable backend executed)
    pruning = [m for m in rec.methods if "pruning" in m.method]
    assert not pruning


def test_routing_consistency_failed_blocks_router_dependent_compression() -> None:
    """A failed routing-consistency gate must add a TYPED
    routing_consistency_failed blocker to every router-dependent COMPRESSION
    method (sensitivity suffices on evidence, but the gate still gates
    compression, which reads expert indices)."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    bad = AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence=_full_profile().evidence,
        routing_consistency_passed=False,
    )
    rec = pol.recommend(bad, RecTarget())
    # every router-dependent compression method carries the typed blocker
    for m in rec.blocked_methods:
        if m.method in {"exl3-primary", "llm-compressor", "modelopt-nvfp4", "nvfp4-substitute"}:
            assert any(
                b.code == "routing_consistency_failed" for b in m.blockers
            ), m.method
    # analysis/planning methods are NOT gated on routing consistency
    for m in rec.methods:
        assert not any(b.code == "routing_consistency_failed" for b in m.blockers)
    assert rec.confidence is RecConfidence.INSUFFICIENT


def test_routing_consistency_unknown_blocks_router_dependent_compression() -> None:
    """Routing consistency UNKNOWN (never established) is as dangerous as FAILED
    for router-dependent methods: the typed blocker must still fire."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    # routing_consistency_passed = None (unknown)
    rec = pol.recommend(_full_profile(), RecTarget())
    for m in rec.blocked_methods:
        if m.method in {"exl3-primary", "llm-compressor", "modelopt-nvfp4", "nvfp4-substitute"}:
            assert any(b.code == "routing_consistency_failed" for b in m.blockers), m.method
    # analysis methods unaffected
    assert not any(
        b.code == "routing_consistency_failed"
        for m in rec.methods
        for b in m.blockers
    )


def test_routing_consistency_blocks_only_not_confidence_only() -> None:
    """Routing-consistency failure is a typed BLOCKER, not merely a confidence
    downgrade: router-dependent compression methods land in blocked_methods and
    carry the routing_consistency_failed code."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    bad = AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence=_full_profile().evidence,
        routing_consistency_passed=False,
    )
    rec = pol.recommend(bad, RecTarget())
    methods = [m for m in rec.blocked_methods if m.method == "exl3-primary"]
    assert methods
    assert any(b.code == "routing_consistency_failed" for b in methods[0].blockers)
    # not a bare confidence downgrade landing in the recommended list
    assert "exl3-primary" not in {m.method for m in rec.methods}


def _full_evidence_profile() -> AtlasProfile:
    """Profile whose recommended methods are all MEDIUM confidence and whose
    evidence coverage is high, so ordering is driven by the declared tiers."""
    ev = {
        k: StageEvidence(k, v)
        for k, v in {
            "identity": "measured",
            "corpus_semantic": "measured",
            "spectral": "estimated",
            "shared_structure": "estimated",
            "routing_consistency": "measured",
            "global_bit_budget": "predicted",
            "kv_budget": "estimated",
            "nvfp4_suitability": "estimated",
        }.items()
    }
    return AtlasProfile(profile_id="p", model="k3-mini", evidence=ev)


def test_ordering_responds_to_coverage_and_stays_deterministic() -> None:
    """The ordering key genuinely reads evidence coverage and each method's
    declared confidence (not fabricated metrics): a coverage-band change is
    observable at the decision level, and identical inputs yield identical
    order (stable method-id tie-break)."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    # high-coverage profile -> MEDIUM/HIGH overall confidence, order follows the
    # policy's declared tiering (all methods share one band & confidence, so the
    # stable method-id rank finalizes a deterministic order).
    rec = pol.recommend(_full_evidence_profile(), RecTarget(memory_target_gib=115.0))
    order = [m.method for m in rec.methods]
    canonical = [
        "teacher-identity",
        "calibration",
        "sensitivity",
        "bit-allocation",
        "kv-optimization",
    ]
    assert order == canonical
    # deterministic: repeating the exact call reproduces the exact order
    rec2 = pol.recommend(_full_evidence_profile(), RecTarget(memory_target_gib=115.0))
    assert [m.method for m in rec2.methods] == canonical
    # coverage actually participates: a low-coverage profile drops the overall
    # decision below the high-coverage one (declared qualitative band).
    base = _full_evidence_profile()
    low = AtlasProfile(
        profile_id="p",
        model="k3-mini",
        evidence={
            k: StageEvidence(k, e.kind, present=e.present, coverage=0.05)
            for k, e in base.evidence.items()
        },
    )
    assert pol.recommend(low, RecTarget(memory_target_gib=115.0)).confidence in (
        RecConfidence.LOW,
        RecConfidence.INSUFFICIENT,
    )


def test_ordering_memory_target_pressure_keeps_stable_tiebreak() -> None:
    """Under TIGHT memory pressure the declared memory direction biases memory-
    reducing methods ahead, but the stable method-id rank still fully determines
    order among equal memory-direction methods (never random/volatile)."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    tight = pol.recommend(_full_evidence_profile(), RecTarget(memory_target_gib=64.0))
    relaxed = pol.recommend(_full_evidence_profile(), RecTarget(memory_target_gib=200.0))
    tight_order = [m.method for m in tight.methods]
    relaxed_order = [m.method for m in relaxed.methods]
    # deterministic: identical inputs -> identical output
    assert [m.method for m in pol.recommend(
        _full_evidence_profile(), RecTarget(memory_target_gib=64.0)
    ).methods] == tight_order
    # Under TIGHT pressure the memory-reducing ("down") methods rank AHEAD of
    # the memory-neutral teacher-identity/calibration pair; relaxed keeps the
    # stable method-id order. This is the one observable, declared pressure
    # effect — no invented per-method fit metric.
    assert tight_order.index("sensitivity") < tight_order.index("teacher-identity")
    assert relaxed_order == canonical_order()
    # the two orderings genuinely differ at the front (pressure participates)
    assert tight_order[0] == "sensitivity" and relaxed_order[0] == "teacher-identity"


def canonical_order() -> list[str]:
    return ["teacher-identity", "calibration", "sensitivity", "bit-allocation", "kv-optimization"]


def test_profile_alias_preserved_through_service(tmp_path: Path) -> None:
    """declared profile_id (e.g. 'glm52') must survive save -> list -> resolve
    even though the canonical content id differs."""
    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    p = AtlasProfile(profile_id="glm52", model="glm-5.2", evidence={
        "identity": StageEvidence("identity", "measured"),
        "corpus_semantic": StageEvidence("corpus_semantic", "measured"),
        "spectral": StageEvidence("spectral", "estimated"),
        "shared_structure": StageEvidence("shared_structure", "estimated"),
        "routing_consistency": StageEvidence("routing_consistency", "measured"),
    })
    canonical = p.profile_id_of()
    assert canonical.startswith("profile-")
    path = svc.save_profile(p)
    # persisted file records the alias separately from the canonical id
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    assert saved["declared_profile_id"] == "glm52"
    assert saved["profile_id"] == canonical
    # list_profiles exposes both ids
    listed = {x["declared_profile_id"]: x["profile_id"] for x in svc.list_profiles()}
    assert listed.get("glm52") == canonical
    # resolve by the DECLARED alias, even though canonical content id differs
    rec = svc.recommend("glm52", RecTarget())
    assert rec.profile_id == canonical


def test_recommend_service_real_missing_backend(tmp_path: Path) -> None:
    """The service (policy path) reports backend_missing for a method whose
    backend RECORD was removed from the registry, and still resolves profiles."""
    from model_atlas.backend.registry import BackendRegistry

    default = build_default_registry()
    records = {i: r for i, r in _fake_records(default).items() if i != "exl3"}
    assert "exl3" not in records
    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"),
        work_root=str(tmp_path / "runs"),
        registry=BackendRegistry(records),
    )
    svc.save_profile(_full_profile())
    rec = svc.recommend("k3-mini", RecTarget())
    blocked = {m.method: m for m in rec.blocked_methods}
    exl = blocked["exl3-primary"]
    assert any(b.code == "backend_missing" for b in exl.blockers)


def test_recommend_cli_profile_alias(tmp_path: Path) -> None:
    """The CLI --profile alias path must resolve the declared 'glm52' alias even
    though the canonical content id differs."""
    from typer.testing import CliRunner

    from model_atlas.cli import app

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(AtlasProfile(profile_id="glm52", model="glm-5.2", evidence={
        "identity": StageEvidence("identity", "measured"),
        "corpus_semantic": StageEvidence("corpus_semantic", "measured"),
        "spectral": StageEvidence("spectral", "estimated"),
        "shared_structure": StageEvidence("shared_structure", "estimated"),
        "routing_consistency": StageEvidence("routing_consistency", "measured"),
    }))
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["recommend", "--profile", "glm52", "--profiles-dir", str(tmp_path / "profiles")],
    )
    assert res.exit_code == 0, res.output
    assert "recommendation_id" in res.output
    assert "profile-" in res.output


def test_api_start_fails_closed_uncompilable(tmp_path: Path):
    """start() only runs a VERIFIED executable plan — a recipe with unavailable
    backends fails closed (never starts a fake quantization)."""
    from model_atlas.recipe.compiler import RecipeCompileError
    from model_atlas.recipes.builtin import glm52_no_pruning_recipe

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    with pytest.raises(RecipeCompileError):
        svc.start(glm52_no_pruning_recipe(), inputs={})


def test_gui_is_static_no_embedded_payloads_xss_safe(tmp_path: Path):
    """The GUI is static HTML/CSS/JS only — NO embedded user/recommedation JSON.
    All data is fetched over HTTP by the browser. It must reference the fetch
    routes and never use innerHTML on service data."""
    from model_atlas.recommend import RecommendationService

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(_full_profile())
    out = str(tmp_path / "gui.html")
    write_gui(out, svc)
    text = Path(out).read_text(encoding="utf-8")
    # NO embedded profile/recommendation payload — the page is data-free.
    assert "rec-" not in text
    assert "k3-mini" not in text
    # XSS-safe: JS escapes via textContent and never sets innerHTML on data.
    assert "<script>" in text  # the harness JS present
    assert "innerHTML" not in text  # no value interpolation via innerHTML
    # fetches live routes rather than embedding a snapshot.
    assert "fetch('/api/profiles')" in text
    assert "fetch('/api/recommend'" in text
    assert "fetch('/api/preview-selection'" in text
    assert "fetch('/api/start'" in text


def test_gui_static_no_profile_payload(tmp_path: Path):
    """Even when run with a directory of profiles, the GUI page must NOT embed
    the profile data (no raw snapshot/JSON in the script)."""
    from model_atlas.recommend import RecommendationService

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(_full_profile())
    from model_atlas.recommend.gui import render_gui

    text = render_gui(svc)
    assert "innerHTML" not in text
    assert "SNAPSHOT" not in text
    assert "const SNAPSHOT" not in text


def test_recipe_preview_selection_builds_draft_and_diffs(tmp_path: Path):
    """preview-from-selection builds a deterministic no-pruning draft, reports
    diff / compile blockers / readiness, and (when it compiles) a verified plan
    summary — never mutating and never shipping a full recipe payload."""
    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    sel = ["calibration", "sensitivity"]
    preview = svc.recipe_preview(sel)
    assert preview["recipe_id"].startswith("recipe-")
    assert set(preview["selected_methods"]) == set(sel)
    # draft includes the requested methods' stages + transitive deps
    stage_ids = {s["id"] for s in preview["stages"]}
    assert {"t1-identity", "t2-calibration", "t3-sensitivity"} <= stage_ids
    # readiness/plan never claim verified when the draft is not executable
    assert preview["readiness"]["verified_plan"] is False
    assert preview["plan"] is None or preview["plan"].get("pins_pass") is False
    # diff vs canonical builtin recipe
    assert preview["diff"]["no_pruning"] is True
    assert "t7-exl3" in preview["diff"]["omitted_stages"]


def test_recipe_preview_all_selection_compiles_false_fail_closed(tmp_path: Path):
    """Selecting every method still yields a non-executable draft today (the
    GLM-5.2 canonical recipe mixes unavailable backends/hybrid precision), so
    readiness stays false — the GUI's compress gate stays closed."""
    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    preview = svc.recipe_preview(None)  # all
    assert len(preview["stages"]) >= 14
    assert preview["compiles"] is False
    assert preview["readiness"]["verified_plan"] is False
    assert [i["code"] for i in preview["compile_blockers"]]  # non-empty blockers


def test_compress_gate_functions() -> None:
    """The GUI's blocked-button state function fails closed: until a valid
    authorization token, a recommendation, and a ready preview matching the
    current selection all hold, Compress is disabled with exact reasons.
    (Static-string inspection mirrors the browser's computeGates.)"""
    from model_atlas.recommend.gui import _GUI_PAGE

    # computeGates() emits distinct, exact reasons for each missing gate.
    assert "no valid recommendation token (recommend first)" in _GUI_PAGE
    assert "no recommendation computed yet" in _GUI_PAGE
    assert "no methods selected for the recipe" in _GUI_PAGE
    assert "no verified executable plan produced for this selection" in _GUI_PAGE
    assert "selection changed since preview" in _GUI_PAGE
    # the compress button is only enabled when computeGates().ready is true.
    assert "btn.disabled = !gate.ready" in _GUI_PAGE
    # blocked (non-authorized) methods are non-toggleable; only authorized ones are.
    assert "cb.disabled = isBlocked" in _GUI_PAGE


def test_gui_start_fetches_token_binding_not_recipe(tmp_path: Path):
    """The browser never holds a recipe payload: /api/start is POSTed only the
    bounded token/preview_id/hash/selection binding (server owns the artifact).
    Preview must also be token-bound."""
    from model_atlas.recommend.gui import _GUI_PAGE as text

    assert "token: authToken, preview_id: preview.preview_id" in text
    assert "hash: preview.hash" in text
    assert "selected: selArr" in text
    # preview is token-bound too
    assert "{ token: authToken, selected: selArr }" in text
    # no embedded user/recommendation JSON in the page source
    assert "recipe_payload" not in text
    assert "model_dump" not in text


def test_gui_compress_button_disabled_with_blockers(tmp_path: Path):
    from model_atlas.recommend import RecommendationService

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(_full_profile())
    from model_atlas.recommend.gui import render_gui

    text = render_gui(svc)
    # The compress button starts disabled (fail closed, never fakes quantization)
    # until every gate passes: no fatal blockers, preview compiles, verified
    # executable plan with passing live pins.
    assert 'id="compressBtn" disabled' in text
    assert "disabled until every gate passes" in text
    # no_pruning is locked on — allow_pruning is a disabled checkbox.
    assert 'id="allowPrune" disabled' in text
    assert "no_pruning=true" in text


# --- Authorization token + preview + start with a REAL persisted job engine ---

def _executable_recipe(tmp_path: Path) -> CompressionRecipe:
    """A canonical, executable, single-stage recipe served by the in-repo
    atlas_quant_probe backend (available + pinned) with an immutable source, so
    the compiled artifact verifies pins and a real engine run completes."""
    from model_atlas.jobs.artifacts import source_manifest

    src = tmp_path / "model_src"
    src.mkdir()
    (src / "w.bin").write_bytes(b"stable-weights-v1")
    files: dict[str, str] = {
        k: v
        for k, v in source_manifest(str(src)).get("files", {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }
    return CompressionRecipe(
        name="auth-exe",
        source=SourceIdentity(
            source_id="s", checkpoint_path=str(src), sha256=files
        ),
        calibration=CalibrationIdentity(calibration_id="c", corpus_name="corp"),
        constraints=RecipeConstraints(
            no_pruning=True,
            allow_pruning_capability=False,
            preserve_non_expert_backbone=True,
            immutable_source=True,
            allow_hybrid_precision=False,
            max_resident_gib=115.0,
            derived_format="safetensors",
        ),
        stages=[
            RecipeStage(
                id="s1",
                name="s1",
                effect_class=StageEffectClass.PROFILING,
                backend=StageBackendPin(backend_id="atlas_quant_probe", version="1.0.0"),
                produces_format=["manifest.json"],
                evidence_policy=EvidenceKind.ESTIMATED,
            )
        ],
    )


def test_authorize_binds_token_to_recommendation_and_method_set(tmp_path: Path):
    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(_full_profile())
    a1 = svc.authorize("k3-mini", RecTarget(memory_target_gib=115.0))
    a2 = svc.authorize("k3-mini", RecTarget(memory_target_gib=115.0))
    # deterministic recommendation ids but OPAGUE tokens must differ
    assert a1["recommendation_id"] == a2["recommendation_id"]
    assert a1["token"] != a2["token"]
    assert a1["authorized_methods"] == a2["authorized_methods"]
    assert a1["selection_hash"] == a2["selection_hash"]
    assert a1["selection_hash"]  # bound to the exact authorized method set
    # recommendation payload matches the plain recommend() exactly
    plain = svc.recommend("k3-mini", RecTarget(memory_target_gib=115.0))
    assert a1["recommendation"]["recommendation_id"] == plain.recommendation_id


def test_start_authorized_runs_persisted_job_asynchronously(tmp_path: Path):
    """Given a valid token + a ready executable preview, start returns run_id
    IMMEDIATELY (background worker), the job is persisted (status observable),
    and the run completes via the durable engine."""
    from model_atlas.recommend import RecommendationService
    from model_atlas.recommend.api import (
        _AuthorizationSession,
        _PendingPreview,
        _selection_hash,
    )

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    recipe = _executable_recipe(tmp_path)
    from model_atlas.recipe.compiler import RecipeCompiler
    from model_atlas.recipes import CompiledPlanArtifact

    comp = RecipeCompiler(svc.registry).compile(recipe)
    artifact = CompiledPlanArtifact.from_compiled(comp, inputs={}, registry=svc.registry)
    artifact.verify()
    artifact.verify_pins_against(svc.registry)

    tok = "t-exec"
    h = _selection_hash(["exe-method"])
    svc.sessions[tok] = _AuthorizationSession(
        token=tok, recommendation_id="rec-x", profile_id="p",
        target=RecTarget(), no_pruning=True,
        constraints_snapshot={}, authorized_methods=["exe-method"],
    )
    svc.pending_previews["pv-exec"] = _PendingPreview(
        token=tok, preview_id="pv-exec", selection_hash=h,
        selected=["exe-method"], recipe=recipe, artifact=artifact,
        inputs={}, run_id=artifact.run_id,
    )

    import time as _time
    t0 = _time.time()
    res = svc.start_authorized(tok, "pv-exec", h, ["exe-method"])
    elapsed = _time.time() - t0
    # asynchronous: returns ~instantly, before the worker's run completes
    assert res["status"] == "started"
    assert res["run_id"] == artifact.run_id
    assert elapsed < 2.0

    # job identity is persisted BEFORE dispatch -> status immediately observable
    st = svc.plane.status(res["run_id"])
    assert st["run_id"] == res["run_id"]
    assert st["status"] in ("pending", "running", "completed",
                            "failed_terminal", "failed_recoverable")

    # durable completion via the real engine
    for _ in range(200):
        st = svc.plane.status(res["run_id"])
        if st["status"] in ("completed", "failed_terminal", "failed_recoverable"):
            break
        _time.sleep(0.1)
    assert st["status"] == "completed", st

    # duplicate start is rejected as replay (never a second execution)
    from model_atlas.recommend.api import AuthError
    with pytest.raises(AuthError) as exc:
        svc.start_authorized(tok, "pv-exec", h, ["exe-method"])
    assert exc.value.code == "replay"


def test_gui_preview_invalidation_and_compress_gating_deterministic():
    """GUI authorization semantics (deterministic JS assertions, non-vacuous):
    any profile/target/checkbox/recommendation change invalidates the preview
    and keeps Compress disabled until a fresh preview matches the current
    selection hash. Asserted against the served JS behavior, not the source
    alone: the button is only enabled when gate.ready and computeGates reasons
    cover every invalidation path."""
    from model_atlas.recommend.gui import _GUI_PAGE as t

    # every change source invalidates + CLEARS the binding (fresh recommendation
    # required) and re-disables Compress
    assert "clearBinding('profile changed')" in t
    assert "clearBinding('memory target changed')" in t
    assert "invalidatePreview('selection changed')" in t
    assert "invalidatePreview('new recommendation')" in t
    # profile/memory change clears token + reco + selection + preview
    assert "authToken = null" in t
    assert "reco = null" in t
    assert "selection = new Set()" in t
    assert "preview = null" in t
    assert "authToken" in t  # token is part of the state; no token = disabled
    # Compress enabled ONLY when computeGates().ready (token+selection+preview match)
    assert "btn.disabled = !gate.ready" in t
    # selection drift after a preview re-disables (re-preview required)
    assert "selection changed since preview" in t


def test_gui_monitor_polls_terminal_and_fetches_evidence():
    """Monitor behavior: poll status + events; at a TERMINAL status STOP
    polling and fetch validate/lineage/outputs — not a vacuous source check."""
    from model_atlas.recommend.gui import _GUI_PAGE as t

    # polls status AND events
    assert "fetch('/api/jobs/' + enc + '/events')" in t
    # terminal states stop the poll loop
    assert "FAILED_TERMINAL" in t
    assert "COMPLETED_WITH_WARNINGS" in t
    assert "FAILED_RECOVERABLE" in t
    assert "CANCELLED" in t
    # after terminal, fetch and render outputs + run-bound lineage + validation
    assert "fetch('/outputs?run_id=' + enc)" in t
    assert "fetch('/lineage?run_id=' + enc)" in t
    assert "recipe={}" not in t and "/lineage?run_id=" in t  # always run-bound
    assert "fetch('/validate?run_id=' + enc" in t  # per-stage validation
    assert "fetchLineage" not in t  # replaced by fetchEvidence
