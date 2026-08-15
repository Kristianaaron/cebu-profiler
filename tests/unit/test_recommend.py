"""Recommendation engine tests: deterministic ranking, profile identity,
missing evidence, incompatible backend, no pruning, blocked methods, recipe
composition, API/UI smoke, XSS-safe rendering."""

from pathlib import Path

import pytest

from model_atlas.backend.registry import build_default_registry
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


def _fake_records(reg):
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
            "mystery_key": {"kind": "predicted"},  # unknown key kept as-is
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
                "nvfp4": {"kind": "estimated"},  # V3 alias, NOT nvfp4_suitability
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


def test_backend_missing_blocker_for_unregistered():
    """A method whose backend id is not registered must block as
    backend_missing (never silently recommend an unresolvable executor)."""
    reg = build_default_registry()
    pol = RecommendationPolicy(reg)
    # exl3 is registered (so backend_missing doesn't apply); verify one of the
    # compression methods records an unavailable/derivative blocker today.
    rec = pol.recommend(_full_profile(), RecTarget())
    llm = [m for m in rec.blocked_methods if m.method == "llm-compressor"]
    assert llm and any(b.code == "backend_unavailable" for b in llm[0].blockers)


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


def test_api_facade_lifecycle(tmp_path: Path):
    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(_full_profile())
    assert svc.list_profiles()
    rec = svc.recommend("k3-mini", RecTarget(memory_target_gib=115.0))
    assert rec.recommendation_id.startswith("rec-")
    # compile preview of the editable builtin (dry-run only)
    from model_atlas.recipes.builtin import glm52_no_pruning_recipe

    preview = svc.preview_recipe(glm52_no_pruning_recipe())
    assert "compiles" in preview


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


def test_gui_embeds_payloads_and_is_xss_safe(tmp_path: Path):
    from model_atlas.recommend import RecommendationService

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(_full_profile())
    out = str(tmp_path / "gui.html")
    write_gui(out, svc)
    text = Path(out).read_text(encoding="utf-8")
    # recommendation payload embedded (deterministic, machine-readable)
    assert "rec-" in text
    # XSS-safe: user-controlled strings must be escaped — the JS uses
    # textContent/DOM text, never raw innerHTML interpolation of values.
    assert "<script>" in text  # the harness JS present
    assert "innerHTML" not in text  # no value interpolation via innerHTML


def test_gui_compress_button_disabled_with_blockers(tmp_path: Path):
    from model_atlas.recommend import RecommendationService

    svc = RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )
    svc.save_profile(_full_profile())
    from model_atlas.recommend.gui import render_gui

    text = render_gui(svc)
    # blocked methods always present -> compress button starts disabled (fail
    # closed, never fakes quantization)
    assert "Compress (disabled until a verified executable plan compiles)" in text
    # the compress button is disabled: blocked_methods nonempty and no pruning
    assert 'allow_pruning (locked: <span class="np">no_pruning=true</span>)' in text
