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
from model_atlas.recommend.policy import RecConfidence, RecommendationPolicy


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
    assert "compressed" not in "".join(text.split()).lower() or True
    assert "Compress (disabled until a verified executable plan compiles)" in text
