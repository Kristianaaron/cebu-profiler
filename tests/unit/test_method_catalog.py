"""Exhaustive deterministic fixtures/tests for the frozen versioned MethodSpec
catalog in ``model_atlas.recommend.policy``.

Guards the catalog as a single fail-closed authority: every method is unique
and resolvable, every spec is fully explicit (no empty execution-identity
fields), unknown/pruning-looking unknowns fail closed, generic effects/intents
are never conflated, and the catalog digest is a stable, ordering-insensitive
identity that flows into recommendation ids. None of these tests exercise a
backend, model, or network — they are pure, deterministic property checks over
the catalog and the digest construction.
"""

import hashlib
from dataclasses import replace

import pytest

from model_atlas.backend.registry import build_default_registry
from model_atlas.recipe.compiler import canonical_json
from model_atlas.recipe.schema import StageEffectClass
from model_atlas.recommend import AtlasProfile, RecTarget, StageEvidence
from model_atlas.recommend.policy import (
    METHOD_CATALOG,
    METHOD_CATALOG_VERSION,
    RECOMMENDATION_POLICY_VERSION,
    CompressionIntent,
    MethodFamily,
    RecommendationPolicy,
    method_catalog_digest,
    method_spec,
)

# Execution-identity fields that feed the catalog digest. Each is paired with a
# MUTATOR that yields a spec whose identity_dict differs from the base spec.
_IDENTITY_MUTATORS = [
    ("method", lambda s: replace(s, method=s.method + "-mutated")),
    ("family", lambda s: replace(s, family=MethodFamily.KV)),
    ("backend_id", lambda s: replace(s, backend_id=s.backend_id + "-mutated")),
    (
        "evidence_stages",
        lambda s: replace(s, evidence_stages=s.evidence_stages + ("extra-stage",)),
    ),
    (
        "recipe_stage_ids",
        lambda s: replace(s, recipe_stage_ids=s.recipe_stage_ids + ("extra-recipe",)),
    ),
    (
        "effect_classes",
        lambda s: replace(s, effect_classes=s.effect_classes + (StageEffectClass.KV,)),
    ),
    (
        "compatible_intents",
        lambda s: replace(
            s,
            compatible_intents=s.compatible_intents
            + (CompressionIntent.PRUNE_ONLY,),
        ),
    ),
    ("memory_direction", lambda s: replace(s, memory_direction="up")),
    ("routing_dependent", lambda s: replace(s, routing_dependent=not s.routing_dependent)),
    ("planning_only", lambda s: replace(s, planning_only=not s.planning_only)),
    (
        "provenance_ids",
        lambda s: replace(s, provenance_ids=s.provenance_ids + ("extra-provenance",)),
    ),
]

_PRUNE_COMPATIBLE_INTENTS = {CompressionIntent.PRUNE_ONLY, CompressionIntent.HYBRID}


def _reconstructed_digest(specs) -> str:
    """Rebuild the catalog digest from a candidate list of specs, mirroring the
    canonical payload construction (sorted by method, ``identity_dict`` per
    spec, canonical JSON so key order is irrelevant)."""
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


def _profiled_profile() -> AtlasProfile:
    return AtlasProfile(
        profile_id="p-catalog",
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


def test_version_constants_are_frozen() -> None:
    """The policy version is the catalog-bearing release and the catalog version
    is pinned for the calibration-aware recipe contract."""
    assert RECOMMENDATION_POLICY_VERSION == "policy-v4-calibration-aware"
    assert METHOD_CATALOG_VERSION == 3


def test_every_method_id_is_unique_and_resolvable() -> None:
    """Contract item 1: all catalog method IDs are unique and each resolves back
    to the exact spec via ``method_spec``."""
    ids = [spec.method for spec in METHOD_CATALOG]
    assert len(ids) == len(set(ids))
    assert len(METHOD_CATALOG) >= 1
    for spec in METHOD_CATALOG:
        assert method_spec(spec.method) is spec


def test_every_spec_is_fully_explicit() -> None:
    """Contract item 2: every spec carries an explicit family, backend, evidence
    stages, recipe stages, effect classes, intents, and memory direction — no
    empty execution-identity field exists."""
    for spec in METHOD_CATALOG:
        assert spec.family in MethodFamily
        assert spec.backend_id and spec.backend_id.strip()
        assert spec.evidence_stages, f"{spec.method}: empty evidence_stages"
        assert spec.recipe_stage_ids, f"{spec.method}: empty recipe_stage_ids"
        assert spec.effect_classes, f"{spec.method}: empty effect_classes"
        assert spec.compatible_intents, f"{spec.method}: empty compatible_intents"
        assert isinstance(spec.memory_direction, str) and spec.memory_direction.strip()


def test_unknown_and_pruning_looking_unknowns_fail_closed() -> None:
    """Contract item 3: unknown method names — including pruning-looking strings
    that are NOT catalogued — must never gain a family; ``method_spec`` raises."""
    for name in (
        "not-a-method",
        "tenp-pruning-not-catalogued",
        "flexmoepruning",
        "structured-pruning-w1",
        "custom-prune",
    ):
        with pytest.raises(KeyError, match="unknown or unclassified"):
            method_spec(name)

    # No catalogued method is secretly pruning-classified by the name alone.
    assert not any("prune" in spec.method.lower() for spec in METHOD_CATALOG)


def test_no_pruning_family_entry_without_pruning_effect_and_intent() -> None:
    """Contract item 4: any spec classified under the PRUNING family must
    declare a PRUNING effect-class AND a pruning-compatible intent. (The frozen
    v1 catalog has none, so no executable pruning MethodSpec exists today.)"""
    pruning_specs = [s for s in METHOD_CATALOG if s.family is MethodFamily.PRUNING]
    # No executable pruning MethodSpec in the current catalog.
    assert pruning_specs == []
    # But the invariant holds universally: a PRUNING-family spec, were one
    # added, would be required to carry both markers.
    for spec in METHOD_CATALOG:
        if spec.family is MethodFamily.PRUNING:
            assert StageEffectClass.PRUNING in spec.effect_classes
            assert bool(set(spec.compatible_intents) & _PRUNE_COMPATIBLE_INTENTS)


def test_quantization_specs_declare_quantization_and_exclude_prune_only() -> None:
    """Contract item 5: every QUANTIZATION-family spec declares the QUANTIZATION
    effect-class and never accepts PRUNE_ONLY as a compatible intent."""
    quant_specs = [s for s in METHOD_CATALOG if s.family is MethodFamily.QUANTIZATION]
    assert quant_specs  # the current catalog does carry quantization methods
    for spec in quant_specs:
        assert StageEffectClass.QUANTIZATION in spec.effect_classes
        assert CompressionIntent.PRUNE_ONLY not in spec.compatible_intents


def test_planning_only_specs_claim_no_quantization_or_pruning_effects() -> None:
    """Contract item 6: analysis/planning-only entries are derivative plans, not
    derivative producers — they must not claim QUANTIZATION or PRUNING effects."""
    planning = [s for s in METHOD_CATALOG if s.planning_only]
    assert planning  # planning methods exist and are exercised below
    for spec in planning:
        assert StageEffectClass.QUANTIZATION not in spec.effect_classes
        assert StageEffectClass.PRUNING not in spec.effect_classes


def test_digest_is_64_lowercase_hex_and_stable() -> None:
    """Contract item 7a: the catalog digest is 64 lowercase hex chars and stable
    across repeated calls."""
    digest = method_catalog_digest()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    for _ in range(3):
        assert method_catalog_digest() == digest


def test_digest_is_insensitive_to_source_tuple_ordering() -> None:
    """Contract item 7b: reordering the catalog tuple does not change the digest,
    so the digest is a canonical identity rather than an order artifact."""
    base_digest = method_catalog_digest()
    shuffled = list(reversed(METHOD_CATALOG))
    # Confirm the reconstruction reproduces production output exactly.
    assert _reconstructed_digest(METHOD_CATALOG) == base_digest
    assert _reconstructed_digest(shuffled) == base_digest


def test_mutating_each_identity_field_changes_reconstructed_digest() -> None:
    """Contract item 8: mutating ANY execution-identity field in a copied spec
    changes the reconstructed catalog digest — the digest is sensitive to every
    field that defines a method's identity."""
    base = method_spec("nvfp4-substitute")
    base_digest = _reconstructed_digest(METHOD_CATALOG)
    for label, mutate in _IDENTITY_MUTATORS:
        mutated = mutate(base)
        assert mutated.identity_dict() != base.identity_dict(), label
        altered_catalog = [
            mutated if spec.method == "nvfp4-substitute" else spec
            for spec in METHOD_CATALOG
        ]
        assert _reconstructed_digest(altered_catalog) != base_digest, label


def test_recommendation_id_changes_when_catalog_digest_changes() -> None:
    """Contract item 9: recommendation ids already differ when the catalog digest
    differs — a digest change flows through the canonical identity seam into the
    rec id. Verified by driving the real ``recommend`` path with the digest
    function safely patched (restored in ``finally``), never by reassigning
    production scoring globals."""
    import model_atlas.recommend.policy as policy_mod

    reg = build_default_registry()
    profile = _profiled_profile()
    target = RecTarget(memory_target_gib=115.0)

    real_digest = method_catalog_digest()
    mutated = replace(method_spec("nvfp4-substitute"), backend_id="some-other-backend")
    mutated_catalog = [
        mutated if s.method == "nvfp4-substitute" else s for s in METHOD_CATALOG
    ]
    mutated_digest = _reconstructed_digest(mutated_catalog)
    assert mutated_digest != real_digest

    pol = RecommendationPolicy(reg)
    baseline = pol.recommend(profile, target).recommendation_id

    original = policy_mod.method_catalog_digest
    try:
        policy_mod.method_catalog_digest = lambda: mutated_digest
        changed = pol.recommend(profile, target).recommendation_id
    finally:
        policy_mod.method_catalog_digest = original

    # The same profile/target under a different catalog digest must yield a
    # different recommendation id (the digest is part of the rec identity).
    assert changed != baseline
