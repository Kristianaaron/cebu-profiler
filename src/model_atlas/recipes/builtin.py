"""Built-in canonical recipes for the Atlas control plane.

Two families are shipped:

1. **GLM-5.2 canonical fidelity-first NO-PRUNING recipe** (the product rule).
   Its fifteen stages follow the registered recipe stages (teacher identity →
   corpus profiling → sensitivity → representation → conditioning → GEMQ/MixQuant
   bit allocation → EXL3 primary quant → ReQuant refinement → residual correction
   → SM121 NVFP4 substitution → FP8/BF16 protected tail → KV → two-Spark runtime
   → Eval Lab). It is a *compiled/dry-run* plan — the compiler will (truthfully)
   mark EXL3 / ModelOpt-NVFP4 / Eval-Lab backends unavailable and the EXL3+
   NVFP4+FP8 hybrid undischarged, so execution stays FAIL-CLOSED until pinned
   dependencies and runtime validation land.

2. **TENP/FlexMoE structural-pruning OPT-IN capability recipe.** Separate,
   never injected, and only runnable with no_pruning=false + the pruning
   capability actually declared. Its backend is unavailable today, so it also
   fails closed.

Both are model-agnostic in structure; the GLM-5.2 identity is one registered
subject, not the only one (AGENTS.md).
"""

from __future__ import annotations

from model_atlas.recipe.schema import (
    CalibrationIdentity,
    CompressionRecipe,
    HardwareEnvelope,
    PublishRule,
    RecipeConstraints,
    RecipeStage,
    RecipeStatus,
    ResourceBounds,
    SourceIdentity,
    StageBackendPin,
    StageEffectClass,
    ValidationGate,
)
from model_atlas.schemas.evidence import EvidenceKind

GLM52_SOURCE_PATH = "/media/glm52/models/nvidia/GLM-5.2-NVFP4"
GLM52_CENSUS = "232385 tensors / 47 shards / ~464.8 GiB (measured census 2026-08-14)"
GLM52_HARDWARE = HardwareEnvelope(
    node_count=2,
    model_arch="glm-5.2",
    compute_arch="gb10-sm121",
    topology="2x-spark",
    per_node_host_gib=120.0,
    per_node_gpu_gib=120.0,
    interconnect="connectx-7",
    runtime_backend="vllm-modelopt",
)


def _stage(
    sid: str,
    name: str,
    effect: StageEffectClass,
    backend_id: str,
    *,
    produces: list[str],
    requires: list[str] | None = None,
    policy: EvidenceKind = EvidenceKind.PREDICTED,
    host_gb: float = 0.0,
    scratch_gb: float = 0.0,
    workers: int = 1,
    gates: list[tuple[str, str, dict[str, str]]] | None = None,
    note: str = "",
) -> RecipeStage:
    return RecipeStage(
        id=sid,
        name=name,
        effect_class=effect,
        backend=StageBackendPin(backend_id=backend_id, version="unpinned"),
        produces_format=produces,
        requires_formats=requires or [],
        seed=0,
        validation_gates=[ValidationGate(gate_id=g, kind=k, params=p) for g, k, p in (gates or [])],
        resources=ResourceBounds(
            max_host_gb=host_gb, max_scratch_gb=scratch_gb, max_workers=workers
        ),
        evidence_policy=policy,
        notes=note,
    )


def glm52_no_pruning_recipe(
    source_path: str = GLM52_SOURCE_PATH,
) -> CompressionRecipe:
    """The canonical fidelity-first, NO-PRUNING GLM-5.2 recipe.

    This is a plan. Backends EXL3 / ModelOpt-NVFP4 / Eval-Lab are not installed
    in this repo, so compiling it fails closed on backend_unavailable + the
    undischarged EXL3+NVFP4+FP8 hybrid — exactly as intended until runtime
    validation.
    """
    stages = [
        # 1. Teacher/reference identity — immutable source hashing
        _stage(
            "t1-identity",
            "Teacher/reference identity",
            StageEffectClass.IDENTITY,
            "atlas_analysis_v3",
            produces=["manifest.json"],
            policy=EvidenceKind.MEASURED,
            host_gb=1.0,
            note="SOURCE MATH: classify tensors + preserve identity (AGENTS 3,7,14)",
        ),
        # 2. Corpus/calibration profiling — balanced labels (AGENTS 6)
        _stage(
            "t2-calibration",
            "Corpus/calibration profiling (balanced)",
            StageEffectClass.PROFILING,
            "atlas_analysis_v3",
            produces=["corpus-profile"],
            requires=["manifest.json"],
            policy=EvidenceKind.ESTIMATED,
            host_gb=8.0,
            note="Underrepresented capability labels silently pruned — balance the corpus",
        ),
        # 3. Sensitivity — semantic + activation + Hessian + routing + spectral
        _stage(
            "t3-sensitivity",
            "Semantic+activation+Hessian+routing+spectral sensitivity",
            StageEffectClass.SENSITIVITY,
            "atlas_analysis_v3",
            produces=["sensitivity-map"],
            requires=["corpus-profile", "manifest.json"],
            policy=EvidenceKind.ESTIMATED,
            host_gb=8.0,
            note="Evidence-only scoring; research metadata is NOT the algorithm",
        ),
        # 4. Shared cross-expert representation analysis
        _stage(
            "t4-representation",
            "Shared cross-expert representation analysis",
            StageEffectClass.REPRESENTATION,
            "atlas_analysis_v3",
            produces=["representation-map"],
            requires=["manifest.json"],
            policy=EvidenceKind.ESTIMATED,
            host_gb=4.0,
        ),
        # 5. Distribution/outlier conditioning
        _stage(
            "t5-conditioning",
            "Distribution/outlier conditioning",
            StageEffectClass.CONDITIONING,
            "modelopt_nvfp4",  # ModelOpt conditioning pass (placeholder; unavailable)
            produces=["conditioned-weights"],
            requires=["sensitivity-map", "representation-map"],
            policy=EvidenceKind.PREDICTED,
            host_gb=16.0,
            workers=2,
            note="Only when a ModelOpt producer/version parity is measured",
        ),
        # 6. GEMQ/MixQuant-informed global EXL3 bit allocation
        _stage(
            "t6-bit-allocation",
            "GEMQ/MixQuant-informed global EXL3 bit allocation",
            StageEffectClass.ALLOCATION,
            "atlas_analysis_v3",
            produces=["bit-allocation"],
            requires=["sensitivity-map", "representation-map"],
            policy=EvidenceKind.PREDICTED,
            host_gb=4.0,
            note="Global (not greedy per-layer) budget; predictions until materialized",
        ),
        # 7. EXL3 primary quantization
        _stage(
            "t7-exl3",
            "EXL3 primary quantization",
            StageEffectClass.QUANTIZATION,
            "exl3",
            produces=["exl3"],
            requires=["conditioned-weights", "bit-allocation"],
            policy=EvidenceKind.PREDICTED,
            host_gb=64.0,
            scratch_gb=64.0,
            workers=4,
            gates=[
                ("eq-control", "eq_control", {"rel_l2": "1e-3"}),
                ("identity-control", "identity_control", {}),
            ],
            note="EXL3 backend factually UNAVAILABLE (no pinned revision/SM121 kernel)",
        ),
        # 8. ReQuant-style fixed-budget refinement
        _stage(
            "t8-refinement",
            "ReQuant-style fixed-budget refinement",
            StageEffectClass.REFINEMENT,
            "atlas_quant_probe",  # in-repo fixed-grid refiner (validated math)
            produces=["exl3"],
            requires=["exl3"],
            policy=EvidenceKind.ESTIMATED,
            host_gb=8.0,
            note="Improves reconstruction at FIXED budget; never changes format",
        ),
        # 9. Selective low-rank/sparse residual correction
        _stage(
            "t9-residual",
            "Selective low-rank/sparse residual correction",
            StageEffectClass.RESIDUAL,
            "atlas_quant_probe",
            produces=["residual-corrected"],
            requires=["exl3"],
            policy=EvidenceKind.PREDICTED,
            host_gb=8.0,
        ),
        # 10. SM121-aware NVFP4 substitution (selective)
        _stage(
            "t10-nvfp4",
            "SM121-aware NVFP4 substitution",
            StageEffectClass.QUANTIZATION,
            "modelopt_nvfp4",
            produces=["modelopt_nvfp4"],
            requires=["exl3"],
            policy=EvidenceKind.PREDICTED,
            host_gb=32.0,
            gates=[("routing-preserving", "validator", {"js_div": "0.05"})],
            note="Backend factually UNAVAILABLE (producer/version parity unproven)",
        ),
        # 11. FP8/BF16 protected sensitive tail
        _stage(
            "t11-tail",
            "FP8/BF16 protected sensitive tail",
            StageEffectClass.QUANTIZATION,
            "llm_compressor",
            produces=["fp8_e4m3"],
            requires=["exl3"],
            policy=EvidenceKind.PREDICTED,
            host_gb=16.0,
            note="Backend factually UNAVAILABLE (LLM Compressor placeholder)",
        ),
        # 12. KV optimization
        _stage(
            "t12-kv",
            "KV optimization (cache/context gate)",
            StageEffectClass.KV,
            "atlas_analysis_v3",
            produces=["kv-plan"],
            requires=["manifest.json"],
            policy=EvidenceKind.ESTIMATED,
            gates=[("context-gate", "validator", {"min_context_tokens": "8192"})],
        ),
        # 13. Real two-Spark memory/runtime profiling (measured only)
        _stage(
            "t13-runtime",
            "Real two-Spark memory/runtime profiling",
            StageEffectClass.RUNTIME,
            "eval_lab",
            produces=["runtime-profile"],
            requires=["exl3", "modelopt_nvfp4", "fp8_e4m3", "kv-plan"],
            policy=EvidenceKind.MEASURED,
            host_gb=1.0,
            note="Needs authorized maintenance window; services never stopped by a script",
        ),
        # 14. Eval Lab / Pareto evaluation
        _stage(
            "t14-eval",
            "Eval Lab / Pareto evaluation",
            StageEffectClass.EVALUATION,
            "eval_lab",
            produces=["eval-results"],
            requires=["runtime-profile"],
            policy=EvidenceKind.MEASURED,
            note="Backend factually UNAVAILABLE (external harness placeholder)",
        ),
    ]
    return CompressionRecipe(
        name="glm52-no-pruning-fidelity",
        description=(
            "Canonical GLM-5.2 fidelity-first compression: NO PRUNING. Fifteen "
            "registered recipe stages. Compiled/dry-run plan only — execution "
            "fails closed until EXL3/ModelOpt/Eval-Lab dependencies are pinned, "
            "producer-version parity measured, and the EXL3+NVFP4+FP8 hybrid is "
            "declared by a runtime/backend."
        ),
        source=SourceIdentity(
            source_id="glm-5.2-nvfp4",
            checkpoint_path=source_path,
            checkpoint_revision="0.46.0.dev65+g977d34dc3",
            sha256={},  # path-bound {rel_path: sha256}, filled by a census step
            params_estimate=0,  # unmeasured here
        ),
        calibration=CalibrationIdentity(
            calibration_id="glm52-calibration-v1",
            corpus_name="balanced-glm52-calibration",
            seed=0,
            partition="atlas_calibration",
            capability_labels=[
                "code_generation",
                "mathematical_reasoning",
                "general_reasoning",
                "long_context_retrieval",
                "multilingual_support",
                "creative_writing",
            ],
            note="balance required: underrepresented labels are silently pruned (AGENTS 6)",
        ),
        hardware=GLM52_HARDWARE,
        constraints=RecipeConstraints(
            no_pruning=True,  # CANONICAL POLICY — default, not a choice here
            allow_pruning_capability=False,
            preserve_non_expert_backbone=True,
            immutable_source=True,
            allow_hybrid_precision=False,  # EXL3+NVFP4+FP8 needs explicit backend support
            max_resident_gib=115.0,
            derived_format="safetensors",
        ),
        stages=stages,
        publish=PublishRule(
            require_all_stages_validated=True,
            require_no_pruning=True,
            evidence_kind_min=EvidenceKind.MEASURED,
            require_runtime_benchmarked=True,
            require_repair_or_validated=True,
        ),
        backend_pins={},
    )


def tenp_pruning_optin_recipe(
    source_path: str = GLM52_SOURCE_PATH,
) -> CompressionRecipe:
    """Separately-registered OPT-IN TENP/FlexMoE structural-pruning capability
    recipe. Never auto-injected; runnable only with no_pruning=false and the
    pruning capability declared. The pruning backend is UNAVAILABLE today, so
    compilation fails closed (no_pruning policy is per-recipe, and this recipe
    explicitly opts out of it)."""
    stages = [
        _stage(
            "p1-trace",
            "Streamed routing trace (TENP input)",
            StageEffectClass.SENSITIVITY,
            "atlas_analysis_v3",
            produces=["routing-trace"],
            requires=["manifest.json"],
            policy=EvidenceKind.ESTIMATED,
        ),
        _stage(
            "p2-tenp",
            "TENP score + keep-map",
            StageEffectClass.SENSITIVITY,
            "atlas_analysis_v3",
            produces=["keep-map"],
            requires=["routing-trace"],
            policy=EvidenceKind.ESTIMATED,
        ),
        _stage(
            "p3-prune",
            "FlexMoE structural pruning (opt-in)",
            StageEffectClass.PRUNING,
            "tenp_pruning",
            produces=["pruned-checkpoint"],
            requires=["keep-map"],
            policy=EvidenceKind.PREDICTED,
            host_gb=64.0,
            scratch_gb=64.0,
            note="OPT-IN: not part of any canonical recipe; backend UNAVAILABLE today",
        ),
    ]
    return CompressionRecipe(
        name="tenp-flexmoe-pruning-optin",
        description=(
            "Separately-registered OPT-IN TENP/FlexMoE pruning capability. It is "
            "NEVER part of the canonical GLM-5.2 (no_pruning=true) recipe; it must "
            "be explicitly compiled with no_pruning=false and a declared pruning "
            "capability. The pruning backend is unavailable in this repo, so this "
            "fails closed and does not run."
        ),
        source=SourceIdentity(
            source_id="glm-5.2-nvfp4",
            checkpoint_path=source_path,
            checkpoint_revision="0.46.0.dev65+g977d34dc3",
        ),
        calibration=CalibrationIdentity(
            calibration_id="glm52-pruning-calibration", corpus_name="balanced", seed=0
        ),
        hardware=GLM52_HARDWARE,
        constraints=RecipeConstraints(
            no_pruning=False,  # explicit opt-out (this capability recipe)
            allow_pruning_capability=True,  # requires the capability be declared
            preserve_non_expert_backbone=True,
            immutable_source=True,
            allow_hybrid_precision=False,
            max_resident_gib=115.0,
            derived_format="safetensors",
        ),
        stages=stages,
        publish=PublishRule(
            require_all_stages_validated=True,
            require_no_pruning=False,
            evidence_kind_min=EvidenceKind.MEASURED,
            require_runtime_benchmarked=True,
            require_repair_or_validated=True,
        ),
    )


def nvfp4_width_slice_optin_recipe(
    source: SourceIdentity,
    width: int,
) -> CompressionRecipe:
    """Explicit uniform-width NVFP4 structural pruning recipe.

    This is a deterministic engineering control, not a quality-aware pruning
    recommendation and not a runtime-loadability claim.
    """
    stage = RecipeStage(
        id="width-slice",
        name="Uniform aligned NVFP4 expert-channel width slice",
        effect_class=StageEffectClass.PRUNING,
        backend=StageBackendPin(
            backend_id="atlas_nvfp4_width_slice",
            version="1.0.0",
            minimum_status=RecipeStatus.VALIDATED,
        ),
        parameters={"width": str(width)},
        produces_format=["pruned-checkpoint", "safetensors"],
        expected_outputs=["config.json", "model.safetensors.index.json"],
        seed=0,
        validation_gates=[
            ValidationGate(gate_id="width-slice-checkpoint", kind="checkpoint", required=True)
        ],
        resources=ResourceBounds(
            max_host_gb=16.0,
            max_scratch_gb=192.0,
            max_workers=1,
            note="bounded streaming exporter plus staged/CAS derivative copies",
        ),
        evidence_policy=EvidenceKind.PREDICTED,
        notes=(
            "OPT-IN uniform width control; no TENP, quality-retention, or runtime "
            "validation claim"
        ),
    )
    return CompressionRecipe(
        name=f"atlas-nvfp4-width-slice-w{width}",
        description=(
            "Explicit uniform aligned expert-channel pruning control. The output "
            "is structurally validated but remains runtime-unvalidated."
        ),
        source=source,
        calibration=CalibrationIdentity(
            calibration_id="not-used-uniform-width-slice",
            corpus_name="none",
            seed=0,
            note="uniform structural control uses no quality calibration",
        ),
        # Artifact-only structural job. Runtime compatibility is promoted only
        # after a real loader/forward canary, never inferred from file schema.
        hardware=GLM52_HARDWARE.model_copy(update={"runtime_backend": "none"}),
        constraints=RecipeConstraints(
            no_pruning=False,
            allow_pruning_capability=True,
            preserve_non_expert_backbone=True,
            immutable_source=True,
            allow_hybrid_precision=False,
            max_resident_gib=115.0,
            derived_format="safetensors",
        ),
        stages=[stage],
        publish=PublishRule(
            require_all_stages_validated=True,
            require_no_pruning=False,
            evidence_kind_min=EvidenceKind.PREDICTED,
            require_runtime_benchmarked=False,
            require_repair_or_validated=False,
        ),
        backend_pins={"atlas_nvfp4_width_slice": "1.0.0"},
    )
