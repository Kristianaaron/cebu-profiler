"""Atlas v3 analysis analyzers (fidelity-first evidence surfaces).

Each analyzer is a pure evidence producer over measured model behavior — it
never mutates weights and never turns a prediction into a measured result
(AGENTS invariants + v3 scientific rules).
"""

from model_atlas.analysis.conditional_sensitivity import (
    ConditionalSensitivity,
    ConditionalSensitivityPoint,
    conditional_sensitivity,
)
from model_atlas.analysis.corpus_semantic import (
    ClusterExpertCoverage,
    CorpusDelta,
    CorpusSemanticReport,
    ExpertClusterActivation,
    SemanticCluster,
    build_corpus_semantic_map,
    project_corpus_delta,
)
from model_atlas.analysis.global_bit_budget import (
    BPW_CHOICES,
    BitAssignment,
    GlobalBitMap,
    enumerate_global_bit_maps,
)
from model_atlas.analysis.kv_memory import (
    KvBudgetResult,
    KvOption,
    MemoryLedger,
    kv_bytes_per_token,
    plan_kv_budget,
)
from model_atlas.analysis.nvfp4_suitability import (
    Nvfp4Candidate,
    Nvfp4SuitabilityReport,
    nvfp4_suitability,
)
from model_atlas.analysis.quant_interaction import (
    InteractionPrediction,
    QuantInteractionModel,
    fit_quant_interaction,
    predict_global_error,
)
from model_atlas.analysis.refiner import RefinementResult, refine_expert_tensors
from model_atlas.analysis.residual_correction import (
    ActionOption,
    ResidualPlan,
    residual_correction_plan,
)
from model_atlas.analysis.routing_consistency import (
    RoutingConsistency,
    RoutingConsistencyReport,
    routing_consistency,
)
from model_atlas.analysis.shared_representation import (
    SharedAnalysis,
    SharedStructure,
    analyze_shared_representation,
)
from model_atlas.analysis.spectral import (
    SpectralAnalysis,
    SpectralProfile,
    analyze_spectral,
)
from model_atlas.analysis.structural_fallback import (
    StructuralFallbackPlan,
    structural_fallback_plans,
)

__all__ = [
    "ConditionalSensitivity",
    "ConditionalSensitivityPoint",
    "conditional_sensitivity",
    "ClusterExpertCoverage",
    "CorpusDelta",
    "CorpusSemanticReport",
    "ExpertClusterActivation",
    "SemanticCluster",
    "build_corpus_semantic_map",
    "project_corpus_delta",
    "BitAssignment",
    "BPW_CHOICES",
    "GlobalBitMap",
    "enumerate_global_bit_maps",
    "KvBudgetResult",
    "KvOption",
    "MemoryLedger",
    "kv_bytes_per_token",
    "plan_kv_budget",
    "Nvfp4Candidate",
    "Nvfp4SuitabilityReport",
    "nvfp4_suitability",
    "InteractionPrediction",
    "QuantInteractionModel",
    "fit_quant_interaction",
    "predict_global_error",
    "RefinementResult",
    "refine_expert_tensors",
    "ActionOption",
    "ResidualPlan",
    "residual_correction_plan",
    "RoutingConsistency",
    "RoutingConsistencyReport",
    "routing_consistency",
    "SharedAnalysis",
    "SharedStructure",
    "analyze_shared_representation",
    "SpectralAnalysis",
    "SpectralProfile",
    "analyze_spectral",
    "StructuralFallbackPlan",
    "structural_fallback_plans",
]
