"""Behaviour and trajectory ontology (v2 §8, §11), as typed controlled enums.

These are the shared vocabularies the Cebu Profiler uses to tag traces so behaviour can
be sliced, contrasted, and linked to experts/coalitions/pathways.
"""

from __future__ import annotations

from enum import StrEnum


class CapabilityLabel(StrEnum):
    CODE_GENERATION = "code_generation"
    REPOSITORY_NAVIGATION = "repository_navigation"
    DEBUGGING = "debugging"
    FRONTEND_FROM_SPEC = "frontend_from_spec"
    FRONTEND_FROM_IMAGE = "frontend_from_image"
    VOXEL_SPATIAL = "voxel_spatial"
    VISUAL_REASONING = "visual_reasoning"
    TOOL_SELECTION = "tool_selection"
    TOOL_ARGUMENT_ACCURACY = "tool_argument_accuracy"
    PLANNING = "planning"
    EXECUTION_MONITORING = "execution_monitoring"
    ERROR_INTERPRETATION = "error_interpretation"
    RECOVERY_REPLANNING = "recovery_replanning"
    LONG_CONTEXT_RETRIEVAL = "long_context_retrieval"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    GENERAL_REASONING = "general_reasoning"
    MULTIMODAL_GROUNDING = "multimodal_grounding"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    CREATIVE_WRITING = "creative_writing"
    PRODUCT_DESIGN_REASONING = "product_design_reasoning"
    MULTILINGUAL_SUPPORT = "multilingual_support"


class TrajectoryStage(StrEnum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    CHOOSE_TOOL = "choose_tool"
    EXECUTE = "execute"
    INSPECT = "inspect"
    DIAGNOSE = "diagnose"
    REPAIR = "repair"
    VERIFY = "verify"
    COMMUNICATE = "communicate"


class SuccessState(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    RECOVERED = "recovered"
    PARTIALLY_RECOVERED = "partially_recovered"
    UNKNOWN = "unknown"


class DataPartition(StrEnum):
    CEBU_CALIBRATION = "cebu_calibration"
    DEVELOPMENT_EVALUATION = "development_evaluation"
    HELD_OUT_EVALUATION = "held_out_evaluation"


class GenerationMode(StrEnum):
    TEACHER_FORCED = "teacher_forced"
    FREE_GENERATION = "free_generation"
    TOOL_TRAJECTORY = "tool_trajectory"
    COUNTERFACTUAL = "counterfactual"
    FAILURE_RECOVERY = "failure_recovery"
    COMPRESSION_PROBE = "compression_probe"
    CAUSAL_ABLATION = "causal_ablation"


class TraceFamily(StrEnum):
    ROUTING = "routing"
    CONTRIBUTION = "contribution"
    REPRESENTATION = "representation"
    INTERVENTION = "intervention"


class InterventionType(StrEnum):
    EXPERT_SUPPRESSION = "expert_suppression"
    GROUP_SUPPRESSION = "group_suppression"
    SUBSTITUTION = "substitution"
    ROUTER_PERTURBATION = "router_perturbation"
    PRECISION_CHANGE = "precision_change"
    CHANNEL_MASKING = "channel_masking"
    TILE_MASKING = "tile_masking"
    FEATURE_STEERING = "feature_steering"
    PATH_DISRUPTION = "path_disruption"
