"""Held-out evaluation and leakage gate."""

from model_atlas.evaluation.cka import CKA, centered_linear_cka
from model_atlas.evaluation.contracts import (
    CorpusSlice,
    DomainKLDAggregate,
    DomainKLDReport,
    EvaluationIdentity,
    EvaluationReport,
    EvidenceKind,
    LayerDivergence,
    MetricEvidence,
    ReproducibilityManifest,
    RouterDivergenceRecord,
    RouterDivergenceSummary,
    SampleAlignment,
    TokenKLDResult,
    TokenKLDRow,
)
from model_atlas.evaluation.eval_lab import (
    EVAL_LAB_REVISION,
    CandidateTaskReport,
    DataPartition,
    EndpointConfigIdentity,
    EndpointTransport,
    EvalLabAdapter,
    EvalLabHandoff,
    EvalLabRequest,
    EvalLabResult,
    EvalLabStatus,
    EvalParameters,
    FrozenHeldOutManifest,
    PerformanceReport,
    TaskScore,
    TeacherRelativeBlocker,
    eval_lab_output_layout,
)
from model_atlas.evaluation.heldout import (
    HeldOutReport,
    LabelRetention,
    evaluate_heldout,
    router_repair_targets,
)
from model_atlas.evaluation.kld import KLDMismatchError, token_kld
from model_atlas.evaluation.leakage import LeakageResult, detect_leakage, promote_allowed

__all__ = [
    "HeldOutReport",
    "LabelRetention",
    "evaluate_heldout",
    "router_repair_targets",
    "LeakageResult",
    "detect_leakage",
    "promote_allowed",
    # contracts
    "EvidenceKind",
    "MetricEvidence",
    "CorpusSlice",
    "EvaluationIdentity",
    "SampleAlignment",
    "TokenKLDRow",
    "DomainKLDAggregate",
    "DomainKLDReport",
    "TokenKLDResult",
    "LayerDivergence",
    "RouterDivergenceRecord",
    "RouterDivergenceSummary",
    "ReproducibilityManifest",
    "EvaluationReport",
    # kld
    "token_kld",
    "KLDMismatchError",
    # cka
    "CKA",
    "centered_linear_cka",
    # candidate-only Eval Lab handoff
    "EVAL_LAB_REVISION",
    "DataPartition",
    "EndpointTransport",
    "EndpointConfigIdentity",
    "FrozenHeldOutManifest",
    "EvalParameters",
    "EvalLabRequest",
    "TeacherRelativeBlocker",
    "TaskScore",
    "PerformanceReport",
    "CandidateTaskReport",
    "EvalLabStatus",
    "EvalLabResult",
    "EvalLabHandoff",
    "EvalLabAdapter",
    "eval_lab_output_layout",
]
