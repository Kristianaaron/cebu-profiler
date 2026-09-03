"""Candidate lineage and candidate graph (v3 "Candidate graph requirements").

Every candidate is an immutable node in a DAG: it stores exactly the operators
and parameters, a tensor/expert representation map, predicted-vs-measured
status, memory breakdown, routing stability, corpus-regression hotspots,
provenance, and hard-floor failures. A candidate is never mutated — derived
candidates create new nodes pointing at parents.

Measured results and predictions are never conflated (evidence discipline):
a candidate whose metrics are predictions can never be marked deployable.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.schemas.evidence import EvidenceKind


class CandidateStage(StrEnum):
    """Pipeline stage a candidate reached (blueprint %5.1 P0-P10)."""

    P0_REFERENCE = "p0_reference"
    P1_EVIDENCE = "p1_evidence"
    P2_ALLOCATION_MODEL = "p2_allocation_model"
    P3_CONDITIONING = "p3_conditioning"
    P4_EXL3 = "p4_exl3"
    P5_REFINEMENT = "p5_refinement"
    P6_SM121_ALLOCATION = "p6_sm121_allocation"
    P7_KV_SYSTEM_BUDGET = "p7_kv_system_budget"
    P8_MATERIALIZED_BENCHMARK = "p8_materialized_benchmark"
    P9_PARETO_DECISION = "p9_pareto_decision"
    P10_STRUCTURAL_FALLBACK = "p10_structural_fallback"
    P11_RECOVERY = "p11_recovery"


class OperatorKind(StrEnum):
    """Named research operators; each links to its paper/provenance."""

    SHARED_REPRESENTATION = "shared_representation"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    CONDITIONAL_SENSITIVITY = "conditional_sensitivity"
    ROUTING_CONSISTENCY_GUARD = "routing_consistency_guard"
    GLOBAL_BIT_BUDGET = "global_bit_budget"
    QUANT_INTERACTION_MODEL = "quant_interaction_model"
    FIXED_GRID_REFINEMENT = "fixed_grid_refinement"
    RESIDUAL_CORRECTION = "residual_correction"
    NVFP4_SUITABILITY = "nvfp4_suitability"
    KV_MEMORY_OPTIMIZER = "kv_memory_optimizer"
    STRUCTURAL_FALLBACK = "structural_fallback"
    EXL3_EXPRESS = "exl3_express"
    ROUTE_PRESERVING = "route_preserving"
    ENUM = "enum"


class RepresentationAssignment(BaseModel):
    """Representation of a single tensor/expert/group (v3 precision map)."""

    model_config = ConfigDict(extra="forbid")

    tensor_or_group: str
    representation: str  # exl3 | nvfp4 | fp8 | bf16 | correction_payload | structural_mask
    bpw: float | None = Field(default=None, ge=0.0)
    memory_bytes: float = Field(default=0.0, ge=0.0)
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED


class RegressionHotspot(BaseModel):
    """A corpus region where a candidate regressed (localized quality loss)."""

    model_config = ConfigDict(extra="forbid")

    semantic_cluster: str
    quality_delta: float  # teacher-relative; negative = regression
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED
    link: str = ""  # human-readable "why"


class CandidateMetricSet(BaseModel):
    """The quality vector + system/routing metrics a candidate carries.

    Every number must be tagged with its production kind so inferred/predicted
    numbers can never be presented as measured.
    """

    model_config = ConfigDict(extra="forbid")

    quality_retention: float | None = Field(default=None, ge=0.0)
    logit_kl: float | None = Field(default=None, ge=0.0)
    topk_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    routing_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    mtp_acceptance: float | None = Field(default=None, ge=0.0, le=1.0)
    decode_tps: float | None = Field(default=None, ge=0.0)
    prefill_tps: float | None = Field(default=None, ge=0.0)
    resident_gib: float | None = Field(default=None, ge=0.0)
    safe_context_tokens: int | None = Field(default=None, ge=0)
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED


class CandidateNode(BaseModel):
    """An immutable candidate-graph node (v3 %9)."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    parent_ids: list[str] = Field(default_factory=list)
    stage: CandidateStage = CandidateStage.P0_REFERENCE
    name: str
    operators: list[OperatorKind] = Field(default_factory=list)
    operator_parameters: dict[str, str] = Field(default_factory=dict)
    tensor_repr_map: list[RepresentationAssignment] = Field(default_factory=list)
    memory_breakdown: dict[str, float] = Field(default_factory=dict)  # bucket -> bytes
    quality_vector: CandidateMetricSet = Field(default_factory=CandidateMetricSet)
    routing_stability: dict[str, float] = Field(default_factory=dict)
    corpus_hotspots: list[RegressionHotspot] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)  # paper/impl/experiment refs
    predicted: bool = True  # predictions are never deployable
    deployed: bool = False
    hard_floor_failures: list[str] = Field(default_factory=list)
    created_at: str = ""
    notes: str = ""


class CandidateGraphError(ValueError):
    """Raised when building/querying a candidate graph violates immutability."""


class CandidateGraph(BaseModel):
    """DAG of candidate nodes with lineage queries (v3 %9)."""

    model_config = ConfigDict(extra="forbid")

    source_teacher_id: str = ""
    nodes: dict[str, CandidateNode] = Field(default_factory=dict)

    def add(self, node: CandidateNode) -> CandidateNode:
        if node.candidate_id in self.nodes:
            raise CandidateGraphError(f"candidate {node.candidate_id!r} already registered")
        for p in node.parent_ids:
            if p not in self.nodes:
                raise CandidateGraphError(f"parent {p!r} of {node.candidate_id!r} missing")
            if self.nodes[p].deployed and not node.predicted:
                raise CandidateGraphError(
                    f"cannot derive {node.candidate_id!r} from a deployable parent "
                    f"{p!r} without an explicit new measured run"
                )
        self.nodes[node.candidate_id] = node
        return node

    def lineage(self, candidate_id: str) -> list[CandidateNode]:
        """Full ancestry, nearest-above first (immutable provenance chain)."""
        if candidate_id not in self.nodes:
            raise CandidateGraphError(f"unknown candidate {candidate_id!r}")
        seen: dict[str, CandidateNode] = {}
        stack = list(self.nodes[candidate_id].parent_ids)
        while stack:
            p = stack.pop()
            if p in seen or p == candidate_id:
                continue
            seen[p] = self.nodes[p]
            stack.extend(self.nodes[p].parent_ids)
        return sorted(seen.values(), key=lambda n: len(self.nodes[n.candidate_id].parent_ids))

    def root(self, candidate_id: str) -> CandidateNode:
        lineage = self.lineage(candidate_id)
        return lineage[-1] if lineage else self.nodes[candidate_id]

    def siblings(self, candidate_id: str) -> list[CandidateNode]:
        """Candidates sharing an immediate parent with ``candidate_id``."""
        node = self.nodes[candidate_id]
        parents = set(node.parent_ids)
        return [
            n
            for n in self.nodes.values()
            if n.candidate_id != candidate_id and set(n.parent_ids) == parents
        ]

    def measured(self) -> list[CandidateNode]:
        """Only measured (non-predicted) candidates — decision surface."""
        return [n for n in self.nodes.values() if not n.predicted]

    def deployable(self) -> list[CandidateNode]:
        """Measured, no hard-floor failures, explicitly deployed."""
        return [
            n
            for n in self.nodes.values()
            if not n.predicted and not n.hard_floor_failures and n.deployed
        ]
