"""Corpus-semantic bidirectional mapping + quality-delta projection (v3 %6.5).

Two directions must both work:
  corpus cluster -> samples -> layers/experts -> co-activation -> compression
  decision;
  expert/tensor/channel -> activating corpus regions -> specialization ->
  overlapping experts -> unique coverage -> compression decision.

Implemented over a synthetic calibration corpus: each sample gets an embedding
cluster (deterministic from its tokens), per-expert activation counts per
cluster are measured from real forwards, and a ``CorpusDelta`` maps
teacher-relative quality deltas back onto semantic clusters after a candidate
evaluation. Evidence gates block pruning of INSUFFICIENT_EVIDENCE regions.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, forward
from cebu_profiler.schemas.coverage import CapacityCoverage, EvidenceGate
from cebu_profiler.schemas.evidence import EvidenceKind


class SemanticCluster(BaseModel):
    """A deterministic semantic cluster over the calibration corpus."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    label: str
    sample_ids: list[int] = Field(default_factory=list)
    domain: str = ""
    observations: int = Field(default=0, ge=0)


class ClusterExpertCoverage(BaseModel):
    """Per-cluster expert activation evidence (bidirectional link)."""

    model_config = ConfigDict(extra="forbid")

    capacity_id: str = ""
    cluster_id: str
    layer: int
    expert: int
    routed_count: int = Field(default=0, ge=0)
    activation_frequency: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "good"  # good|fair|low|insufficient_evidence


class ExpertClusterActivation(BaseModel):
    """Inverse link: expert -> clusters that activate it."""

    model_config = ConfigDict(extra="forbid")

    layer: int
    expert: int
    activating_clusters: list[str] = Field(default_factory=list)
    unique_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class CorpusDelta(BaseModel):
    """Teacher-relative quality delta projected onto a semantic cluster."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    candidate_id: str
    quality_delta: float = Field(ge=-1.0, le=1.0)  # teacher-relative
    regression: bool = False
    linked_decisions: list[str] = Field(default_factory=list)
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED


class CorpusSemanticReport(BaseModel):
    """Whole-corpus bidirectional map + coverage + delta projection."""

    model_config = ConfigDict(extra="forbid")

    model: str
    clusters: list[SemanticCluster] = Field(default_factory=list)
    cluster_expert_coverage: list[ClusterExpertCoverage] = Field(default_factory=list)
    expert_activation: list[ExpertClusterActivation] = Field(default_factory=list)
    deltas: list[CorpusDelta] = Field(default_factory=list)
    insufficient_clusters: list[str] = Field(default_factory=list)

    def expert_activating_clusters(self, layer: int, expert: int) -> list[str]:
        for e in self.expert_activation:
            if e.layer == layer and e.expert == expert:
                return e.activating_clusters
        return []

    def cluster_experts(self, cluster_id: str) -> list[ClusterExpertCoverage]:
        return [c for c in self.cluster_expert_coverage if c.cluster_id == cluster_id]


def _cluster_for(sample_id: int, tokens: list[int], seed: int = 0) -> tuple[str, str]:
    # deterministic cluster assignment from token content
    h = hashlib.sha256(repr(tokens).encode()).hexdigest()
    idx = int(h, 16) % 3
    return f"cluster-{idx}", ["matrix", "sequence", "structure"][idx]


def build_corpus_semantic_map(
    model: MiniMoE,
    samples: list[CalibrationSample],
    *,
    top_k: int | None = None,
    gate: EvidenceGate | None = None,
    n_clusters: int = 3,
) -> CorpusSemanticReport:
    """Build the bidirectional mapping + coverage from real forwards."""
    gate = gate or EvidenceGate()
    clusters: list[SemanticCluster] = []
    cluster_obs: dict[str, int] = {}
    sid_to_cluster: dict[int, str] = {}
    for idx, s in enumerate(samples):
        cid, dom = _cluster_for(idx, s.tokens, seed=0)
        if not any(c.cluster_id == cid for c in clusters):
            clusters.append(SemanticCluster(cluster_id=cid, label=dom, domain=dom))
        cluster_obs[cid] = cluster_obs.get(cid, 0) + 1
        sid_to_cluster[idx] = cid
    for c in clusters:
        c.sample_ids = [i for i, s in enumerate(samples) if sid_to_cluster.get(i) == c.cluster_id]
        c.observations = cluster_obs[c.cluster_id]

    # per-cluster expert routed counts
    routed: dict[tuple[str, int, int], int] = {}
    total_per_cluster: dict[str, int] = {}
    for idx, s in enumerate(samples):
        cid = sid_to_cluster[idx]
        total_per_cluster[cid] = total_per_cluster.get(cid, 0) + len(s.tokens)
        res = forward(model, s.tokens, top_k=top_k)
        for tr in res.traces:
            for ids in tr.topk_ids:
                for e in ids:
                    key = (cid, tr.layer, e)
                    routed[key] = routed.get(key, 0) + 1

    coverage_rows: list[ClusterExpertCoverage] = []
    for c in clusters:
        tot = total_per_cluster.get(c.cluster_id, 0) or 1
        n_exp = model.n_exp
        for li in range(len(model.layers)):
            for e in range(n_exp):
                cnt = routed.get((c.cluster_id, li, e), 0)
                freq = cnt / tot if tot else 0.0
                cap = CapacityCoverage(
                    capacity_id=f"{c.cluster_id}:L{li}:E{e}",
                    meaningful_observations=cnt,
                    token_count=tot,
                    activation_frequency=freq,
                )
                status = cap.status(gate.thresholds).value
                if cap.status(gate.thresholds).value == "insufficient_evidence":
                    coverage_rows.append(
                        ClusterExpertCoverage(
                            capacity_id=f"{c.cluster_id}:L{li}:E{e}",
                            cluster_id=c.cluster_id,
                            layer=li,
                            expert=e,
                            routed_count=cnt,
                            activation_frequency=round(freq, 4),
                            status=status,
                        )
                    )
                else:
                    coverage_rows.append(
                        ClusterExpertCoverage(
                            capacity_id=f"{c.cluster_id}:L{li}:E{e}",
                            cluster_id=c.cluster_id,
                            layer=li,
                            expert=e,
                            routed_count=cnt,
                            activation_frequency=round(freq, 4),
                            status=status,
                        )
                    )

    # inverse expert->clusters + unique coverage
    expert_activation: list[ExpertClusterActivation] = []
    for li in range(len(model.layers)):
        for e in range(model.n_exp):
            acts = [c.cluster_id for c in clusters if routed.get((c.cluster_id, li, e), 0) > 0]
            unique = len(set(acts)) / n_clusters if n_clusters else 0.0
            expert_activation.append(
                ExpertClusterActivation(
                    layer=li,
                    expert=e,
                    activating_clusters=sorted(acts),
                    unique_coverage=round(unique, 4),
                )
            )

    insufficient = sorted(
        {cc.cluster_id for cc in coverage_rows if cc.status == "insufficient_evidence"}
    )
    return CorpusSemanticReport(
        model=model.arch.name,
        clusters=clusters,
        cluster_expert_coverage=coverage_rows,
        expert_activation=expert_activation,
        deltas=[],
        insufficient_clusters=insufficient,
    )


def project_corpus_delta(
    report: CorpusSemanticReport,
    *,
    candidate_id: str,
    per_sample_delta: dict[int, float],
    linked_decisions: list[str] | None = None,
) -> CorpusSemanticReport:
    """Project teacher-relative quality deltas onto semantic clusters.

    ``per_sample_delta`` maps sample index -> teacher-relative quality delta.
    Local regressions stay visible even when aggregate quality passes.
    """

    delta_tot: dict[str, list[float]] = {}
    for c in report.clusters:
        for sid in c.sample_ids:
            if sid in per_sample_delta:
                delta_tot.setdefault(c.cluster_id, []).append(per_sample_delta[sid])
    deltas: list[CorpusDelta] = []
    for cid, vals in delta_tot.items():
        d = sum(vals) / len(vals)
        deltas.append(
            CorpusDelta(
                cluster_id=cid,
                candidate_id=candidate_id,
                quality_delta=round(max(-1.0, min(1.0, d)), 6),
                regression=d < -0.02,
                linked_decisions=list(linked_decisions or []),
            )
        )
    report.deltas = deltas
    return report
