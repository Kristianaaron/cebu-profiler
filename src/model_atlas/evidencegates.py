"""Eval + Pareto evidence gates (Phase 7).

A candidate can only be labelled MEASURED (and be deployable / on the measured
Pareto frontier) after it is (1) materialized to bytes, (2) held-out evaluated
on an immutable harness, and (3) runtime-benchmarked on the real two-node stack.
Anything short of that is PREDICTED and can never be marked deployable or put
on the measured frontier.

This module encodes those three gate predicates, so downstream Pareto promotion
and deployment logic cannot label a candidate MEASURED without the full chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_atlas.schemas.evidence import EvidenceKind


@dataclass
class MeasuredGateStatus:
    candidate_id: str
    materialized: bool = False  # derivative shards exist + hashes recorded
    heldout_evaluated: bool = False  # immutable held-out harness run
    runtime_benchmarked: bool = False  # real two-node vllm/NCCL run measured
    evidence_kind: EvidenceKind = EvidenceKind.PREDICTED
    gates: dict[str, bool] = field(default_factory=dict)

    def compute(self) -> None:
        all_pass = self.materialized and self.heldout_evaluated and self.runtime_benchmarked
        self.evidence_kind = EvidenceKind.MEASURED if all_pass else EvidenceKind.PREDICTED
        self.gates = {
            "materialized": self.materialized,
            "heldout_evaluated": self.heldout_evaluated,
            "runtime_benchmarked": self.runtime_benchmarked,
        }

    def is_measured(self) -> bool:
        self.compute()
        return self.evidence_kind is EvidenceKind.MEASURED

    def to_dict(self) -> dict[str, object]:
        self.compute()
        return {
            "candidate_id": self.candidate_id,
            "evidence_kind": self.evidence_kind.value,
            "gates": self.gates,
            "measured": self.evidence_kind is EvidenceKind.MEASURED,
        }


@dataclass
class FrontierRecorder:
    """Records measured vs predicted frontier points, never conflated."""

    measured: list[dict[str, object]] = field(default_factory=list)
    predicted: list[dict[str, object]] = field(default_factory=list)

    def add_candidate(
        self,
        candidate_id: str,
        *,
        quality: float | None,
        resident_gib: float | None,
        decode_tps: float | None,
        context_tokens: int | None,
        materialized: bool = False,
        heldout_evaluated: bool = False,
        runtime_benchmarked: bool = False,
        provenance: str = "",
    ) -> str:
        """Register a candidate onto the correct frontier.

        Returns the evidence kind it landed under. Only full-chain candidates go
        measured; everything else is predicted and never deployable.
        """
        gate = MeasuredGateStatus(
            candidate_id=candidate_id,
            materialized=materialized,
            heldout_evaluated=heldout_evaluated,
            runtime_benchmarked=runtime_benchmarked,
        )
        gate.compute()
        point: dict[str, object] = {
            "candidate_id": candidate_id,
            "quality": quality,
            "resident_gib": resident_gib,
            "decode_tps": decode_tps,
            "context_tokens": context_tokens,
            "evidence_kind": gate.evidence_kind.value,
            "provenance": provenance,
        }
        if gate.is_measured():
            self.measured.append(point)
        else:
            self.predicted.append(point)
        return gate.evidence_kind.value

    def measured_frontier(self) -> list[dict[str, object]]:
        return self.measured

    def predicted_frontier(self) -> list[dict[str, object]]:
        return self.predicted

    def to_dict(self) -> dict[str, object]:
        return {"measured": self.measured, "predicted": self.predicted}
