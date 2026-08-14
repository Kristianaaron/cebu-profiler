"""V3 canonical pipeline orchestrator (v3 %5 / blueprint §4.3).

The canonical fidelity-first pipeline:
  teacher -> corpus coverage -> traces -> semantic/spectral/shared-structure
  evidence -> conditional sensitivity -> global EXL3 allocation -> optional
  conditioning -> EXL3 materialization -> refinement/correction -> NVFP4/FP8/BF16
  hardware allocation -> KV/system budget -> two-Spark benchmark -> corpus
  quality-delta projection -> Pareto/knee -> optional structural fallback ->
  optional recovery -> deployment manifest.

This orchestrator wires all v3 analyzers into a single reproducible run over a
synthetic mini-MoE, producing a machine-readable V3Run output. Every stage flags
measured-vs-predicted status honestly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.analysis.corpus_semantic import (
    CorpusSemanticReport,
    build_corpus_semantic_map,
)
from model_atlas.analysis.global_bit_budget import enumerate_global_bit_maps
from model_atlas.analysis.kv_memory import MemoryLedger, plan_kv_budget
from model_atlas.analysis.nvfp4_suitability import nvfp4_suitability
from model_atlas.analysis.quant_interaction import fit_quant_interaction
from model_atlas.analysis.routing_consistency import routing_consistency
from model_atlas.analysis.shared_representation import analyze_shared_representation
from model_atlas.analysis.spectral import analyze_spectral
from model_atlas.analysis.structural_fallback import structural_fallback_plans
from model_atlas.atlas.reap import CalibrationSample
from model_atlas.atlas.runtime import MiniMoE
from model_atlas.experiments.pareto_v3 import restrict_frontier
from model_atlas.schemas.coverage import CapacityCoverage, EvidenceGate
from model_atlas.schemas.evidence import EvidenceKind


class V3Run(BaseModel):
    """One full v3 pipeline execution's machine-readable output."""

    model_config = ConfigDict(extra="forbid")

    model: str
    seed: int
    stages_run: list[str] = Field(default_factory=list)
    corpus_semantic: CorpusSemanticReport | None = None
    shared_structure: object | None = None  # SharedAnalysis
    spectral: object | None = None
    bit_maps: dict[str, object] = Field(default_factory=dict)
    nvfp4: object | None = None
    routing_consistency_passed: bool | None = None
    kv_plan: object | None = None
    pareto: object | None = None
    structural_fallback: list[object] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)  # stage -> kind
    notes: str = ""


def V3Run_from(obj: object) -> V3Run:
    """Type alias helper (keeps mypy strict happy for object fields)."""
    return obj  # type: ignore[return-value]


def run_v3_pipeline(
    model: MiniMoE,
    corpus: list[CalibrationSample],
    *,
    seed: int = 0,
    top_k: int | None = None,
    budgets: tuple[float, ...] = (0.002, 0.003, 0.004),
    ledger: MemoryLedger | None = None,
) -> V3Run:
    """Execute the canonical v3 pipeline over the synthetic mini-MoE."""
    stages: list[str] = []
    result = V3Run(model=model.arch.name, seed=seed)
    gate = EvidenceGate()

    # 1. corpus evidence (coverage + bidirectional map)
    semantic = build_corpus_semantic_map(model, corpus, top_k=top_k, gate=gate)
    result.corpus_semantic = semantic
    stages.append("corpus_semantic")

    # 2. spectral + shared-structure evidence
    spectral = analyze_spectral(model, heavy_tail_modes=4)
    shared = analyze_shared_representation(model)
    result.spectral = spectral
    result.shared_structure = shared
    stages += ["spectral", "shared_structure"]

    # 3. global EXL3 bit-budget allocation (predictions)
    bit_maps = enumerate_global_bit_maps(model, budgets=budgets)
    result.bit_maps = {str(k): v for k, v in bit_maps.items()}
    stages.append("global_bit_budget")

    # 4. NVFP4 suitability probe
    nvfp4 = nvfp4_suitability(model, corpus, layers=None, experts=None)
    result.nvfp4 = nvfp4
    stages.append("nvfp4_suitability")

    # 5. routing-consistency identity gate (source == source must pass)
    identity = routing_consistency(model, model, corpus[:2], top_k=top_k)
    result.routing_consistency_passed = identity.passed
    stages.append("routing_consistency")

    # 6. KV/system-memory budget
    kv = plan_kv_budget(
        ledger or MemoryLedger(rank="node_a", physical_bytes=128 * 1024**3),
        arch_hidden=model.hidden,
        n_layers=len(model.layers),
        context_target_tokens=32000,
    )
    result.kv_plan = kv
    stages.append("kv_budget")

    # 7. quant-interaction surrogate fit (prediction model)
    fit_quant_interaction(model, sample_layers=[0], sample_experts=[0, 1])
    stages.append("quant_interaction")

    # 8. structural fallback (evidence-gated; under-observed never reduced)
    widths = {
        (li, e): model.mid for li in range(len(model.layers)) for e in range(model.n_exp)
    }
    _cov_map: dict[str, CapacityCoverage] = {}
    for c in semantic.cluster_expert_coverage:
        _cov_map[c.capacity_id] = CapacityCoverage(
            capacity_id=c.capacity_id,
            meaningful_observations=c.routed_count,
            token_count=c.activation_frequency and 0 or 0,
            activation_frequency=c.activation_frequency,
        )
    fallback = structural_fallback_plans(
        widths,
        coverage=_cov_map,
        gate=gate,
        reductions=(0.05, 0.10),
    )
    result.structural_fallback = list(fallback)
    stages.append("structural_fallback")

    # 9. Pareto over a small synthetic candidate family. These are IDEALIZED
    #    numbers for demonstration on the miniature, NOT measurements: they are
    #    tagged PREDICTED (never MEASURED) and can never be marked deployable.
    #    Real measured numbers from materialized + evaluated + runtime-measured
    #    candidates replace these behind the measured data path.
    from model_atlas.experiments.pareto_v3 import FrontierPoint

    candidates = []
    family = (
        (0.250, 0.945, 24.0, 600000),  # compact: low GiB, low quality
        (0.275, 0.950, 23.5, 560000),
        (0.300, 0.960, 22.0, 480000),
        (0.350, 0.985, 21.0, 360000),
        (0.375, 0.990, 20.5, 320000),
        (0.400, 0.995, 19.5, 260000),  # fidelity: high GiB, high quality
    )
    for i, (size_gb, q, decode, ctx) in enumerate(family):
        candidates.append(
            FrontierPoint(
                candidate_id=f"mk-{i}",
                values={
                    "quality": q,
                    "resident_gib": size_gb,
                    "decode_tps": decode,
                    "context": ctx,
                },
                evidence_kind=EvidenceKind.PREDICTED,
            )
        )
    pareto = restrict_frontier(candidates)
    result.pareto = pareto
    stages.append("pareto")

    result.stages_run = stages
    result.evidence = {
        "corpus_semantic": EvidenceKind.MEASURED.value,
        "spectral": EvidenceKind.ESTIMATED.value,
        "shared_structure": EvidenceKind.ESTIMATED.value,
        "global_bit_budget": EvidenceKind.PREDICTED.value,
        "nvfp4": EvidenceKind.ESTIMATED.value,
        "routing_consistency": EvidenceKind.MEASURED.value,
        "kv_budget": EvidenceKind.ESTIMATED.value,
        "structural_fallback": EvidenceKind.PREDICTED.value,
        "pareto": EvidenceKind.PREDICTED.value,
    }
    result.notes = (
        "fidelity-first pipeline; pareto points are PREDICTED (idealized demo "
        "family), not measured — never deployable until materialized + "
        "held-out evaluated + runtime measured"
    )
    return result


def v3_run_to_jsonable(run: V3Run) -> dict[str, object]:
    """JSON-safe extraction for dashboard embedding (tuple keys -> str)."""

    def _strip(obj: object) -> object:
        if isinstance(obj, dict):
            return {str(k): _strip(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        if hasattr(obj, "model_dump"):
            return _strip(obj.model_dump())
        if isinstance(obj, tuple):
            return [_strip(v) for v in obj]
        return obj

    return _strip(run.model_dump())  # type: ignore[return-value]


__all__ = ["V3Run", "run_v3_pipeline", "v3_run_to_jsonable"]
