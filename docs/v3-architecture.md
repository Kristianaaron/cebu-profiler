# Atlas v3 — Fidelity-First Analyzer Architecture

Status: implemented 2026-08-13 against the synthetic MiniMoE (model-agnostic).

## Canonical pipeline
1. corpus evidence (coverage + bidirectional map, evidence gate)
2. spectral + shared-structure evidence
3. global EXL3 bit-budget allocation (predictions)
4. NVFP4 suitability probe (tolerance + routing impact + recovery flag)
5. routing-consistency identity gate
6. KV/system-memory budget (per-rank ledger)
7. quant-interaction surrogate fit (confidence certificate)
8. structural fallback (evidence-gated, never default-prune)
9. Pareto frontier + knee region
Orchestrated in `atlas/v3_pipeline.py` as `run_v3_pipeline(...)`.

## Analyzers (`model_atlas/analysis/`)
Each analyzer is a pure evidence producer; it never mutates weights and never
turns a prediction into a measured result.

| Analyzer | File | Evidence kind |
|---|---|---|
| SharedRepresentationAnalyzer | shared_representation.py | estimated |
| SpectralQualityAnalyzer | spectral.py | estimated |
| ConditionalSensitivityModel | conditional_sensitivity.py | estimated |
| RoutingConsistencyGuard | routing_consistency.py | measured |
| GlobalBitBudgetOptimizer | global_bit_budget.py | predicted |
| QuantizationInteractionModel | quant_interaction.py | predicted |
| FixedGridRefiner | refiner.py | measured |
| ResidualCorrectionPlanner | residual_correction.py | predicted |
| NVFP4SuitabilityAnalyzer | nvfp4_suitability.py | estimated |
| KVMemoryOptimizer + MemoryLedger | kv_memory.py | estimated |
| StructuralFallbackPlanner | structural_fallback.py | predicted |

## Evidence-discipline rules (non-negotiable)
- Predictions are never deployable; every candidate records predicted vs measured.
- A measured (non-predicted) candidate cannot derive from a deployable parent.
- INSUFFICIENT_EVIDENCE capacity is never reduced unless an explicit override is
  recorded (`schemas/coverage.py`).
- Absence of activation is not evidence of irrelevance.

## Candidate graph
`model_atlas/candidates/graph.py`: immutable DAG (parent→child), operator +
provenance, per-tensor representation map, memory breakdown, routing stability,
corpus hotspots.

## Pareto (v3)
`model_atlas/experiments/pareto_v3.py`: nondominated frontier over active
objectives, knee as a scored REGION (never a single point), and per-candidate
neighbor deltas (fidelity/compact) with marginal quality/GiB.

## Integration
- Dashboard: Researcher → V3 Analyzers / Candidate Graph / Corpus Evidence.
- CLI: `model-atlas analyze`, `v3-pareto`, `v3-candidates`, `v3-corpus`.
- §27 output contract: `v3_run.json`, `v3_candidate_graph.json`,
  `v3_corpus_evidence.json`, `shared_representation.json`, `spectral_quality.json`,
  `routing_consistency.json`, `global_bit_budget.json`, `kv_ledger.json`,
  `pareto_frontier.json`.

## Gated on real infrastructure (not synthetic)
NVFP4 SM121 kernel-path measurement, actual EXL3 materialization, BF16-parent
authoritative rankings, and two-Spark runtime profiling.
