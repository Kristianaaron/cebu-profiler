# Changelog

## 0.6.0 — 2026-08-12

Real-bytes derivative-candidate planner (blueprint §24/§25, F10), grounded in
the measured GLM-5.2 NVFP4 census —

- `planning/realbytes.py`: `account_manifest` aggregates a checkpoint manifest
  into measured backbone (BF16) vs routed-expert bytes from each tensor's
  `byte_size` (never `numel × dtype_bytes` — the NVFP4 experts are ~8.19 bpw).
  `plan_candidates` produces per-envelope derivative candidates (default
  190/210/225 GB): pruning retention + target expert/backbone bpw, stored and
  per-node resident bytes, coverage, and a measured-vs-estimated tag (§31:20).
- `CLI real-candidates <checkpoint_dir>` prints the candidate report against a
  real mounted checkpoint.
- Real numbers (GLM-5.2 NVFP4, wired drive): 432.9 GiB measured (experts
  397.7 GiB @ 8.19 bpw, backbone 35.2 GiB @ 16 bpw). 190 GiB → keep 60% at FP8;
  210 GiB → keep 70% at FP8; 225 GiB → keep 50% at full 8.19 bpw — the measured
  echo of the prune-vs-uniform-low-bit thesis.
- Tests: `test_f20_realbytes` (7) incl. a drive-gated GLM integration; 162 total;
  ruff + mypy clean.

## 0.5.0 — 2026-08-12

Six-level atlas hierarchy (v2 §9) — traceable up and down, closing the gap
where only the L1 ownership layer existed —

- `atlas/hierarchy.py`: `AtlasLevel` (weights → units → experts → coalitions →
  pathways → behaviour), `HierarchyMap` with up (`ancestors` / `behaviours_of`)
  and down (`descendants` / `project_down`) traceability, and a measured
  `prevalence = #behaviours supported` signal per contributor.
- `build_hierarchy(model, samples)`: builds all six levels from real forwards —
  expert/channel aggregates (L1–L3), per-layer coalitions from measured path
  signatures (L4), cross-layer pathways (L5), and per-label success behaviours
  (L6). Every node tagged `measured`; `validate()` enforces adjacency + no
  dangling refs.
- Wired into the §27 machine-readable contract: `hierarchy_map.json` is now a
  guaranteed artifact at the enhanced evidence level and is emitted by
  `export_run()`.
- Tests: `test_f19_hierarchy` (8) + exported-artifact coverage in
  `test_atlas_export`; 155 total; ruff + mypy clean.

## 0.4.0 — 2026-08-07

Complete the offline-buildable blueprint phase-2 modules (§8.1/8.3/8.4, §10, Priority 4#5, §17 Control C) —

- `schemas/trace_records.py` + `atlas/traces.py`: normalized §10 trace records
  (`RouterRecord` / `ExpertAggregate` / `ChannelAggregate`), all measured.
- `scoring/semantic.py` (§8.1): capability-label → expert semantic associations
  (protection/explanation signal) fed into the manifest as `scores.semantic`.
- `scoring/redundancy.py` (§8.3): channel uniqueness from output-projection
  correlation + `KEEP_VALUE = importance·causal·uniqueness·stability`.
- `scoring/quant_sensitivity.py` (§8.4): per-expert quantization sensitivity →
  per-expert `quant_recommendation.bpw` recommendation.
- `planning/optimizer.py` (Priority 4#5): budget-constrained rate-distortion
  allocation → `CompressionManifest`.
- `experiments` (Control C, §17): depth-aware TENP-only arm + `compare_controls`
  (uniform / control_c / hetero) at matched budgets.
- Pipeline: `run_compression_pipeline` now enriches manifests with semantic /
  uniqueness / kvalue scores and measured quant bpw; planner propagates them.
- `ChannelScore` / `ExpertScores` extended (semantic, uniqueness, kvalue).
- Tests: `test_f18_blueprint_phase2` (7); 143 total; ruff + mypy clean.

## 0.3.1 — 2026-08-07

Wire the compression milestone into the export bridge —

- `atlas/export.py`: `export_run()` now also emits `compression_manifest.json`
  (trace → TENP → stability → causal → Taylor → SM121 width planner) over the
  same eval-lab calibration corpus.
- `output_layout.py`: `compression_manifest.json` added to the §27 canonical
  `ATLAS_RUN_FILES` set and guaranteed at the enhanced/causal evidence levels.
- Tests: export + output-layout coverage for the new artifact; 127 total.

## 0.3.0 — 2026-08-07

First end-to-end Atlas compression milestone (GLM-5.2 neuron/EXL3 blueprint §7–12, §25) —

- `atlas/collector.py` + `runtime` streaming channel collector (`ChannelStatsAccumulator`)
  — Module A: online per-(layer, expert, channel) FFN activation stats, no raw-tensor persistence.
- First-class scorers (`scoring/`): TENP (forward-only, NVFP4-ready), grouped-Taylor surrogate,
  targeted causal boundary triage, stability/confidence/rank aggregation (blueprint §9.1 base + §7 B–E).
- `planning/widths.py` + `width_buckets.py` (SM121 vocab): variable-width planner with
  coverage-target bucketing, confidence-penalized composite, protected-channel hard constraints.
- `schemas/manifest.py`: versioned, self-validating `CompressionManifest` (blueprint §11).
- `executor/structural.py`: coupled gate/up/down slicing executor + all six §12.2 tests
  (dry-run, permutation equivalence, topology, replay, protected).
- `integrations/glm52.py` / `integrations/k3.py`: architecture adapters (structural layout contract;
  real tensor sizes gated on checkpoint census).
- `atlas/compress.py`: `run_compression_pipeline()` — trace → TENP → stability → causal → Taylor
  → planner → manifest over the synthetic MiniMoE.
- Tests: 15 new `test_f16_compression_pipeline`; 125 total; ruff + mypy clean.

## 0.2.0 — 2026-08-04

Atlas export bridge (cross-repo manifest contract v1) —

- `atlas/export.py`: `export_run()` runs the real REAP pipeline (mini-MoE →
  eval-lab calibration corpus → saliency → candidate plans → optional
  derivative) and **writes** the canonical `atlas_runs/<id>/` artifacts
  (`run_manifest.json`, `layer_saliency.json`, `plans.json`, `derivative.json`),
  realizing the declared `output_layout.ATLAS_RUN_FILES` contract.
- CLI: `model-atlas export --eval-lab-root <path> --out <root> [--build]`.
- Source expert identity preserved end-to-end in keep-map + derivative maps.
- Tests: 5 new; 110 total; ruff + mypy strict clean.

## 0.1.0 — 2026-08-01

Initial scaffold (blueprint §21 first commit, model-agnostic):

- Package skeleton (`src/model_atlas`), hatchling build, `model-atlas` CLI.
- `ArchitectureSpec` + registry; Kimi K3 registered as the first subject, plus
  `k3-mini` synthetic model for deterministic unit tests.
- Tensor census + ownership (layerwise, source-identity preserving, no
  unclassified tensors), with coverage validation.
- Byte-accurate memory-planner stub that rejects plans exceeding either node's
  local budget (go/no-go).
- `AGENTS.md` invariants, architecture / tensor-ownership / memory-model docs.
- Deterministic unit tests; ruff + mypy config.

Not yet implemented (later milestones): streamed REAP atlas runtime, trace
capture, keep-map / derivative planning UI, checkpoint conversion, repair /
distillation, two-node serving runtime, eval-lab plugin bridge.
