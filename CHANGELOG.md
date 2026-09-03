# Changelog

## 1.0.0 — 2026-08-13

Cebu Profiler v3 fidelity-first architecture (per the v3 implementation instruction +
GLM-5.2 blueprint) —

- New `analysis/` analyzers (all evidence-disciplined, never mutate weights):
  - `SharedRepresentationAnalyzer` (KLT/PCA-style shared-vs-unique energy),
    `SpectralQualityAnalyzer` (effective rank / energy / heavy tail / uniqueness),
    `ConditionalSensitivityModel` (upstream-conditional sensitivity, MixQuant-style),
    `RoutingConsistencyGuard` (top-k / rank / boundary / JS divergence gate, VSRAQ-style),
    `GlobalBitBudgetOptimizer` (GEMQ-style EXL3 allocation per system budget),
    `QuantizationInteractionModel` (additive+pairwise surrogate with confidence certificate),
    `FixedGridRefiner` (ReQuant-style fixed-storage refinement),
    `ResidualCorrectionPlanner` (+bpw vs residual-correction vs NVFP4/FP8 at matched memory),
    `NVFP4SuitabilityAnalyzer` (tolerance + routing-impact + QAD/CKA-QAD recovery flag),
    `KVMemoryOptimizer` + per-rank `MemoryLedger` (weights/runtime/KV/OS/headroom),
    `StructuralFallbackPlanner` (5–20% fold-in variants, evidence-gated).
- New `schemas/coverage.py`: `INSUFFICIENT_EVIDENCE` gate — absence of activation is
  not evidence of irrelevance; destructive actions on under-observed capacity are
  blocked unless an explicit auditable override is recorded. Rare-specialist
  capacity is distinguished from redundant capacity.
- New `candidates/` graph: immutable candidate lineage with parent→child DAG,
  predicted-vs-measured status, operator + provenance, per-tensor representation
  map, memory breakdown, routing stability, corpus hotspots. Predictions can never
  be deployable; derived measured candidates from a deployable parent are blocked.
- New `experiments/pareto_v3.py`: nondominated frontier, **knee as a scored
  region**, neighbor deltas (fidelity/compact moves) with marginal quality/GiB —
  predictions and measured points are never conflated.
- `analysis/corpus_semantic.py`: bidirectional corpus↔model map (cluster→experts,
  expert→activating clusters) with per-cluster coverage statuses and teacher-relative
  quality-delta projection onto clusters.
- `profiler/v3_pipeline.py`: canonical fidelity-first orchestrator streaming all analyzers.
- Dashboard: new **Researcher** nav section with **V3 Analyzers / Candidate Graph /
  Corpus Evidence** tabs over measured data; prediction-vs-measured labels on every
  surface.
- CLI: `analyze`, `v3-pareto`, `v3-candidates`, `v3-corpus` subcommands.
- §27 output contract extended with `v3_run.json`, `v3_candidate_graph.json`,
  `v3_corpus_evidence.json`, `shared_representation.json`, `spectral_quality.json`,
  `routing_consistency.json`, `global_bit_budget.json`, `kv_ledger.json`,
  `pareto_frontier.json`; `export_run` emits the v3 artifacts.
- Tests: `test_f22_v3` (13) + `test_f23_v3_system` (6) + export/dashboard updates;
  189 total; ruff + mypy (`src`) clean.

# Changelog

## 0.8.3 — 2026-08-13

Success−Failure tab readability pass —

- `ContrastAccumulator.cell_saliency(label, layer, expert, pos, neg)` exposes
  the raw per-cell mean saliencies on successful vs failed runs; `_contrast_voxels`
  now emits `pos`/`neg` alongside `delta` so the panel shows the actual
  components, not just the derived number.
- Contrast canvas gets per-capability **column headers** with more row spacing,
  cell text colour by polarity (bright on success-favoured filled tiles), and
  the hover panel adds a plain-language finding ("routing tied to good
  outcomes" vs "goes with failures") plus a success/failure saliency table.
- Fix: `canvas#cap3d-contrast` gets an explicit responsive `width:100%` +
  `aspect-ratio` rule so it never collapses to the intrinsic 300×150 canvas size
  (mirrors the capability canvas).
- Tests: test_f14 asserts raw pos/neg bounds and the responsive canvas rules;
  170 total; ruff + mypy (`src`) clean.

## 0.8.2 — 2026-08-12

Capability tab: monochrome 3D voxel view + sticky detail panel, agent-friendly —

- `_capability_voxels`: compact per-(label, layer, expert) voxel payload
  (score normalised per label; bounded by geometry).
- Vanilla, dependency-free 3D voxel canvas (no three.js): **monochrome
  grayscale with a 1-bit Bayer ordered-dither look** on a near-black canvas;
  drag-to-rotate, wheel-zoom, hover, click-to-pin. **Render-on-interaction
  only** (no animation loop) and **devicePixelRatio-scaled with integer pixel
  snapping**, so it is crisp/high-res and stays compute-light on laptops.
- **Sticky right panel** updates on hover/select: shows the focused capability,
  per-layer expert rows with a density bar, and truncates long text. Clicking a
  **layer row opens that layer's detail view** (per-expert rows, w/ back link).
- **True 3D cubes** (all 8 corners + visible faces projected) so rotating stays
  readable; layers spaced wider; hover/ease with a generous hit radius that
  highlights the whole layer slice.
- **Isometric stack of translucent sheet panels, dark mode**: each
  (capability, layer) is a thin translucent sheet panel, and panels stack up in
  a fixed iso view (capabilities/experts on the two diagonals, layers up) on the
  near-black canvas — no cubes, no outlines. Cell opacity = saliency; spacious
  cells (GAP 0.6) stay disjoint so hover is exact (interior-verified).
- **Tier tags** on the panel rows and the canonical capability table: a
  colour-coded tag per score — strong ≥0.75, good ≥0.5, moderate ≥0.25, weak
  <0.25 (relative per label). Capability top-scores are normalised per label so
  table and panel agree; fixed `${…}` escaping across the capability/contrast/
  coalition tables so values render.
- **Translucent light cubes + top-left controls**: cubes render as translucent
  light-grey solids with faint edges and low default opacity so the
  selected/hovered cube (bright fill + white double-edge) clearly contrasts;
  all 192 cubes stay visible. A top-left overlay has **zoom −/＋ buttons** and a
  **"layers" slider** that spreads the layers vertically apart (1×–4×) so
  hovering/selecting an individual expert is easier. Drag-rotate + scroll-zoom
  limits kept.
- **Grid + vignette behind the canvas** (`#07070a` 48px grid + radial vignette
  fading the edges, beneath the right panel, o-cina recipe); the connecting
  sheet-outline was removed so cells read as separate panels.
- **Layer filter** on the right panel: L0 / L1 / … / all chips to view each
  layer separately or together, and it filters both the canvas and the panel.
- **Hover = the tile under the cursor**: point-in-polygon over each tile's
  top rhombus; where translucent overlapped tiles both contain the cursor it
  picks the nearest center (the tile you're aiming at). Cursor becomes a
  **pointer** over a tile.
- See-through grayscale cubes with **opacity-as-saliency** (hot = more opaque),
  no dither/outline; high-DPI/integer-snapped, no animation loop.
- Agent-friendly: `<caption>` + `th scope="col"` on every table; a
  `<script type="application/json" id="cap3d-json">` structured block; panel is
  `aria-live`; the canonical table remains the ground truth.
- Tests: test_f14 asserts the voxel payload; JS validated in node (syntax +
  runtime: 192 voxels dithered, panel rows/groups render).

## 0.8.1 — 2026-08-12

Native Cebu↔Eval bridge: eval-lab's Cebu Lab now natively consumes the full
profiler picture (no data dropped) —

- `profiler/export.py` emits a consolidated `planning_maps.json` (the seven
  granular §25 maps + per-candidate precision/residency/coverage), registered
  in the §27 output contract so nothing is flagged as an unknown artifact.
- Prepared for the eval-harness consumer: bridge carries precision/residency/
  coverage + §25 maps + real-bytes candidates end to end.

## 0.8.0 — 2026-08-12

Cohesive Cebu Profile Platform IA (Profile vs Quantization & Fit) —

- Dashboard reorganised into three nav sections + an ecosystem link: **Overview**
  (Summary), **Profiling** (Capability, Success−Failure, Coalitions, Paths,
  Hierarchy, Planning Maps), **Quantization & Fit** (Compression, Derivatives,
  Held-out, Real-bytes), and a nav link to the **Eval Harness** (standalone
  benchmarking app).
- New **Hierarchy** view: the six-level §9 hierarchy node counts + a trace-down
  example (components realising the first behaviour, peak shared-channel
  prevalence).
- New **Real-bytes** view: §24/§25 derivative envelopes from measured
  checkpoint bytes — the mounted GLM-5.2 NVFP4 census when present, else a
  synthetic caret. So Cebu Profiler = the profiling/fit platform; Eval stays the
  benchmark app.
- Tests: test_f14 extended to defend the new payloads; 169 total; ruff + mypy clean.

## 0.7.0 — 2026-08-12

Complete the §25 planning-artifact set (all 13 maps now exist) —

- `planning/maps.py`: 7 new typed maps — ChannelMap, TileMap, NodeOwnershipMap,
  OverflowPackMap, RouterRepairMap, ResidualRepairMap, DistillationTargetMap —
  joining the existing keep/precision/residency/coalition-protection/
  path-preservation/substitute maps.
- `planning/maps_build.py`: `build_planning_maps` produces all 7 from a
  synthetic model + measured REAP saliency. Channel/tile maps grounded in
  measured channel-uniqueness (§8.3); node-ownership from the census placement;
  router-repair reindexes expert↔router slots contiguously with route_bias
  locked on (§31:18); overflow/residual/distillation derive from measured
  saliency and are tagged estimated.
- Dashboard: new "Planning Maps" tab rendering all 7 maps for browser QA.
- Tests: `test_f21_maps` (7); 169 total; ruff + mypy clean.

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

Six-level profiler hierarchy (v2 §9) — traceable up and down, closing the gap
where only the L1 ownership layer existed —

- `profiler/hierarchy.py`: `ProfilerLevel` (weights → units → experts → coalitions →
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
  `test_profiler_export`; 155 total; ruff + mypy clean.

## 0.4.0 — 2026-08-07

Complete the offline-buildable blueprint phase-2 modules (§8.1/8.3/8.4, §10, Priority 4#5, §17 Control C) —

- `schemas/trace_records.py` + `profiler/traces.py`: normalized §10 trace records
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

- `profiler/export.py`: `export_run()` now also emits `compression_manifest.json`
  (trace → TENP → stability → causal → Taylor → SM121 width planner) over the
  same eval-lab calibration corpus.
- `output_layout.py`: `compression_manifest.json` added to the §27 canonical
  `CEBU_RUN_FILES` set and guaranteed at the enhanced/causal evidence levels.
- Tests: export + output-layout coverage for the new artifact; 127 total.

## 0.3.0 — 2026-08-07

First end-to-end Cebu Profiler compression milestone (GLM-5.2 neuron/EXL3 blueprint §7–12, §25) —

- `profiler/collector.py` + `runtime` streaming channel collector (`ChannelStatsAccumulator`)
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
- `profiler/compress.py`: `run_compression_pipeline()` — trace → TENP → stability → causal → Taylor
  → planner → manifest over the synthetic MiniMoE.
- Tests: 15 new `test_f16_compression_pipeline`; 125 total; ruff + mypy clean.

## 0.2.0 — 2026-08-04

Cebu Profiler export bridge (cross-repo manifest contract v1) —

- `profiler/export.py`: `export_run()` runs the real REAP pipeline (mini-MoE →
  eval-lab calibration corpus → saliency → candidate plans → optional
  derivative) and **writes** the canonical `profiler_runs/<id>/` artifacts
  (`run_manifest.json`, `layer_saliency.json`, `plans.json`, `derivative.json`),
  realizing the declared `output_layout.CEBU_RUN_FILES` contract.
- CLI: `cebu-profiler export --eval-lab-root <path> --out <root> [--build]`.
- Source expert identity preserved end-to-end in keep-map + derivative maps.
- Tests: 5 new; 110 total; ruff + mypy strict clean.

## 0.1.0 — 2026-08-01

Initial scaffold (blueprint §21 first commit, model-agnostic):

- Package skeleton (`src/cebu_profiler`), hatchling build, `cebu-profiler` CLI.
- `ArchitectureSpec` + registry; Kimi K3 registered as the first subject, plus
  `k3-mini` synthetic model for deterministic unit tests.
- Tensor census + ownership (layerwise, source-identity preserving, no
  unclassified tensors), with coverage validation.
- Byte-accurate memory-planner stub that rejects plans exceeding either node's
  local budget (go/no-go).
- `AGENTS.md` invariants, architecture / tensor-ownership / memory-model docs.
- Deterministic unit tests; ruff + mypy config.

Not yet implemented (later milestones): streamed REAP profiling runtime, trace
capture, keep-map / derivative planning UI, checkpoint conversion, repair /
distillation, two-node serving runtime, eval-lab plugin bridge.
