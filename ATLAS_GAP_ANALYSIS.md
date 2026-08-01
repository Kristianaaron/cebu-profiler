# ATLAS_GAP_ANALYSIS.md

**Contract:** Parent-to-Derivative Model Foundry v2 (Kimi K3 Semantic, Causal,
Geometric, Compression, and Deployment Atlas).
**Scope reviewed:** current `model-atlas` repository (the Atlas subsystem),
plus `eval-lab` (the evaluation harness, referenced as the plugin/shared
substrate). Generated 2026-08-01.
**Status:** gap analysis only. No code changes beyond the scaffold already
present. Stopping for review per contract §30.

---

## 1. Current architecture (actual state)

`model-atlas` is a brand-new, model-agnostic package. As of this analysis only
the foundation slice exists — no git repo, no tests, no CLI, no runtime:

```
src/model_atlas/
  schemas/architecture.py     # DType, LayerKind, TensorRole, MoELayout, ArchitectureSpec
  schemas/__init__.py
  census/tensor_ownership.py  # TensorOwnership, PhysicalLocation, PlacementPolicy, OwnershipManifest
  census/census.py            # build_manifest(): enumerate tensors -> ownership manifest
  __init__.py
```

What the foundation actually delivers:

- **ArchitectureSpec** — model-agnostic structural descriptor (layers, MoE
  geometry, vocab, dtypes, tensor-role -> numel table). K3 is not yet registered.
- **TensorRole / DType / byte accounting** — every tensor maps to one role; bytes =
  numel × dtype-bytes. Foundational for "no unclassified tensors."
- **Census / ownership** — enumerates per-layer + global tensors into an
  `OwnershipManifest` with uniqueness validation, source identity preserved, and
  placement policy (expert-parallel + shared-on-A). Honest
  `needs_source_measurement` path when real tensor sizes are unknown.

Not yet present: memory planner, architecture registry, synthetic mini-MoE,
CLI, tests, docs beyond the two foundation modules.

`eval-lab` (separate repo, option-A dependency) already contains: model-asset
schema/registry, task/suite + label ontology, evaluation service, job
orchestrator with persistent state machine, leakage guard, comparison engine,
and a Svelte GUI (M1/M2). It is the evaluation harness the Atlas plugs into;
it is **not** the Atlas.

---

## 2. Target architecture (contract)

The v2 contract requires the full cycle:

```
Oversized parent checkpoint
  -> structural inventory -> labelled tasks/trajectories
  -> layerwise routing/contribution/representation tracing
  -> semantic profiles -> counterfactual routing & causal interventions
  -> coalitions & cross-layer pathways -> weight/channel/tile/compression analysis
  -> derivative architecture search -> checkpoint construction
  -> router/residual/behavioural repair -> held-out evaluation
  -> two-DGX-Spark deployment -> regression-driven iteration
```

Backed by six linked atlas levels (weights→units→experts→coalitions→pathways→
behaviour), ten required maps, seven trace families, a staging funnel, evidence
grades + negative controls, an extensible compression-backend interface
(source MXFP4 / BF16 / FP16 / FP8 / NVFP4 / INT8 / INT4 / EXL3 / AQLM /
structured pruning / removed / NVMe overflow), an automated architecture-search
engine, derivative construction/repair, ~45 machine-readable outputs, and a
complete GUI.

---

## 3. Implemented / partial / missing matrix

Legend: **IMP** implemented · **PAR** partial · **MISS** missing
Where a capability exists, its owner is noted (MA = model-atlas, EL = eval-lab).

| Contract capability (§30 list) | Status | Notes |
|---|---|---|
| Persistent job orchestration | PAR | EL has a persistent eval-job state machine; **none** in MA for atlas jobs |
| Structural model / schemas | PAR | MA: ArchitectureSpec + ownership; no full §6 structural graph, vision/quant classes |
| Routing **beyond frequency** | MISS | No atlas runtime; no routing traces |
| Contribution tracing | MISS | — |
| Residual tracing | MISS | — |
| Representation tracing | MISS | — |
| Success / failure / recovery contrasts | MISS | §12 not built |
| Free-generation traces | MISS | incl. teacher-forced/free/counterfactual modes |
| Counterfactual routing | MISS | §13 not built |
| Route regret | MISS | — |
| Expert substitution | MISS | §10.6 not built |
| Coalition analysis | MISS | §10.7 / §17 |
| Multi-component causal tracing | MISS | §17 synergy/redundancy |
| Cross-layer pathway mapping | MISS | §10.8 / §14 |
| Sparse features / dictionary learning | MISS | §15 |
| Vocabulary projections | MISS | §16 |
| Neuron / channel / block / tile maps | MISS | §18 |
| EXL3 and AQLM probes | MISS | no compatibility matrix, no reports |
| Expert response curves | MISS | §23 |
| Uncertainty and negative controls | MISS | §20 none |
| Automated derivative architecture search | MISS | §24 |
| All required machine-readable outputs | MISS | MA emits only ownership manifest; no `atlas_runs/<id>/` layouts |
| Complete GUI workflows | PAR | EL has M1/M2 GUI; no Atlas Lab views |

### Foundational candidate (pre-milestone) — what the current scaffold buys

| Foundation | Status | Notes |
|---|---|---|
| Model-agnostic architecture spec | IMP | ArchitectureSpec |
| Byte accounting by dtype | IMP | DTYPE_BYTES |
| Tensor classification by role | PAR | covers experts/shared/latent/attention/norm/embed/lm_head; **missing** vision, quantization-metadata, and neuron/channel/tile granularity |
| Tensor census + ownership + no-unclassified | IMP | build_manifest + validation |
| Source-identity preservation | IMP | expert_index + key scheme (extends to source↔derivative in §26) |
| Memory / go-no-go planner | MISS | planned stub, not yet written |
| Architecture registry (K3, mini) | MISS | not yet written |
| Synthetic miniature MoE | MISS | needed for deterministic tests |

---

## 4. Data-contract gaps

1. **No AtlasRun / AtlasTrace schemas in MA.** eval-lab `schemas/atlas.py`
   reserves (`AtlasRunManifest`, `AtlasTraceField`, `ExpertIdentity`,
   `EvidenceLevel/Kind`) — must be adopted/extended in MA (or shared via the
   bridge) to match v2 §8 trace record (adds `generation_mode`,
   `success_state`, `trace_schema_version`) and §9 six-level hierarchy.
2. **No trace families.** §11 defines routing/contribution/representation/
   intervention traces; no schemas exist.
3. **No maps.** §10 requires structural, activation, contribution, functional,
   causal, redundancy/substitution, coalition, cross-layer, compression,
   deployment maps. Only an ownership manifest exists.
4. **No behaviour/trajectory ontology in MA.** Labels + stages exist in eval-lab
   (§8 v1 ontology); v2 adds success states (`success/failure/recovered/
   partially_recovered/unknown`). Needs to be shared via the plugin bridge.
5. **No data-partition model in MA.** `atlas_calibration / development_evaluation /
   held_out_evaluation` + leakage detection live in eval-lab; not surfaced here.
6. **No evidence-grades model.** §20 grades and §31 no-fabrication rules have no
   typed carrier in MA.
7. **No machine-readable output contract.** §27 `atlas_runs/<id>/` layout
   (run_manifest, parquet set, evidence_registry, uncertainty_report) absent.
8. **No model-asset lifecycle in MA.** §5 fields exist in eval-lab; derivative
   asset types (`derivative_checkpoint`, `student_model`) need extension.

---

## 5. Runtime gaps

1. **No layerwise checkpoint execution** — the core Atlas runtime. Must stream
   one layer at a time from an oversized checkpoint without materializing the
   full model (REAP-style loop, skill `reap-compression`).
2. **No streaming census** against a real oversized checkpoint (safe-tensor
   header reads only; needs shard enumeration + hashing + structural graph).
3. **No memory-planner** (byte-accurate, per-node go/no-go) — priority gap.
4. **No job orchestrator scoped to atlas jobs** — MA has none; eval-lab's
   orchestrator supports safe pause/resume/cancel at chunk/layer boundaries but
   is wired for eval jobs, not atlas layerwise chunks.
5. **No compression backend interface** — no quantization/EXL3/AQLM support,
   no support-status model (conversion vs probe vs inference).
6. **No tracing I/O** — token-trace index, parquet writers, artifact store,
   provenance linking. Not even JSONL.
7. **No two-Spark / elastic serving** — §deployment entirely absent.
8. **No repair / distillation runtime** — §26 separate jobs absent.

---

## 6. Research-method gaps

1. **Everything after routing discovery is unbuilt:** contribution,
   representation, residual tracing (§11–§12).
2. **No counterfactual routing / route regret** (§13) — disproving "router chose
   best" requires this.
3. **No coalitions or multi-component causal tracing** (§17) — the A+B
   catastrophic-cascade case is untestable.
4. **No cross-layer pathway discovery** (§14) — path signatures, branch points,
   success/failure divergence.
5. **No sparse-feature / dictionary learning** (§15, §16) — polysemanticity,
   feature redundancy, vocabulary projection.
6. **No neuron/channel/tile sensitivity** (§18) — the finest intervention grain.
7. **No staged funnel** (§19) A→G, so no evidence-level accounting.
8. **No evidence grades / negative controls** (§20) — frequency-matched controls,
   random masks, route-preserving quantization, alternate corpora.
9. **No keep/precision/residency/coalition/path/substitute maps** or candidate
   optimiser (§24/§25).

---

## 7. Compression-backend gaps

1. **No `CompressionBackend` protocol** (§21) — the interface itself is absent.
2. **Zero backends implemented** — `source_mxfp4`, `bf16/fp16`, `fp8`, `nvfp4`,
   `int8/int4`, `exl3`, `aqlm`, `structured_pruning`, `removed`,
   `nvme_overflow`, `custom_research_backend` are all unbuilt.
3. **No EXL3/AQLM compatibility matrix or reports** (§22). `AQLM`, `EXL3` are
   **not** assumed to support K3; pinned-revision audits required.
4. **No per-expert response curves** (§23) across FP8/NVFP4/INT4/EXL3/AQLM/
   structured pruning/removal/substitution.
5. **No support-status discipline** — conversion must never be presented as
   deployable inference (§31: 12, 13, 14, 24).

---

## 8. GUI gaps

All §28 views are missing in MA: atlas summary, capability/success-failure/
trajectory/layer/expert/neuron-feature/coalition/path/route-regret/
compression/quant-compat/response-curve/keep-map/architecture-search/memory /
regression-to-atlas / evidence explorers.

`eval-lab` provides the dashboard shell (M1/M2) and model-asset + job-monitor
views. The Atlas Lab is a distinct surface to be added — per v2 §31:22, long
atlas jobs must not depend on the GUI; heavy analysis never runs in frontend
components (§31:23).

---

## 9. Testing gaps

1. **No tests yet** in MA (no pytest config, no runner). Foundation modules are
   untested as written (census logic is untested).
2. **No synthetic miniature MoE** — the deterministic unit-test substrate
   required for every milestone (real K3 is 1.56 TB and must never be used in
   fast tests).
3. No golden fixtures for census/ownership, no numeric-equivalence/identity
   control tests (§31:5), no evidence-chain tests.
4. No integration harness for layerwise streaming (mock/synthetic model).
5. No paired/AB, seed-stability, or corpus-resample tests for research claims
   (§20), nor leakage-blocking promotion tests (present in eval-lab; needs
   surfacing here).

---

## 10. Risk classification

| Risk | Severity | Mitigation |
|---|---|---|
| Building ranking-only Atlas (violates §31:1, "one saliency score") | **High** | Keep contribution/representation/counterfactual/coalition in scope from milestone 2 onward |
| Correlation presented as causation (§31:4) | **High** | Evidence grades + negative controls as typed, gated outputs |
| Losing source expert identity during renumbering (§31:17,18) | **High** | Ownership already identity-preserving; extend to maps + builder |
| Real checkpoint resist measurement (census parity) | **Medium** | Model-agnostic spec + synthetic fixtures; verify 100% tensor-key on a real sample before relying |
| EXL3/AQLM on K3 unsupported (§31:12) | **Medium** | Probe minimum viable experts first; never report skipped tests as passing |
| Derivative promoted without held-out validation (§31:25) | **Medium** | Held-out + leakage gate mandatory before promotion |
| Scope bloat ("one huge application") | Medium | Keep eval-lab = harness, model-atlas = Atlas/derivative; plugin bridge one-way |
| Two-Spark latency/memory assumptions untested | Medium | Byte-accurate planner + go/no-go before any candidate build |

---

## 11. Recommended milestone sequence

Derived from v2 §29, re-scoped to the current near-empty MA repo. Milestones M[1-9]
are foundation/atlas; M10+ are the derivative/repair/deployment surface.

1. **F0 — Finish commit-1 foundation:** memory-planner (go/no-go), architecture
   registry (register K3 lay-out + mini synthetic MoE), CLI (`doctor`,
   `census`, `plan`), pytest config + deterministic tests, tensor-ownership and
   benchmark. *(existing §21 first commit, unfinished)*
2. **F1 — Data-contract substrate (v2 §10/§8):** AtlasRun/AtlasTrace, trace
   families, behaviour/stage/success ontology (share via eval-lab bridge),
   evidence grades, negative controls, `atlas_runs/<id>/` output contract,
   model-asset lifecycle extension.
3. **F2 — Streaming layerwise census + structural graph** (§6): shard
   enumeration, hashing, tensor classification incl. vision/quant metadata,
   structural_model_graph.json. No full model materialization.
4. **F3 — Streamed REAP routing atlas** (§19 stage A/B, \routing + contribution):
   per-label/per-stage saliency, activation & contribution maps; memory-safe
   layerwise loop.
5. **F4 — Representation + success/failure/recovery contrasts** (§11–§12);
   free-generation + teacher-forced modes.
6. **F5 — Counterfactual routing + route regret** (§13).
7. **F6 — Sparse features + vocabulary projections** (§15–§16).
8. **F7 — Compression-backend protocol + probes + response curves** (§21–§23);
   EXL3/AQLM feasibility milestone (§29 M8).
9. **F8 — Coalitions + multi-component causal tracing** (§17); substitution map
   (§10.6).
10. **F9 — Cross-layer pathway mapping** (§14); neuron/channel/tile analysis (§18).
11. **F10 — Derivative architecture search + planning artifacts** (§24–§25):
    keep/precision/residency/coalition/path/substitute maps; 190/210/225 GB
    candidates.
12. **F11 — Derivative builder + renumbering + validation** (§26).
13. **F12 — Repair/distillation runtime** (§26) + held-out evaluation + leakage gate.
14. **F13 — Two-DGX-Spark serving + elasticity**; latency/memory evidence gates.
15. **F14 — Complete Atlas Lab GUI** (§28) over stored artifacts (never heavy
    compute in browser).

Early priorities (biggest unblocking, lowest risk):
- **F0 finish** (foundation correctness + mini MoE gives every later milestone a
  test substrate),
- **F2 streaming census + structural graph** (everything downstream reads it),
- **F1 contracts** (typed evidence discipline so nothing later is fabricated).

### Cross-cutting decisions to confirm before building
- Confirm `model-atlas` is the sole home of the Atlas subsystem (option A) and
  eval-lab integration is a one-way plugin (labels, suites, eval reporting).
- Confirm which trace/research meat is in-scope for the first atlas milestone:
  F3 (routing+contribution) is the honest minimal "basic layerwise Atlas," not
  just frequency.
- Memory budget envelopes (190/210/225 GB) and two-Spark assumptions come from
  v2 §3; keep them config, not code.

---

---

## 12. V2 feature-requirement completeness checklist

Every requirement group in the pasted v2 contract, with current status and
where it should land. IMP/PAR/MISS = implemented / partial / missing today.
Owner: MA = `model-atlas`, EL = `eval-lab`. Milestone = recommended F-stage from §11.

| v2 § | Requirement group | Status | Notes / location |
|---|---|---|---|
| §1 | Product definition: parent→derivative foundry (not just pruning/GUI) | MISS | scope is a platform, not a single ranking script |
| §1 | Derivative-candidate family: coding, agentic, frontend/visual, voxel/spatial, balanced daily driver, general-retention, elastic (resident+NVMe overflow), distilled K2.x/student | MISS | must be targets the search engine can emit (v2 §24/§25); none planned explicitly |
| §1 | End-to-end workflow dependency chain | MISS | no runtime; only census slice |
| §2 | Core hypothesis + **explicit preservation of negative findings** (distributed functions, frequency≠importance, local-router suboptimality, label→expert mismatch, polysemanticity, similarity≠substitutability, cross-layer dependence, removal hurting coordination, low-bit slower, student beating pruned parent) | MISS | §20 evidence + negative-controls regime must record these; currently nothing |
| §3.1 | Parent K3: 2.8T/104B, 93 layers, KDA/MLA, 896×top-16, shared, Stable LatentMoE, Attention Residuals, MXFP4 | PAR | facts captured in ArchitectureSpec design, **K3 not yet registered** |
| §3.2 | Hardware: DGX Spark A/B, GB10, ConnectX-7; reserve-memory list; envelopes 190/210/225 GB | PAR | factored into planned memory planner; envelopes not yet config/dates |
| §4 | Platform architecture tree (GUI, model-asset, task/suite registry, harness, Atlas Lab, search, builder, repair, serving, orchestrator, artifact store, provenance) | MISS | diagram-level only; no modules beyond census |
| §4 | **Critical subsystem boundary**: Atlas same product, not the eval runner | PAR | principle documented in MA README/AGENTS; not enforced in code (no runtime) |
| §5 | Model asset lifecycle types + minimum fields | PAR | EL has model-asset registry; MA needs derivative/student/teacher types + atlas fields |
| §6 | Checkpoint registration + structural mapping (13-step, metadata-first, no body loads) | MISS | F2; EL does header-only SafeTensors inspection (PAR), no structural graph/hash/shard logic |
| §6 | Outputs: checkpoint_manifest(.json/.parquet), hashes, tensor_relationships, structural_model_graph, validation report | MISS | F2 |
| §7 | Data partitions (atlas_calibration / development_eval / held_out_eval) | PAR | EL has leakage guard + `data_partition`; must surface in MA plans |
| §7 | Leakage detection (dup/reuse/overlap/lineage/near-dup) + block promotion | PAR | EL `leakage.py`; not in MA |
| §8 | Behaviour ontology (21 labels) + trajectory stages (10) | PAR | EL v1 labels; v2 adds success states |
| §8 | Trace record fields (task, sample, suite, partition, labels, stage, token_range, mode, success, source ids, layer, expert, run_id, schema_version) | MISS | F1 |
| §8 | Success states (success/failure/recovered/partially_recovered/unknown) | MISS | F1 |
| §9 | **Six-level hierarchy**: L1 weights/tensors, L2 units (channels/neurons/tiles/latent dirs/sparse features), L3 experts, L4 coalitions, L5 cross-layer paths, L6 behaviour; traceable up/down | MISS | L1 foundation only (ownership); L2–L6 all absent |
| §10 | Ten maps: structural, activation, contribution, functional, causal, redundancy/substitution, coalition, cross-layer, compression, deployment | MISS | only structural-equivalent ownership manifest |
| §10 | Functional-map evidence statements (disproportionately salient, confidence, causal level, partners, substitute) — no simplistic `Expert 417 = Python` | MISS | §20 discipline required |
| §11 | Trace families: routing, contribution, representation, intervention | MISS | F1+F3+ |
| §11 | Representation storage options: full/FP16/FP8/random projection/PCA/statistics/principal directions/sparse-feature activations | MISS | F4 |
| §11 | Generation modes: teacher_forced / free_generation / tool_trajectory / counterfactual / failure_recovery / compression_probe / causal_ablation | MISS | F4–F8 |
| §12 | Success/failure/recovery contrasts (separate maps, contrast stats, path divergence, recovery-damage) | MISS | F4 |
| §13 | Counterfactual routing + route regret (alt routes, local quality, downstream loss, route regret, fragility, repair-vs-keep-map separation) | MISS | F5 |
| §14 | Cross-layer route/path analysis (signatures, branch points, compression deltas, path records) | MISS | F9 |
| §15 | Sparse features / dictionary learning (SAE), feature↔label↔layer↔expert↔outcome links, redundancy | MISS | F6 |
| §16 | Semantic projection through output vocabulary (top promoted/suppressed tokens, clusters, context dependence, validated) | MISS | F6 |
| §17 | Multi-component causal tracing (joint/synergy/redundancy/min-sufficient/min-destructive sets; A+B catastrophic case) | MISS | F8 |
| §18 | Neuron/channel/block/tile analysis (gate/up/down, latent-shared/expert factors; magnitude/contribution/gradient/patching/masking/error) | MISS | F9; K3 component distinction required from structural graph |
| §19 | Adaptive tracing funnel A–G (broad scan→contribution→repr/features→counterfactual→compression probes→causal/coalition→derivative validation) + evidence-stage tagging | MISS | governs phasing (F3–F12) |
| §20 | Evidence grades (7 levels) + uncertainty fields + controls (random masks, quantization, freq-matched, label-shuffled, paraphrases, alternate corpora, route-preserving/changing quant, random coalitions) | MISS | F1 + discipline across all |
| §21 | `CompressionBackend` protocol + support-statuses (unsupported/conversion/probe/inference/training/custom-kernel) | MISS | F7 |
| §21 | Backend registry: source_mxfp4, bf16, fp16, fp8, nvfp4, int8, int4, exl3, aqlm, structured_pruning, removed, nvme_overflow, custom_research_backend | MISS | F7 |
| §22 | EXL3 + AQLM probes: pinned revision, compatibility matrix, isolated-then-representative experts, 17-measure evaluation, conversion≠inference, two reports | MISS | F7/F8 |
| §23 | Per-expert compression-response curves (bytes/effbits/overhead/errors/drift/KL/routing/regression/runtime/kernel/repair) | MISS | F7 |
| §24 | Derivative Architecture Search Engine (objective, constraints, decision variables, multiple Pareto candidates) | MISS | F10 |
| §25 | **13 planning artifacts**: keep_map, precision_map, residency_map, channel_map, tile_map, coalition_protection_map, path_preservation_map, substitute_map, node_ownership_map, overflow_pack_map, router_repair_map, residual_repair_map, distillation_target_map | MISS | F10; only keep/precision/residency/substitute currently noted — must add the other nine |
| §25 | Candidate families at 190/210/225 GB + per-candidate report (bytes by node, active bytes/token, coverage, risks, kernel compat, repair needs, uncertainty) | MISS | F10 |
| §26 | Derivative construction (17 stages) + repair (router/bias/residual/repr/sparse-feature/expert-adapt/LoRA/PV-tuning/white-box distill) | MISS | F11/F12 |
| §27 | **Machine-readable outputs**: ~45-file `atlas_runs/<id>/` layout (manifests, parquet set incl. routing_traces, counterfactuals, regret, activation, contribution, sparse_features, vocabulary_projection, layer/label saliency, contrasts, coactivation, transitions, similarity, substitutes, coalitions, multi_component_causal, cross_layer_paths, quantization_probes, expert_response_curves, channel/neuron/tile maps, projection_sensitivity, ablation, negative_controls, evidence_registry, uncertainty_report, resource_telemetry, warnings, summary, reproducibility_command.sh) | MISS | F1 defines contract; produced across F3–F12 |
| §28 | GUI: 25 views + label glossary (measured/estimated/predicted/correlated/locally causal/downstream causal/held-out/unsupported/not tested) | MISS | F14; EL shell reusable |
| §29 | 15 milestones | — | mapped to F0–F14 in §11 / §14 |
| §30 | Gap analysis this report | IMP | this document |
| §31 | 27 non-negotiable rules | see §13 compliance matrix | |

---

## 13. §31 non-negotiable rules → compliance status

| §31 rule | Status | Enforced by |
|---|---|---|
| 1. Atlas ≠ one saliency score | MISS | evidence families §10–§11 (none yet) |
| 2. Routing frequency ≠ causal importance | MISS | counterfactual + causal traces |
| 3. Don't assume router chose best route | MISS | §13 counterfactual/regret |
| 4. Correlation ≠ causation | PAR(design) | evidence grades §20 must gate all claims |
| 5. No simplistic permanent expert labels | MISS | functional-map evidence statements |
| 6. Similarity ≠ substitution | MISS | redundancy/substitution map |
| 7. Don't ignore success/failure/recovery | MISS | §12 contrasts |
| 8. Don't ignore coalitions | MISS | §10.7/§17 |
| 9. Don't ignore cross-layer paths | MISS | §14 |
| 10. Don't ignore sparse features / polysemanticity | MISS | §15/§16 |
| 11. No one global quant format | MISS | per-expert/tensor/channel choice §23/§24 |
| 12. Don't assume EXL3/AQLM support K3 | MISS | §22 compatibility matrix |
| 13. Conversion ≠ deployable inference | MISS | support-statuses §21 |
| 14. Lower bits ≠ faster | MISS | runtime/throughput measurement §23 |
| 15. Don't uniformly prune every expert | MISS | keep-map search §24 |
| 16. Immutable parent checkpoint | PAR(design) | manifest/hash + builder never mutates source (F2/F11) |
| 17. Don't lose source expert identity | PAR(design) | ownership already preserves; extend to maps/builder |
| 18. Don't hide router/bias remapping | MISS | F11 renumber/remap must be auditable |
| 19. Don't mix calibration + held-out | PAR | EL leakage guard; MA must honor |
| 20. Predictions ≠ measured results | MISS | evidence grades + glossary |
| 21. No custom kernels before profiling | PAR(design) | AGENTS invariant; funnel stage E before any kernel |
| 22. Long jobs ≠ depend on GUI | PAR | EL orchestrator pattern; MA atlas jobs must match |
| 23. No heavy Atlas analysis in frontend | MISS | F14 serves stored artifacts only |
| 24. Don't report skipped compat tests as passing | MISS | §22/§21 support-status discipline |
| 25. No promotion without held-out validation | MISS | F12 leakage gate + held-out required |
| 26. Winner need not be pruned K3 | PAR(design) | student route must stay first-class |
| 27. Preserve student-distillation as first-class route | PAR(design) | F12; must not be deprioritized vs surgery |

---

## 14. Summary of newly confirmed gaps (added in this audit)

1. Derivative-candidate family (§1) + negative-findings preservation (§2) — not yet explicit targets.
2. Six-level atlas hierarchy (§9) — L1 only (ownership); L2 units, L3 experts, L4 coalitions, L5 pathways, L6 behaviour all absent.
3. Representation storage options (§11) — PCA/random-projection/principal-directions/sparse-feature avenues unspecified.
4. Full planning-artifact set (§25) — nine maps beyond keep/precision/residency/substitute are unlisted: channel, tile, coalition-protection, path-preservation, node-ownership, overflow-pack, router-repair, residual-repair, distillation-target.
5. Machine-readable output catalogue (§27) — the ~45-file `atlas_runs/<id>/` contract is not yet scoped in MA.
6. §31 compliance — 27 rules, mostly PAR(design) or MISS; no current code enforces the discipline.

*End of gap analysis. Stopping for review per contract §30.*
