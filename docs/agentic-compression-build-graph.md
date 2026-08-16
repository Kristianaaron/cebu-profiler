# Atlas compression platform: agentic build graph

Status: implementation contract for the post-v3 compression platform.

## Product invariant

Atlas must let a user compare `quantize_only`, `prune_only`, `hybrid`, and
`custom` compression strategies against the same immutable teacher, corpus,
hardware envelope, evaluation definition, and fit ledger. Predictions and
measurements are never conflated. No method is executable until a pinned
backend, typed artifact contract, validator, and focused parity test exist.

The primary optimization is:

> Minimize teacher-relative quality divergence, with token KLD as a first-class
> axis, subject to hardware fit, context, throughput, capability-floor, and
> user-policy constraints.

KLD is necessary but not sufficient. CKA, routing divergence, task retention,
runtime correctness, memory, context, and throughput remain independent axes.

## Agent roles

| Role | Owner | Responsibility |
|---|---|---|
| Integration lead | primary Codex agent | graph ownership, schema seams, merges, acceptance, live-system safety |
| Bounded builder | DSV4/OMP isolated worktree | one node, exact paths/interfaces/tests, no integration decisions |
| Adversarial reviewer | independent agent | read-only contract review, negative cases, evidence truthfulness |
| Canary monitor | independent agent | process/progress/service identity, no edits |
| Runtime operator | primary agent only | model/service/large-artifact actions after preflight |

## Direct-build versus OMP decision rule

The integration lead builds directly when any condition is true:

1. The change crosses a safety, identity, authorization, artifact-promotion, or
   schema-version boundary.
2. Two active branches would edit the same contract or serialization format.
3. The smallest correct patch is three files or fewer and requires contextual
   judgment rather than mechanical implementation.
4. A builder misses the first-edit deadline, repeats broad inspection, or runs
   redundant full suites.
5. The work touches a live model, production service, destructive checkpoint
   transformation, or multi-hundred-GB artifact.
6. A review found a P0 integration defect; the lead closes it before delegation
   resumes.

Delegate to OMP when all conditions are true:

1. The node has stable typed inputs/outputs and an isolated file surface.
2. Its focused tests run without a live GPU service or external mutation.
3. Acceptance can be stated as deterministic assertions.
4. The branch can be discarded without affecting another node.
5. The prompt can name explicit anti-goals and a hard wall-clock bound.

## Concurrency policy

- DSV4 runs at `--thinking high`, never `max`. Bounded implementation benefits
  from deliberate reasoning, but max-thinking has produced inspection stalls
  and does not justify its latency here. A healthy in-flight worker is not
  restarted solely to change its thinking level.
- Maximum active implementation lanes: two OMP builders plus the integration
  lead. A third agent may review or monitor read-only.
- Never run two builders against the same worktree.
- Never parallelize schema producer and schema consumer changes unless the
  consumer uses a checked-in fixture of the proposed schema.
- Focused tests may run concurrently. Full fast suite runs once after merge.
- CUDA builds, full-model scans, and large checkpoint writes are serialized.
- A builder gets four minutes to make its first edit and 25 minutes for a small
  node. Missing the first-edit gate triggers interruption and direct takeover.

## Dependency graph

```text
G0  Accepted baseline and safety snapshot
 |
 +--> G1  Product intent + method catalog -----------------------+
 |       quantize_only | prune_only | hybrid | custom            |
 |       method family, lifecycle, evidence, capability          |
 |                                                               |
 +--> G2  Evaluation contracts + pure metrics ----------------+  |
 |       token KLD, domain KLD, CKA, router divergence         |  |
 |                                                            |  |
 +--> G3  Recipe/artifact schema v2 <--------------------------+--+
         typed ports, producer binding, structured params,
         structural mode != precision mode, tensor scopes
              |
              +--> G4 Compiler DAG/linker
              |    exact edges, coverage/overlap, taint,
              |    backend/method preconditions
              |
              +--> G5 Job/artifact transport
              |    upstream CAS refs, recursive bundles,
              |    resume hash binding, adapter evidence
              |
              +--> G6 Backend capability contract
                   algorithm/version, modes, ports, tensor roles,
                   resumability, training/runtime requirements
                         |
          +--------------+-------------------+-------------------+
          |                                  |                   |
          v                                  v                   v
 G7 Quantization lane               G8 Structural lane     G9 Recovery/KV lane
 EXL3, NVFP4, GEMQ,                 REAP, TENP,             QAD+CKA, residual,
 MixQuant, VSRAQ,                   FlexMoE, surgery        AP-aware KV VQ
 conditioning, ReQuant
          |                                  |                   |
          +------------------+---------------+-------------------+
                             v
                    G10 Recipe composers
              quantize | prune | hybrid | custom
                             |
               +-------------+--------------+
               |                            |
               v                            v
       G11 Recommendation/API         G12 Candidate evaluation
       intent authorization,          report, fit ledger,
       blockers, preview              Pareto/marginals
               |                            |
               +-------------+--------------+
                             v
                     G13 GUI comparison
          method controls, candidate matrix, lineage,
          exact disabled reasons, promote-to-preview
                             |
                             v
                     G14 Eval Lab handoff
             canonical request/result and digest validation
                             |
                             v
                     G15 Integration acceptance
                             |
                             v
                     G16 GLM-5.2 NVFP4 canary
                             |
                             v
                     G17 Kimi K3 real experiment
```

## Node contracts

### G0 — Accepted baseline

- Record repository commit and clean status.
- Record production service PIDs/GPU allocation without mutating them.
- Preserve the GLM-5.2 NVFP4 source identity and existing W64 engineering
  artifact as non-scientific provenance.
- Gate: focused baseline tests green; no unowned worktree changes.

### G1 — Product intent and method catalog

- Add `CompressionIntent`: `quantize_only`, `prune_only`, `hybrid`, `custom`.
- Add method-family metadata: analysis, conditioning, allocation,
  quantization, structural, refinement, residual, recovery, KV, evaluation.
- Catalog named methods with paper/provenance IDs, required evidence, required
  backend capabilities, lifecycle status, compatible intents, and output kind.
- Bind intent and constraints into recommendation/token/preview identity.
- Gate: changing intent changes identity; family mismatch rejects; unavailable
  methods remain visible with exact blockers.

### G2 — Evaluation contracts and metrics

- Versioned `EvaluationReport`, `ReproducibilityManifest`, and evidence refs.
- Teacher-forced per-token KL `KL(teacher || candidate)` with stable float64
  reduction, mask/alignment checks, token-weighted domain aggregation.
- Linear centered CKA with explicit degenerate blockers.
- Router JS/top-k/rank/margin metrics preserving ordered ranks and carrying
  sample/domain/layer/token identity.
- Gate: analytic KL fixture, identity zero, mask test, alignment rejection,
  CKA identity/degenerate cases, router order regression.

### G3 — Recipe/artifact schema v2

- Orthogonal structural and precision composition modes.
- Typed stage input/output ports and exact producer binding.
- Tensor scopes, ownership, overlap precedence, and complete-coverage policy.
- Structured parameters rather than `dict[str, str]` only.
- Versioned recursive artifact bundle manifest.
- Migration reads v1; new writes use v2. No silent reinterpretation.

### G4 — Compiler DAG/linker

- Resolve a unique producer output for every consumer port.
- Reject ambiguity, cycles, missing ports, illegal scope overlap, uncovered
  required tensors, invalid structural taint, and unsupported composition.
- Preserve no-pruning transitivity when intent is quantize-only.

### G5 — Job/artifact transport

- Deliver verified upstream CAS references to adapters.
- Bind run/resume identity to upstream hashes and bundle manifests.
- Stage recursively, validate before promotion, publish atomically.
- Accept backend-reported evidence only when schema and artifact digests verify.

### G6 — Backend capability contract

- Declare exact algorithm/version, supported intents and precision modes,
  tensor roles, ports, deterministic/resumable status, calibration/training
  needs, derivative kind, validators, and runtime compatibility.
- Registry remains fail-closed for placeholder or unpinned methods.

### G7–G9 — Method lanes

Each real method integration must include:

1. Pinned backend adapter.
2. Typed request and output bundle.
3. Bounded resource declaration.
4. Deterministic/resume semantics.
5. Structural/checkpoint validator.
6. Golden bounded-tensor or tiny-checkpoint parity test.
7. Explicit unsupported architecture/format errors.

Planning heuristics must be labeled `planning_only`; they cannot produce a
deployable candidate.

### G10 — Recipe composers

- Compose only methods whose port/capability/evidence contracts close.
- Quantize-only never imports structural outputs.
- Prune-only produces a runnable structurally modified checkpoint.
- Hybrid materializes structural output first, then re-profiles and quantizes.
- Custom remains authorization-bound and cannot bypass compiler gates.

### G11–G13 — Product surfaces

- Recommendation returns per-intent availability and exact disabled reasons.
- Unselected blocked methods do not globally block an authorized subset.
- Candidate comparison shows KLD, worst-domain KLD, CKA, routing divergence,
  benchmark retention, bytes, peak/runtime/KV/headroom, speed, and context.
- Every metric displays measured/estimated/predicted evidence independently.
- Preview exposes recipe stages, pins, issues, lineage, and expected outputs.

### G14 — Eval Lab handoff

- Canonical content-addressed request and result schemas.
- Pin teacher, candidate, corpus/tokenizer/template, metric config, harness,
  backend/container, seed, hardware, and input/output hashes.
- Atlas validates the returned schema/digest; it does not pretend an
  unavailable external evaluator ran.

### G15 — Integration acceptance

- Focused suites after each merge.
- Adversarial tests for identity drift, authorization mismatch, missing
  artifacts, forged evidence, resume mismatch, partial bundles, unknown
  methods, and unavailable backends.
- One full non-slow suite after all focused suites are green.
- Independent GO/NO-GO audit before a real checkpoint job.

### G16 — GLM-5.2 NVFP4 reliability canary

- Purpose: validate the application pipeline, not establish teacher-quality
  compression science.
- Mark source as ModelOpt NVFP4 and any further compression as lossy
  requantization testing.
- Exercise profile, recommendation, preview, authorization, execution,
  interruption/resume, validation, Eval Lab handoff, comparison, and lineage.
- Start with metadata, then bounded real tensors, then a small derivative.

### G17 — Kimi K3 real experiment

- Fingerprint the D10 source precision and immutable identity.
- Compare quantization-only, pruning-only, and hybrid candidates.
- Use the public 330.2 GB IQ1_S all-expert GGUF as a reference, not a recipe.
- Structural candidates include expert-preserving variable-width pruning,
  expert removal, and hybrid policies with semantic-specialist constraints.
- Repair with teacher-guided QAD/CKA before final quantization and evaluation.

## OMP builder prompt template

Every builder prompt must contain all sections below; no broad “finish Atlas”
prompt is permitted.

```text
ROLE
You are the bounded implementer for node <ID>. Integration decisions belong to
the parent agent.

BASELINE
Repository/worktree: <absolute isolated path>
Starting commit: <sha>

OBJECTIVE
One sentence describing the observable result.

ALLOWED PATHS
Exact files/directories. Editing any other path is scope drift.

REQUIRED CONTRACT
Exact types, functions, fields, invariants, serialization versions, errors,
and evidence semantics.

ANTI-GOALS
No live services, no package installs, no global formatting, no full suite,
no placeholder success, no unpinned backend claims, no unrelated cleanup.

TESTS
Exact focused commands and adversarial cases. State which existing regressions
must remain green.

TIMEBOX
Write a five-item checklist, inspect once, first edit within four minutes,
finish within <N> minutes. If blocked, stop and report the exact interface.

DELIVERY
Commit only if focused checks pass. Report commit, files, tests, limitations,
and remaining integration assumptions.
```

## Reviewer prompt template

```text
Audit commit <sha> read-only against node <ID>.
Verify contract completeness, negative cases, identity/provenance, evidence
truthfulness, fail-closed behavior, serialization compatibility, and test
non-vacuity. Do not edit. Return GO or NO-GO with severity, exact files/lines,
and the smallest safe correction.
```

## Builder push/review loop

```text
builder focused green
  -> independent read-only audit
  -> NO-GO: integration lead patches small seam or returns one bounded fix
  -> focused regression rerun
  -> merge into integration branch
  -> downstream contract fixture regenerated
```

Maximum two correction loops per builder. On the third issue or a repeated
no-edit stall, the integration lead takes over directly.

## Fastest implementation waves

1. **Wave A:** G1 intent/catalog and G2 contracts/pure metrics in parallel.
2. **Wave B:** G3 schema v2 directly owned by integration lead; G2 fit ledger
   and Pareto corrections can continue independently.
3. **Wave C:** G4 compiler and G5 transport sequentially; G6 capability model
   can run in parallel from the frozen G3 fixture.
4. **Wave D:** real EXL3/GLM bounded lane plus Eval Lab request/result adapter;
   pruning integrations stay fail-closed.
5. **Wave E:** recommendation/API and candidate comparison, then GUI.
6. **Wave F:** merged acceptance and GLM-5.2 NVFP4 reliability canary.
7. **Wave G:** Kimi K3 profiling and real Pareto experiment.

This ordering exposes user flexibility early, fixes KLD correctness early, and
does not delay the GLM canary on unfinished pruning algorithms while still
building the artifact graph required for real hybrid compression.
