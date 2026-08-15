# Atlas compression control plane — architecture

The control plane is the deterministic execution/planning surface a future Atlas
UI and other agents call to run an **interface-agnostic compression backend**.
It is a recipe → compile → durable job engine, with typed backends, capability
registry, deterministic repair, and full evidence discipline. It does **not**
implement compression algorithms itself — it schedules them, verifies their
claims, and records exactly what happened.

```
  authored recipe (CompressionRecipe)
      │  deterministic canonical id (recipe_id, plan_id)
      ▼
  RecipeCompiler ── no_pruning transitively enforced, illegal/unsupported
      │              orderings + hybrid compositions rejected, unavailable
      │              backend => fail-closed compile error
      ▼
  CompiledRecipe (immutable plan: pinned backends, statuses, hashes)
      │
      ▼
  JobEngine        durable run dir:
      │              plan.json · job.json (atomic replace)
      ├── backend registry (typed adapters; available/unavailable)
      ├── stage execution (idempotent, resume-safe, stage.lock excluded)
      ├── content-addressed store (objects/…  sha256)
      ├── events.jsonl (append-only journal)
      └── manifest.json + reproduce.sh  → UI/agent records
```

## Modules

| Module | Responsibility |
|---|---|
| `recipe/schema.py` | canonical `CompressionRecipe`/`RecipeStage` (+ `RecipeConstraints` incl. `no_pruning`), `SourceIdentity`, `CalibrationIdentity`, `HardwareEnvelope`, `PublishRule` |
| `recipe/compiler.py` | deterministic ids; ordering; no-pruning *transitive*; hybrid rejection; backend fail-closed availability; immutable `CompiledRecipe` |
| `backend/contract.py` | `BackendRecord`, typed adapter protocol, `CommandBackedAdapter`, `ParameterSpec`, availability probes (fail closed) |
| `backend/registry.py` | `BackendRegistry` + in-repo truthful adapters, EXL3/ModelOpt/LLM-Compressor/Eval-Lab command-backed placeholders, TENP pruning opt-in capability, plugin loader |
| `jobs/schema.py` | `Job`/`StageOutput`/`OutputRef`/`RepairRecord` + state machine |
| `jobs/artifacts.py` | content-addressed store, staging→atomic promotion, file locks, source-immutability snapshot |
| `jobs/engine.py` | durable engine: journals, idempotent replay, crash-safe resume, provenance non-escalation, cancellation |
| `repair/gate.py` | typed repair proposals, deterministic allowlist, before/after hashes, rollback |
| `controlplane/api.py` | `ControlPlane` facade (capabilities/compile/dry-run/start/status/resume/validate/lineage) |
| `recipes/builtin.py` | canonical GLM-5.2 no-pruning recipe + TENP opt-in pruning recipe |

## Canonical product rule (GLM-5.2)

* **Fidelity-first: `no_pruning=true` is the default.** No stage whose effect
  class is `pruning` (or which transitively consumes a pruning-produced format)
  may appear in a no-pruning recipe — enforced by the compiler, transitively.
* **Pruning (TENP/FlexMoE) is a separately-registered opt-in capability.**
  `tenp_pruning` backend declares the `pruning` capability; a recipe using it
  must set `no_pruning=false` + `allow_pruning_capability=true`, AND each
  pruning stage must be served by a capability-declaring backend. No such recipe
  is auto-injected; none runs today (its backend is unavailable → fail closed).
* **The W64 structural canary is not a scientific candidate.** It is a
  structural-only export used for exporter validation; corpus-driven keep-maps
  and a maintenance-window load/forward are still required before any
  scientific/runtime-readiness claim.

## Evidence & provenance

* Every stage declares an `evidence_policy` ceiling (predicted/estimated/…).
* The recorded `evidence_kind` is `min(policy, reported)`. A backend that
  reports `measured` on a stage whose policy is `predicted` records `predicted`
  and emits a `evidence.non_escalation` journal event. Provenance is never
  upgraded silently.
* The manifest separates `predicted` resource estimates from `measured`
  footprints; nothing is reported as measured without the materialized +
  held-out + runtime-benchmarked chain.

## Fail-closed guarantees

* An unavailable backend (no dependency, no adapter, no probe) is a **compile
  error** — `backend_unavailable` (`backend_missing` if unregistered). It is
  never silently skipped or simulated.
* An unsupported precision hybrid (EXL3+NVFP4+FP8 …) is a **compile error**
  unless the *selected* backend (available + version-pinned) *explicitly
  declares* that exact combination via `hybrid:<fmt1>+<fmt2>…` capability.
  `allow_hybrid_precision` records author intent only and never demotes an
  unsupported composition to a warning.
* A `CommandBackedAdapter` with no wired command raises `BackendUnavailable` at
  execution time; it cannot fabricate an output.
* The built-in `atlas_quant_probe`/`atlas_analysis_v3` adapters are
  **probe/analysis-only** (`produces_derivative=False`). A compression stage
  (quantization/refinement/residual/conditioning) pinned to them is a compile
  error — no compression stage/job can succeed without a real derivative.
* `require_available=false` is dry-run-only: the stage compiles for planning but
  is explicitly non-executable at run time.

## Concurrency & durability

* A run dir is bound to its **actual** input identity (`run_id = hash(plan,
  inputs)`); distinct inputs → distinct deterministic run dirs. `engine_for`
  rebuilds a run dir by EXACT path + full identity validation (no glob); a
  resume recomputes the run_id from the persisted inputs and refuses a mismatch.
* Stage outputs are staged in an isolated scratch space, content-addressed by
  **full sha256** (collision-guarded), and atomically published; nothing is
  mutated in place. Content addressing makes replay idempotent and cross-run
  dedup possible.
* Every transition is an append-only **write-ahead** JSONL event (fsynced)
  before the atomic `job.json` replace. A crash mid-stage leaves the job
  `RUNNING`/`RESUMING`; resume re-executes in-flight/failed stages idempotently
  and **re-verifies every already-DONE stage's published output hashes** before
  trusting them.
* A run dir is guarded by an **advisory OS flock** (held on an open fd,
  auto-released on crash — no stale markers to recover); a second engine refuses
  entry.
* Immutable source is enforced twice: a stat snapshot is taken at run start and
  re-verified at every stage boundary, and any declared `SourceIdentity.sha256`
  content hashes are verified by full digest; a mismatch fails the run.

## Validation gates & evidence

* Every stage runs **every declared validation gate before DONE**. A declared
  gate that cannot be executed (missing validator/eq-control/identity-control
  wiring) fails closed — a stage with an unvalidated gate is never DONE and
  never promotes.
* Recorded `evidence_kind` is `min(policy, reported)`; a backend reporting
  `measured` on a `predicted`-policy stage records `predicted` and emits a
  `evidence.non_escalation` journal event. Provenance is never upgraded silently.
* `require_available=false` makes a stage **dry-run-only**: it can compile for
  planning but execution is explicitly non-executable (fails closed at run time).

## Compendium of resolved audit findings

| Audit item | Resolution |
|---|---|
| Repair verification `or True` | real full-digest re-read of the produced CAS blob; target ref atomically updated; rollback restores bytes+state |
| Arbitrary unregistered apply_fn | transforms are registered with a VERSIONED identity; `apply` refuses a version mismatch (no arbitrary apply_fn) |
| Source path->sha256 gaps | complete recursive relative-path manifest (no first-eight cap); exact membership + hash verified path-bound; mismatch terminal |

## Compiled-plan artifact & reproduction

`compile-recipe --out` writes a **versioned immutable compiled-plan artifact**
(`CompiledPlanArtifact`): the canonical recipe, `recipe_sha256`/`recipe_id`/
`plan_id`, stage->backend exact resolved pins, the backend status snapshot,
canonical job `inputs`, and the deterministic `run_id`. `artifact.verify()`
recomputes ids + run_id and fails closed on any inconsistency. `model-atlas job
start --plan <artifact>` loads + verifies it and starts with the artifact's
canonical inputs; `reproduce.sh` (written into every run dir) embeds the exact
inputs so a fresh run reproduces the NONEMPTY run id.

## Execution ordering (per stage)

1. Isolated staging dir (never visible to the run until published).
2. The adapter `context` receives `staging_dir`/`output_sink`.
3. `expected_outputs` gate: every declared output must exist in staging.
4. Compression stages require a **real non-evidence derivative**: the adapter's
   `produces_derivative` AND the record's `produces_derivative` AND a staged
   weight/serialization file must all agree — else the stage fails closed.
5. Validation gates run against STAGING first (then again against published
   outputs pre-DONE).
6. Only then are the verified content-addressed outputs + stage manifest
   published and the stage marked DONE.

## Hardware axes (four separate fields)

`HardwareEnvelope` now separates **model_arch** (glm-5.2/k3) from
**compute_arch** (gb10-sm121) from **topology** (2x-spark) from
**runtime_backend** (vllm-modelopt). The compiler checks each axis against its
own backend field (`architectures`, `compute_archs`, `topologies`,
`runtime_compat`) — glm-5.2 is never compared to gb10-sm121 or vllm-modelopt to
sm121.

## Executability & hybrid

* Executable stages require an **exact resolved version**; `unpinned` compiles
  only for planning and is dry-run-only/non-executable (also enforced at run
  time by the engine).
* A hybrid (EXL3+NVFP4+FP8 …) compiles only when the declaration comes from a
  **selected, available, version-resolved backend that actually PRODUCES one of
  the precision formats** — a profiling/eval analysis backend can never
  authorize it.

## Structures

* `CompiledRecipe.resolved_backends`/`backend_status_snapshot` are frozen
  read-only Mappings.
* no-pruning propagation retains ALL producer edges (a format produced by
  multiple stages keeps every producer in the reachability DAG).
* the capability view includes `produces_derivative` + per-backend resource
  limits.
