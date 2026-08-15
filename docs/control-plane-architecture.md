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
| Repair verification `or True` | real full-digest verification (`RepairGate.verify`) + validator execution before DONE |
| Toy/probe quant backend | `BackendAdapter.produces_derivative=False`; compression stages require a real derivative producer at compile |
| Declared validation gates not executed | every required gate runs pre-DONE; missing/unrunnable gate fails closed |
| Source content hashes / `SourceIdentity.sha256` | verified by full digest; mismatch terminal |
| O_EXCL lock | replaced with advisory flocks (fd-held, crash-safe, no stale state) |
| Resume binding + run_id/inputs | resume recomputes run_id from persisted inputs; mismatch refuses |
| `engine_for` glob | exact-path run dir + full identity validation |
| Non-functional reproduce | CLI loads + verifies `--plan`; unsupported `--recipe-id` lineage command removed |
| Backend contract fields | min status, exact version pin, formats, params, resources, arch/runtime all enforced |
| Hybrid declaration source | only from the selected, available, version-pinned backend |
| Repair atomic rollback | restores a persisted CAS ref (verified), never flags; registered typed transforms |
| Evidence downgrade / channels | monotonic (never upgrade) + channel-range enforced |
| DONE-hash resume verification | re-hash every DONE stage output before trusting a resume |
| Staging isolation | staging is private; commit publishes only verified CAS refs + a stage manifest |
| Journal write-ahead | event (fsync) precedes each atomic job.json snapshot |
| CAS collision guard | full-digest keys; a mismatched slot is never overwritten |
| no-pruning taint | full DAG reachability from every pruning stage |
| Deep immutability | compiled plan stored as canonical payload, reconstructed fresh per access |
| `created_at` docs | clarified time-addressed vs content-addressed id semantics |
| Builtin quant probe | typed non-compression; no compression stage can succeed without a real derivative |
