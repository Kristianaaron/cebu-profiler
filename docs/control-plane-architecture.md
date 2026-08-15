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
  unless a runtime/backend *explicitly declares* that exact combination via
  `hybrid:<fmt1>+<fmt2>…` capability. `allow_hybrid_precision` records author
  intent only and never demotes an unsupported composition to a warning.
* A `CommandBackedAdapter` with no wired command raises `BackendUnavailable` at
  execution time; it cannot fabricate an output.

## Concurrency & durability

* A run dir is bound to its **actual** input identity (`run_id = hash(plan,
  inputs)`); distinct inputs → distinct deterministic run dirs.
* Stage outputs are staged then atomically promoted into a content-addressed
  store; nothing is mutated in place. Content addressing makes replay
  idempotent and cross-run dedup possible.
* Every transition is an append-only JSONL event **before** an atomic `job.json`
  replace (write-ahead). A crash mid-stage leaves the job `RUNNING`/`RESUMING`;
  resume re-executes in-flight/failed stages idempotently from the journal.
* A run dir is guarded by an exclusive lock; a second engine refuses entry.
* Immutable source is enforced: a stat snapshot is taken at run start and
  re-verified at every stage boundary; any change fails the run.
