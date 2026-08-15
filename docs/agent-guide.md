# Atlas compression control plane — agent operating guide

This is the operating manual for an agent (or a human) driving the Atlas
compression control plane safely and honestly.

## Ground rules

1. **Do not fabricate.** If a backend is unavailable, its compile/run fails
   closed. Never report a method as executed, validated, or measured without a
   journal event proving it.
2. **Do not touch the source model.** The engine snapshots the source and fails
   any run that observes a change under `immutable_source=true`. Never write
   into the mounted GLM-5.2 path.
3. **Never upgrade provenance.** Recorded evidence_kind is the policy ceiling,
   not what a backend claimed.
4. **Pruning is opt-in only.** Under `no_pruning=true` (the canonical default)
   a pruning stage/consumer is a compile error. A TENP/FlexMoE run requires the
   separately-registered capability and an explicit no-pruning opt-out.
5. **Don't claim EXL3/NVFP4/fp8 hybrid works** unless the selected backend
   declares exactly that combination AND the availability gate passes.

## Typical workflow

```bash
# 1. See what exists and what's runnable
model-atlas backend-capabilities

# 2. Inspect the canonical GLM-5.2 no-pruning recipe (dry-run, honest blockers)
model-atlas compile-recipe --recipe glm52-no-pruning        # shows why it fails closed

# 3. Lineage / ids / reproducibility (no execution needed)
model-atlas job lineage --recipe glm52-no-pruning

# 4. Start a REAL run on a recipe whose backends are available
model-atlas job start --recipe <family-that-compiles> --out controlplane_runs
# -> prints run_id, status

# 5. Inspect / resume / validate / cancel
model-atlas job status   --run-id <id> --out controlplane_runs
model-atlas job resume   --run-id <id> --out controlplane_runs
model-atlas job validate --run-id <id> --stage <sid> --out controlplane_runs
model-atlas job cancel   --run-id <id> --reason "<why>" --out controlplane_runs
```

## Reading a run

- `job.json` — state machine. `RUNNING`/`RESUMING` after a crash → resume
  re-executes in-flight/failed stages idempotently.
- `events.jsonl` — authoritative event stream (stage outcomes, output hashes,
  provenance holds, terminals).
- `manifest.json` — final user record. `readiness.runtime_benchmarked` is
  `false` until a real two-Spark profile; `published` is `false` until the
  publish rule passes.
- `objects/…` — content-addressed stage outputs; `validate` re-hashes them.

## Writing a new recipe

Author a `CompressionRecipe` (see `recipes/builtin.py` for the worked example).
Keep the register: `_stage(id, name, effect_class, backend_id, produces=…,
requires=…)`. The compiler checks ordering, hybrid legitimacy, no-pruning
transitivity, and backend availability (fail-closed) — a recipe that can't
compile today is a *valid plan expressed honestly*, not a bug.

## Repair

Validators may emit typed `RepairProposal`s. Only allowlisted deterministic
kinds (`router_bias_reorder`, `keep_channels_normalize`, `bit_count_rebaseline`,
`index_total_size_rebuild`, `evidence_downgrade`) can be compiled/applied.
Agent suggestions are proposals until compiled; before/after hashes are kept and
rollback is hash-verified. Never apply a non-allowlisted repair.

## Reporting

State evidence kind per result and name the gate that produced it. Never report
a speedup/retention without cold-start, warm-up, prefill, token latency, and
NVMe/network stall context. Keep `main` and production services untouched; all of
this work lives on `atlas-glm52-experiment-runtime`.
