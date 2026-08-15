# Atlas control plane — UI integration contract

The control plane is deliberately framework-light so a future UI or service can
embed it (FastAPI/uvicorn already optional deps) or shell out to the CLI. All
state is on disk under the work root (`controlplane_runs/`).

## Capabilities (read-only, cheap)

`GET /capabilities` (or `model-atlas backend-capabilities`) returns:

```json
{
  "backends": { "<id>": { "backend_id", "display_name", "method_family",
                          "formats": [...], "status": "discovered|…",
                          "version", "declared_capabilities": [...],
                          "supported_formats": [...], "fail_closed": true,
                          "parameters": [...] } },
  "capabilities": { "hybrid:…": ["modelopt_nvfp4"], "pruning": ["tenp_pruning"] },
  "available": ["atlas_quant_probe", "atlas_analysis_v3"],
  "controls": { "compiler": "recipe-compiler-v1",
                "default-policy": "no_pruning=true (fidelity-first)", "...": "…" }
}
```

A UI may render `status` as a badge (unavailable → grey, discovered → blue,
experimental → amber, validated → green, recommended → gold). Nothing more.

## Compile / dry-run

`model-atlas compile-recipe --recipe <family> [--dry-run]`.

* dry-run reports `recipe_id`, `compiles`, and typed `issues` — no execution.
* without `--dry-run` it writes an immutable compiled plan.
* A recipe that fails to compile (unavailable backend, unsupported hybrid,
  no-pruning violation) prints typed errors and exits non-zero.

## Start / status / resume / validate / cancel / lineage

`model-atlas job <action> …`. Each run is a directory:

```
controlplane_runs/runs/<run_id>/
  plan.json          immutable compiled plan (CompressionRecipe)
  job.json           atomic-replaced Job record (state machine)
  events.jsonl       append-only JSONL (agent/UI event stream)
  manifest.json      final run manifest (status, stages, evidence, readiness)
  reproduce.sh       exact reproduction command
  objects/…          content-addressed stage outputs (sha256-keyed)
  stage/<id>/…       per-stage staging + finalized outputs
```

`job status --run-id <id>` returns the manifest, the event list, stage records
(status/outputs/evidence), and readiness flags (`all_stages_done`,
`runtime_benchmarked=false` until a two-Spark profile, `published=false`).

## Events (JSONL) — the stream a UI tailed

```json
{"ts":"…","event":"stage.start","stage":"t7-exl3","backend":"exl3"}
{"ts":"…","event":"stage.output","stage":"t7-exl3","name":"…","sha256":"…"}
{"ts":"…","event":"evidence.non_escalation","stage":"…","reported":"measured","recorded":"predicted","policy_ceiling":"predicted"}
{"ts":"…","event":"run.terminal","status":"failed_terminal","stage":"t7-exl3"}
```

## Reproducibility

`lineage` returns recipe/plan/run ids, source+calibration identity, and an exact
`reproduce_command`. `reproduce.sh` in a run dir replays the run.

## Integration notes

* Never run heavy analysis in a frontend; the plane serves stored artifacts
  (AGENTS invariant 22/23).
* A UI may call `resume`/`validate`/`cancel`; it must not bypass the engine's
  locks (a second engine on the same run is refused).
* `status` is safe to poll; `start`/`resume` are exclusive actions.
