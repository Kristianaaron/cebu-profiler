# Worked NO-PRUNING GLM-5.2 recipe — compiled/dry-run plan (not executed)

> Status: **PLAN ONLY.** This is the canonical fidelity-first GLM-5.2 recipe
> rendered by `model-atlas compile-recipe --recipe glm52-no-pruning`. It is a
> compiled/dry-run artifact: EXL3 / ModelOpt-NVFP4 / LLM-Compressor / Eval-Lab
> backends are **not installed/validated in this repo**, so execution **fails
> closed**. Nothing below claims a compression was performed. A real run begins
> only when pinned dependencies exist and each backend's availability probe
> passes.

## Recipe family: `glm52-no-pruning`

* Source: `/media/glm52/models/nvidia/GLM-5.2-NVFP4`
  (GLM-5.2-NVFP4, producer `0.46.0.dev65+g977d34dc3`, quant_algo NVFP4)
  — immutable mount, never rewritten.
* Calibration: `glm52-calibration-v1` balanced corpus (labels: code, math,
  general reasoning, long-context, multilingual, creative writing).
* Hardware: 2× DGX Spark (GB10 SM121), 120 GiB host/node target, ConnectX-7.
* Constraints: `no_pruning=true`, `preserve_non_expert_backbone=true`,
  `immutable_source=true`, `allow_hybrid_precision=false`,
  `max_resident_gib=115.0`.
* Publish: requires all stages validated, evidence ≥ measured, runtime
  benchmarked, repair/validated.

## 15 ordered stages

| # | stage id | effect | backend | produces | evidence policy |
|---|---|---|---|---|---|
| 1 | t1-identity | identity | atlas_analysis_v3 | manifest.json | measured |
| 2 | t2-calibration | profiling | atlas_analysis_v3 | corpus-profile | estimated |
| 3 | t3-sensitivity | sensitivity | atlas_analysis_v3 | sensitivity-map | estimated |
| 4 | t4-representation | representation | atlas_analysis_v3 | representation-map | estimated |
| 5 | t5-conditioning | conditioning | modelopt_nvfp4* | conditioned-weights | predicted |
| 6 | t6-bit-allocation | allocation | atlas_analysis_v3 | bit-allocation | predicted |
| 7 | t7-exl3 | quantization | exl3* | exl3 | predicted |
| 8 | t8-refinement | refinement | atlas_quant_probe | exl3 (refined) | estimated |
| 9 | t9-residual | residual | atlas_quant_probe | residual-corrected | predicted |
| 10 | t10-nvfp4 | quantization | modelopt_nvfp4* | modelopt_nvfp4 | predicted |
| 11 | t11-tail | quantization | llm_compressor* | fp8_e4m3 | predicted |
| 12 | t12-kv | kv | atlas_analysis_v3 | kv-plan | estimated |
| 13 | t13-runtime | runtime | eval_lab* | runtime-profile | measured |
| 14 | t14-eval | evaluation | eval_lab* | eval-results | measured |

`*` = command-backed placeholder; UNAVAILABLE until a pinned dependency is
wired + probed + validated. Stages 1,2,3,4,6,8,9,12 use the in-repo validated
Atlas adapters (`atlas_analysis_v3`, `atlas_quant_probe`), which can run today
against the synthetic MiniMoE; they are the only parts that rehearse.

## Honest dry-run result (measured from the compiler, 2026-08-15)

```
recipe: glm52-no-pruning
  recipe_id: recipe-4ca3287e9b31c87c564f7256
  compiles: False
  [error] unsupported_hybrid_precision ... EXL3+NVFP4+FP8 ... no backend declares it
          (allow_hybrid_precision does not authorize it)
  [error] backend_unavailable 'modelopt_nvfp4' ... fail closed ...
  [error] backend_unavailable 'exl3'          ... fail closed ...
  [error] backend_unavailable 'modelopt_nvfp4' (t10-nvfp4) ... fail closed ...
  [error] backend_unavailable 'llm_compressor' ... fail closed ...
  [error] backend_unavailable 'eval_lab' (t13-runtime) ... fail closed ...
  [error] backend_unavailable 'eval_lab' (t14-eval)    ... fail closed ...
```

(6 unavailable-backend errors + 1 unsupported-hybrid error = 7 total compile
blockers, verified.) These blockers are intentional and truthful. They record that the recipe is
correct *as a plan* and that real execution does not exist today. When each
dependency lands (EXL3 pinned build with SM121 audit; ModelOpt producer-parity
measured; LLM Compressor wired; Eval Lab harness reachable), and a hybrid
combination is explicitly declared + tested, the same recipe compiles and runs.

## What this recipe is / is not

- IS: a versioned, deterministic, serializable plan with source/calibration
  identity, hardware envelope, no-pruning constraint, ordered stages, backend
  pins, seeds, expected formats, validation gates, publish policy, and a stable
  `recipe_id` — available for a UI and agents to inspect.
- IS NOT: an executed compression, a validated derivative, a measured runtime
  profile, or a recommendation. `runtime_benchmarked` and `published` stay false
  in any manifest produced from it.
