# Changelog

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
