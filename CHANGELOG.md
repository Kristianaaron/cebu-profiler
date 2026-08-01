# Changelog

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
