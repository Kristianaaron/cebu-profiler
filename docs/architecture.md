# Architecture

Model-agnostic parent-to-derivative platform. Atlas is part of the same product
goal as the evaluation harness but is a **separate execution subsystem**: the
harness asks "what can a completed model do?", the Atlas asks "which internal
parent-model components are associated with / necessary for / causally
responsible for those behaviours?"

## Core packages

- `schemas/architecture.py` — `ArchitectureSpec`, `MoELayout`, `TensorRole`,
  `DType`, byte accounting. Structural layout of any transformer/MoE family.
- `registry/` — `ArchitectureRegistry`, the extension point for future parents.
  Kimi K3 (first subject) + a synthetic miniature `k3-mini` are built in.
- `census/` — tensor census + ownership:
  - `tensor_ownership.py` — `TensorOwnership` (role, dtype, numel, layer,
    expert, physical location), `OwnershipManifest` (unique keys, derived
    byte summaries).
  - `census.py` — `build_manifest()` enumerates tensors layerwise, preserving
    source identity and refusing to fabricate sizes (`needs_source_measurement`).
- `planning/memory_planner.py` — byte-accurate, per-node go/no-go plan
  (resident vs stored vs active-bytes-per-token; runtime reserve).
- `synthetic/mini_moe.py` — deterministic miniature K3-shaped MoE for tests.
- `cli.py` — `doctor`, `list-architectures`, `census`, `plan`.

## Model-agnostic extension

To add a new parent: register an `ArchitectureSpec`. Structural layout (layers,
MoE geometry, dtypes) is provided up front; exact per-tensor sizes come from
that checkpoint's census and are `None` until measured — never invented.

## Boundary discipline

- Atlas must not become the evaluation runner.
- Never do heavy Atlas analysis in frontend components; serve stored artifacts.
- Long Atlas jobs must not depend on the GUI (persistent orchestrator pattern).

## Target atlas hierarchy (not yet implemented)

Weights/tensors → internal units (channels/neurons/tiles/latent directions/
sparse features) → experts → coalitions → cross-layer pathways → behaviour.
Census/ownership today realizes Level 1; the rest are later milestones.
