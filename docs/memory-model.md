# Memory model

Byte-accurate accounting; a disk fit is not a runtime fit.

## Tiers (always kept separate)

- **Stored bytes** — all tensor bytes regardless of location.
- **Resident bytes per node** — Node A/B own tensors + replicated tensors
  (counted on both). NVMe tier is stored but **not** resident.
- **Active expert bytes per token** — `num_layers × top_k × per_expert_bytes`
  (an estimate of the MoE read per generated token).

## Runtime reserve

The OS, CUDA/NCCL workspaces, activations, KDA/MLA state, KV/recurrent state,
telemetry, and emergency headroom are separate from resident weights. The
planner adds a configurable `runtime_reserve` (default 30 GiB) before comparing
to each node budget.

## Go/no-go

A plan is `unsafe` when `resident(node) + runtime_reserve > budget(node)` for
either node, with the failing node and amounts reported.

## Target envelopes (from the v2 blueprint)

Resident-weight envelopes to explore: **190 GB / 210 GB / 225 GB**, on two DGX
Spark nodes of **128 GB unified memory each**. These are config inputs, not
code constants.

## Current status

`planning/memory_planner.py` implements the assessment for a manifest and gives
a deterministic go/no-go. Real per-tensor sizes for K3 are not yet measured, so
planning K3 currently correctly fails closed (`needs_source_measurement`).
