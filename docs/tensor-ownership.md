# Tensor ownership

Ownership is the invariant backbone of the Atlas: **every tensor is accounted
for, mapped to exactly one role and one physical location, and its source
identity is preserved.**

## Concepts

- **Role** (`TensorRole`): what the tensor does (router, expert, shared expert,
  latent projection, attention, MLA state, KDA decay, norm, embedding, LM head,
  vision). One role per tensor — nothing is "unclassified."
- **Source identity**: `layer_index` + `expert_index` (plus the canonical
  `key`) remain tied to the source checkpoint. Candidate/local slot IDs, router
  aliases, keep-map entries, and physical locations are separate concepts and
  never overwrite source identity.
- **Physical location** (`PhysicalLocation`): node_a / node_b / nvme_a / nvme_b /
  replicated.
- **Placement policy** (`PlacementPolicy`): `replicate_shared_on_a` (default) —
  routed experts split across A/B in expert-parallel fashion; non-expert and
  shared tensors on node A.

## Invariants (enforced)

1. Tensor keys in a manifest are unique.
2. Every record has a non-null `role`.
3. Byte accounting is `numel × dtype_bytes`; stored ≠ resident ≠ active.
4. No tensor size is fabricated: an architecture without `tensor_params`
   produces a `needs_source_measurement` manifest, not guessed numbers.

## Current status

`build_manifest()` realizes Level 1 (weights/tensors) for synthetic fixtures
and reports `needs_source_measurement` for real checkpoints. Structural graph,
shard enumeration, and hashing for actual checkpoints are a later milestone.
