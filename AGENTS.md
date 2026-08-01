# AGENTS.md — operating invariants for model-atlas

These are non-negotiable. Read fully before changing code. Model-agnostic by
construction; Kimi K3 is one registered instance, never the only one.

## Invariants

1. **Measure before cutting.** No expert is removed solely because it is
   infrequently routed. Remove only on evidence (saliency, contribution,
   causal tests).
2. **Compute follows weights.** Move activations between nodes; never remotely
   fetch large weight matrices per token.
3. **Preserve the non-expert backbone first** (attention, MLA/KDA state, latent
   projections, routers, norms, embeddings, LM head, shared experts, vision
   pathway) in initial rounds.
4. **Router indices and correction biases must be reordered exactly with expert
   renumbering** — never drift out of sync.
5. **Every lossy intervention needs a deterministic identity/no-op control and
   a numerical-equivalence test.**
6. **Calibration data is part of the model design.** Underrepresented
   capability labels will be silently pruned — balance the calibration corpus.
7. **Separate IDs strictly:** source expert IDs, local/candidate slot IDs,
   router aliases, keep-map entries, and physical tensor locations are distinct
   and never conflated.
8. **Separate serialized size, resident weight memory, runtime scratch,
   allocator reserve, KV/cache state, and promotion-tier duplication.** Disk fit
   != runtime fit. Byte-accurate accounting, not param-count estimates.
9. **Do not begin with custom kernels.** Prove correctness and measure
   bottlenecks with existing primitives first.
10. **Never report a speedup without cold start, warm-up, prefill, token
    latency, and NVMe/network stall context.**
11. **Keep the full unmodified source checkpoint immutable.** All derivatives
    derive from plans + manifests.
12. **Evidence is typed.** `measured / estimated / predicted / inferred /
    causally_tested`. Never present inferred or predicted values as measured,
    and never report a speedup or retention claim without the gate/run that
    produced it.
13. **Reproducibility.** Every experiment reproduces from config, source
    commit, checkpoint hashes, calibration manifest, and seed.
14. **Optionality.** Atlas functionality must never break generic evaluation of
    other architectures.

## Phasing

Canonical order (evidence chain is never skipped):

source inventory & architecture audit → calibration corpus + holdouts →
chunked trace → trace validation + saliency → keep-map + risk → candidate
materialization → structural audit → tiny behavior gate → per-component
quantization screens → healing/distillation if red → retained-quality gate →
runtime/cache/context gates → performance/speculation gates → promotion.

Commit 1 (done): foundation only — registry, census/ownership, memory-planner
stub that rejects over-budget plans, synthetic mini-MoE, deterministic tests,
docs. **No pruning, no networking hot paths, no kernels.** Those arrive in
later milestones against real evidence.

## Methods

The atlas is compression-method-agnostic. Candidate families (REAP; AQLM
low-bit; EXL3/EXO quant; maestro-style orchestration) are scored on the same
evidence base; none is hard-wired. When a method is evaluated, record the exact
method, version, config, and its own evidence/error — do not claim one method
"worked" from another method's run.
