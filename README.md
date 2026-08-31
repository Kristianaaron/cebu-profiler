# model-atlas

**Model-agnostic inside-the-model analysis and derivative-building platform.**

`model-atlas` measures what is inside a large transformer/MoE model and turns
that evidence into a smaller, evidence-driven derivative candidate: tensor
census → streamed REAP atlas → keep-map planning → checkpoint conversion →
repair/distillation → serving.

It is **not** tied to any one model. Kimi K3 is the first registered subject
(2.78T params, 104B active, 896 routed experts/layer top-16, Stable LatentMoE,
MXFP4 experts, ~1.56 TB). The core is driven by a configurable
`ArchitectureSpec`; other large models register as additional architectures.

## Intent (one clear purpose per app)

- **`eval-lab`** — *measure* model/agent capability with deterministic evidence.
  Separate project; stays a harness only.
- **`model-atlas`** — *shrink or reshape a large base model into a smaller,
  evidence-driven derivative.* This repo.
- **Bridge** — an optional plugin, added only when a user wants `eval-lab`'s
  label ontology / holdout evals to validate an Atlas derivative. Not a baked-in
  dependency. Dependency edge is one-way: `model-atlas → eval-lab`.

## Method-agnostic evidence

The atlas records a rich, per-expert / per-layer evidence base (saliency,
routing frequency, contribution norms, routing entropy, quantization
sensitivity, substitution). Any compression method — REAP, AQLM low-bit,
EXL3/EXO quant formats, or a maestro-style orchestrator — is scored against the
same evidence rather than hard-wired.

## Subsystems (crisp modules, not one blob)

1. `census` — tensor census + ownership (every tensor, layerwise, source
   identity preserved; no unclassified).
2. `atlas` — streamed layerwise REAP analysis (saliency by label/stage).
   *Not yet implemented — this is the core milestone.*
3. `planning` — byte-accurate memory + keep-map planner.
4. `convert` / `repair` — checkpoint conversion, repair & distillation.
5. `serving` — two-node / elastic runtime.

## Current status

Scaffold (commit 1): package skeleton, architecture registry (K3 + synthetic
mini-MoE), tensor census + ownership, byte-accurate memory planner stub that
rejects over-budget plans, deterministic unit tests, docs. No pruning,
networking hot paths, or custom kernels yet.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
model-atlas doctor
model-atlas list-architectures
model-atlas census k3-mini
model-atlas plan k3-mini --budget-gb 0.001 --node-budget-a-gb 0.0006
```

Run `model-atlas --help` for all commands.

## Runtime-kernel evidence

Atlas imports versioned kernel benchmark receipts without owning the CUDA
implementation. Exact measured direct-kernel observations can gate candidate
runtime selection; CPU/reference checks and estimates remain visible but cannot
be ranked as speed evidence. Model IDs, operator names, phases, representations,
and ABIs are data rather than allowlists, so new models do not require bridge
code changes. See the
[Kernel Evidence Bridge](docs/kernel-evidence-bridge.md).

## License

Apache-2.0.
