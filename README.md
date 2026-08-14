# model-atlas

**Model-agnostic inside-the-model analysis and derivative-building platform.**

`model-atlas` measures what is inside a large transformer/MoE model and turns
that evidence into a smaller, evidence-driven derivative candidate: tensor
census → streamed REAP atlas → keep-map planning → checkpoint conversion →
repair/distillation → serving.

It is **not** tied to any one model. Registered subjects include **GLM-5.2**
(NVIDIA NVFP4 / DeepSeek-style DSA sparse-attention MoE, 78 layers, 256 routed
experts, top-8) and **Kimi K3** (2.78T params, 104B active, 896 routed
experts/layer top-16). The core is driven by a configurable `ArchitectureSpec`;

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
same evidence rather than hard-wired. Evidence is typed
(`measured` / `estimated` / `predicted` / `inferred` / `causally_tested`);
predictions are never presented as measured and are never deployable.

## Subsystems

1. `census` — tensor census + ownership (every tensor, layerwise, source
   identity preserved; no unclassified).
2. `checkpoint` — Safetensors header census, source manifest (hash-verified,
   AppleDouble `._*` aware), structural graph, synthetic fixture builder.
3. `atlas` — streamed layerwise REAP analysis, channel trace collector
   (`ChannelStatsAccumulator`), v3 fidelity-first pipeline (spectral /
   shared-structure / conditional / routing-consistency / bit-budget / NVFP4 /
   KV / structural fallback / Pareto).
4. `scoring` — TENP, grouped-Taylor surrogate, causal, stability, semantic,
   redundancy, quant-sensitivity.
5. `planning` — byte-accurate memory planner, real-bytes candidate planner,
   width-bucket / rate-distortion optimizer, typed `CandidatePlan` maps,
   INSUFFICIENT_EVIDENCE protection gate.
6. `builder` — derivative builder from a `CandidatePlan` (coupled gate/up/down
   surgery, renumbering + router remap, identity provenance).
7. `serving` — two-node expert-parallel assignment + fit, elastic NVMe-overflow
   simulation, KV ledger.
8. `preflight` — machine-readable capability/preflight report.

## Current status

The codebase provides a **synthetic research/control plane** plus a **real
checkpoint metadata census** (GLM-5.2-NVFP4: 232,385 tensors / 47 shards /
~465 GB, 100% classified, structural graph valid). It does **not** yet run real
tensor bodies end-to-end: no torch/transformers/vLLM/SGLang/ModelOpt/EXL3
execution adapter, no real forward/trace on GLM weights, and no distributed
runtime. Those are the active milestones on the `atlas-glm52-experiment-runtime`
branch. No custom kernels, no networking hot paths.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
model-atlas doctor
model-atlas list-architectures
model-atlas census k3-mini
model-atlas plan k3-mini --node-a-gb 0.001 --node-b-gb 0.001
model-atlas preflight --out capability_report.json
```

Run `model-atlas --help` for all commands. Fast tests (default, excludes
`--slow`/`--gpu`):
`pytest`; full suite: `pytest -m ""`; integration: `pytest -m integration`.

## License

Apache-2.0.
