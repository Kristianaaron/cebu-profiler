# Cebu Profiler

![Cebu Profiler](docs/banner.svg)

**Shrink a huge model. Prove it still works. Fail closed.**

Cebu Profiler is a model-agnostic, inside-the-model analysis and
derivative-building platform. It measures what is inside a large transformer /
MoE checkpoint and turns that evidence into a smaller, evidence-driven
derivative: tensor census → streamed REAP profiling → keep-map planning →
checkpoint conversion → repair/distillation → serving.

Four properties define the pipeline:

- **Evidence first.** Nothing is pruned, quantized, or reshaped on a hunch —
  every intervention cites measured saliency, contribution, or causal evidence.
- **Fail closed.** Missing evidence, malformed keep-maps, or a failed quality
  gate stops the pipeline with a recorded rejection. Estimated values are never
  presented as measured ones.
- **Model-agnostic.** Kimi K3 is the first registered subject (2.78T params,
  104B active, 896 routed experts/layer top-16, Stable LatentMoE, MXFP4
  experts, ~1.56 TB). The core is driven by a configurable `ArchitectureSpec`;
  other large models register as additional architectures.
- **Serving-ready for vLLM, SGLang & TRT-LLM.** Derivatives are laid out for
  direct vLLM / SGLang / TRT-LLM serving — NVFP4-coherent expert groups,
  router-consistent — not GGUF exports. Runtime load/forward validation is an
  explicit gate, never assumed.

## How it works

1. **Profile** — census every tensor and layer; establish ownership so no
   tensor is unclassified and source identity is preserved.
2. **Recommend** — pick the compression method the evidence supports.
3. **Compress** — saliency-ranked 16-channel expert width-slicing with router
   indices and correction biases reordered exactly in step (NVFP4-coherent
   layout), or quantization where the evidence calls for it.
4. **Verify & hand off** — structural checks, a teacher KLD/CKA quality gate
   against the original model on the same prompts, runtime load/forward
   validation under vLLM / SGLang, then a hash-verified run bundle.

## Intent (one clear purpose per app)

- **`eval-lab`** — *measure* model/agent capability with deterministic
  evidence. Separate project; stays a harness only.
- **`cebu-profiler`** — *shrink or reshape a large base model into a smaller,
  evidence-driven derivative.* This repo.
- **Bridge** — an optional plugin, added only when a user wants `eval-lab`'s
  label ontology / holdout evals to validate a Cebu Profiler derivative. Not a
  baked-in dependency. Dependency edge is one-way: `cebu-profiler → eval-lab`.

## Method-agnostic evidence

The profiler records a rich, per-expert / per-layer evidence base (saliency,
routing frequency, contribution norms, routing entropy, quantization
sensitivity, substitution). Any compression method — REAP, AQLM low-bit,
EXL3/EXO quant formats, or a maestro-style orchestrator — is scored against the
same evidence rather than hard-wired.

## Subsystems (crisp modules, not one blob)

1. `census` — tensor census + ownership (every tensor, layerwise, source
   identity preserved; no unclassified).
2. `profiler` — streamed layerwise REAP analysis (saliency by label/stage).
3. `analysis` — v3 fidelity-first analyzers (spectral, semantic,
   quantization-sensitivity, routing consistency, and more).
4. `planning` — byte-accurate memory + keep-map planner, width buckets,
   protection of attention/shared/backbone components.
5. `compression` — backend registry, quantization math, per-expert response
   curves.
6. `executor` — structural application of plans to a model clone.
7. `serving` — two-node / elastic runtime.
8. `dashboard` — interactive HTML dashboard rendered from measured artifacts.
9. `kernels` — versioned runtime-kernel evidence receipts (fail-closed oracle).

## Current status

Working end-to-end on the synthetic mini-MoE subject: tensor census,
architecture registry, streamed profiling, v3 fidelity-first analysis
pipeline, width-slice planning under byte-accurate budgets, structural
execution, quality gates, and the dashboard. Real-checkpoint compression
arrives against measured evidence in later milestones — no pruning decisions
are made on this scaffold alone.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cebu-profiler doctor
cebu-profiler list-architectures
cebu-profiler census k3-mini
cebu-profiler plan k3-mini --budget-gb 0.001 --node-budget-a-gb 0.0006
```

Run `cebu-profiler --help` for all commands.

## Runtime-kernel evidence

Cebu Profiler imports versioned kernel benchmark receipts without owning the
CUDA implementation. Exact measured direct-kernel observations can gate
candidate runtime selection; CPU/reference checks and estimates remain visible
but cannot be ranked as speed evidence. Model IDs, operator names, phases,
representations, and ABIs are data rather than allowlists, so new models do not
require bridge code changes. See the
[Kernel Evidence Bridge](https://github.com/Kristianaaron/cebu-profiler/blob/main/docs/kernel-evidence-bridge.md).

## Evidence-grade instrumentation

Beyond the base census and REAP profile, the profiler ships evidence-grade
instrumentation used when a derivative claim must survive scrutiny:

- **Rank-trust protocol** (`stability/protocol.py`) — split-half rank
  reliability (Spearman) between calibration halves, keep-set Jaccard at fixed
  sizes, and named proxy controls (count / mass / proxy), with an evidence-typed
  verdict: `measured` is reserved for strong split-half agreement on real runs,
  never granted to proxies.
- **Causal prune-arm stress matrix** (`stress/arms.py`) — remove low-score
  experts at several fractions against random-removal and high-score controls,
  plus a bit-exact identity arm; damage is scored by measured logit-KL, output
  cosine, and argmax stability — not by the proxy that chose the victims.
- **Deep census** (`census/deep.py`) — opt-in measured per-tensor quantization
  damage (INT8 per-channel, INT4 group-128, FP8 e4m3 SQNR in dB) plus
  distribution and spectral statistics, decoded straight from safetensors
  bytes in bounded-memory chunks; the header-only census remains the default.
- **Coverage gate & limitations ledger** — every run manifest carries a
  machine-checkable evidence-coverage report and an explicit limitations block;
  missing or empty artifacts fail the gate instead of passing silently.

## Credits & methodology inspiration

Several instrumentation patterns here were independently re-implemented after
studying [alesha-pro/atlas](https://github.com/alesha-pro/atlas) (MIT) and its
published GLM-5.3-Flash NVFP4 evidence bundle — specifically: split-half rank
stability with keep-set Jaccard and proxy controls, the five-arm causal prune
stress test with random/high controls, per-tensor measured SQNR scans
(INT8/INT4-g128/FP8), and machine-readable coverage + limitations blocks in
model evidence bundles. All code in this repository is original: the
implementations run on Cebu Profiler's own manifest model, scorer interfaces,
and frozen-model intervention API, with evidence-typing, fail-closed identity
arms, bounded-memory chunked decoding, and config-driven architecture specs as
additions of our own. Ideas credited where due; nothing copied.

## License

Apache-2.0.
