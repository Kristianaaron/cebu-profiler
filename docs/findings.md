# Initial findings — unknowns that require measurement

These are open items that must be measured or resolved before they can be
encoded as facts. Nothing here is guessed.

## Kimi K3 (first registered parent)

- **Vocabulary size** — not given in the blueprint; must come from the
  checkpoint config/tokenizer. Currently `None` (not fabricated).
- **Exact per-tensor shapes** (router, gate/up/down per expert, latent
  projections, attention/KDA/MLA, norms, embeddings, LM head) — require the
  real checkpoint census, not layout arithmetic.
- **Exact per-expert parameter count** — needed for byte-accurate active
  bytes/token. Unknown today.
- **Attention Residuals / SiTU-GLU / Stable LatentMoE internals** — must be
  verified against the checkpoint + a pinned architectural reference
  (PipeNetwork/kimi-k3-mlx), not assumed.
- **Vision pathway** — presence/shape unverified.

## Reference/baseline

- Pin `PipeNetwork/kimi-k3-mlx` commit; verify tensor-name mapping against the
  official checkpoint (Phase-0-style census parity: 100% tensor-key coverage,
  no unclassified tensors).
- Establish baselines to compare derivatives against: full-K3 teacher/API,
  DeepSeek V4 Flash, a strong Qwen local model.

## Hardware

- Confirm both DGX Spark nodes' exact unified-memory availability and what the
  OS/CUDA/NCCL workspace actually reserves, so the runtime-reserve default is
  grounded, not assumed.

## Methods

- `AQLM` / `EXL3` / `maestro` were named by the user as candidate
  compression/orchestration approaches but are **not** documented in the
  blueprint sources; each needs a pinned-revision feasibility audit (they are
  not assumed to support K3). REAP is the documented saliency method.
