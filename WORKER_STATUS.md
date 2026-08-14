# WORKER_STATUS — atlas-glm52-experiment-runtime

_Updated 2026-08-14 (round 3, coherent rewrite — no stale overclaims)._

## Mission / current state

Implement the real GLM-5.2 two-DGX-Spark experiment capability. Round 3 added a
**full loadable uniform-width derivative materializer** (all sparse layers x all
experts, transactional/resumable/streamed) + a **measured size plan** from the
464.8 GB census, and corrected every second-review finding. Two irreducible
gates remain and are honestly reported (see Blockers): (i) no loadable NLFP4
decode path in the installed stack, and (ii) the real forward/benchmark requires
an operator maintenance window (services must be stopped by the operator, and
current availability is a separate live gate).

## Review round 3 — applied

- **A (refs)** — reference tensors now go to UNIQUE shards (`layer{l}-ref-{i}`)
  so the second never overwrites the first; fixture now includes both the router
  correction bias AND `input_layernorm.weight`; test asserts no name is dropped
  or duplicated.
- **B (validation)** — `expected`/`found` now compare EXACT (dtype, shape,
  byte_size) triples (not dtype+byte only); duplicate tensor names are rejected;
  output shard hashes computed after write; `keep_channels` normalized exactly
  once up front and fails closed on empty / duplicates / negatives / out-of-range
  / inconsistent packed+scale dims / non-16-multiple for down.
- **C (runbook + vLLM/Ray)** — removed stale `per-{}` placeholder and unquoted
  derivative path; validated server flags against vllm 0.21 `--help`
  (`--nnodes`/`--node-rank`/`--enable-expert-parallel`, NO server `--node-ip`);
  Ray journal-inspection shows the installed stack has **no Ray package**, so
  the emitted `ray start --node-ip-address` is Ray's documented spelling, not a
  local-help-verified flag (recorded; Ray must be installed separately).
- **D (causal + measured gate)** — `causal_ablation_scores` is now a genuine
  baseline-vs-ablated-output scorer with shape checks (no false causal proxy);
  `RealActivationHook` measures ONLY with an explicit real-corpus run id
  (offline replay never); `TorchScoringResult` rejects MEASURED with synthetic or
  missing provenance. Verified under `.venv-exec` (torch) + repo venv.
- **F (loadable materializer, NEW)** — `loader.py` `materialize_uniform_width`:
  all sparse layers x all experts, uniform width `W` (18-multiple-of-16 verified
  fail-closed), expert-specific channel selections only when width == W, every
  non-target tensor copied verbatim, SAFETENSORS index + `config.json`
  (`moe_intermediate_size=W`) rebuilt, quant/tokenizer/code assets preserved,
  exact census validation (names/dtype/shape/byte totals/hashes) + source
  immutability, transactional (temp -> journal -> validate -> promote),
  resumable (`done-shard` journal), never loads a source shard wholesale
  (per-window bounded reads). `plan_uniform_widths` computes the size plan from
  the metadata census BEFORE writing.
- **G (real size plan)** — computed from the mounted 464.8 GB census for aligned
  widths (below) and shows only widths fitting two ~120 GiB physical nodes AFTER
  an explicitly authorized maintenance-window removal of production occupancy.

## Real size plan (measured census, 2026-08-14)

Source: 232,385 tensors / 47 shards / ~464.8 GB; hidden 6144, 2048 moe_intermediate,
75 sparse layers x 256 experts, group 16, NVFP4.

| W | total GiB | per-rank GiB (EP) | fits 2x~120 GiB (window) |
|---|---|---|---|
| 64 | 47.6 | 41.4 | yes |
| 128 | 60.0 | 47.6 | yes |
| 256 | 84.9 | 60.0 | yes |
| 512 | 134.6 | 84.9 | yes |
| 1024 | 234.0 | 134.6 | marginal (needs measured free >135 GiB/rank) |
| 2048 | 432.9 | 234.0 | no |

Backbone (copied) bytes ≈ 35.2 GiB; routed-expert bytes ≈ 397.7 GiB at full width.
After an authorized maintenance window (production occupancy removed), physical
capacity is ~120 GiB/node; at `per_rank ~60 GiB` node W=256 fits with ~60 GiB
headroom. Current availability (production still runnning) is measured
separately by `model-atlas two-node` and is NOT the planning capacity.

## Commits (round 1–3)

`97f23b0` P0 · `2846b84` P1 · `e072896`+`d3a8b22` P2 · `972dbc3` handoff · `1a8de9a`
P3–7 · `9a164e9` review-fix 1 · (round-3 commit, below).

## Current status — two irreducible gates

1. **Loadable decode path (schema/runtime blocker, demonstrated from installed
   source)**: vllm 0.21 + transformers 5.9.0 here expose NO ModelOpt NVFP4
   decoder (no `modelopt` quant_method mapping — verified: `modelopt-aware
   path present: False`; `nvfp4 modules in vllm quant: []`; `compressed_tensors`
   files for nvfp4: none; `CompressedTensorsLinearMethod` does not handle
   `modelopt`). The mounted checkpoint is `quant_method=modelopt, quant_algo=NVFP4`.
   So the standard stack cannot decode/execute this NVFP4 derivative as-is; the
   closest valid artifact our materializer can produce is the FULL LOADABLE
   uniform-width safetensors+index+config derivative (valid HF/census structure)
   plus an honest note that the NVFP4 *decode/run* requires either a ModelOpt
   decode adapter or a BF16 reference that is unavailable. The materializer
   writes a correct `moe_intermediate_size` and preserves every tensor's dtype/
   shape/quant metadata, so it is a valid checkpoint for any stack that CAN
   decode ModelOpt NVFP4 (e.g. the exact producer version). We do NOT claim
   vllm can run it today.
2. **Service-window execution**: real forward/benchmark requires the operator to
   remove production occupancy (DeepSeek two-rank vLLM + llama-server) — never
   done automatically. `docs/glm52-runbook.md` gives exact commands; the gate
   stays CLOSED until an authorized window.

## Tests / lint

- Round-3 regressions green (repo venv): test_materialize, test_glm52trace,
  test_twonode, test_quantbackends, test_evidencegates, test_streaming, + torch
  tests verified under `.venv-exec` (D semantics). `ruff` + `mypy` clean.
