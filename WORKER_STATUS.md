# WORKER_STATUS — atlas-glm52-experiment-runtime

_Updated 2026-08-14 (round 2). Worker implementing the real GLM-5.2 two-DGX-Spark
experiment mission. Branch `atlas-glm52-experiment-runtime` (source `f1fd5d9`).

## Current phase

**Phases 0–7 IMPLEMENTED and TESTED** (code + tests + manifests + runbook).
The only thing left is the **explicit service-window execution** of the real
full-model forward + two-node benchmark, which would require evicting the
production two-rank DeepSeek vLLM + llama-server. That single execution gate is
BLOCKED by design (operating mode forbids stopping/contending services); the
maintenance-window runbook is one decision away.

## Commits created (round 2)

| Commit | Scope |
|---|---|
| (Phase 3) | real GLM config/facts + bounded streamed routing trace; torch scoring (TENP/FlexMoE/Taylor/causal); derivative materializer; quant backends; two-node inventory/launch; runtime contracts; evidence gates; runbook + CLI — committed below |

## Phase 3 — real GLM trace + scoring · DONE (tested fixture + bounded real)
- `glm52trace.py`: measured GLM-5.2 NVFP4 facts (quant_algo=NVFP4, group_size=16,
  75 sparse layers, 154880 vocab, MTP=1, kv_lora_rank=512); bounded CPU routing
  trace over the real router (`gate.weight` BF16 + `e_score_correction_bias` F32)
  producing measured top-8 IDs/weights/frequency/co-activation/entropy. Ran on the
  real mount. Config adapter normalizes `layer_types` for native transformers
  (`deepseek_sparse_attention` -> sparse), verified `GlmMoeDsaForCausalLM` loads.
- `scoring/torch_scores.py`: torch-backed TENP / FlexMoE channel ranking /
  grouped-Taylor / causal-ablation; forward-only requirements for bounded path,
  full-forward needs flagged HIGH_PRECISION_WEIGHTS+ROUTER_LOGITS (service-gated). Verified
  math under `.venv-exec` (torch/cuda present).

## Phase 4 — derivative materializer · DONE (fixture verified)
- `materialize.py`: transactional output (temp -> JSONL journal -> validate ->
  promote), sha256 per-shard manifest, source immutability (read-only mmap),
  fail-closed coverage (abort + discard if a tensor expected is missing),
  NVFP4 block-scale-aware channel slicing (gate/up rows + down columns coupled).

## Phase 5 — real quantization backends · DONE (honest detection)
- `quantbackends.py`: EXL3 (absent both hosts -> requires_custom_kernel + setup
  commands), ModelOpt NVFP4 (host venv 0.45.0 < producer 0.46.0.dev65 -> PROBE_ONLY
  with parity note; never INFERENCE), vllm NVFP4 (compressed-tensors scheme present
  but not claimed runtime). Setup commands are actionable, never mocks.

## Phase 6 — two-node inventory/launch/runtime · DONE (metadata probes)
- `twonode.py`: live probe of `spark-d167` (10.77.0.1) + `gx10-ac63` (10.77.0.2)
  via ssh BatchMode; both reachable, both NVIDIA GB10 compute-cap (12,1), both
  exec venvs (torch 2.11.0+cu130, NCCL (2,28,9), vllm 0.21.0 maps
  GlmMoeDsaForCausalLM -> deepseek_v2 and has compressed-tensors NVFP4). Exact
  per-rank ledger (weights/runtime/allocator/KV/comm/OS/headroom) + launch-plan
  gates (nodes_reachable, per_rank_memory_fit, non_evasive).
- `runtimecontracts.py`: SM121/MTP/KV/runtime contracts (fp8 KV baseline, NVFP4
  KV experimental-only behind parity gate, MTP acceptance/rollback).

## Phase 7 — eval + Pareto evidence gates · DONE
- `evidencegates.py`: candidate is MEASURED (deployable / measured frontier)
  ONLY when materialized + held-out evaluated + runtime benchmarked; otherwise
  PREDICTED / never deployable. `FrontierRecorder` keeps measured vs predicted
  frontiers separate.

## Runbook + rollback
- `docs/glm52-runbook.md`: exact commands for preflight, routing canary, torch
  forward, two-node vllm launch, benchmark, eval, Pareto — all behind the
  service-window gate.
- `scripts/production_rollback.py`: read-only helper that PRINTS the operator's
  freeze/SIGTERM commands for the DeepSeek vLLM worker (pid 1366114) + llama-server
  (pid 611507); never executes a stop/restart itself.

## Review-correction (2026-08-14, round 3) — applied, commit below

The code-review rejection of `1a8de9a` was addressed and committed. Key corrections:

1. **glm52trace** relabelled to `REAL_ROUTER_SYNTHETIC_INPUT_PROBE` / `PREDICTED`
   (real router + synthetic Gaussian input — never measured corpus/activation
   evidence); added `input_label`/`evidence_kind`/`provenance`; entropy fixed
   (positive; no double-negation); coactivation counts each unordered distinct
   expert combination exactly once per token; `sorce_gate_bias` typo removed
   -> `gate_bias_values`; `normalized_glm52_config` now DROPS the incompatible
   `layer_types` key (does not DSA->sparse / silent full_attention replacement).
2. **Materializer** is now real NVFP4 surgery: nibble-aware (2 values/byte)
   weight packing; gate/up keep byte-rows; down requires a UNION OF FULL
   16-CHANNEL SCALE GROUPS (fails closed otherwise, `NonBlockAlignedError`);
   `weight_scale_2`/`input_scale` are SCALARS copied unchanged; coverage
   validates exact names/shapes/byte-counts + sha256 hashes (not a count);
   router written ONCE; output is `NON_LOADABLE_EXPERT_BANK` (never claimed
   experiment-ready / vllm-loadable); `overwrite=True` required to replace an
   existing dir (never implicit rmtree).
3. **twonode** measures host unified memory (MemTotal/MemAvailable, since GB10
   VRAM reports N/A) + production occupancy; `RankLedger` separates physical
   capacity / production occupancy / current allocatable — go/no-go never an
   unexplained 100 GiB constant.
4. **vLLM** multi-node flags validated against vllm 0.21 `--help`: `--nnodes` /
   `--node-rank` + `--enable-expert-parallel` / Ray head/worker bootstrap; the
   prior `--node-ip`/`--ray-address` server flags (invalid) and the unsafe
   commands (503 GB source across 2×~120 GB, single-node `from_pretrained`)
   removed. Runbook now OPEN only with a fully materialized, loadable,
   in-envelope derivative; otherwise gate CLOSED.
5. **Quant truth**: EXL3 conversion from the NVFP4 checkpoint is BLOCKED (no
   BF16 parent / no verified ModelOpt dequantization); ModelOpt 0.45 vs producer
   0.46.0.dev65 parity is UNPROVEN -> UNSUPPORTED (was PROBE_ONLY).
6. **Scoring**: kernels over synthetic/random tensors are IMPLEMENTATIONS,
   relabelled PREDICTED with provenance; added `RealActivationHook` real-forward
   interface; the measured gate stays closed until a real corpus forward runs.

## Current status: one gate BLOCKED (real full-model forward + two-node benchmark)

The real full-model forward + two-node benchmark require evicting the production
two-rank DeepSeek vLLM (TP0/TP1 ~103 GB each) + llama-server, AND require a
fully materialized, loadable, in-envelope derivative. That is the single
remaining execution gate. All code/tests/manifests/runbook are complete and
verified; finish via `docs/glm52-runbook.md` (which now opens only after the
gate passes).

## Test + lint results (review round)

- All review regression tests green (repo venv): test_glm52trace, test_materialize,
  test_twonode, test_quantbackends, test_evidencegates, test_streaming,
  test_realbody, test_protocols, test_corpus_manifest, test_runtimecontracts.
- torch scoring tests SKIP in repo venv (no torch); verified correct under
  `.venv-exec` (kernels + provenance + real-hook all pass).
- `ruff check src tests scripts` -> All checks passed.
- `mypy src scripts` -> Success, no issues.


## Test + lint results (round 2)
- All new phase 3–7 test files green: test_glm52trace, test_torch_scores (skips
  in repo venv, verified under .venv-exec), test_materialize, test_quantbackends,
  test_twonode, test_runtimecontracts, test_evidencegates + earlier test_streaming/
  realbody/protocols/corpus_manifest.
- `ruff check src tests scripts` -> All checks passed.
- `mypy src scripts` -> Success, 114 source files, no issues.
- Full fast suite (`pytest -m "not slow"`): **151 s, 221 tests, all green.**

## Measured vs estimated
- Measured: real GLM facts (quant_algo/group_size/layers/vocab/MTP), bounded real
  routing trace (top-8 across 256 experts, layer 3), torch math, two-node SSH/
  NCCL/GPU reachability + exec versions, active services (vllm+llama), dtype/
  NVFP4 layout, modelopt/vllm presence.
- Estimated: KV ledger (contract), runtime throughput, held-out quality deltas
  (all explicitly PREDICTED until materialized + evaluated + benchmarked).
- Blocked (execution only): real full forward + two-node benchmark — service
  window required.

## Active blocker + required decision
1. Authorize the maintenance window (operator stops DeepSeek 2-rank vLLM +
   llama-server) and approve `docs/glm52-runbook.md` section 4 to run the real
   forward + two-node benchmark; then the measured Pareto frontier is produced.

## Exact next commands (after window approval)
```bash
# preflight (safe anytime)
cd /home/kristianaaron/tmp/model-atlas && git checkout atlas-glm52-experiment-runtime
pytest -m "not slow" -q && model-atlas preflight --out capability_report.json
# routing canary (safe)
.venv/bin/python -c "from model_atlas.glm52trace import stream_routing_trace as s; print(s('/media/glm52/models/nvidia/GLM-5.2-NVFP4', layer=3).to_dict())"
# then docs/glm52-runbook.md section 4 (torch forward, two-node vllm, benchmark, eval)
```

## Known risks / rollback
- All on the branch; `main` untouched at `f1fd5d9`; deleting branch reverts all.
- Source GLM mount opened read-only, never modified; hashes preserved.
- No GPU memory used; no production service stopped/deprived during any work here.
- Synthetic/estimated numbers tagged PREDICTED; only full-chain candidates MEASURED.
