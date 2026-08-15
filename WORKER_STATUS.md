# WORKER_STATUS — atlas-glm52-experiment-runtime

_Updated 2026-08-15 (round 8: W64 derivative canary materialized on the fixed exporter)._

## Mission / current state

Implement the real GLM-5.2 two-DGX-Spark experiment capability. Round 3 added a
**structurally-complete uniform-width derivative exporter** (all sparse layers x all
experts, transactional/resumable/streamed) + a **measured size plan** from the
464.8 GB census, and corrected every second-review finding. Two irreducible
gates remain and are honestly reported (see Blockers): (i) no runtime-loadable ModelOpt-NVFP4
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
- **F (structural exporter, NEW)** — `loader.py` `materialize_uniform_width`: produces a STRUCTURALLY-COMPLETE tree. `runtime_compatibility='schema-supported-unvalidated'`; `runtime_validated=False` until a real materialized derivative load/forward is validated. The installed vLLM 0.21 DOES contain a ModelOpt-NVFP4 path (ModelOptNvFp4Config + Linear/FusedMoE + kernels/emulation) — NOT decoder-blocked.
  all sparse layers x all experts, uniform width `W` (multiple of 16, verified fail-closed on non-multiples
  fail-closed), expert-specific channel selections only when width == W, every
  non-target tensor copied verbatim, SAFETENSORS index + `config.json`
  (`moe_intermediate_size=W`) rebuilt, quant/tokenizer/code assets preserved,
  exact census validation (names/dtype/shape/byte totals/hashes) + output hash
  compared to write-time journal; source immutability STAT-verified (size/mtime,
  not full-file hashing); transactional overwrite (old output preserved until
  validated, atomic swap+backup); resumable (`done-shard` journal), never loads
  a source shard wholesale (bounded `_IO_CHUNK` reads). Real mounted GLM-5.2 NVFP4
  geometry precheck PASSES (header-only metadata, no body reads; hidden=6144,
  full=2048, gate/up U8 [2048,3072]+F8 [2048,384], down U8 [6144,1024]+F8
  [6144,128], scalars F32; layer-78 final shared-head block excluded from the
  routed-expert target set). `plan_uniform_widths` computes the size plan from
  the metadata census BEFORE writing.
- **G (real size plan)** — computed from the mounted 464.8 GB census for aligned
  widths (below) and shows only widths fitting two ~120 GiB physical nodes AFTER
  an explicitly authorized maintenance-window removal of production occupancy.

## Real size plan (measured census, 2026-08-14)

Source: 232,385 tensors / 47 shards / ~464.8 GB; hidden 6144, 2048 moe_intermediate,
75 sparse layers x 256 experts, group 16, NVFP4.

| W | total GiB | per-rank GiB (EP) | fits 2x~120 GiB (window) |
|---|---|---|---|
| 64 | 59.1 | yes |
| 128 | 65.1 | yes |
| 256 | 76.9 | yes |
| 512 | 100.6 | yes |
| 1024 | 148.1 | no |
| 2048 | 243.0 | no |

Recomputed with the EXACT exporter size plan (scalar quant tensors do NOT
scale): W64/W128/W256/W512 fit a 115 GiB usable per-node planning budget;
W1024/W2048 do not. W=256 => per_rank ~76.9 GiB. After an authorized
maintenance window (production occupancy removed) the two-node run is
possible; current live availability (production still running) is measured
separately by `model-atlas two-node` and is NOT the planning capacity.

## W64 derivative canary — materialized (round 8)

Fixed the W64 exporter performance defect (`loader.py`): the old
`_stream_body_window` opened each source shard once PER window — a real down
tensor with 6,144 rows kept 64 channels => 24,576 windows/tensor, ~236M source
opens across the export. The exporter now uses a lazy idempotently-closable
per-output-shard body provider (one source handle per shard, always closed via
`finally`), coalesces ordered sparse windows into bounded (<=4MiB) source spans
(one seek/read per span, absolute offsets handled correctly above 4MiB), splits
>4MiB contiguous intervals into <=4MiB chunks, and rejects
overlapping/reversed/zero-length windows. Exact bytes/order and
journal/promotion/resume semantics unchanged. (commit `cabf8f1`).

The preserved canary resumed and completed:
- **Output**: `derivatives/glm-uniform-w64-canary/` — 47 shards, 232,385
  tensors, **69,849,116,672 data bytes** (independent header parse), exactly the
  expected totals.
- **Resume-skip proof**: finalized shards 1–2 skipped unchanged
  (journal `skip-shard` steps); pre- vs post-resume sha256 identical:
  `f01ccdc7…e26b448` (shard 1), `474db88e…1fb5` (shard 2), and size+mtime
  unchanged. Discardable partial shard 3 rebuilt from source.
- **Validation**: exporter's own exact validation `ok 232385 tensors`;
  journal ends `slice → validate → promote`; `promoted=True`.
- **Independent checks**: 47 shards, 69,849,116,672 data bytes, 232,385
  tensors, index weight_map keys == source census, expert tensors sliced to
  width=64 (e.g. `down_proj.weight [6144, 32]`), non-target tensors
  byte-identical to source (20 spot-checked), `moe_intermediate_size=64`.
- **Runtime probe** (installed vLLM 0.21, config-only, no weights/GPU):
  `schema_supported=True`, `quant_config_recognized=True`,
  `decoder_path_present=True`, `kernel_paths_present=True`, linears
  `ModelOptNvFp4LinearMethod`/`FusedMoE`; **`derivative_load_validated=False`,
  `runtime_ready=False`** (Ray/external modelopt not installed; no real
  derivative load/forward run). Config-only report at
  `/tmp/runtime_cap_w64.json`.

> The arbitrary first-64 channels are a **structural canary only** — NOT a
> scientific result. Measured corpus keep-maps and a maintenance-window
> load/forward remain before any scientific/runtime-readiness claim.

**Post-promotion fix (round 8a):** the promoted canary's
`model.safetensors.index.json` `metadata.total_size` was a stale copied source
value (464,795,267,072, the full 503 GB checkpoint). This was a real exporter
packaging bug. `loader._write_index` now rebuilds `metadata.total_size` to the
exact output tensor-data bytes; the already-promoted index was repaired to
69,849,116,672 and revalidated: 47 shards, 232,385 tensors, exact
69,849,116,672 data bytes, exact census match, `weight_map` == all shard
headers, `moe_intermediate_size=64`. Shard payloads were NOT rewritten (shard 1
hash still `f01ccdc7…`); `total_parameters` preserved.

## Commits (round 1–3)

`97f23b0` P0 · `2846b84` P1 · `e072896`+`d3a8b22` P2 · `972dbc3` handoff · `1a8de9a`
P3–7 · `9a164e9` review-fix 1 · (round-3 commit, below).

## Current status — two irreducible gates

1. **Runtime validation gate (not decoder-absent)**: the installed vLLM 0.21
   DOES contain a ModelOpt-NVFP4 path: `modelopt.py` provides
   `ModelOptNvFp4Config`, `ModelOptNvFp4LinearMethod`, `ModelOptNvFp4FusedMoE`;
   the registry maps `modelopt_fp4 -> ModelOptNvFp4Config`; the mounted config
   (`quant_method=modelopt, quant_algo=NVFP4`) returns override `modelopt_fp4`,
   and `ModelOptNvFp4Config.from_config(mounted_qc)` succeeds selecting the
   NVFP4 Linear+FusedMoE methods; NVFP4 kernels/emulation are present in source.
   What is absent/unproven: Ray package (not installed), external `modelopt`
   package (not installed; likely not required for vLLM inference), producer
   parity still unproven, and crucially a real materialized-derivative
   load/forward is NOT yet validated. `runtime_validated=False`.
2. **Service-window execution**: real forward/benchmark requires the operator to
   remove production occupancy (DeepSeek two-rank vLLM + llama-server) — never
   done automatically. `docs/glm52-runbook.md` gives exact commands; the gate
   stays CLOSED until an authorized window.

## Readiness (not decoder-infeasible)

The Atlas structural exporter is implemented and mounted-header-compatible
(real geometry precheck PASS). Candidate widths W64/W128/W256/W512 fit a 115-GiB
usable per-node planning budget (exact table above); W1024/W2048 do not. The
next safe technical step is to materialize a chosen derivative (disk/time/storage
gate) and validate config/index on the resulting tree, then, once Ray is
available and a maintenance window is authorized, load/forward the derivative on
the two nodes. The decoder path IS present (probe output in this file + JSON in
`/tmp/runtime_cap.json`); it is simply not yet end-to-end validated.

## Tests / lint

- runtimeprobe (fake-adapter + installed-vLLM mounted-config probe), loader_r4,
  torch round-7, fitplan, quantbackends (ModelOpt-NVFP4 detection) all green;
  `ruff` + `mypy` (src + scripts) clean; full fast suite green.
