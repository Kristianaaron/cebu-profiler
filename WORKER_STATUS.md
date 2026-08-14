# WORKER_STATUS — atlas-glm52-experiment-runtime

_Updated 2026-08-14. Worker implementing the real GLM-5.2 two-DGX-Spark experiment
mission. Branch `atlas-glm52-experiment-runtime` (source commit `f1fd5d9`).

## Current phase

**Phases 0–2 COMPLETE and committed.** Phase 3+ reached a **concrete
external-resource blocker** (no execution backend) and is reported, not
fabricated. Full capability matrix below.

## Commits created

| Commit | Phase | Summary |
|---|---|---|
| `97f23b0` | 0 | fast/slow test markers, synthetic Pareto evidence fix (PREDICTED), preflight report, version/README truth, types-PyYAML |
| `2846b84` | 1 | indexed Safetensors body streaming, real GLM NVFP4 bounded validation, runtime protocols, validate-bodies CLI |
| `e072896` | 2 | partitioned corpus manifest/loader + resumable run manifest |
| `d3a8b22` | 2 | preflight + real-GLM canary status (honest blocked forward) |

## Phase 0 — baseline & contracts · DONE
- Branch `atlas-glm52-experiment-runtime`, source `f1fd5d9`.
- Env: Python 3.12.3 / aarch64, 20 cores, 121 GiB host RAM; **1× NVIDIA GB10**;
  CUDA 13.0; `nolog GPU` compute apps = a vLLM worker (~10.2 GiB) + a
  `llama-server` (~3.7 GiB) — production workloads, left running.
- **pytest markers** `slow` / `integration` / `gpu`; default `addopts` excludes
  `slow` so the default suite is genuinely fast (~148 s vs >6 min full). Marked
  the spectral / full-v3-pipeline / dashboard / export tests `slow` (they hit
  the pure-Python spectral hotspot).
- **Synthetic Pareto evidence bug fixed**: idealized demo family + CLI both now
  tag `evidence_kind=PREDICTED` (were MEASURED); v3 evidence map `pareto ->
  predicted`; `FrontierPoint` default `evidence_kind` is now `PREDICTED` so
  accidental measurements can never be labelled measured.
- **Packaging/type/stale-fix**: version `0.1.0 -> 1.0.0`; `types-PyYAML` dev dep;
  truthful README; new `docs/capability-report.md`.
- **Machine-readable capability report**: new `preflight.py` + `model-atlas
  preflight --out capability_report.json` (measured GPU/SM, disk, mounted GLM,
  exec-backend present/absent, active services).
- Gate: **ruff + mypy green** (0.38 s mypy, all src+tests clean); fast suite
  green.

## Phase 1 — real checkpoint / tensor streaming substrate · DONE
- `checkpoint/streaming.py`: mmap-backed `BoundedShardReader` /
  `CheckpointStream`. Reads exactly the requested byte range (data-section base
  offset from the 8-byte header length); decodes BF16/F16/F32 reference bodies
  with pure-Python bit math; tracks **peak resident bytes**; never materializes
  a shard; `identity_copy` for read/write/copy proof. `write_safetensors_flat`
  added to the fixture-only writer for identity tests.
- `checkpoint/realbody.py`: `validate_real_bodies` bounded, read-only validation
  over a real checkpoint; read cap `MAX_BODIES`; reports dtype histogram +
  NVFP4 constituent layout + coverage / unclassified.
- `runtimeprotocols.py`: `MoEModelLike` / `LayerWeightsLike` / `ExpertWeights` /
  `TensorLike` protocols decouple structural analysis from `MiniMoE`; adapter in
  tests proves a real-GLM adapter is a drop-in.
- `cli validate-bodies` command.
- **Measured on real GLM-5.2-NVFP4** (`/media/glm52/models/nvidia/GLM-5.2-NVFP4`,
  ~503 GiB mounted):
  - 232,385 tensors / 47 shards / ~464.8 GB
  - dtype histogram: `BF16=1909` (reference: embed, lm_head, attention, shared
    experts, router `gate`, dense mlps), `F32=115,276` (NVFP4 scales +
    norms), `U8=57,600` (NVFP4 weight tokens), `F8_E4M3=57,600` (weight scales)
  - unclassified = **0**, coverage = **1.0** (fail-closed no-unclassified gate)
  - bounded body peak = 6 MiB; **process max RSS ~824 MiB**
  - NVFP4 expert layout measured: `gate_proj.weight U8[2048,3072]` +
    `weight_scale F8_E4M3[2048,384]` + `weight_scale_2 F32[]` +
    `input_scale F32[]` (and same for `up_proj`/`down_proj`) → confirms ModelOpt
    NVFP4 block-scaled 4-bit, NOT uniform INT4. Router per sparse layer:
    `gate.weight BF16[256,6144]` + `e_score_correction_bias F32[256]`.
- Source hashes preserved / never mutated (read-only mmap).

## Phase 2 — real GLM-5.2 corpus + trace runner (substrate) · DONE
- `corpus.py`: immutable partitioned corpus manifest/loader. Reads JSONL /
  plain-text from `calib/` `dev/` `heldout/` (or explicit per-partition file
  lists); filesystem only (no network). Samples never reused across partitions.
- `runmanifest.py`: `RunManifest` + `ChunkProgress` persisting small cumulative
  aggregates at layer/chunk boundaries (never raw activations);
  `ChunkedRunner` resumes by skipping completed chunks, saves after each
  boundary (crash-resume).
- `canary.py` + `cli canary`: real-GLM canary status → census coverage 1.0,
  unclassified 0, bounded body validated `True`, **forward_trace=blocked**
  (honest — this venv has no torch/transformers/vllm/modelopt/EXL3, and GLM
  NVFP4 bodies need the ModelOpt decoder).

## Phase 3+ — BLOCKED (concrete external-resource / exec-dependency)

Real scoring (TENP/Taylor/causal to torch tensors/traces), derivative
materialization of NVFP4 weights, real quantization backends (EXL3, ModelOpt
NVFP4), the two-Spark runtime, and the measured Pareto frontier all require an
**execution dependency that is not present** in this venv, plus a second node:

1. **No torch / transformers / vllm / sglang / modelopt / exllamav2 / safetensors /
   numpy** (measured by `model-atlas preflight`). Installing torch/vllm/modelopt
   is a large install whose GPU/unified-memory use would disrupt the running
   DeepSeek-v4 vLLM worker and llama-server (operating mode forbids stopping or
   consuming their capacity). **Required user decision**: authorize a venv
   (separate from the running services) with the exec stack installed, or
   confirm resource headroom.
2. **No second Spark node** (runtime Phase 6 requires real two-node vLLM/SGLang/
   EXL3/SM121 environment + a second node for exact per-rank memory/
   correctness gates).
3. **NVFP4 decoding** needs NVIDIA ModelOpt/its kernel path; the layout measured
   here (U8 tokens + F8_E4M3 block scales, group_size=16) is the ModelOpt NVFP4
   scheme — a real adapter is blocked until ModelOpt is installed.
4. **No higher-precision BF16 parent shards** beyond the BF16 backbone/attention
   references already present; the frontier's destructive decisions need streamed
   BF16 validation which requires a full BF16 source or torch decode of the
   reference tensors.

I did not simulate/mock these to appear complete — they are reported blocked.

## Tests / commands and exact outcomes

- `pytest` (default fast, excludes slow): **~148 s, all green** (189 tests
  collect; a bounded earlier full run hits the slow spectral hotspot → now
  correctly marked slow).
- New Phase-0/1/2 test files, all green (each run above):
  - `tests/unit/test_streaming.py` (bounded read, identity copy, BF16/F16 decode)
  - `tests/unit/test_realbody.py` (coverage fail-closed, bounded peak, NVFP4
    layout, GLM-style fixture)
  - `tests/unit/test_protocols.py` (MiniMoE adapter satisfies protocols)
  - `tests/unit/test_corpus_manifest.py` (immutable partitions, resume-skip)
- `ruff check .` → **All checks passed**
- `mypy src` → **Success, 106 source files, no issues**
- Real-GLM runs (measured, not estimated):
  - `model-atlas inspect /media/glm52/models/nvidia/GLM-5.2-NVFP4`
    → coverage 1.0, valid, 0 unclassified
  - `model-atlas validate-bodies <dir>` → bounded bodies, NVFP4 layout, peak 6 MiB
  - `model-atlas canary` → census+body OK, forward blocked (honest)
  - `model-atlas preflight` → exec backends all absent

## Measured vs estimated evidence

- **Measured**: bounded body reads/decode, identity copy, census/ownership/
  coverage, dtype histogram, NVFP4 constituent layout, peak resident bytes, real
  GLM mounted census facts, exec-backend presence, active GPU services.
- **Estimated/predicted**: v3 Pareto demo family (explicitly PREDICTED, never
  deployable); real-bytes keep_frac/backbone slice projections in
  `planning/realbytes.py` (already tagged `estimated`).
- **Blocked**: real forward/trace, channel scoring on real weights, derivative
  materialization, quantization backends, distributed runtime, measured Pareto.

## Active blockers + required user decision

1. **Exec stack absent** — authorize installing torch + transformers + (opt.)
   vllm/modelopt/exllamav2 in a **separate venv** (not touching the running
   services), or confirm resource headroom so full-model/native decode runs
   don't disrupt DeepSeek-v4 / llama-server.
2. **No second Spark node** — needed for Phase 6 two-node gates.
3. **No BF16 parent shards** beyond reference tensors — needed for streamed
   BF16 validation on destructive decisions (or authorize torch decode of the
   mounted NVFP4).

## Exact next commands

```bash
# (after user authorizes exec stack into a fresh venv, e.g.)
python -m venv .venv-exec && . .venv-exec/bin/activate
pip install "torch" "transformers" "safetensors" "numpy"
# then resume Phase 3:
cd /home/kristianaaron/tmp/model-atlas
git checkout atlas-glm52-experiment-runtime
.venv-exec/bin/python -c "from model_atlas.preflight import build_capability_report as r; print([m.present for m in r().modules if m.module in ('torch','transformers','safetensors','numpy')])"
pytest -m "not slow" -q          # keep fast suite green
```

## Known risks / rollback

- **Rollback**: all commits are local-only on the branch; `main` untouched and
  at `f1fd5d9`. Deleting the branch reverts everything; no pushes/PRs.
- **Source committed never modified**: mounted GLM dir opened read-only (mmap);
  hashes preserved.
- **GPU safety**: the only GPU-touching work was metadata/body reads (CPU mmap —
  no device memory). No services were stopped or disturbed.
- **Synthetic Pareto now PREDICTED** — no candidate can be marked deployable
  from the demo family; deployment requires materialized + held-out evaluated +
  runtime-measured candidates (still blocked on exec stack).
