# GLM-5.2 two-DGX-Spark maintenance-window runbook (review-corrected)

> **Round-8 note.** The W64 structural canary is now materialized
> (`derivatives/glm-uniform-w64-canary/`, 47 shards / 232,385 tensors /
> 69,849,116,672 bytes) using the fixed exporter (one source handle per output
> shard; bounded coalesced spans; commit `cabf8f1`). Its arbitrary first-64
> channels are a **structural canary only** — measured corpus keep-maps and a
> maintenance-window load/forward remain, so the execution gate below stays
> CLOSED.

**Status: code/tests/manifests implemented. The real runbook BEGIN only when a
fully materialized, structurally-complete (and runtime-loadable for a ModelOpt-capable backend) derivative at or under the per-node envelope
exists. Until such a derivative passes materialized + held-out + runtime gates,
the execution gate stays CLOSED (no direct full-model run of the 503 GB source
is possible on 2×~120 GB, and a single-node Transformers `from_pretrained` of it
is infeasible — both are removed here as unsafe).**

---

## 0. Preflight (safe anytime; metadata only)

```bash
cd /home/kristianaaron/tmp/model-atlas
git checkout atlas-glm52-experiment-runtime
.venv/bin/python -m pytest -m "not slow" -q      # fast suite green
model-atlas preflight --out capability_report.json
model-atlas canary                                # census+body OK, forward gated (honest)
```

## 1. Bounded canaries (no service eviction; no full-model load)

```bash
# REAL_ROUTER_SYNTHETIC_INPUT_PROBE — real router, synthetic hidden input, PREDICTED
.venv/bin/python -c "
from model_atlas.glm52trace import stream_routing_trace
t = stream_routing_trace('/media/glm52/models/nvidia/GLM-5.2-NVFP4', layer=3, n_hidden_rows=64)
print(t.input_label, t.evidence_kind.value)   # probe, NOT measured
"

# torch scoring path (verified under .venv-exec; full-forward flagged)
PYTHONPATH=src .venv-exec/bin/python - <<'PY'
from model_atlas.scoring.torch_scores import needs_for_real_scoring
print('bounded forward_only', needs_for_real_scoring('bounded_cpu').forward_only)
print('full_forward flags', not needs_for_real_scoring('full_forward').forward_only)
PY

# two-node inventory (host unified memory + production occupancy, non-evasive)
.venv/bin/python - <<'PY'
from model_atlas.twonode import run_inventory
inv = run_inventory()
for h, n in inv.nodes.items():
    print(h, 'host_GiB', round(n.host_mem_total_gib,1), 'avail', round(n.host_mem_available_gib,1),
          'production_occupied_GiB', round(n.production_occupied_gib,1))
PY
```

## 2. Derivative materialization → the FULL uniform-width derivative exporter

Use `loader.materialize_uniform_width` (the full structural exporter). It exports
a STRUCTURALLY-COMPLETE checkpoint tree (all sparse layers × all experts, uniform
width, safetensors + index + config rebuilt, quant/tokenizer/code preserved) and
sets `runtime_compatibility='schema-supported-unvalidated'` and
`runtime_validated=False`. The installed vllm 0.21 DOES contain a ModelOpt-NVFP4
path (ModelOptNvFp4Config + Linear/FusedMoE + kernels/emulation); what is absent
is Ray, external `modelopt`, producer-version parity, and validated
materialized-derivative load/forward. It is fed to a ModelOpt-capable loader once such a
backend probe passes; it is never fed to the installed vllm as runtime-ready.

```bash
.venv/bin/python - <<'PY'
from model_atlas.checkpoint.source_manifest import load_manifest
from model_atlas.loader import materialize_uniform_width, plan_exact_sizes

SRC = '/media/glm52/models/nvidia/GLM-5.2-NVFP4'
manifest = load_manifest(SRC)
import json
source_cfg = json.loads(open(SRC + '/config.json').read())
# pick a width that fits both nodes after an authorized maintenance window
# (e.g. W=256 -> per_rank ~60 GiB); compute an exact size plan FIRST (scalars
# do NOT scale, so this is exact, not a width/full factor):
out = '/home/kristianaaron/tmp/model-atlas/derivatives/glm-uniform-w256'
res = materialize_uniform_width(SRC, out, width=256)   # resumable, transactional
print('structurally_complete', res.structurally_complete, 'promoted', res.promoted)
print('runtime_compatibility', res.runtime_compatibility)
print('runtime_validated', res.runtime_validated)   # False until end-to-end validated
PY
```

## 3. The ONLY valid runbook start: a fully materialized structural-complete derivative + a ModelOpt-capable loader

The exporter always sets `runtime_loadable=False` because the installed stack
has a ModelOpt-NVFP4 decoder path but a real derivative load is unvalidated. The
execution gate stays CLOSED until BOTH hold:
1. a **structurally-complete** derivative checkpoint is materialized at `<=
   target envelope` (index+config rebuilt, exact-validation passed);
2. an authoritative backend probe proves a ModelOpt-capable loader can decode/run
   it — until then no vllm launch / full forward is issued. No route skips this
   gate.

### 3a. When the gate is open — two-node vllm serving of the LOADED derivative

Exactly one operator maintenance window (services stopped by the operator; never
by a script).

```bash
# Step 0 — external Ray cluster (validated vllm 0.21):
cd /home/kristianaaron/ai-lab/venvs/vllm
ray start --head --num-gpus 1 --node-ip-address 10.77.0.1 &   # on spark
ssh gx10-ac63 'cd /home/kristianaaron/ai-lab/venvs/vllm && ray start --address 10.77.0.1:6379 --num-gpus 1'
# Step 1 — head node server, expert-parallel, on the LOADED derivative:
python -m vllm.entrypoints.openai.api_server \
  --model /home/kristianaaron/tmp/model-atlas/derivatives/<loadable-derivative> \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --nnodes 2 --node-rank 0 \
  --trust-remote-code --max-model-len 8192
```

### 3b. Benchmark + eval + Pareto (measured only)

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"<loadable-derivative>","messages":[{"role":"user","content":"summarize in 5 bullets"}],"max_tokens":256}'
# record: cold-start, warm-up, prefill tok/s, decode tok/s, KV GiB, MTP acceptance
.venv/bin/python - <<'PY'
from model_atlas.evidencegates import FrontierRecorder
fr = FrontierRecorder()
fr.add_candidate('<deriv-id>', quality=..., resident_gib=..., decode_tps=..., context_tokens=...,
  materialized=True, heldout_evaluated=True, runtime_benchmarked=True,
  provenance='bench:<loadable-derivative> window <date>')
print([p['candidate_id'] for p in fr.measured_frontier()])
PY
```

## 4. Rollback / safety

- All code on `atlas-glm52-experiment-runtime`; `main` untouched at `f1fd5d9`.
- Source GLM mount opened read-only (mmap); never rewritten.
- Derivative output writes to temp + JSONL journal; promote only after
  `validate`; requires `overwrite=True` to replace; no implicit rmtree.
- ModelOpt / vllm exec venvs reused read-only; only repo-local `.venv-exec` ours.
- The measured per-rank ledger separates physical capacity, production
  occupancy, and current availability — go/no-go is never an unexplained constant.
- Any custom SM121 kernel is prohibited until existing primitives prove
  correctness + shape coverage + rollback (AGENTS.md + runtime contract).
- `scripts/production_rollback.py` prints the operator's freeze/SIGTERM
  commands; it never stops/restarts services itself.
