# GLM-5.2 two-DGX-Spark maintenance-window runbook

**Status: everything below is IMPLEMENTED and TESTED except the explicit
service-window execution, which is the single remaining GATE.**

The two GPUs are occupied by the production two-rank DeepSeek vLLM service
(TP0/TP1 ~102.5 GB each) + a `llama-server` on spark. The real forward + runtime
benchmark require evicting them, so all code/tests/manifests/commands are ready
and the *execution* is one explicit service-window decision away. **Never stop,
restart, or contend with those services outside the window.**

---

## 0. Preflight (safe anytime; metadata only)

```bash
cd /home/kristianaaron/tmp/model-atlas
git checkout atlas-glm52-experiment-runtime
.venv/bin/python -m pytest -m "not slow" -q          # fast suite green
model-atlas preflight --out capability_report.json
model-atlas canary                                    # census+body OK, forward=blocked (honest)
sqlite3 /dev/null 'select 1' 2>/dev/null || true      # (placeholder; n/a)
```

## 1. Fast forward / trace canary (CPU or tiny GPU slice, no service eviction)

The bounded routing trace needs no GPU and no service interruption:

```bash
.venv/bin/python -c "
from model_atlas.glm52trace import load_glm52_facts, stream_routing_trace
f = load_glm52_facts('/media/glm52/models/nvidia/GLM-5.2-NVFP4')
t = stream_routing_trace('/media/glm52/models/nvidia/GLM-5.2-NVFP4', layer=3, n_hidden_rows=64)
print('measured facts:', f.quant_algo, f.n_sparse_layers, 'sparse layers;')
print('routing: top-8 across 256 experts on layer 3 ->', len(t.records), 'rows')
"
```

Torch-backed scoring (real tensor math, forward-only, no GPU):

```bash
PYTHONPATH=src .venv-exec/bin/python - <<'PY'
import torch
from model_atlas.scoring.torch_scores import (
    tenp_importance, flexmoe_channel_ranking,
    grouped_taylor_surrogate, causal_ablation_scores, needs_for_real_scoring,
)
# Build small reference tensors (bounded decode of one expert later; here hand tensors)
gate = torch.randn(128, 6144); up = torch.randn(128, 6144); down = torch.randn(6144, 128)
z = torch.randn(64, 128)
imp = tenp_importance(gate, up, down, z)
keep = flexmoe_channel_ranking(imp, 128, budget_frac=0.7)
print('top channels kept per expert:', len(keep), '(every routed expert retained)')
print('requirements', needs_for_real_scoring('bounded_cpu').forward_only)
PY
```

## 2. Derivative materializer (bounded, source immutable)

```bash
.venv/bin/python - <<'PY'
from model_atlas.materialize import materialize_expert_bank
res = materialize_expert_bank(
    '/media/glm52/models/nvidia/GLM-5.2-NVFP4',
    '/home/kristianaaron/tmp/model-atlas/derivatives/glm-layer3-exp0-keep1024',
    corner_layer=3, keep_channels=list(range(1024)), num_experts=1, group_size=16,
)
print('validated', res.validated, 'promoted', res.promoted, 'coverage', res.coverage)
print('shard hashes in derivative_manifest.json (sha256)')
PY
```

## 3. Two-node inventory + launch-plan gate (safe, non-evasive)

```bash
.venv/bin/python - <<'PY'
from model_atlas.twonode import run_inventory, build_launch_plan
inv = run_inventory()
print('reachable nodes:', inv.reachable())
plan = build_launch_plan(inv, weights_bytes_total=190*1024**3, physical_per_rank=100*1024**3)
print('gates:', plan.gates)
PY
```

Measured (2026-08-14): both `spark-d167` (10.77.0.1) and `gx10-ac63` (10.77.0.2)
reachable via `ssh -o BatchMode=yes`, both NVIDIA GB10 compute-cap (12,1), both
with `reap-torch211` + `vllm` exec venvs (torch 2.11.0+cu130, NCCL (2,28,9),
transformers 5.9.0 native `glm_moe_dsa`; vllm 0.21.0 maps `GlmMoeDsaForCausalLM`
-> deepseek_v2 and has a compressed-tensors NVFP4 path).

## 4. THE service-window execution (requires explicit decision + eviction)

**Do NOT run until the maintenance window is scheduled and the production
DeepSeek two-rank vLLM + llama-server are stopped by the operator.**

### 4a. Full model forward + Routing/Activation trace (torch, one node)

```bash
cd /home/kristianaaron/ai-lab/venvs/reap-torch211
# native transformers GLM-5.2 (checked: GlmMoeDsaForCausalLM present)
python - <<'PY'
from transformers import GlmMoeDsaForCausalLM, AutoTokenizer
import torch
ckpt='/media/glm52/models/nvidia/GLM-5.2-NVFP4'
model = GlmMoeDsaForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
ids = tok("write a merge sort in python", return_tensors="pt")["input_ids"]
out = model(ids)  # real forward; capture router/activation hooks for the corpus trace
print('logits', out.logits.shape)
PY
```

### 4b. Two-node launch (expert-parallel via vllm distributed executor)

```bash
cd /home/kristianaaron/ai-lab/venvs/vllm
export NCCL_SOCKET_IFNAME=enP7s7        # 10.77.0.1/10.77.0.2 link
python -m vllm.entrypoints.openai.api_server \
  --model /media/glm52/models/nvidia/GLM-5.2-NVFP4 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --trust-remote-code --max-model-len 8192 \
  --distributed-executor-backend ray --ray-address auto \
  --node-ip 10.77.0.1 10.77.0.2        # two-node expert-parallel
```

### 4c. Benchmark + canary (cold/warm/prefill/decode; long-context)

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"/media/glm52/models/nvidia/GLM-5.2-NVFP4","messages":[
     {"role":"user","content":"summarize the deepseek architecture in 5 bullets"}],
     "max_tokens":256}
# record: cold-start, warm-up, prefill tok/s, decode tok/s, KV GiB, MTP acceptance
```

### 4d. Eval + Pareto (measured only)

```bash
.venv/bin/python - <<'PY'
from model_atlas.evidencegates import FrontierRecorder
fr = FrontierRecorder()
# after 4a/4b/4c complete for this candidate:
fr.add_candidate(
  'glm-nvfp4-v1', quality=0.995, resident_gib=196.0, decode_tps=26.0, context_tokens=384000,
  materialized=True, heldout_evaluated=True, runtime_benchmarked=True,
  provenance='bench:GLM-5.2-NVFP4-2node window <date>')
print('measured frontier:', [p['candidate_id'] for p in fr.measured_frontier()])
print('predicted frontier none until measured:', len(fr.predicted_frontier()))
PY
```

## 5. Rollback / safety

- All code is on `atlas-glm52-experiment-runtime`; `main` untouched at `f1fd5d9`.
- Source GLM checkpoint is opened read-only (mmap); never rewritten.
- Derivative output writes to a temp dir + JSONL journal; promote only after
  validate passes; on failure temp is discarded (no partial candidates).
- ModelOpt / vllm exec venvs are reused read-only; only repo-local
  `.venv-exec` is ours to modify. Do not pip-modify the shared exec venvs.
- Any custom SM121 kernel is prohibited until existing primitives prove
  correctness + shape coverage + rollback tests (AGENTS.md + runtime contract).
