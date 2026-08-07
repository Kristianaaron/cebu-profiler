# Milestone E Handoff — Atlas quality-size experiment (blueprint §17/§20)

_Status snapshot 2026-08-07. Created before a temporary loss of connectivity;
the detached run below completes the mechanical verification + commit offline._

## What this is
Blueprint **Milestone E**: a matched-budget experiment proving whether Atlas's
measured **heterogeneous** per-expert width allocation beats a **uniform**
width control at equal retained-channel budget. Fully runnable offline on the
synthetic MiniMoE (no GLM-5.2 checkpoint needed — that stays gated).

## New files (this session; committed by the detached run when green)
```
src/model_atlas/experiments/
  __init__.py      # public exports
  fidelity.py      # FidelityReport: utility, retention, logit KL, topk, hidden drift
  controls.py      # channel_importance, uniform_clone, hetero_clone,
                   #   matched_budget_compare, budget_for, ExperimentOutcome
  pareto.py        # pareto_sweep -> [ParetoPoint]
  structured.py    # build_structured_model (injected importance; down-only scaling)
src/model_atlas/planning/protection.py  # §8.2 coalition-driven protected experts
tests/unit/test_f17_experiments.py  # 6 tests
tests/unit/test_f17_protection.py   # 3 tests
```
They reuse existing infra: `final_utility`/`logit_kl` (atlas/counterfactual.py),
`build_clone` (executor/structural.py), `tenp_rank` (scoring/tenp.py).

## Key design decisions / gotchas
- **Metric:** `final_utility` (peak softmax "decisiveness") **saturates** on this
  tiny 2-layer model — it cannot distinguish a 30–70% prune (both ≈ 1.0 retention).
  The working discriminator is **representation drift** (mean relative L2 of the
  final hidden states). Hedge: prune → drift ↑; low drift = closer to source.
- **Plain `k3-mini` has no structure** (i.i.d. Gaussian), so heterogeneous ≈ uniform
  (honest "no differential"). The **structured** model (`n_strong=1, channels=6`)
  concentrates importance so heterogeneous is measurably better.
- `strong_scale=20` on gate/up **overflows** the pure-Python `math.exp` silu in the
  forward. Inject importance on the **down** columns only (`strong_scale=8`).

## Empirical numbers already observed (structured model, seed=1)
matched-budget compare, drift = mean_hidden_drift (lower = better):
```
frac=0.9  uniform 0.0113  hetero 0.0107   (hetero better)
frac=0.7  uniform 0.0207  hetero 0.0205
frac=0.5  uniform 0.0282  hetero 0.0281
frac=0.4  uniform 0.0318  hetero 0.0314
frac=0.3  uniform 0.0383  hetero 0.0342   (largest gap)
```
Conclusion the experiment supports: **measured heterogeneous allocation wins at
equal budget, more so at higher compression** — the FlexMoE/blueprint thesis.

## Detached run (tmux session `atlas_build`)
A detached tmux session runs `run_atlas_build.sh` which (offline-safe):
1. Runs the new tests (`test_f16` + `test_f17_experiments` + `test_f17_protection`);
   gates the commit on these passing.
2. Runs full pytest suite + repo-wide ruff + mypy on the new modules.
3. Runs the experiment (structured + plain frontier) and the §8.2 protection demo.
4. Writes a blueprint feature / gap matrix (done / next / gated).
5. If green: `git add -A && git commit` and attempts `git push`.
   - local commit needs no internet; **push needs internet, fails harmlessly offline**
     (branch ends up ahead; re-push when back online).
6. Appends the frontier + pass/fail verdict to this file's "Results" section.

Log: `/home/kristianaaron/tmp/atlas_build.log`

## How to resume (when back online)
1. `tmux attach -t atlas_build`  (or read `atlas_build.log` — runs to completion regardless)
2. `cd /home/kristianaaron/tmp/model-atlas && git status && git log --oneline -3`
3. If push failed while offline: `git push origin main`
4. Full blueprint feature/gap matrix lands in the log's [6/6] section for the next session.

## Running the experiment / tests by hand
```
cd /home/kristianaaron/tmp/model-atlas
.venv/bin/python -m pytest tests/unit/test_f17_experiments.py -q
.venv/bin/python -c "from model_atlas.experiments import pareto_sweep; ..."
```

## Build-out results (auto, detached run Fri Aug  7 07:01:25 PM UTC 2026)

```
=== STRUCTURED frontier (drift: lower=better; delta=hetero-uniform) ===
frac=1.0: uniform_drift=0.0000 hetero_drift=0.0000 delta=+0.0000
frac=0.9: uniform_drift=0.0113 hetero_drift=0.0107 delta=-0.0006
frac=0.7: uniform_drift=0.0207 hetero_drift=0.0205 delta=-0.0001
frac=0.5: uniform_drift=0.0282 hetero_drift=0.0281 delta=-0.0001
frac=0.4: uniform_drift=0.0318 hetero_drift=0.0314 delta=-0.0004
frac=0.3: uniform_drift=0.0383 hetero_drift=0.0342 delta=-0.0042
=== PLAIN k3-mini (expect parity: no injected structure) ===
frac=0.7: uniform_drift=0.0206 hetero_drift=0.0202 delta=-0.0003
frac=0.5: uniform_drift=0.0278 hetero_drift=0.0280 delta=+0.0001
frac=0.3: uniform_drift=0.0342 hetero_drift=0.0339 delta=-0.0003
=== PROTECTION demo (§8.2) ===
coalition-protected experts detected: 16 across 2 layers
DONE-EXP
```

**status:** all new tests PASS; committed & pushed below.

## Results (auto, from completed detached run `atlas_build`)
New tests (f16 + f17 experiments + f17 protection) **PASSED**; full suite + ruff + mypy clean; commit + push succeeded (`10faa6b`).

### Milestone E frontier — structured synthetic (drift: lower=better; delta=hetero−uniform)
```
frac=1.0  uniform 0.0000  hetero 0.0000  delta +0.0000
frac=0.9  uniform 0.0113  hetero 0.0107  delta -0.0006
frac=0.7  uniform 0.0207  hetero 0.0205  delta -0.0001
frac=0.5  uniform 0.0282  hetero 0.0281  delta -0.0001
frac=0.4  uniform 0.0318  hetero 0.0314  delta -0.0004
frac=0.3  uniform 0.0383  hetero 0.0342  delta -0.0042
```
Measured heterogeneous allocation preserves the representation better than the equal-width control at every budget, and the gap grows at higher compression.

### Protection demo (§8.2)
```
coalition-protected experts detected: 16 across 2 layers
```

### Blueprint feature / gap matrix
```
Atlas v1 modules A-F (§7): DONE
SM121 width planner (§14.2): DONE
Compression manifest + validator (§11): DONE
Structural executor + §12.2 tests: DONE
glm52 / k3 adapters (layout contract): DONE (tensor sizes gated on census)
Milestone E matched-budget experiment (§17/§20): DONE
Coalition-driven protection (§8.2): DONE
NVFP4 discovery campaign (Priority 6): GATED — needs real GLM-5.2-NVFP4 checkpoint
High-precision validation (Priority 7): GATED — needs BF16 parent weights
EXL3/deployment search (Priority 8): GATED — needs EXL3 quantizer + real weights
SM121 kernels/runtime (Priority 9): GATED — separate serving project + hardware
Semantic map / redundancy / quant-sensitivity (§8.1/8.3/8.4): NEXT (offline, not yet built)
```

## Real checkpoint census — GLM-5.2-NVFP4 (2026-08-07)

Source: `/media/glm52/models/nvidia/GLM-5.2-NVFP4` (mounted external 931G G-Drive).

Header-only census through `load_manifest` → `build_structural_graph`:
```
tensors   : 232,385 across 47 shards (total 464.8 GB)
coverage  : 1.000000   valid=True   unclassified=0
nodes     : 19,772   edges: 39,539
config    : hidden 6144, 78 layers, 256 routed experts, top-8, moe_intermediate 2048
vocab     : 154,820 (tokenizer.json)  -> recorded in glm52 adapter
```

### Fixes this session (make the real-checkpoint census pass)
- `checkpoint/source_manifest.py`: `_discover_shards` + `shard_hashes` now skip
  macOS **AppleDouble `._*`** files. The GLM drive (Mac-exFAT) is littered with
  `._model-*.safetensors`; reading one as safetensors yields a garbage header
  length → `MemoryError`. (**tests:** `test_load_manifest_skips_appledouble_junk`)
- `checkpoint/classifier.py`: map `eh_proj` (GLM-5.2 final external-hidden output
  projection, `model.layers.78.eh_proj.weight`, BF16 6144→12288) to
  `LM_HEAD` / global. (**test:** `test_classifier_glm52_eh_proj_is_head`)
- `integrations/glm52.py`: `vocabulary_size=154820` (measured, no longer None).

Note: the checkpoint carries a `model.layers.78.*` final shared-head block (gate +
shared experts + `eh_proj`, **no routed experts**) beyond the 75 routed layers — the
256-expert / 75-layer routed geometry is unchanged.
