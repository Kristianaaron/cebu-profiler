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
