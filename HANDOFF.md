# ATLAS — HANDOFF / RESUME FILE

Last updated: session 20260817 (before laptop handover). Live repo is the source
of truth; run `git log --oneline -6` and `git status -sb` first — don't trust
HEAD listed here over the repo's actual state.

## 1. Where we are (committed + green)

Branch `atlas-glm52-experiment-runtime`, repo `/home/kristianaaron/tmp/model-atlas`
on spark-d167 (100.96.194.44). Working tree clean (untracked artifacts/ + profiles/
are pre-existing dry-run stubs, not ours).

Recent commits (above foundation `c91d85e`):
- e47de4b  prune math slice: channel_saliency / ranked_keeper / kl_gate
- cefbee6  deterministic safetensors derivative bundle (`.atlasbundle`, SHA-verified)
- 02e0cd7  Stage 1: header-level width census (`width_sizing`) + `plan_uniform_width`
- 808fb2f  width-slice MethodSpec registered + api recipe dispatch (method executable)
- 7456a54  handoff generalized to width-slice `.atlasbundle` (GGUF backward-compatible)
- 6af496e  explicit teacher-relative KLD/CKA quality gate (`quality_gate`)
- d727724  MethodSpec plugin seam (`ATLAS_METHOD_PLUGIN_DIR` + `register_methods`)
- 44f893f  docs §8 drop-in method authoring
- f6e0312  ModelOpt NVFP4 plugin (real probe + adapter; gate-validation deferred)
- 1700bd2  maintenance lifecycle event stream (drain -> produce -> restore)
- 143b58b  maintenance-watch live renderer
- 327a209  maintenance-watch per-shard progress bar (+ `extract_shard_progress`)

Broad fast suite is green. The width-slice **tool is finished**: registered,
advertised, executable end-to-end (recommend -> recipe dispatch -> validated
backend -> `.atlasbundle` -> verified handoff -> quality gate).

## 2. The goal

Produce an **eval-ready output** for GLM-5.2 (the canary). Live completion of the
pipeline (produce derivative -> KLD/CKA gate -> Eval Lab) requires the **maintenance
window** (drains DSV4/VLLM for ~1.5-2 h) + **explicit user approval** — see §6.

## 3. Cluster access (from this Mac OR Termius-from-phone, same SSH)

- Host: `kristianaaron@100.96.194.44` (spark-d167, Tailscale). gx10-ac63/gx10-tail offline.
- SSH key path has SPACES — always quote: `KEY="...nvsync.key"` then `-i "$KEY"`.
  opts: `-o BatchMode=yes -o ConnectTimeout=10 -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=accept-new`
- Repo: `/home/kristianaaron/tmp/model-atlas`; venv `.venv` (Python 3.12).
- Fast tests: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/<f> -q -p no:cacheprovider`
- Ruff `.venv/bin/ruff check <paths>`; mypy `.venv/bin/mypy <src>`.

## 4. The optimized resume loop (run one pass each time you pick this up)

1. `git log --oneline -6` + `git status -sb` (clean?).
2. Verify green: the fast suites covering channel_pruning_math, channel_pruning_integration,
   bundle, planner, width_sizing, runtime_artifact_handoff_widthslice, quality_gate,
   method_plugin_seam, modelopt_adapter, maintenance_events, maintenance_watch,
   recommend, method_catalog, nvfp4_width_slice_adapter, maintenance_runner,
   run_glm52_canary_maintenance, run_glm52_capture_maintenance, glm52_candidate_eval.
   If anything fails: STOP, fix root cause, retest.
3. Advance the **next safe step** on the eval-readiness checklist (§5).
4. On a REPEATED blocker: diagnose root cause (read the code), fix, retest, add a
   guard/test, then move forward. Never blindly retry; never work around safety.
5. Commit only complete green increments; keep the tree clean.
6. End with a brief: HEAD, green/broken, advanced, next step, awaiting-approval.

## 5. Eval-readiness checklist (SAFE without approval)

- [ ] Full suite green (final gate; slow >8 min — run once in background).
- [ ] Drain/restore REHEARSAL via coordinator **dry-run only** (`execute=False`) —
      proves service-restore order, touches no production.
- [ ] Stage the width-slice derivative (CPU+NVMe, no GPU needed) IF the GLM source
      is reachable and disk allows: produce to a staging dir, verify `.atlasbundle`
      + handoff via `load_verified_width_slice_handoff`.
- [ ] Eval plan/candidate-only report (already built in `glm52_candidate_eval`).

## 6. HARD SAFETY GATES (never bypass)

- **Never run the live maintenance drain** (stop DSV4/VLLM), ModelOpt quant,
  KLD/CKA capture on the box, or Eval Lab WITHOUT the user's explicit approval
  phrase: `Approved: run the full GLM maintenance sequence`.
- Never start/stop/restart production services (DSV4, VLLM, llama, gateway,
  vision, qwen) autonomously.
- GPU is occupied (~106/128 GB by vLLM+llama); never assume free VRAM.
- Do NOT install nvidia-modelopt into the shared TEST venv (dependency drift);
  install into a dedicated runtime env only. GLM won't fit anyway until drain.

## 7. When you reach the approval gate

Output a crisp brief: everything green + rehearsed + derivative staged, the EXACT
approval sentence above, estimated ~1.5-2 h wall-clock, and that DSV4 auto-restores
afterward. Then STOP and wait for the user — do not proceed into the drain.

## 8. Useful commands (phone/Termius)

```bash
# watch the live drain/produce/restore UI (once = replay; omit = live tail):
cd /home/kristianaaron/tmp/model-atlas
PYTHONPATH=src .venv/bin/python scripts/maintenance_watch.py --journal-dir <journal_dir> --once

# inspect the event stream directly:
tail -f <journal_dir>/maintenance-events.jsonl

# see commits / status / staged artifacts:
git log --oneline -6; git status -sb; ls -la artifacts/
```
