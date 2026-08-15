# Atlas recommend → compress workflow

Deterministic, versioned, local-first profile-to-recommendation engine plus a
server-less single-file GUI. Nothing here runs real quantization; current
placeholder adapters are shown as unavailable and Compress **fails closed**
(disabled with the exact blockers) until a pinned derivative-producing backend
exists.

```
Profile ─▶ Recommend ─▶ Review Recipe ─▶ Compress ─▶ Monitor ─▶ Output
 (click)     (policy)     (edit recipe)    (start only a   (job events /
               versioned      + diff/preview   VERIFIED plan)  validation)
```

## Run

```bash
# 1. persist a profile (Evidence-weighted JSON) under ./profiles
python - <<'PY'
from model_atlas.recommend import RecommendationService, AtlasProfile, StageEvidence
svc = RecommendationService(profile_root="profiles", work_root="controlplane_runs")
p = AtlasProfile(profile_id="glm52", model="glm-5.2", evidence={
  "identity": StageEvidence("identity", "measured"),
  "corpus_semantic": StageEvidence("corpus_semantic", "measured", coverage=0.9),
  "spectral": StageEvidence("spectral", "estimated"),
  "shared_structure": StageEvidence("shared_structure", "estimated"),
  "routing_consistency": StageEvidence("routing_consistency", "measured"),
  "global_bit_budget": StageEvidence("global_bit_budget", "predicted"),
  "kv_budget": StageEvidence("kv_budget", "estimated"),
  "nvfp4_suitability": StageEvidence("nvfp4_suitability", "estimated"),
})
print(svc.save_profile(p))
PY

# 2. CLI recommendation (machine-readable)
model-atlas recommend --profile glm52 --profiles-dir profiles \
    --memory-target-gib 115 --out rec.json

# 3. GUI (a single HTML file; open in your browser)
model-atlas recommend-gui --out atlas_recommend.html \
    --profiles-dir profiles --work-root controlplane_runs
```

## Policy (deterministic, versioned `policy-v1`)

* Ranks methods/stages with reasons, evidence references, declared
  memory/quality direction, confidence, blockers, protected sensitive regions
  (attention / MLA / norms / embeddings / LM head / router), and an immutable
  **`no_pruning=true` default**.
* Stable `recommendation_id` / `profile_id` derived from canonical content.
* **Missing evidence reduces confidence and BLOCKS the decision** — never
  invents metrics.
* Compression methods whose backend is unavailable or probe-only (EXL3,
  LLM-Compressor, ModelOpt-NVFP4) are **blocked**; analysis/planning methods
  (teacher identity, calibration, sensitivity, bit-allocation, KV) remain
  recommendable on the in-repo adapters.

## API facade (`RecommendationService`)

list/import profiles · recommend · preview/compile editable recipe · start a
**verified executable plan only** (compiles, verifies pins against the live
registry, then runs) · job status/events/validate/lineage/output.

## Authorization

Recommendations and recipe compiles are deterministic and versioned; the
explanations are agent-readable but never authorize anything. Repair
application/approval is **not** exposed on this facade — it lives behind the
repair gate + engine transaction, so agents cannot silently mutate or approve
repairs.

## GUI

Server-less single HTML: profile+target selection, recommendation cards/table,
explain-why, enable/disable allowed methods, immutable no-pruning indicator,
blocked/experimental badges, memory target + protected-tail controls, recipe
diff/preview, Compress confirmation (button disabled with exact blockers while
any blocked method remains), live job progress, failure evidence, output
listing. All dynamic values render via an XSS-safe DOM-text `esc()` helper —
never raw `innerHTML` interpolation of untrusted values.
