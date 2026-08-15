# Atlas recommend → compress workflow

Deterministic, versioned, local-first profile-to-recommendation engine plus a
**server-served browser GUI**. The server is stdlib-only
(`http.server.ThreadingHTTPServer`); nothing here runs real quantization.
Current placeholder adapters are shown as unavailable and Compress **fails
closed** (disabled with the exact blockers) until a pinned derivative-producing
backend exists and a verified executable plan passes its live-pin gate.

```
Browser GUI ──HTTP──▶ RecommendationServer ──▶ RecommendationService
  fetch /api/profiles, /api/recommend, /api/preview-selection, /api/start
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

# 3. Serve the browser GUI + API (static HTML/CSS/JS; no embedded data)
model-atlas recommend-gui --host 127.0.0.1 --port 8080 \
    --profiles-dir profiles --work-root controlplane_runs
#    -> open http://127.0.0.1:8080/  (fetches /api/profiles, /api/recommend)
```

The GUI serves static HTML/CSS/JS only — no user/recipe JSON is embedded in the
page. The browser fetches `/api/profiles` and `/api/recommend` from the server,
lets you pick a profile and memory target, then recomputes. Method cards /
evidence / confidence / blockers are rendered with DOM `textContent` through an
XSS-safe `esc()` helper — never `innerHTML` from service data.

`recommend-gui` binds loopback only by default; pass
`--unsafe-allow-non-loopback` to bind `0.0.0.0` on a trusted network.

## Toggling + authorization

Only **policy-authorized, unblocked** methods are togglable. Blocked methods
(backend unavailable / probe-only / unpinned / missing evidence) render disabled
with the exact blocker codes. `no_pruning` is locked on (a real verified
pruning-capable backend would be required to unlock it, and none is registered).

## Recipe draft preview-from-selection

`POST /api/preview-selection {"selected": ["calibration", …]}` (or the GUI's
**Preview selection as recipe draft**) builds a deterministic no-pruning draft
from the selected recommendation methods + their transitive stage dependencies,
then reports:

* **diff** (enabled/omitted stage ids vs the canonical builtin recipe),
* **compile blockers** (typed issues),
* **readiness** (`verified_plan` / `pins_pass` / `executable`),
* **verified plan artifact** summary (`plan_id`, `reproduce_command`) **iff** the
  draft compiles. The full pinned artifact is never shipped to the browser (no
  embedded recipe payload).

## Compress gate (fail closed)

Compress is **enabled only when all** of these hold, else disabled with the
exact reason(s) shown:

1. the recommendation has **no fatal blockers** (no blocked methods),
2. the selected recipe **draft compiles**,
3. a **verified executable plan** exists **and its live pins pass**.

Confirmation `POST`s the actual `/api/start`. The GUI then polls
`/api/jobs/<run_id>` status + events, and renders job validation/lineage/outputs
and any failures. Currently, placeholder adapters (EXL3 / ModelOpt-NVFP4 /
LLM-Compressor / Eval-Lab) are unavailable / unpinned and the builtin recipe
fails the source-identity + hybrid gates, so no verified plan exists and the
Compress button stays **disabled** — the system never fakes quantization.

## Policy (deterministic, versioned `policy-v1`)

* Ranks methods/stages with reasons, evidence references, declared
  memory/quality direction, confidence, blockers, protected sensitive regions
  (attention / MLA / norms / embeddings / LM head / router), and an immutable
  **`no_pruning=true` default**.
* Stable `recommendation_id` / `profile_id` derived from canonical content.
* **Missing evidence reduces confidence and BLOCKS the decision** — never
  invents metrics.
* **Router-dependent compression methods require the routing-consistency
  identity gate to have PASSED.** When the gate FAILED or is UNKNOWN (never
  established), expert/router indices may be stale, so EXL3, LLM-Compressor,
  ModelOpt-NVFP4, and NVFP4-substitute carry a typed `routing_consistency_failed`
  blocker — evidence danger, not a confidence nuance. Analysis/planning methods
  (teacher identity, calibration, sensitivity, bit-allocation, KV) run in-repo
  and are not blocked on this gate.
* **Deterministic ordering responds to declared qualitative pressure, not
  invented fit metrics.** Recommended methods are ordered by (1) an evidence
  coverage band (high/adequate/low), (2) the method's own declared confidence
  (HIGH > MEDIUM > LOW), and — only under a tight memory target — (3) a memory
  direction tie-break (memory-reducing ahead), with (4) the policy's stable
  method-id rank as the final definite tie-break. Coverage and the memory target
  therefore really move the order, but the same inputs always give the same
  order.
* **Profiles keep a declared alias.** Each profile carries a human-chosen
  `declared_profile_id` (e.g. `glm52`) separate from its canonical content hash
  id. It is preserved through `save_profile`/`list_profiles`/`_resolve_profile`,
  so `--profile glm52` resolves even though the canonical content id differs.
* Compression methods whose backend is unavailable or probe-only (EXL3,
  LLM-Compressor, ModelOpt-NVFP4) are **blocked**; analysis/planning methods
  (teacher identity, calibration, sensitivity, bit-allocation, KV) remain
  recommendable on the in-repo adapters.

## API facade (`RecommendationService`)

list/import/save profiles (canonical content id + preserved declared alias) ·
recommend (resolves by canonical id, declared alias, or model) ·
preview/compile editable recipe ·
`recipe_preview(selected)` (draft builder) · start a **verified executable plan
only** (compiles, verifies pins against the live registry, then runs) · job
status/events/validate/lineage/output.

`/api/start` accepts either a full `{"recipe": …}` body OR a
`{"selected": […], "inputs": …}` body (the GUI's path — the server rebuilds the
draft, so the browser never holds a recipe payload).

## HTTP routes (server.py)

| Method | Route | Notes |
|---|---|---|
| GET | `/`, `/index.html`, `/gui` | static GUI page (no embedded data) |
| GET | `/api/profiles` | list profiles |
| POST | `/api/profiles/import` | server-side path under profile_root (403 traversal) |
| POST | `/api/recommend` | body `{profile_id, memory_target_gib, constraints}` |
| POST | `/api/preview-selection` | body `{"selected": […]}` → draft diff/blockers/readiness |
| POST | `/api/preview` | full recipe dry-run |
| POST | `/api/start` | `{"recipe":…}` or `{"selected":…,"inputs":…}` → verified plan only |
| GET | `/api/jobs/<id>` · `/api/jobs/<id>/events` | status / event stream |
| GET | `/validate?run_id&stage` | stage validation |
| GET | `/outputs?run_id[&stage_id][&name]` | content-addressed outputs |
| GET | `/lineage?recipe=…` | run lineage |

## Authorization

Recommendations and recipe compiles are deterministic and versioned; the
explanations are agent-readable but never authorize anything. Repair
application/approval is **not** exposed on this facade — it lives behind the
repair gate + engine transaction, so agents cannot silently mutate or approve
repairs.

## GUI

Server-served static HTML/CSS/JS: profile+target selection, recommendation
cards (method/evidence/confidence/blockers), enable/disable only authorized
methods, immutable no-pruning indicator, memory target recompute, recipe
diff/preview, Compress confirmation (disabled with exact blockers while any
gate fails), live job progress, failure evidence, output listing. All dynamic
values render via an XSS-safe DOM-text `esc()` helper — never raw `innerHTML`
interpolation of untrusted values, and no raw snapshot embedded in the page.
