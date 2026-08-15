# Atlas recommend → compress workflow

Deterministic, versioned, local-first profile-to-recommendation engine plus a
**server-served browser GUI**. The server is stdlib-only
(`http.server.ThreadingHTTPServer`); nothing here runs real quantization.
Current placeholder adapters are shown as unavailable and Compress **fails
closed** (disabled with the exact blockers) until a pinned derivative-producing
backend exists and a verified executable plan passes its live-pin gate.

```
Browser GUI ──HTTP──▶ RecommendationServer ──▶ RecommendationService
  fetch /api/profiles, /api/recommend (mints token),
  /api/preview-selection (token-bound), /api/start (token+preview-bound)
Profile ─▶ Recommend ─▶ Review Recipe ─▶ Compress ─▶ Monitor ─▶ Output
 (click)    (policy +     (token-bound      (background    (job events /
             opaque token)  preview/artifact)  worker)        validation)
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

# 2. CLI recommendation: emits the OPAQUE AUTHORIZATION TOKEN bound to the
#    recommendation (canonical method set + selection hash) — machine-readable.
model-atlas recommend --profile glm52 --profiles-dir profiles \
    --memory-target-gib 115 --out rec.json
#    rec.json contains: token, recommendation_id, profile_id, no_pruning,
#    authorized_methods, selection_hash, recommendation

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

`/api/recommend` returns an **opaque authorization token** owned by the server.
The token is bound to the canonical recommendation id, the resolved profile id,
the target, the constraints snapshot and the **exact authorized method set** (a
selection hash). A recommendation with no token is inert — it authorizes
nothing. Explanations never authorize anything.

## Recipe draft preview-from-selection (token-gated)

`POST /api/preview-selection {"token": …, "selected": […], "inputs": {}}`

Only a live token may preview. The selection MUST be a **non-empty**, exact
match of the token's authorized method set (its method-set hash binds it). The
server rejects:

* **empty** selections (`selection_empty`, 400),
* **unknown tokens** (`token_unknown`, 401),
* **not-authorized / mismatched** selections (`selection_not_authorized`, 403),
* selections whose draft has **no verified executable plan** (readiness stays
  `verified_plan:false`).

On success the deterministic compiled artifact is **verified and stored
server-side**, keyed by the preview. The response is a bounded handle — never the
full recipe/artifact payload:

```json
{ "preview_id": "pv-…", "plan_id": "recipe-…", "hash": "<selection sha256>",
  "readiness": {"verified_plan": true, "pins_pass": true, "executable": true},
  "selected_methods": ["calibration", …] }
```

When the selected draft cannot yet produce a verified executable plan (today's
placeholder adapters), the preview is still stored with
`readiness.executable:false` and start is refused (`preview_not_executable`).

## Compress gate (fail closed)

Compress is **enabled only when all** of these hold, else disabled with the
exact reason(s) shown:

1. a valid authorization **token** exists (recommended),
2. the recommendation has **no fatal blockers** (no blocked methods),
3. a **verified executable plan** exists for the current selection,
4. the selection hash still **matches** the last ready preview (no
   profile/target/constraint/checkbox change since).

Any change to profile, target, memory, or any checkbox **invalidates the
preview** and disables Compress until a fresh preview matches the new selection.

Confirmation `POST`s `/api/start` with the bounded handle. The server verifies
the token is live, the preview is pending, and the supplied `hash` + exact
`selected` match what the preview was compiled for (rejecting stale/mismatch/
replay/unknown/empty), persists the job identity **before** dispatch, returns
`run_id` immediately, and executes the plan in a **managed background worker**
(never synchronous request blocking). The GUI then polls `/api/jobs/<run_id>`
status + `/events`, stops at the terminal state, and fetches/renders
validate/lineage/outputs.

Currently, placeholder adapters (EXL3 / ModelOpt-NVFP4 / LLM-Compressor /
Eval-Lab) are unavailable / unpinned and the builtin recipe fails the
source-identity + hybrid gates, so no verified plan exists and the Compress
button stays **disabled** — the system never fakes quantization.

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
`authorize(profile, target)` → recommendation + opaque token (bound to the
exact authorized method set) ·
`preview_selection(token, selected)` → bounded preview_id/plan_id/hash +
server-side verified compiled artifact ·
`start_authorized(token, preview_id, hash, same_selection, inputs)` → immediate
`run_id` + background worker (replay/mismatch/stale rejected) ·
preview/compile editable recipe ·
`recipe_preview(selected)` (draft builder) · job
status/events/validate/lineage/output.

## HTTP routes (server.py)

| Method | Route | Notes |
|---|---|---|
| GET | `/`, `/index.html`, `/gui` | static GUI page (no embedded data) |
| GET | `/api/profiles` | list profiles |
| POST | `/api/profiles/import` | server-side path under profile_root (403 traversal) |
| POST | `/api/recommend` | body `{profile_id, memory_target_gib, constraints}` → **recommendation + token + authorized_methods + selection_hash** |
| POST | `/api/preview-selection` | body `{token, selected, inputs}` → preview_id/plan_id/hash; accepts any NONEMPTY subset of the authorized methods, rejects empty/unknown/not-authorized |
| POST | `/api/preview` | full recipe dry-run (no token; read-only) |
| POST | `/api/start` | body `{token, preview_id, hash, selected, inputs}` → run_id immediately, background worker; NO arbitrary-recipe or raw-selection start |
| GET | `/api/jobs/<id>` · `/api/jobs/<id>/events` | status / event stream |
| GET | `/validate?run_id&stage` | stage validation |
| GET | `/outputs?run_id[&stage_id][&name]` | content-addressed outputs |
| GET | `/lineage?run_id=…` | run lineage (actual completed run; no recipe={}) |

## Authorization (token model)

Recommendations, previews and starts are each gated:

1. `/api/recommend` mints an **opaque, server-side token** bound to the canonical
   recommendation/profile/target/constraints and the **exact authorized method
   set** (its deterministic selection hash). A bare recommendation mints no
   token and authorizes nothing.
2. `/api/preview-selection` **requires** a valid token and any NONEMPTY subset
   of the authorized methods. It stores a **verified compiled artifact
   server-side**, keyed by the token + the subset's own deterministic hash, and
   returns only a bounded `preview_id` / `plan_id` / `hash` handle — no recipe
   payload. The subset hash (not the full-set hash) is what start re-verifies.
3. `/api/start` **accepts only** `(token, preview_id, hash, exact same
   selection, inputs)`, all re-verified against the stored package.
   Stale/unknown/replay/mismatch/empty starts are rejected deterministically.
   The job identity is persisted **before** dispatch; dispatch runs in a
   managed background worker; the HTTP request returns `run_id` immediately
   (never synchronous request blocking); duplicate starts are rejected as
   replay or return deterministically.

Repair application/approval is **not** exposed on this facade — it lives behind
the repair gate + engine transaction, so agents cannot silently mutate or
approve repairs.

## GUI

Server-served static HTML/CSS/JS: profile+target selection, recommendation
cards (method/evidence/confidence/blockers), enable/disable only authorized
methods, immutable no-pruning indicator, memory target recompute, recipe
diff/preview, Compress confirmation (disabled with exact blockers while any gate
fails), live job progress, failure evidence, output listing. The selection is
initialized from the methods actually checked (authorized, unblocked). A
profile/target/constraint change clears token/reco/selection/preview (so Compress
requires a fresh recommendation); a checkbox change invalidates the preview until
a fresh preview matches the current selection. All dynamic values render via an
XSS-safe DOM-text `esc()` helper — never raw `innerHTML` interpolation of
untrusted values, and no raw snapshot embedded in the page.
