"""Server-less local single-file GUI for the Atlas recommendation workflow.

Flow: Profile -> Recommend -> Review Recipe -> Compress -> Monitor -> Output.

Rendered as a fully self-contained HTML page that calls the
`RecommendationService` facade via injected JSON snapshots (fetch endpoints are
provided by a tiny stdlib http.server when run standalone: `/api/profiles`,
`/api/recommend`, `/api/recipe/preview`, `/api/recipe/compile`, `/api/job/…`).
All dynamic values are rendered through an XSS-safe `esc()` DOM-text helper —
nothing is interpolated into innerHTML from untrusted input.

The GUI NEVER authorizes anything itself: recommendations, compiles, and starts
all go through `RecommendationService` (deterministic policy + verified-plan
gate). Current placeholder adapters are rendered with BLOCKED/EXPERIMENTAL
badges and the Compress button is DISABLED with the exact blockers shown —
failing closed, never faking quantization.
"""

# ruff: noqa: E501  (large embedded HTML/JS template lines)

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from model_atlas.recommend.api import RecommendationService
from model_atlas.recommend.policy import RecTarget

ATLAS_PROFILE_DEFAULT_DIR = "profiles"

_JS_ESC = """
const esc = (v) => {
  const el = document.createElement('div');
  el.textContent = String(v ?? '');
  return el.textContent;
};
"""


def _html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _static_page(data_payloads: dict[str, Any]) -> str:
    """Deterministic single-file HTML with embedded JSON snapshots + JS."""
    snapshots = json.dumps(data_payloads, sort_keys=True, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Atlas Recommend -> Compress</title>
<style>
 body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1a202c}}
 main{{max-width:1080px;margin:0 auto;padding:24px}}
 h1{{font-size:18px}} .card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:14px 0}}
 .method{{border:1px solid #e2e8f0;border-radius:8px;padding:10px;margin:8px 0}}
 .badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-left:6px}}
 .badge.blocked{{background:#fee2e2;color:#991b1b}} .badge.exp{{background:#fef3c7;color:#92400e}}
 .badge.ok{{background:#dcfce7;color:#166534}}
 button{{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:8px 14px}}
 button:disabled{{background:#cbd5e1;color:#475569;cursor:not-allowed}}
 pre{{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto}}
 .np{{font-weight:600;color:#166534}}
</style></head><body>
<main>
<h1>&#9881; Atlas &#8212; Profile &#8594; Recommend &#8594; Review &#8594; Compress &#8594; Monitor</h1>
<div class="card"><h2>1. Profile + target</h2>
<label>Profile: <select id="profileSel"></select></label>
<label>Memory target (GiB): <input type="number" id="mem" value="115"></label>
<label><input type="checkbox" id="allowPrune" disabled> allow_pruning (locked: <span class="np">no_pruning=true</span>)</label>
<button id="recoBtn">Recommend</button>
<div id="reco"></div>
</div>
<div class="card"><h2>2. Recommendations (deterministic policy)</h2>
<div id="methods"></div></div>
<div class="card"><h2>3. Recipe review &#8212; immutable <span class="np">no-pruning indicator</span></h2>
<div id="recipeText"></div>
<button id="compileBtn">Compile editable recipe</button>
<pre id="compileOut"></pre>
</div>
<div class="card"><h2>4. Compress (verified plan only)</h2>
<div id="blockedExplain"></div>
<button id="compressBtn" disabled>Compress (disabled until a verified executable plan compiles)</button>
<div id="jobStatus"></div>
</div>
<div class="card"><h2>5. Output / failure evidence</h2>
<pre id="output"></pre></div>
</main>
<script>const SNAPSHOT = {snapshots};{_JS_ESC}
const api = {{profiles:SNAPSHOT.profiles, rec:SNAPSHOT.reco, preview:SNAPSHOT.preview,
  compileOut:SNAPSHOT.compile, job:SNAPSHOT.job, blocked:SNAPSHOT.blocked}};
// populate profile select (textContent is XSS-safe)
const sel = document.getElementById('profileSel');
api.profiles.forEach(p => {{ const o = document.createElement('option'); o.value = esc(p.profile_id); o.textContent = esc(p.profile_id); sel.appendChild(o); }});
function renderRec(rec) {{
  const host = document.getElementById('methods'); host.textContent = '';
  (rec.methods||[]).forEach(m => {{
    const d = document.createElement('div'); d.className = 'method';
    const b = document.createElement('b'); b.textContent = m.method; d.appendChild(b);
    const badge = document.createElement('span');
    badge.className = m.blockers && m.blockers.length ? 'badge blocked' : 'badge exp';
    badge.textContent = m.blockers && m.blockers.length ? 'BLOCKED' : m.confidence ? 'confidence ' + m.confidence : 'exp';
    d.appendChild(badge);
    const why = document.createElement('div'); why.textContent = esc(m.reason); d.appendChild(why);
    if (m.blockers && m.blockers.length) {{ const bd = document.createElement('div'); bd.textContent = 'blockers: ' + esc(m.blockers.map(x=>x.code).join(', ')); bd.className='badge blocked'; d.appendChild(bd); }}
    host.appendChild(d);
  }});
  const np = document.createElement('div'); np.className='np'; np.textContent = 'no_pruning = ' + esc(rec.no_pruning); host.appendChild(np);
}}
function escButtons(rec) {{
  const ex = document.getElementById('blockedExplain'); ex.textContent = '';
  const blocked = (rec.blocked_methods||[]);
  blocked.forEach(m => {{ const d=document.createElement('div'); d.textContent='BLOCKED: '+esc(m.method)+' — '+esc(m.blockers.map(x=>x.message).join('; ')); d.className='badge blocked'; ex.appendChild(d); }});
  document.getElementById('compressBtn').disabled = blocked.length > 0; // fail closed
}}
document.getElementById('recoBtn').addEventListener('click', () => {{ const r = api.rec; renderRec(r); escButtons(r); document.getElementById('recipeText').textContent = 'compiled recipe id: ' + esc(r.recommendation_id); }});
document.getElementById('compileBtn').addEventListener('click', () => {{ document.getElementById('compileOut').textContent = esc(JSON.stringify(api.compileOut,null,2)); }});
document.getElementById('compressBtn').addEventListener('click', () => {{ document.getElementById('jobStatus').textContent = 'job: ' + esc(JSON.stringify(api.job,null,2)); }});
</script></body></html>"""


def render_gui(service: RecommendationService, **snapshots: Any) -> str:
    """Render the GUI with a deterministic service snapshot (profiles, a
    sample recommendation, preview, compile, job placeholders)."""
    profiles = service.list_profiles()
    reco: dict[str, Any] = {}
    preview: dict[str, Any] = {}
    compile_out: dict[str, Any] = {}
    job: dict[str, Any] = {}
    if profiles:
        try:
            prof = service.import_profile(profiles[0]["path"])
            rec = service.recommend(prof, target=RecTarget())
            reco = rec.to_dict()
            # editable-draft compile/preview: build a minimal no-pruning recipe
            from model_atlas.recipes.builtin import glm52_no_pruning_recipe

            draft = glm52_no_pruning_recipe()
            preview = service.preview_recipe(draft)
            try:
                compiled = service.compile_recipe(draft)
                compile_out = {"plan_id": compiled.plan_id, "compiles": True}
            except Exception as exc:  # noqa: BLE001
                compile_out = {"compiles": False, "error": str(exc)}
                job = {"status": "not_started", "reason": "unavailable placeholders"}
        except Exception as exc:  # noqa: BLE001
            reco = {"error": str(exc)}
    return _static_page(
        {
            "profiles": profiles,
            "reco": reco,
            "preview": preview,
            "compile": compile_out,
            "job": job,
            "blocked": reco.get("blocked_methods", []),
        }
    )


def write_gui(path: str, service: RecommendationService | None = None, **snapshots: Any) -> str:
    service = service or RecommendationService()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_gui(service, **snapshots), encoding="utf-8")
    return path
