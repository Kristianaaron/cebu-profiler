"""Static browser GUI for the Atlas recommendation workflow.

Server-served static HTML/CSS/JS only — no embedded user JSON. The browser
fetches ``/api/profiles`` and ``/api/recommend`` from the local
``RecommendationServer``, lets the user pick a profile and memory target, then
recomputes the recommendation. Method cards/evidence/confidence/blockers are
rendered with ``textContent`` through a DOM-text ``esc()`` helper — never
``innerHTML`` from service data (XSS-safe). Only policy-authorized methods can
be toggled; ``no_pruning`` is locked on.

Flow: Profile -> Recommend -> Review Recipe (preview-from-selection) -> Compress
-> Monitor -> Output.

The GUI NEVER authorizes anything itself: recommendations, previews, and starts
all go through ``RecommendationService`` (deterministic policy + verified-plan
gate). The Compress button is enabled only when EVERY gate passes — no fatal
recommendation blockers, the preview compiles, and a verified executable plan
with passing live pins exists — otherwise it is disabled with the exact
blockers shown (fail closed, never faking quantization).
"""

# ruff: noqa: E501  (large embedded HTML/JS template lines)

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_atlas.recommend.api import RecommendationService

ATLAS_PROFILE_DEFAULT_DIR = "profiles"

# Static, data-free HTML/CSS/JS. No snapshot/JSON is ever interpolated here; the
# page fetches everything over HTTP and renders only via textContent esc().
_GUI_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlas Recommend -&gt; Compress</title>
<style>
 body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1a202c}
 main{max-width:1080px;margin:0 auto;padding:24px}
 h1{font-size:18px} h2{font-size:15px;margin-top:0}
 .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:14px 0}
 .method{border:1px solid #e2e8f0;border-radius:8px;padding:10px;margin:8px 0}
 .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-left:6px}
 .badge.blocked{background:#fee2e2;color:#991b1b} .badge.exp{background:#fef3c7;color:#92400e}
 .badge.ok{background:#dcfce7;color:#166534} .badge.muted{background:#edf2f7;color:#4a5568}
 .np{font-weight:600;color:#166534}
 button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer}
 button:disabled{background:#cbd5e1;color:#475569;cursor:not-allowed}
 pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;white-space:pre-wrap}
 .row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 label.gate{display:block;margin:6px 0}
 .reason{color:#4a5563;font-size:13px;margin:4px 0}
 .evid{color:#718096;font-size:12px}
 .check{list-style:none;padding:0;margin:6px 0}
 .check li::before{content:"\\2714  ";color:#166534}
 .check li.x::before{content:"\\2718  ";color:#991b1b}
</style></head><body>
<main>
<h1>&#9881; Atlas &#8212; Profile &#8594; Recommend &#8594; Review &#8594; Compress &#8594; Monitor</h1>
<div class="card"><h2>1. Profile + target</h2>
<div class="row">
<label>Profile:
  <select id="profileSel"><option value="">(no profiles)</option></select>
</label>
<label>Memory target (GiB):
  <input type="number" id="mem" value="115" min="1" step="0.5">
</label>
<label>Strategy:
  <select id="intent">
    <option value="quantize_only">Quantize only</option>
    <option value="prune_only">Prune only</option>
    <option value="hybrid">Quantize + prune</option>
    <option value="custom">Custom</option>
  </select>
</label>
<label><input type="checkbox" id="allowPrune"> authorize pruning capability</label>
</div>
<button id="recoBtn">Recommend</button>
<span id="recoMeta" class="evid"></span>
<div id="reco"></div>
</div>
<div class="card"><h2>2. Recommendations (deterministic policy)</h2>
<p class="evid">Toggle only the blue-checked (policy-authorized, unblocked) methods;
blocked methods are greyed with the exact reason.</p>
<div id="methods"></div>
</div>
<div class="card"><h2>3. Recipe review &#8212; immutable <span class="np">no-pruning</span></h2>
<div class="row">
<button id="previewBtn">Preview selection as recipe draft</button>
<span id="previewStatus" class="evid"></span>
</div>
<pre id="recipeText"></pre>
</div>
<div class="card"><h2>4. Compress (verified executable plan only)</h2>
<div id="blockedExplain" class="evid"></div>
<button id="compressBtn" disabled>Compress (disabled until every gate passes)</button>
<div id="jobStatus"></div>
</div>
<div class="card"><h2>5. Output / failure evidence</h2>
<pre id="output"></pre></div>
</main>
<script>
const esc = (v) => { const el = document.createElement('div'); el.textContent = String(v ?? ''); return el.textContent; };
const $ = (id) => document.getElementById(id);

let authToken = null; // opaque token bound to the current recommendation
let reco = null;      // last recommendation payload
let preview = null;   // last preview-from-selection payload
let selection = new Set(); // user-toggled authorized methods

// ANY change to profile/target/constraints/checkbox/recommendation invalidates
// the preview AND the token binding: Compress stays disabled until a fresh
// recommend -> preview round-trip matches the current selection hash.
function invalidatePreview(reason) {
  preview = null;
  updateCompress();
  $('previewStatus').textContent = reason ? 'preview invalidated: ' + reason : '';
  $('recipeText').textContent = '';
}

function setRecoBtn(msg) { $('recoMeta').textContent = msg ? ' ' + msg : ''; }

async function loadProfiles() {
  const r = await fetch('/api/profiles');
  const data = await r.json();
  if (!r.ok) throw new Error((data && data.error) || ('profiles ' + r.status));
  const sel = $('profileSel');
  sel.textContent = '';
  (data.profiles || []).forEach(p => {
    const o = document.createElement('option');
    o.value = esc(p.profile_id);
    o.textContent = esc(p.profile_id);
    sel.appendChild(o);
  });
  if (sel.options.length) sel.selectedIndex = 0;
}

async function recommend() {
  const pid = $('profileSel').value;
  if (!pid) { setRecoBtn('no profile selected'); return; }
  const mem = parseFloat($('mem').value) || 115;
  const intent = $('intent').value;
  const allowPruning = $('allowPrune').checked;
  setRecoBtn('computing…');
  let data;
  try {
    const r = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile_id: pid, memory_target_gib: mem, intent: intent, constraints: { allow_pruning: allowPruning } })
    });
    data = await r.json();
    if (!r.ok) throw new Error((data && data.error) || ('recommend ' + r.status));
  } catch (e) {
    setRecoBtn('recommend failed: ' + esc(e.message));
    authToken = null;
    return;
  }
  authToken = data.token;       // opaque authorization token (server-bound)
  reco = data.recommendation || {};
  selection = new Set();        // fresh selection; token now awaited
  invalidatePreview('new recommendation');
  renderRec(reco);
  setRecoBtn('' + (reco.recommendation_id || ''));
}

function renderRec(rec) {
  const host = $('methods'); host.textContent = '';
  const authorized = (rec.methods || []).slice();
  const blocked = (rec.blocked_methods || []).slice().map(m => ({ ...m, blocked: true }));
  const all = authorized.concat(blocked);
  all.sort((a, b) => (a.rank || 99) - (b.rank || 99));
  all.forEach(m => {
    const card = document.createElement('div'); card.className = 'method';
    const top = document.createElement('div');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    const isBlocked = !!m.blocked || ((m.blockers || []).length > 0);
    cb.disabled = isBlocked;
    cb.checked = !isBlocked;
    cb.addEventListener('change', () => {
      if (cb.checked) { selection.add(m.method); invalidatePreview('selection changed'); }
      else { selection.delete(m.method); invalidatePreview('selection changed'); }
    });
    top.appendChild(cb);
    const b = document.createElement('b'); b.textContent = m.method; top.appendChild(b);
    const badge = document.createElement('span');
    badge.className = isBlocked ? 'badge blocked' : 'badge ok';
    badge.textContent = isBlocked
      ? 'blocked'
      : 'confidence ' + (m.confidence || 'medium');
    top.appendChild(badge);
    card.appendChild(top);
    const reason = document.createElement('div'); reason.className = 'reason';
    reason.textContent = esc(m.reason || ''); card.appendChild(reason);
    if (m.evidence_refs && m.evidence_refs.length) {
      const ev = document.createElement('div'); ev.className = 'evid';
      ev.textContent = 'evidence: ' + esc(m.evidence_refs.join(', ')); card.appendChild(ev);
    }
    if (isBlocked && m.blockers) {
      const bd = document.createElement('div');
      bd.textContent = 'blockers: ' + esc(m.blockers.map(x => x.code).join(', '));
      bd.className = 'badge blocked';
      card.appendChild(bd);
    }
    host.appendChild(card);
  });
  const np = document.createElement('div'); np.className = 'np';
  np.textContent = 'no_pruning = ' + esc(rec.no_pruning);
  np.textContent += '; intent = ' + esc(rec.intent || 'quantize_only');
  host.appendChild(np);
  // Initialize the Profile+Compress selection from the methods the browser
  // actually checked (the authorized, unblocked ones). The recipe is then the
  // exact checked subset — matching what preview/start will verify.
  selection = new Set(
    all.filter(m => !(m.blocked || ((m.blockers || []).length > 0))).map(m => m.method)
  );
}

async function previewSelection() {
  if (!authToken) { $('previewStatus').textContent = 'preview requires a valid recommendation token (recommend first)'; return; }
  const selArr = Array.from(selection);
  $('previewStatus').textContent = 'previewing ' + selArr.length + ' method(s)…';
  try {
    const r = await fetch('/api/preview-selection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: authToken, selected: selArr })
    });
    const data = await r.json();
    if (!r.ok) throw new Error((data && data.error) || ('preview ' + r.status));
    preview = data;
    renderPreview(preview);
  } catch (e) {
    preview = null;
    $('previewStatus').textContent = 'preview failed: ' + esc(e.message);
    updateCompress();
  }
}

function renderPreview(p) {
  $('recipeText').textContent = JSON.stringify({
    preview_id: p.preview_id,
    plan_id: p.plan_id,
    hash: p.hash,
    readiness: p.readiness || {},
    intent: p.intent || '',
    intent_blockers: p.intent_blockers || [],
    selected_methods: p.selected_methods || []
  }, null, 2);
  const ok = p.readiness && p.readiness.executable && p.plan_id;
  $('previewStatus').textContent = ok
    ? 'ready: verified plan ' + ((p.plan_id) || '')
    : 'preview stored (selected id=' + (p.preview_id || '?') + ') but no verified executable plan yet';
  updateCompress();
}

function updateCompress() {
  const gate = computeGates();
  const btn = $('compressBtn');
  btn.disabled = !gate.ready;
  const ex = $('blockedExplain'); ex.textContent = '';
  if (!gate.ready) {
    const ul = document.createElement('ul'); ul.className = 'check';
    gate.reasons.forEach(re => {
      const li = document.createElement('li'); li.className = 'x'; li.textContent = re; ul.appendChild(li);
    });
    ex.appendChild(ul);
  }
}

function computeGates() {
  const reasons = [];
  if (!authToken) reasons.push('no valid recommendation token (recommend first)');
  if (!reco) reasons.push('no recommendation computed yet');
  if (reco && (!reco.methods || reco.methods.length === 0)) reasons.push('no policy-authorized methods recommended');
  if (!selection || selection.size === 0) reasons.push('no methods selected for the recipe');
  if (!preview) reasons.push('no preview: build a recipe draft from a selection');
  if (preview && preview.hash && preview.selected_methods
      && JSON.stringify([...selection].sort()) !== JSON.stringify([...preview.selected_methods].sort()))
    reasons.push('selection changed since preview — re-preview');
  if (preview && !(preview.readiness && preview.readiness.executable))
    reasons.push('no verified executable plan produced for this selection');
  if (preview && preview.intent_blockers)
    preview.intent_blockers.forEach(b => reasons.push((b.code || 'intent_blocked') + ': ' + (b.message || '')));
  const ready = reasons.length === 0;
  return { ready, reasons };
}

$('recoBtn').addEventListener('click', recommend);
$('previewBtn').addEventListener('click', previewSelection);
// profile / target / constraints changes INVALIDATE the entire authorization
// + selection + preview (fresh recommendation required): they clear token,
// reco, selection, preview so Compress is always gated on a current token.
function clearBinding(reason) {
  authToken = null;      // the old authorization token no longer applies
  reco = null;           // the old recommendation no longer applies
  selection = new Set(); // clear selection — Compress waits for re-select
  invalidatePreview(reason);
}
$('profileSel').addEventListener('change', () => clearBinding('profile changed'));
$('mem').addEventListener('input', () => clearBinding('memory target changed'));
$('intent').addEventListener('change', () => {
  if ($('intent').value === 'quantize_only') $('allowPrune').checked = false;
  clearBinding('strategy changed');
});
$('allowPrune').addEventListener('change', () => clearBinding('pruning authorization changed'));
$('compressBtn').addEventListener('click', async () => {
  if (!authToken || !preview) return;
  const g = computeGates();
  if (!g.ready) { updateCompress(); return; }
  $('jobStatus').textContent = 'starting verified plan…';
  const selArr = Array.from(selection);
  try {
    const r = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: authToken, preview_id: preview.preview_id,
        hash: preview.hash,
        plan_id: preview.plan_id,
        recipe_sha256: preview.recipe_sha256,
        selected: selArr,
        inputs: {}
      })
    });
    const data = await r.json();
    if (!r.ok) throw new Error((data && data.error) || ('start ' + r.status));
    const runId = data.run_id;
    $('jobStatus').textContent = 'started run ' + runId;
    pollRun(runId);
  } catch (e) {
    $('jobStatus').textContent = 'start failed: ' + esc(e.message);
    updateCompress();
  }
});

function setOutput(text) { $('output').textContent = text || ''; }

async function pollRun(runId) {
  const enc = encodeURIComponent(runId);
  try {
    const all = await Promise.allSettled([
      fetch('/api/jobs/' + enc),
      fetch('/api/jobs/' + enc + '/events')
    ]);
    const statusData = all[0].status === 'fulfilled' ? await all[0].value.json() : { error: 'status unavailable' };
    const events = all[1].status === 'fulfilled' ? await all[1].value.json() : { events: [] };
    const st = statusData.status || '';
    setOutput(JSON.stringify({ status: statusData, events: events.events || [] }, null, 2));
    $('jobStatus').textContent = 'run ' + runId + ' status: ' + st;
    const term = String(st).toUpperCase();
    if (term === 'DONE' || term === 'COMPLETED' || term === 'COMPLETED_WITH_WARNINGS' || term === 'FAILED' || term === 'FAILED_TERMINAL' || term === 'FAILED_RECOVERABLE' || term === 'CANCELLED' || term === 'CANCELED') {
      fetchEvidence(runId);
      return; // STOP polling at terminal state
    }
    setTimeout(() => pollRun(runId), 1500);
  } catch (e) {
    $('jobStatus').textContent = 'poll error: ' + esc(e.message);
    setTimeout(() => pollRun(runId), 1500);
  }
}

async function fetchEvidence(runId) {
  // fetch + render validate/lineage/outputs for the TERMINAL run only
  const enc = encodeURIComponent(runId);
  const statusData = await (await fetch('/api/jobs/' + enc)).json();
  const outputs = await (await fetch('/outputs?run_id=' + enc)).json();
  // run_id-bound lineage (actual completed stages)
  let lineage = { note: 'unavailable' };
  try { lineage = await (await fetch('/lineage?run_id=' + enc)).json(); } catch (e) { /* best-effort */ }
  // per-stage validation for actually-COMPLETED stages in the run
  let validation = [];
  const stage_map = (statusData.stages) || {};
  const doneStages = Object.keys(stage_map).filter(sid => {
    const so = stage_map[sid]; const sstat = so && so.status ? String(so.status) : '';
    return sstat === 'done' || sstat === 'completed' || sstat === 'running' || sstat === 'skipped';
  });
  for (const sid of doneStages.slice(0, 5)) {
    try {
      const v = await (await fetch('/validate?run_id=' + enc + '&stage=' + encodeURIComponent(sid))).json();
      if (v && v.run_id) validation.push({ stage: sid, ...v });
    } catch (e) { /* best-effort */ }
  }
  setOutput(JSON.stringify({ status: statusData, outputs: outputs.outputs || [], lineage, validation }, null, 2));
}

updateCompress();
loadProfiles().catch(e => setRecoBtn('profile load failed: ' + esc(e.message)));
</script></body></html>
"""


def render_gui(service: RecommendationService | None = None, **snapshots: object) -> str:
    """Return the static, data-free GUI page.

    ``service`` and any ``snapshots`` are accepted for backwards-compatible
    signature parity, but the page NEVER embeds a service snapshot — all data is
    fetched over HTTP by the browser. (XSS-safe: no raw payload in the page.)
    """
    return _GUI_PAGE


def write_gui(path: str, service: RecommendationService | None = None, **snapshots: object) -> str:
    """Write the static GUI page to ``path`` (data-free; browser fetches all)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_gui(service, **snapshots), encoding="utf-8")
    return path


def _html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)
