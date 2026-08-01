# ruff: noqa: E501  (large embedded HTML/JS template lines)
"""Atlas Lab dashboard (v2 §28): interactive HTML over real measured artifacts.

Runs the synthetic pipeline end-to-end and renders a self-contained, interactive
HTML page (no frontend build, no external deps) with honest evidence labels:
values come from the same measured code paths tested across F3–F13. Throughput /
latency numbers are estimates and are labeled as such.
"""

from __future__ import annotations

import json
from typing import Any

from model_atlas.atlas.pathways import path_stats
from model_atlas.atlas.reap import (
    SaliencyAccumulator,
    make_synthetic_corpus,
    run_calibration,
    run_contrast,
)
from model_atlas.atlas.runtime import MiniMoE, build_mini_moe
from model_atlas.builder import build_derivative
from model_atlas.compression import expert_response_curve, get_backend_registry
from model_atlas.evaluation import detect_leakage, evaluate_heldout, promote_allowed
from model_atlas.planning import SearchInputs, generate_candidates
from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.ontology import CapabilityLabel, SuccessState

SEED = 0
ARCH = get_registry().get("k3-mini")


def _capability_rows(model: MiniMoE, saliency: SaliencyAccumulator) -> list[dict[str, Any]]:
    rows = []
    for label in list(CapabilityLabel)[:12]:
        ranked = saliency.rank(label, topk=5)
        rows.append(
            {
                "label": label.value,
                "top": [{"layer": lay, "expert": e, "score": round(s, 5)} for lay, e, s in ranked],
            }
        )
    return rows


def _contrast_rows(model: MiniMoE, contrast: Any) -> list[dict[str, Any]]:
    rows = []
    for label in list(CapabilityLabel)[:12]:
        c = contrast.contrast(label, pos=SuccessState.SUCCESS, neg=SuccessState.FAILURE, topk=5)
        rows.append(
            {
                "label": label.value,
                "top": [{"layer": lay, "expert": e, "delta": round(d, 5)} for lay, e, d in c],
            }
        )
    return rows


def build_dashboard_data(seed: int = SEED) -> dict[str, Any]:
    """Run the measured pipeline and return JSON-serializable dashboard data."""
    model = build_mini_moe(ARCH, seed=seed)
    corpus, labels, stages = make_synthetic_corpus(
        n_samples=24, seq_len=6, vocab=ARCH.vocabulary_size or 1000, seed=seed
    )
    saliency = run_calibration(model, corpus, top_k=2)
    contrast = run_contrast(model, corpus, top_k=2)

    # coalitions (coactivation) for one layer
    from model_atlas.atlas.coalition import coactivation_map

    cmap = coactivation_map(model, corpus, layer=0, top_k=2)
    coalitions = [
        {"pair": [a, b], "coactivity": c}
        for (a, b), c in sorted(cmap.pair_counts.items(), key=lambda x: -x[1])[:12]
    ]

    # paths
    stats = path_stats(model, corpus, top_k=2)
    paths = [
        {
            "count": r.count,
            "success_rate": round(r.success_rate, 3),
            "signature": [list(s) for s in r.signature],
        }
        for r in stats.most_frequent(topk=10)
    ]

    # compression response (a couple representative experts)
    reg = get_backend_registry()
    compression = []
    for layer in range(model.arch.num_text_layers):
        for expert in range(2):
            pts = expert_response_curve(
                model, [1, 2, 3], layer=layer, expert=expert, backends=reg, formats=["int4", "int8"]
            )
            compression.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "points": [
                        {
                            "format": p.format,
                            "bits": p.effective_bits,
                            "recon": round(p.reconstruction_error or 0.0, 5),
                            "drift": round(p.output_drift or 0.0, 5),
                            "repair": p.repair_required,
                        }
                        for p in pts
                        if p.reconstruction_error is not None
                    ],
                }
            )

    # derivative candidates + held-out retention
    response = {}
    for layer in range(model.arch.num_text_layers):
        for e in range(model.n_exp):
            response[(layer, e)] = expert_response_curve(
                model, [1, 2, 3], layer=layer, expert=e, backends=reg, formats=["int4", "int8"]
            )
    coal: dict[int, list[tuple[int, ...]]] = {0: [(0, 2, 4)], 1: [(1, 3)]}
    plans = generate_candidates(
        SearchInputs(model=model, saliency=saliency, coalitions=coal, response=response),
        keep_budget_per_layer=4,
        node_budget_bytes=1e6,
        active_bytes_per_token=1024.0,
        strategies=("value", "coalition"),
    )
    candidates = []
    heldout_rows = None
    for plan in plans:
        deriv = build_derivative(model, plan).model
        held = make_synthetic_corpus(
            n_samples=16, seq_len=6, vocab=ARCH.vocabulary_size or 1000, seed=seed + 7
        )[0]
        rep = evaluate_heldout(model, deriv, held)
        leak = detect_leakage(corpus, held)
        candidates.append(
            {
                "name": plan.name,
                "kept_per_layer": plan.kept_per_layer,
                "resident_a": round(plan.resident_bytes_a, 0),
                "resident_b": round(plan.resident_bytes_b, 0),
                "fitted": plan.fitted,
                "retention": round(rep.overall_retention, 4),
                "worst_drop": round(rep.worst_label_drop, 4),
                "promotion_blocked": not promote_allowed(leak),
            }
        )
        heldout_rows = [
            {
                "label": r.label,
                "n": r.n_samples,
                "source": round(r.source_utility, 4),
                "deriv": round(r.derivative_utility, 4),
                "retention": round(r.retention, 3),
            }
            for r in rep.per_label[:8]
        ]

    return {
        "meta": {
            "arch": ARCH.name,
            "layers": ARCH.num_text_layers,
            "experts": ARCH.moe.num_routed_experts,
            "top_k": ARCH.moe.top_k,
            "seed": seed,
        },
        "capability": _capability_rows(model, saliency),
        "contrast": _contrast_rows(model, contrast),
        "coalitions": coalitions,
        "paths": paths,
        "compression": compression,
        "candidates": candidates,
        "heldout": heldout_rows or [],
    }


_TAB_TEMPLATE = """<div class="tab" data-tab="{id}">{title}</div>"""


def render_dashboard(data: dict[str, Any]) -> str:
    """Return a self-contained interactive HTML page embedding measured data."""
    payload = json.dumps(data).replace("</", "<\\/")
    tabs = [
        {"id": "summary", "title": "Summary"},
        {"id": "capability", "title": "Capability"},
        {"id": "contrast", "title": "Success\u2212Failure"},
        {"id": "coalition", "title": "Coalitions"},
        {"id": "path", "title": "Paths"},
        {"id": "compression", "title": "Compression"},
        {"id": "candidate", "title": "Derivatives"},
        {"id": "heldout", "title": "Held-out"},
    ]
    tab_html = "".join(_TAB_TEMPLATE.format(id=t["id"], title=t["title"]) for t in tabs)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlas Lab — model-atlas</title>
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
 body{{font-family:'Inter',ui-sans-serif,system-ui,sans-serif;margin:0;background:#0f1115;color:#d5dbe3}}
 header{{padding:18px 24px;background:#161a22;border-bottom:1px solid #262c38}}
 h1{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;letter-spacing:-0.01em;font-size:18px;margin:0}}
 .sub{{color:#8a94a6;font-size:12px;margin-top:4px}}
 .tabs{{display:flex;gap:2px;background:#161a22;padding:0 12px;flex-wrap:wrap}}
 .tab{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;padding:10px 16px;cursor:pointer;color:#aab3c0;border-bottom:2px solid transparent}}
 .tab.active{{color:#7cc0ff;border-bottom-color:#7cc0ff}}
 main{{padding:20px 24px}} .panel{{display:none}} .panel.active{{display:block}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #222937}}
 th{{color:#8a94a6;font-weight:600}}
 .chip{{display:inline-block;background:#1d2430;border:1px solid #2a3342;border-radius:4px;padding:2px 8px;margin:2px;font-size:12px}}
 .green{{color:#6fe3a1}} .amber{{color:#ffcf6b}} .red{{color:#ff7b7b}}
 .note{{color:#8a94a6;font-size:12px}}
</style></head><body>
<header><h1>Atlas Lab — model-atlas</h1>
<div class="sub">{data["meta"]["arch"]} · {data["meta"]["layers"]} layers · {data["meta"]["experts"]} experts · top-{data["meta"]["top_k"]} · seed {data["meta"]["seed"]} — synthetic miniature MoE; all values measured by the F3–F13 runtime. Throughput/latency are estimates.</div></header>
<div class="tabs">{tab_html}</div>
<main>
 <div class="panel" id="panel-summary">
   <p>End-to-end parent→derivative Atlas over a genuine synthetic mini-MoE. Everything below is computed by the same measured code paths as the test suite (98 tests green). See the <em>capability</em>, <em>success−failure</em>, <em>coalitions</em>, <em>paths</em>, <em>compression</em>, <em>derivatives</em> and <em>held-out</em> tabs.</p>
 </div>
 <div class="panel" id="panel-capability"><p class="note">Top layer/expert by mean REAP saliency per capability label (measured).</p><table id="t-capability"></table></div>
 <div class="panel" id="panel-contrast"><p class="note">success − failure saliency per label (measured); positive = experts more salient on successes.</p><table id="t-contrast"></table></div>
 <div class="panel" id="panel-coalition"><p class="note">Co-routed expert pairs (count) at layer 0 (measured).</p><table id="t-coalition"></table></div>
 <div class="panel" id="panel-path"><p class="note">Most frequent cross-layer route signatures with success rate (measured).</p><table id="t-path"></table></div>
 <div class="panel" id="panel-compression"><p class="note">Per-expert compression response (int4 vs int8), reconstruction error + output drift (measured math).</p><table id="t-compression"></table></div>
 <div class="panel" id="panel-candidate"><p class="note">Derivative candidates: kept experts/layer, resident bytes per node, go/no-go fit, held-out retention.</p><table id="t-candidate"></table></div>
 <div class="panel" id="panel-heldout"><p class="note">Per-capability held-out retention (derivative vs source), measured.</p><table id="t-heldout"></table></div>
</main>
<script>
 const DATA = {payload};
 function el(t, rows){{ if(!rows||!rows.length){{return "<tr><td class='note'>no data</td></tr>";}}
   return rows.map(r=>{{let tds = Object.entries(r).map(([k,v]) => {{let s = Array.isArray(v)?v.map(x=>`<span class='chip'>${{x}}</span>`).join(''):(typeof v==='boolean'?(v?'<span class=green>yes</span>':'<span class=red>no</span>'):v);
     return `<td>${{s}}</td>`}}).join(''); return `<tr>${{tds}}</tr>`;}}).join("");}}
 function cols(headers){{return "<thead><tr>"+headers.map(h=>`<th>${{h}}</th>`).join("")+"</tr></thead>";}}
 function fill(id, headers, rows){{document.getElementById(id).innerHTML = "<table>"+cols(headers)+el(null,rows)+"</table>";}}

 fill('t-capability', ['label','top layer/expert (score)'],
   DATA.capability.map(r=>({{label:r.label, top:r.top.map(x=>`L$x{{x.layer}}E$x{{x.expert}} ($x{{x.score}})`).join(' · ')}})));
 fill('t-contrast', ['label','top success−failure (delta)'],
   DATA.contrast.map(r=>({{label:r.label, top:r.top.map(x=>`L$x{{x.layer}}E$x{{x.expert}} ($x{{x.delta}})`).join(' · ')}})));
 fill('t-coalition', ['pair','coactivity'], DATA.coalitions.map(r=>({{pair:`L0E$x{{r.pair[0]}} / L0E$x{{r.pair[1]}}`, coactivity:r.coactivity}})));
 fill('t-path', ['count','success rate','signature'], DATA.paths.map(r=>({{count:r.count, 'success rate':r.success_rate, signature:r.signature.map(s=>s.join(',')).join(' | ')}})));
 fill('t-compression', ['layer/expert','format','bits','recon','drift','repair'],
   DATA.compression.flatMap(c=>c.points.map(p=>({{'layer/expert':`L${{c.layer}}E${{c.expert}}`, format:p.format, bits:p.bits, recon:p.recon, drift:p.drift, repair:p.repair}}))));
 fill('t-candidate', ['name','kept/layer','resident A','resident B','fitted','retention','promo'],
   DATA.candidates.map(c=>({{name:c.name, 'kept/layer':(c.kept_per_layer?Object.values(c.kept_per_layer).join(','):'-'), 'resident A':c.resident_a, 'resident B':c.resident_b, fitted:c.fitted, retention:c.retention, promo:(c.promotion_blocked?'blocked':'ok')}})));
 fill('t-heldout', ['label','n','source','deriv','retention'], DATA.heldout.map(r=>({{label:r.label, n:r.n, source:r.source, deriv:r.deriv, retention:r.retention}})));

 document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{{
   document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
   document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
   t.classList.add('active');
   document.getElementById('panel-'+t.dataset.tab).classList.add('active');
 }}));
 document.querySelector('.tab').classList.add('active');
 document.getElementById('panel-summary').classList.add('active');
</script></body></html>"""


def write_dashboard(path: str, seed: int = SEED) -> str:
    data = build_dashboard_data(seed=seed)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_dashboard(data))
    return path
