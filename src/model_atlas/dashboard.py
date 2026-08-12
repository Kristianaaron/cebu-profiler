# ruff: noqa: E501  (large embedded HTML/JS template lines)
"""Atlas Lab dashboard (v2 §28): interactive HTML over real measured artifacts.

Runs the synthetic pipeline end-to-end and renders a self-contained, interactive
HTML page (no frontend build, no external deps) with honest evidence labels:
values come from the same measured code paths tested across F3–F13. Throughput /
latency numbers are estimates and are labeled as such.
"""

from __future__ import annotations

import json
import os
from typing import Any

from model_atlas.atlas.hierarchy import build_hierarchy
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


# Lucide icon inner-SVG (viewBox 0 0 24 24) for each side-nav tab.
_LUCIDE = "fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\""
_ICONS: dict[str, str] = {
    "summary": '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
    "capability": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "contrast": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "coalition": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>',
    "path": '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
    "compression": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "candidate": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "heldout": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1 1 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "maps": '<polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21 3 6"/><polyline points="3 6 9 9 15 6 21 9"/><line x1="9" x2="9" y1="9" y2="18"/>',
    "hierarchy": '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="2"/><circle cx="4" cy="12" r="2"/><circle cx="4" cy="18" r="2"/>',
    "reality": '<path d="M12 2v4M12 18v4M2 12h4M18 12h4"/><circle cx="12" cy="12" r="6"/>',
}


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


def _capability_voxels(model: MiniMoE, saliency: SaliencyAccumulator) -> dict[str, Any]:
    """Compact 3D voxel payload for the Capability view.

    One voxel per scored (label, layer, expert) cell, score normalised per label
    to 0..1. Bounded by the geometry (2*8=16 cells/label), so the renderer stays
    tiny and cheap to draw.
    """
    labels = [lbl.value for lbl in list(CapabilityLabel)[:12]]
    n_layers = len(model.layers)
    n_exp = model.n_exp
    voxels: list[dict[str, Any]] = []
    for li, label in enumerate(list(CapabilityLabel)[:12]):
        ranked = saliency.rank(label, topk=n_layers * n_exp)
        ranked = [(lay, e, s) for lay, e, s in ranked if s > 0]
        if not ranked:
            continue
        mx = max(s for _, _, s in ranked)
        for lay, e, s in ranked:
            voxels.append(
                {"label": li, "layer": lay, "expert": e, "score": round(s / mx, 3)}
            )
    return {"labels": labels, "n_layers": n_layers, "n_experts": n_exp, "voxels": voxels}


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


def _real_bytes_payload() -> dict[str, Any]:
    """Real-bytes derivative envelopes (§24/§25) for the Quantization & Fit view.

    Uses the mounted GLM-5.2 NVFP4 census when present, else a synthetic caret,
    so the section always renders. Retention fractions are estimates (a routing
    census needs inference); the byte math is measured.
    """
    from model_atlas.planning.realbytes import GIB, account_manifest, plan_candidates

    _REAL = "/media/glm52/models/nvidia/GLM-5.2-NVFP4"
    out: dict[str, Any] = {"source": None, "measured_gib": 0.0, "candidates": []}
    try:
        if os.path.isfile(os.path.join(_REAL, "config.json")):
            from model_atlas.checkpoint.source_manifest import load_manifest

            acc = account_manifest(load_manifest(_REAL))
            out["source"] = _REAL
        else:  # synthetic caret so the section still renders offline
            from model_atlas.checkpoint.source_manifest import CheckpointManifest

            def _tensor(name: str, size_gib: float, bpw: float) -> Any:
                from model_atlas.checkpoint.source_manifest import TensorEntry

                size = int(size_gib * GIB)
                numel = int(size * 8 / bpw)
                return TensorEntry(
                    name=name, dtype="bf16", shape=[numel], numel=numel,
                    byte_size=size, shard="s", offset_start=0, offset_end=size,
                )

            tensors = [
                _tensor(f"model.layers.0.experts.{e}.gate.weight", 20.0 / 4, 8.19)
                for e in range(4)
            ]
            tensors += [
                _tensor("model.layers.0.self_attn.q_proj.weight", 3.33, 16.0),
                _tensor("model.embed_tokens.weight", 3.33, 16.0),
                _tensor("model.lm_head.weight", 3.34, 16.0),
            ]
            cf = CheckpointManifest(
                checkpoint_dir="(synthetic)",
                tensors=tensors,
                total_bytes=sum(t.byte_size for t in tensors),
                tensor_count=len(tensors),
            )
            acc = account_manifest(cf)
            out["source"] = "(synthetic caret)"
        out["measured_gib"] = round(acc.total_bytes / GIB, 1)
        out["candidates"] = [
            {
                "envelope": round(c.envelope_gb, 0),
                "keep": c.keep_frac,
                "precision": c.expert_precision,
                "bpw": c.mean_expert_bpw,
                "stored": round(c.stored_gib(), 1),
                "resident_a": round(c.resident_a_gib(), 1),
                "resident_b": round(c.resident_b_gib(), 1),
                "risk": c.risk,
            }
            for c in plan_candidates(acc)
        ]
    except Exception as exc:  # never break the whole dashboard over the fit section
        out["source"] = f"(unavailable: {type(exc).__name__})"
    return out


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

    # §25 planning-artifact maps (measured/estimated, see maps_build)
    from model_atlas.planning.maps_build import build_planning_maps

    maps = build_planning_maps(model, saliency)
    maps_payload = {
        "channel": [e.model_dump() for e in maps.channel.entries],
        "tile": [e.model_dump() for e in maps.tile.entries],
        "node_ownership": [e.model_dump() for e in maps.node_ownership.entries],
        "overflow_pack": [e.model_dump() for e in maps.overflow_pack.entries],
        "router_repair": [e.model_dump() for e in maps.router_repair.entries],
        "residual_repair": [e.model_dump() for e in maps.residual_repair.entries],
        "distillation_target": [e.model_dump() for e in maps.distillation_target.entries],
    }

    # §9 six-level hierarchy (profile)
    hm = build_hierarchy(model, corpus, top_k=2)
    hierarchy_payload = {
        "model_id": hm.model_id,
        "counts": hm.counts(),
        "levels": ["weights", "units", "experts", "coalitions", "pathways", "behaviour"],
        "example": {},
    }
    # a concrete trace-down example: contributors to the first behaviour
    from model_atlas.atlas.hierarchy import AtlasLevel

    behs = hm.nodes_at(AtlasLevel.BEHAVIOUR)
    if behs:
        proj = hm.project_down(behs[0].key)
        hierarchy_payload["example"] = {
            "behaviour": behs[0].label,
            "experts": len(proj.get("experts", [])),
            "units": len(proj.get("units", [])),
            "weights": len(proj.get("weights", [])),
            "top_shared_unit_prevalence": (
                proj["units"][0]["prevalence"] if proj.get("units") else None
            ),
        }

    # real-bytes derivative envelopes (fit) — real GLM census when mounted
    reality_payload = _real_bytes_payload()

    ecosystem_payload = {
        "eval_host": 8100,
        "note": "Eval Harness = standalone benchmarking app, Atlas = profiling/fit platform",
    }

    return {
        "meta": {
            "arch": ARCH.name,
            "layers": ARCH.num_text_layers,
            "experts": ARCH.moe.num_routed_experts,
            "top_k": ARCH.moe.top_k,
            "seed": seed,
        },
        "capability": _capability_rows(model, saliency),
        "capability3d": _capability_voxels(model, saliency),
        "contrast": _contrast_rows(model, contrast),
        "coalitions": coalitions,
        "paths": paths,
        "compression": compression,
        "candidates": candidates,
        "heldout": heldout_rows or [],
        "maps": maps_payload,
        "hierarchy": hierarchy_payload,
        "reality": reality_payload,
        "ecosystem": ecosystem_payload,
    }


_CAP3D_JS = r"""
// Capability 3D voxel view — dependency-free, render-on-interaction only.
// FIXED isometric (3D POV) stacked-sheet view, no rotation UI. Each
// capability L is a diamond-stack column; experts E run along one diagonal,
// capabilities along the other, and layers l stack upward (vertical). Every
// voxel is a translucent isometric tile plus a shaded side face so stacks read
// as 3D. Each cell's TOP diamond projects to a fixed screen parallelogram
// disjoint from every other top diamond => hover stays exact/unambiguous.
// Wheel zooms. (No drag-rotate.)
(function () {
  var cv = document.getElementById('cap3d');
  if (!cv || !cv.getContext || !DATA.capability3d || !DATA.capability3d.voxels) return;
  var panelEl = document.getElementById('cap3d-panel');
  var V = DATA.capability3d, labels = V.labels, nl = V.n_layers, ne = V.n_experts, vox = V.voxels;
  var dpr = (window.devicePixelRatio || 1);
  var W = cv.clientWidth || 680, H = cv.clientHeight || 420;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  var ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  var zoom = 1.0, S = 6, T = S, du0 = S * 0.87, dw0 = S * 0.5, OX = 0, OY = 0;   // iso params
  var hover = null, pin = null, focus = 0, selLayer = null;
  var layerOn = []; for (var _l0 = 0; _l0 < nl; _l0++) layerOn.push(true);
  var cells = [];   // { v, top: [4 pts screen], side: [4 pts screen] }

  // Isometric rhombus tile for cell (E,L,l): perfect lattice tiling via two
  // basis vectors, vE = one expert step (down-right), vL = one capability step
  // (down-left). Layer l shifts the whole tile up by l*T so stacks are vertical.
  // Isometric RHOMBUS tile for cell (E,L,l) — a true tiling parallelogram
  // (translation lattice) spanned by expert step vE=(du,dw) and capability
  // step vL=(-du,dw); layer l lifts it by l*T so stacks are vertical.
  function tile(E, L, l) {
    var x = (E - L) * du0, y = (E + L) * dw0 - l * T;
    return [
      [x - du0, y + dw0],   // a + vL
      [x, y],               // a
      [x + du0, y + dw0],   // a + vE
      [x, y + 2 * dw0]      // a + vE + vL
    ];
  }
  // Vertical wall under the tile's left +front edge down to the next layer.
  function side(E, L, l) {
    var a = tile(E, L, l), b = tile(E, L, l + 1);
    return [[a[0][0], a[0][1]], [a[3][0], a[3][1]], [b[3][0], b[3][1]], [b[0][0], b[0][1]]];
  }
  function layout() {
    var sc2 = Math.min((W - 48) / ((ne + labels.length) * 0.87 + 2), (H - 56) / (nl + (ne + labels.length) * 0.55));
    S = sc2; du0 = S * 0.87; dw0 = S * 0.5; T = S * 1.15;
    // bounds of all top tiles + side faces (origin 0,0)
    var x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    function scan(pt) { if (pt[0] < x0) x0 = pt[0]; if (pt[0] > x1) x1 = pt[0]; if (pt[1] < y0) y0 = pt[1]; if (pt[1] > y1) y1 = pt[1]; }
    for (var li = 0; li < labels.length; li++) {
      for (var ly = 0; ly < nl; ly++) {
        if (!layerOn[ly]) continue;
        for (var ex = 0; ex < ne; ex++) {
          var t = tile(ex, li, ly); for (var q = 0; q < 4; q++) scan(t[q]);
          var s2 = side(ex, li, ly); for (var q2 = 0; q2 < 4; q2++) scan(s2[q2]);
        }
      }
    }
    OX = W / 2 - (x0 + x1) / 2; OY = H / 2 - (y0 + y1) / 2;
    function move(pts, ox, oy) { var o = []; for (var i = 0; i < pts.length; i++) o.push([pts[i][0] + ox, pts[i][1] + oy]); return o; }
    cells = [];
    for (var label2 = 0; label2 < labels.length; label2++) {
      for (var layer2 = 0; layer2 < nl; layer2++) {
        if (!layerOn[layer2]) continue;
        for (var e2 = 0; e2 < ne; e2++) {
          var v2 = vox[0];
          for (var k2 = 0; k2 < vox.length; k2++) if (vox[k2].label === label2 && vox[k2].layer === layer2 && vox[k2].expert === e2) { v2 = vox[k2]; break; }
          var tp = move(tile(e2, label2, layer2), OX, OY);
          cells.push({ v: v2, top: tp, side: move(side(e2, label2, layer2), OX, OY),
            cx: (tp[0][0] + tp[1][0] + tp[2][0] + tp[3][0]) / 4, cy: (tp[0][1] + tp[1][1] + tp[2][1] + tp[3][1]) / 4 });
        }
      }
    }
  }

  function isOn(v, l) { return v.label === l.label && v.layer === l.layer && v.expert === l.expert; }
  function fillQuad(pts, style) { ctx.fillStyle = style; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (var q = 1; q < 4; q++) ctx.lineTo(pts[q][0], pts[q][1]); ctx.closePath(); ctx.fill(); }
  function strokeQuad(pts) { ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (var q = 1; q < 4; q++) ctx.lineTo(pts[q][0], pts[q][1]); ctx.closePath(); ctx.stroke(); }
  function aa(v4) { return Math.max(0, Math.min(1, v4)).toFixed(3); }
  function draw() {
    layout();
    ctx.fillStyle = '#0a0d13'; ctx.fillRect(0, 0, W, H);   // dark background retained
    // one translucent sheet panel per (capability, layer) behind its cells
    for (var g = 0; g < cells.length;) {
      var gl = cells[g].v.label, gy = cells[g].v.layer, g0 = g;
      while (g < cells.length && cells[g].v.label === gl && cells[g].v.layer === gy) g++;
      var f = cells[g0], l = cells[g - 1];
      var panel = [[f.top[0][0], f.top[0][1]], [l.top[1][0], l.top[1][1]], [l.top[2][0], l.top[2][1]], [f.top[3][0], f.top[3][1]]];
      fillQuad(panel, 'rgba(140,165,210,' + aa(0.06 + 0.10 * f.v.score) + ')');
      ctx.strokeStyle = 'rgba(140,165,210,0.20)'; ctx.lineWidth = 1; strokeQuad(panel);
    }
    // painter-sort cells back-to-front, then translucent cell panels (opacity=saliency)
    var order = cells.slice();
    order.sort(function (a, b) { return (a.cy - b.cy) || (a.v.layer - b.v.layer); });
    for (var i = 0; i < order.length; i++) {
      var c = order[i], v = c.v, top = c.top;
      var isSel = pin && isOn(v, pin), isHov = hover && isOn(v, hover);
      var alpha = Math.max(0.05, 0.14 + 0.7 * v.score);
      if (isSel || isHov) {
        fillQuad(top, 'rgba(226,236,250,' + aa(Math.min(1, alpha + 0.4)) + ')');
        ctx.strokeStyle = isSel ? '#ffffff' : 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1.5; strokeQuad(top);
      } else {
        fillQuad(top, 'rgba(216,228,246,' + aa(alpha) + ')');
      }
    }
    ctx.fillStyle = 'rgba(140,160,200,0.8)'; ctx.font = '11px system-ui'; ctx.textAlign = 'left';
    ctx.fillText('diagonals = experts & capabilities · layers stack up as sheets · score = opacity · filter on the right', 6, H - 6);
  }

  function inPoly(px, py, poly) {
    var inside = false;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      var xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
      if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  }
  function pick(e) {
    var r = cv.getBoundingClientRect ? cv.getBoundingClientRect() : { left: 0, top: 0 };
    var bx = e.clientX - r.left, by = e.clientY - r.top;
    var best = null, bd = 1e18;
    for (var i = 0; i < cells.length; i++) {
      if (!inPoly(bx, by, cells[i].top)) continue;
      var dd = (cells[i].cx - bx) * (cells[i].cx - bx) + (cells[i].cy - by) * (cells[i].cy - by);
      if (dd < bd) { bd = dd; best = cells[i].v; }
    }
    return best;
  }

  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
  function bar(v) { return '<span class="p-bar"><i style="width:' + Math.round(v.score * 100) + '%"></i></span> ' + Math.round(v.score * 100) + '%'; }
  function rowHtml(v) {
    var isSel = pin && isOn(v, pin);
    var isHov = (!pin) && hover && isOn(v, hover);
    return '<div class="p-row' + (isSel ? ' sel' : isHov ? ' hov' : '') +
      '" onclick="window.capSel(' + v.label + ',' + v.layer + ',' + v.expert + ')">E' + v.expert + ' ' + bar(v) + '</div>';
  }
  function filterHtml() {
    var h = '<div class="p-filt"><span class="p-filt-l">layers</span>';
    for (var l = 0; l < nl; l++) {
      h += '<span class="p-chip' + (layerOn[l] ? ' on' : '') + '" onclick="window.capLayerOn(' + l + ')">L' + l + '</span>';
    }
    h += '<span class="p-chip' + (layerOn.every(Boolean) ? ' on' : '') + '" onclick="window.capAllLayers()">all</span></div>';
    return h;
  }
  function renderPanel() {
    var li = hover ? hover.label : (pin ? pin.label : focus);
    if (selLayer) li = selLayer.label;
    focus = li;
    var name = labels[li] || '';
    var head = filterHtml();
    if (selLayer && selLayer.label === li) {
      var rows = vox.filter(function (v) { return v.label === li && v.layer === selLayer.layer && layerOn[v.layer]; })
        .sort(function (a, b) { return a.expert - b.expert; });
      var html = head + '<div class="p-head" title="' + name + '">\u25B8 ' + esc(name) + ' <span class="mut">· layer L' + selLayer.layer + '</span></div>';
      html += '<div class="p-sub"><a class="p-back" href="javascript:void(0)" onclick="window.capClearLayer()">\u2190 all layers</a> · ' + rows.length + ' experts</div>';
      for (var j = 0; j < rows.length; j++) html += rowHtml(rows[j]);
      if (panelEl) panelEl.innerHTML = html;
      return;
    }
    var all = vox.filter(function (v) { return v.label === li && layerOn[v.layer]; })
      .sort(function (a, b) { return (a.layer - b.layer) || (a.expert - b.expert); });
    var html = head + '<div class="p-head" title="' + name + '">' + (pin ? '\u25CF ' : '\u25CB ') + esc(name) + '</div>';
    html += '<div class="p-sub">' + all.length + ' visible cells · hover or click</div>';
    var lastL = -1;
    for (var i = 0; i < all.length; i++) {
      var v = all[i];
      if (v.layer !== lastL && layerOn[v.layer]) {
        html += '<div class="p-grp" onclick="window.capLayer(' + li + ',' + v.layer + ')">layer L' + v.layer + '<span class="p-grp-caret">&#9656;</span></div>';
        lastL = v.layer;
      }
      html += rowHtml(v);
    }
    if (panelEl) panelEl.innerHTML = html;
  }

  function redraw() { draw(); renderPanel(); }
  window.capSel = function (label, layer, expert) { pin = { label: label, layer: layer, expert: expert }; focus = label; redraw(); };
  window.capLayer = function (label, layer) { selLayer = { label: label, layer: layer }; pin = null; focus = label; redraw(); };
  window.capClearLayer = function () { selLayer = null; redraw(); };
  window.capLayerOn = function (layer) { layerOn[layer] = !layerOn[layer]; if (pin && !layerOn[pin.layer]) pin = null; redraw(); };
  window.capAllLayers = function () { for (var l = 0; l < nl; l++) layerOn[l] = true; redraw(); };

  function cur(p) { if (cv.style) cv.style.cursor = p ? 'pointer' : 'default'; }
  cv.addEventListener('pointermove', function (e) { var p = pick(e); hover = p; draw(); renderPanel(); cur(p); });
  cv.addEventListener('pointerdown', function (e) { var p = pick(e); if (p) { pin = (pin && isOn(pin, p)) ? null : p; focus = p.label; draw(); renderPanel(); } });
  cv.addEventListener('wheel', function (e) { e.preventDefault(); zoom = Math.max(0.5, Math.min(2.5, zoom * (e.deltaY < 0 ? 1.05 : 0.95))); draw(); }, { passive: false });
  window.addEventListener('resize', function () {
    var W2 = cv.clientWidth || W, H2 = cv.clientHeight || H;
    cv.width = Math.round(W2 * dpr); cv.height = Math.round(H2 * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); W = W2; H = H2; draw();
  });

  var hi = vox[0];
  for (var vi = 1; vi < vox.length; vi++) if (vox[vi].score > hi.score) hi = vox[vi];
  focus = hi ? hi.label : 0;
  draw(); renderPanel();
})();
"""


_TAB_TEMPLATE = """<div class="tab" data-tab="{id}"><svg viewBox="0 0 24 24" width="15" height="15" {_LUCIDE} style="flex:0 0 auto">{icon}</svg><span>{title}</span></div>"""



def render_dashboard(data: dict[str, Any]) -> str:
    """Return a self-contained interactive HTML page embedding measured data."""
    payload = json.dumps(data).replace("</", "<\\/")
    cap3d_json = json.dumps(data.get("capability3d", {})).replace("</", "<\\/")
    nav: list[dict[str, Any]] = [
        {
            "section": "Overview",
            "tabs": [{"id": "summary", "title": "Summary"}],
        },
        {
            "section": "Profiling",
            "tabs": [
                {"id": "capability", "title": "Capability"},
                {"id": "contrast", "title": "Success\u2212Failure"},
                {"id": "coalition", "title": "Coalitions"},
                {"id": "path", "title": "Paths"},
                {"id": "hierarchy", "title": "Hierarchy"},
                {"id": "maps", "title": "Planning Maps"},
            ],
        },
        {
            "section": "Quantization & Fit",
            "tabs": [
                {"id": "compression", "title": "Compression"},
                {"id": "candidate", "title": "Derivatives"},
                {"id": "heldout", "title": "Held-out"},
                {"id": "reality", "title": "Real-bytes"},
            ],
        },
    ]
    nav_html = []
    for group in nav:
        nav_html.append(f"<div class='navsec'>{group['section']}</div>")
        for t in group["tabs"]:
            nav_html.append(
                _TAB_TEMPLATE.format(
                    id=t["id"], title=t["title"], icon=_ICONS.get(t["id"], ""), _LUCIDE=_LUCIDE
                )
            )
    nav_html.append(
        "<a class='navlink' href='http://${{location.hostname}}:8100/' target='_blank'>"
        "Eval Harness &#8599;</a>"
    )
    tab_html = "".join(nav_html)
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
 .layout{{display:flex;min-height:100vh}}
 .col{{flex:1;display:flex;flex-direction:column;min-width:0}}
 nav.side{{width:200px;flex:0 0 200px;display:flex;flex-direction:column;gap:2px;background:#161a22;border-right:1px solid #262c38;padding:14px 10px;position:sticky;top:0;height:100vh;overflow-y:auto;box-sizing:border-box}}
 nav.side .tab{{display:flex;align-items:center;gap:9px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;padding:8px 10px;cursor:pointer;color:#aab3c0;border-radius:6px}}
 nav.side .tab svg{{flex:0 0 auto}}
 nav.side .tab:hover{{background:#1d2430}}
 nav.side .tab.active{{color:#7cc0ff;background:#1d2430}}
 main.main{{flex:1;padding:22px 26px}}
 .panel{{display:none}} .panel.active{{display:block}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #222937}}
 th{{color:#8a94a6;font-weight:600}}
 .chip{{display:inline-block;background:#1d2430;border:1px solid #2a3342;border-radius:4px;padding:2px 8px;margin:2px;font-size:12px}}
 .green{{color:#6fe3a1}} .amber{{color:#ffcf6b}} .red{{color:#ff7b7b}}
 .note{{color:#8a94a6;font-size:12px}}
 .panel h3{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:13px;color:#aab3c0;margin:18px 0 6px}}
 .navsec{{color:#5d6673;font-size:10.5px;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;margin:14px 10px 4px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}}
 .navlink{{display:block;margin:14px 10px 0;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;color:#7cc0ff;text-decoration:none;padding:6px 0;border-top:1px solid #262c38}}
 .navlink:hover{{color:#a8d6ff}}
 .stat{{display:inline-block;background:#1d2430;border:1px solid #2a3342;border-radius:6px;padding:10px 14px;margin:4px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}}
 .stat .k{{color:#8a94a6;font-size:11px;display:block}} .stat .v{{font-size:18px;color:#d5dbe3}}
 .cap3d-wrap{{position:relative;display:flex;gap:12px;align-items:flex-start;background:#05070a;border:1px solid #262c38;border-radius:8px;padding:10px}}
 canvas#cap3d{{flex:1;min-width:0;width:100%;aspect-ratio:680/420;height:auto;display:block;touch-action:none;cursor:default;border-radius:6px;background:#0a0d13}}
 canvas#cap3d.dragging{{cursor:grabbing}}
 .cap3d-panel{{flex:0 0 300px;position:sticky;top:0;align-self:flex-start;background:#0a0c10;border-left:1px solid #262c38;padding-left:12px;max-height:420px;overflow:auto;font-size:12px;color:#cfd6e0}}
 .cap3d-panel .p-head{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12.5px;color:#e9edf3;margin:2px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
 .cap3d-panel .p-sub{{color:#5d6673;font-size:10.5px;margin:2px 0 8px}}
 .cap3d-panel .p-grp{{color:#7cc0ff;font-size:11px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;margin:8px 0 2px;cursor:pointer;display:flex;align-items:center;gap:6px}}
 .cap3d-panel .p-grp:hover{{color:#a8d6ff;text-decoration:underline}}
 .cap3d-panel .p-grp-caret{{font-size:9px;opacity:.8;transition:transform .1s}}
 .cap3d-panel .p-back{{color:#7cc0ff;cursor:pointer;text-decoration:none;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}}
 .cap3d-panel .p-back:hover{{text-decoration:underline}}
 .cap3d-panel .p-filt{{display:flex;align-items:center;gap:4px;border-bottom:1px solid #262c38;padding-bottom:8px;margin-bottom:8px}}
 .cap3d-panel .p-filt-l{{color:#5d6673;font-size:10px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;margin-right:2px;text-transform:uppercase;letter-spacing:.06em}}
 .cap3d-panel .p-chip{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:10.5px;color:#8a94a6;background:#12161d;border:1px solid #2a3342;border-radius:999px;padding:2px 9px;cursor:pointer}}
 .cap3d-panel .p-chip.on{{color:#0a0c10;background:#e9edf3;border-color:#e9edf3}}
 .cap3d-panel .p-chip:hover{{border-color:#7cc0ff;color:#7cc0ff}}
 .cap3d-panel .p-row{{display:flex;align-items:center;gap:8px;padding:3px 4px;border-radius:4px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:11px}}
 .cap3d-panel .p-row:hover{{background:#161a22}}
 .cap3d-panel .p-row.sel{{background:#1d2430;color:#fff}}
 .cap3d-panel .p-bar{{flex:0 0 42px;height:5px;background:#1a1f28;border-radius:3px;overflow:hidden}}
 .cap3d-panel .p-bar i{{display:block;height:100%;background:#e9edf3}}
</style></head><body>
<div class="layout">
<nav class="side">{tab_html}</nav>
<div class="col">
<header><h1>Atlas Profile Platform — model-atlas</h1>
<div class="sub">{data["meta"]["arch"]} · {data["meta"]["layers"]} layers · {data["meta"]["experts"]} experts · top-{data["meta"]["top_k"]} · seed {data["meta"]["seed"]} — synthetic miniature MoE; all values measured by the F3–F13 runtime. Two surfaces: <em>Profiling</em> (how the model routes &amp; what components carry) and <em>Quantization &amp; Fit</em> (how to shrink it + real-bytes envelopes).</div></header>
<main class="main">
 <div class="panel" id="panel-summary">
   <p>End-to-end parent→derivative Atlas over a genuine synthetic mini-MoE. Everything below is computed by the same measured code paths as the test suite. This is the <strong>Atlas Profile Platform</strong>: use <em>Profiling</em> to understand the model, <em>Quantization &amp; Fit</em> to shrink/score it, and the <strong>Eval Harness</strong> link (bottom of the nav) for independent benchmarking.</p>
 </div>
 <div class="panel" id="panel-capability">
   <p class="note">3D voxel view: one voxel per scored <code>(layer, expert)</code> cell, capability labels run along the depth axis; colour = measured saliency (per-label normalised). <strong>Drag to rotate · scroll to zoom · hover for values.</strong> The tables below are the canonical, agent-readable form — no vision needed to read the data.</p>
   <div class="cap3d-wrap">
     <canvas id="cap3d"
       role="img"
       aria-label="3D voxel saliency map: x-axis = expert, y-axis = layer, depth = capability label; voxel brightness (grayscale ordered dither) = measured saliency. Interact or read the panel and tables for exact values."></canvas>
     <aside class="cap3d-panel" id="cap3d-panel" aria-live="polite">
       <p class="mut">Hover a voxel to inspect · click to pin · drag to rotate · scroll to zoom.</p>
     </aside>
   </div>
   <script type="application/json" id="cap3d-json">{cap3d_json}</script>
   <table id="t-capability"></table>
 </div>
 <div class="panel" id="panel-contrast"><p class="note">success − failure saliency per label (measured); positive = experts more salient on successes.</p><table id="t-contrast"></table></div>
 <div class="panel" id="panel-coalition"><p class="note">Co-routed expert pairs (count) at layer 0 (measured).</p><table id="t-coalition"></table></div>
 <div class="panel" id="panel-path"><p class="note">Most frequent cross-layer route signatures with success rate (measured).</p><table id="t-path"></table></div>
 <div class="panel" id="panel-hierarchy"><p class="note">Six-level atlas hierarchy (v2 §9): L1 weights → L2 units → L3 experts → L4 coalitions → L5 pathways → L6 behaviour, traceable up and down. Per-level node counts are measured; the example shows how many lower-level components realise the first behaviour (and the peak shared-channel prevalence — how load-bearing).</p><div id="hier-stats"></div><table id="t-hierarchy"></table></div>
 <div class="panel" id="panel-compression"><p class="note">Per-expert compression response (int4 vs int8), reconstruction error + output drift (measured math).</p><table id="t-compression"></table></div>
 <div class="panel" id="panel-candidate"><p class="note">Derivative candidates: kept experts/layer, resident bytes per node, go/no-go fit, held-out retention.</p><table id="t-candidate"></table></div>
 <div class="panel" id="panel-heldout"><p class="note">Per-capability held-out retention (derivative vs source), measured.</p><table id="t-heldout"></table></div>
 <div class="panel" id="panel-reality"><p class="note">Real-bytes derivative envelopes (§24/§25) computed from measured checkpoint bytes — the mounted GLM-5.2 NVFP4 when present, else a synthetic caret. Retention fractions are estimates (a routing census needs inference); the byte math is measured.</p>
    <div><span class="stat"><span class="k">source</span><span class="v" id="rl-source"></span></span><span class="stat"><span class="k">measured GiB</span><span class="v" id="rl-total"></span></span></div>
    <table id="t-reality"></table></div>
 <div class="panel" id="panel-maps"><p class="note">§25 planning artifacts. Channel/tile maps are grounded in <em>measured</em> channel-uniqueness (v2 §8.3); node-ownership from the census placement; router-repair preserves expert↔router index coupling (v2 §31:18); overflow/residual/distillation derive from measured saliency. Removal-impact fields are estimates pending causal traces.</p>
    <h3>Channel map</h3><table id="t-channel"></table>
    <h3>Tile map</h3><table id="t-tile"></table>
    <h3>Node ownership</h3><table id="t-ownership"></table>
    <h3>Overflow pack (NVMe)</h3><table id="t-overflow"></table>
    <h3>Router repair</h3><table id="t-router"></table>
    <h3>Residual repair</h3><table id="t-residual"></table>
    <h3>Distillation targets</h3><table id="t-distill"></table>
 </div>
 </main>
</div>
</div>
<script>
 const DATA = {payload};
 function el(t, rows){{ if(!rows||!rows.length){{return "<tr><td class='note'>no data</td></tr>";}}
   return rows.map(r=>{{let tds = Object.entries(r).map(([k,v]) => {{let s = Array.isArray(v)?v.map(x=>`<span class='chip'>${{x}}</span>`).join(''):(typeof v==='boolean'?(v?'<span class=green>yes</span>':'<span class=red>no</span>'):v);
     return `<td>${{s}}</td>`}}).join(''); return `<tr>${{tds}}</tr>`;}}).join("");}}
 function cols(headers){{return "<thead><tr>"+headers.map(h=>`<th scope="col">${{h}}</th>`).join("")+"</tr></thead>";}}
 function fill(id, headers, rows){{document.getElementById(id).innerHTML = "<table><caption>"+(headers.join(' · '))+"</caption>"+cols(headers)+el(null,rows)+"</table>";}}

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
 fill('t-channel', ['layer','expert','chan','importance','keep'], DATA.maps.channel.map(r=>({{'layer':r.layer_index,'expert':r.source_expert_id,'chan':r.channel_id,'importance':r.importance,'keep':r.keep}})));
 fill('t-tile', ['layer','expert','tile','start','importance','keep'], DATA.maps.tile.map(r=>({{'layer':r.layer_index,'expert':r.source_expert_id,'tile':r.tile_index,'start':r.channel_start,'importance':r.importance,'keep':r.keep}})));
 fill('t-ownership', ['tensor','role','layer','expert','node'], DATA.maps.node_ownership.map(r=>({{'tensor':r.tensor_key,'role':r.role,'layer':(r.layer_index??'-'),'expert':(r.source_expert_id??'-'),'node':r.node}})));
 fill('t-overflow', ['layer','expert','tier','reason'], DATA.maps.overflow_pack.map(r=>({{'layer':r.layer_index,'expert':r.source_expert_id,'tier':r.tier,'reason':r.reason}})));
 fill('t-router', ['layer','old','new','action','route_bias'], DATA.maps.router_repair.map(r=>({{'layer':r.layer_index,'old':r.old_index,'new':(r.new_index??'-'),'action':r.action,'route_bias':r.route_bias}})));
 fill('t-residual', ['layer','expert','component','severity','target'], DATA.maps.residual_repair.map(r=>({{'layer':r.layer_index,'expert':r.source_expert_id,'component':r.component,'severity':r.severity,'target':r.target}})));
 fill('t-distill', ['layer','expert','type','priority'], DATA.maps.distillation_target.map(r=>({{'layer':r.layer_index,'expert':r.source_expert_id,'type':r.target_type,'priority':r.priority}})));
 fill('t-hierarchy', ['level','nodes'], DATA.hierarchy.levels.map(l=>({{'level':l, 'nodes':DATA.hierarchy.counts[l]}})));
 document.getElementById('hier-stats').innerHTML =
   (DATA.hierarchy.example && DATA.hierarchy.example.behaviour)
     ? `<span class='stat'><span class='k'>trace-down example</span><span class='v'>${{DATA.hierarchy.example.behaviour}}</span></span>`+
       Object.entries(DATA.hierarchy.example).filter(([k])=>k!=='behaviour').map(([k,v])=>`<span class='stat'><span class='k'>${{k}}</span><span class='v'>${{v}}</span></span>`).join('')
     : '<p class="note">no behaviour nodes measured</p>';
 document.getElementById('rl-source').textContent = DATA.reality.source;
 document.getElementById('rl-total').textContent = DATA.reality.measured_gib + ' GiB';
 fill('t-reality', ['envelope','keep','precision','bpw','stored','resident A','resident B','risk'],
   DATA.reality.candidates.map(r=>({{'envelope':r.envelope+' GiB','keep':(r.keep*100)+'%','precision':r.precision,
     'bpw':r.bpw,'stored':r.stored,'resident A':r.resident_a,'resident B':r.resident_b, 'risk':r.risk}})));

 {_CAP3D_JS}
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
