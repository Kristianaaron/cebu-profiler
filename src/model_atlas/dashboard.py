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
        ranked = [(lay, e, s) for lay, e, s in ranked if s > 0]
        mx = max((s for _, _, s in ranked), default=1)
        rows.append(
            {
                "label": label.value,
                "top": [{"layer": lay, "expert": e, "score": round(s / mx, 3)} for lay, e, s in ranked],
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
// Wheel zooms. Drag to rotate (yaw/pitch).
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

  var SX = 1.7, SY = 2.8, SZ = 2.4, HFE = 0.62;
  var zoom = 1.0, spread = 1.0, ry = 0.70, rx = 0.70, OX = 0, OY = 0;   // zoom, layer spread, rotation + centre
  var hover = null, pin = null, focus = 0, selLayer = null;
  var layerOn = []; for (var _l0 = 0; _l0 < nl; _l0++) layerOn.push(true);
  var cells = [];   // { v, pts[8], nz[6], vispolys[], depth, cx, cy }
  var dragging = false, moved = 0, lastX = 0, lastY = 0;

  var CORNERS = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
  var FACES = [
    { c: [0,1,5,4], n: [0,-1,0] },
    { c: [3,2,6,7], n: [0,1,0] },
    { c: [0,3,7,4], n: [-1,0,0] },
    { c: [1,2,6,5], n: [1,0,0] },
    { c: [0,1,2,3], n: [0,0,-1] },
    { c: [4,5,6,7], n: [0,0,1] }
  ];

  function rot3(x, y, z) {
    var cy = Math.cos(ry), sy = Math.sin(ry), cx = Math.cos(rx), sx = Math.sin(rx);
    var px = x * cy + z * sy, pz = -x * sy + z * cy, py = y;
    var y2 = py * cx - pz * sx, z2 = py * sx + pz * cx;
    return [px, y2, z2];
  }

  function layout() {
    var cw = cv.clientWidth || W, chh = cv.clientHeight || H;
    if (cw !== W || chh !== H) { W = cw; H = chh; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
    var sc = Math.min((W - 70) / (Math.max(ne * SX, nl * SY, labels.length * SZ) + 4), (H - 70) / (nl * SY + labels.length * SZ + 3)) * 0.95 * zoom;
    function projAll(v, ox0, oy0) {
      var cx = (v.expert - (ne - 1) / 2) * SX, cy = -(v.layer - (nl - 1) / 2) * SY * spread, cz = (v.label - (labels.length - 1) / 2) * SZ;
      var rc = rot3(cx, cy, cz), pts = [];
      for (var i = 0; i < 8; i++) { var r = rot3(cx + CORNERS[i][0] * HFE, cy + CORNERS[i][1] * HFE, cz + CORNERS[i][2] * HFE); pts.push([ox0 + r[0] * sc, oy0 + r[1] * sc, r[2] * sc]); }
      return { rc: rc, pts: pts };
    }
    var flat = vox.filter(function (v) { return layerOn[v.layer]; });
    var x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9, i;
    function scan(pp) { if (pp[0] < x0) x0 = pp[0]; if (pp[0] > x1) x1 = pp[0]; if (pp[1] < y0) y0 = pp[1]; if (pp[1] > y1) y1 = pp[1]; }
    for (i = 0; i < flat.length; i++) { var o = projAll(flat[i], 0, 0); for (var j = 0; j < 8; j++) scan(o.pts[j]); }
    OX = W / 2 - (x0 + x1) / 2; OY = H / 2 - (y0 + y1) / 2;
    cells = [];
    for (i = 0; i < flat.length; i++) {
      var v = flat[i], o = projAll(v, OX, OY), nz = [], vis = [];
      for (var f = 0; f < 6; f++) { var nr = rot3(FACES[f].n[0], FACES[f].n[1], FACES[f].n[2]); nz.push(nr[2]); if (nr[2] > 0) vis.push(FACES[f].c.map(function (ci) { return o.pts[ci]; })); }
      cells.push({ v: v, pts: o.pts, nz: nz, vispolys: vis, depth: o.rc[2] * sc, cx: OX + o.rc[0] * sc, cy: OY + o.rc[1] * sc });
    }
  }

  function isOn(v, l) { return v.label === l.label && v.layer === l.layer && v.expert === l.expert; }
  function facePath(pts, f) { return FACES[f].c.map(function (ci) { return pts[ci]; }); }
  function tracePath(poly) { ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]); for (var q = 1; q < 4; q++) ctx.lineTo(poly[q][0], poly[q][1]); ctx.closePath(); }
  function aa(v4) { return Math.max(0, Math.min(1, v4)).toFixed(3); }
  // Translucent light cube faces with faint edges; the selected/hovered cube
  // is much brighter + white double-edge so it clearly stands out.
  function drawCube(cb) {
    var v = cb.v;
    var active = (pin && isOn(v, pin)) || (hover && isOn(v, hover));
    var hot = 0.25 + 0.75 * v.score;
    for (var f = 0; f < 6; f++) {
      if (cb.nz[f] <= 0) continue;
      var shade = 0.3 + 0.7 * Math.max(0, cb.nz[f]);          // face-orientation lighting
      var a = (active ? 0.5 + 0.4 * hot * shade : 0.05 + 0.20 * hot * shade); // low opacity by default
      var poly = facePath(cb.pts, f);
      tracePath(poly);
      ctx.fillStyle = 'rgba(235,235,235,' + aa(a) + ')';
      ctx.fill();
      if (active) { ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 2; }
      else { ctx.strokeStyle = 'rgba(218,218,218,' + aa(0.35 * shade) + ')'; ctx.lineWidth = 1; }
      ctx.stroke();
    }
  }

  function draw() {
    layout();
    ctx.clearRect(0, 0, W, H);   // transparent canvas -> grid + vignette behind
    var order = cells.slice();
    order.sort(function (a, b) { return a.depth - b.depth; });
    for (var i = 0; i < order.length; i++) drawCube(order[i]);
    // subtle floating layer labels (L0, L1, ...) anchored by each layer's centroid
    var lp = [], la;
    for (la = 0; la < nl; la++) lp.push({ x: 0, y: 0, n: 0 });
    for (la = 0; la < cells.length; la++) { var cc = cells[la]; lp[cc.v.layer].x += cc.cx; lp[cc.v.layer].y += cc.cy; lp[cc.v.layer].n++; }
    ctx.font = '10.5px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (la = 0; la < nl; la++) {
      if (!lp[la].n) continue;
      ctx.fillStyle = 'rgba(163,163,163,0.5)';
      ctx.fillText('L' + la, lp[la].x / lp[la].n - 34, lp[la].y / lp[la].n);
    }
    ctx.textBaseline = 'alphabetic';
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
    var r = cv.getBoundingClientRect ? cv.getBoundingClientRect() : { left: 0, top: 0, width: W, height: H };
    var bx = (e.clientX - r.left) * (W / r.width), by = (e.clientY - r.top) * (H / r.height);
    // visually-topmost cube under the cursor: last in painter order containing it
    var order = cells.slice();
    order.sort(function (a, b) { return a.depth - b.depth; });
    for (var i = order.length - 1; i >= 0; i--) {
      var vsps = order[i].vispolys;
      for (var f = 0; f < vsps.length; f++) { if (inPoly(bx, by, vsps[f])) return order[i].v; }
    }
    return null;
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
  function tier(sc) { if (sc >= 0.75) return 'strong'; if (sc >= 0.5) return 'good'; if (sc >= 0.25) return 'moderate'; return 'weak'; }
  function tierBadge(v) { return '<span class="p-tier ' + tier(v.score) + '">' + tier(v.score) + '</span>'; }
  function bar(v) { return '<span class="p-bar"><i style="width:' + Math.round(v.score * 100) + '%"></i></span> ' + Math.round(v.score * 100) + '%'; }
  function rowHtml(v) {
    var isSel = pin && isOn(v, pin);
    var isHov = (!pin) && hover && isOn(v, hover);
    return '<div class="p-row' + (isSel ? ' sel' : isHov ? ' hov' : '') +
      '" onclick="window.capSel(' + v.label + ',' + v.layer + ',' + v.expert + ')">E' + v.expert + ' ' + bar(v) + ' ' + tierBadge(v) + '</div>';
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

  function cur(p) { if (cv.style) cv.style.cursor = dragging ? 'grabbing' : (p ? 'pointer' : 'default'); }
  function onMove(e) {
    layout();              // rebuild cells at the canvas's CURRENT size so pick never drifts from draw
    if (dragging) {
      var dx = e.clientX - lastX, dy = e.clientY - lastY;
      moved += Math.abs(dx) + Math.abs(dy);
      lastX = e.clientX; lastY = e.clientY;
      ry += dx * 0.012; rx += dy * 0.012; rx = Math.max(0.06, Math.min(1.5, rx));
      hover = null; draw(); cur(null);
    } else {
      var p = pick(e);
      hover = p; draw(); renderPanel(); cur(p);
    }
  }
  function onDown(e) {
    dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY;
    if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId);
    cv.classList.add('dragging'); cur(null);
  }
  function onUp(e) {
    if (!dragging) return;
    dragging = false; cv.classList.remove('dragging');
    if (moved < 6) {               // it was a click, not a drag -> toggle pin
      layout(); var p = pick(e); if (p) { pin = (pin && isOn(pin, p)) ? null : p; focus = p.label; draw(); renderPanel(); }
    } else cur(null);
  }
  cv.addEventListener('pointermove', onMove);
  cv.addEventListener('pointerdown', onDown);
  cv.addEventListener('pointerup', onUp);
  cv.addEventListener('pointercancel', function () { dragging = false; cv.classList.remove('dragging'); });
  cv.addEventListener('wheel', function (e) { e.preventDefault(); zoom = Math.max(0.4, Math.min(3.0, zoom * (e.deltaY < 0 ? 1.05 : 0.95))); draw(); }, { passive: false });
  window.addEventListener('resize', function () {
    var W2 = cv.clientWidth || W, H2 = cv.clientHeight || H;
    cv.width = Math.round(W2 * dpr); cv.height = Math.round(H2 * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); W = W2; H = H2; draw();
  });

  var hi = vox[0];
  for (var vi = 1; vi < vox.length; vi++) if (vox[vi].score > hi.score) hi = vox[vi];
  focus = hi ? hi.label : 0;
  draw(); renderPanel();
  (function wireControls() {
    var zi = document.getElementById('czoomin'), zo = document.getElementById('czoomout'), sp = document.getElementById('cspread');
    if (zi) zi.addEventListener('click', function () { zoom = Math.min(3, zoom * 1.25); draw(); });
    if (zo) zo.addEventListener('click', function () { zoom = Math.max(0.4, zoom / 1.25); draw(); });
    if (sp) sp.addEventListener('input', function () { spread = parseFloat(sp.value); draw(); });
  })();
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
                {"id": "capability", "title": "Experts"},
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
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<title>Atlas Lab — model-atlas</title>
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
 body{{font-family:'Inter',ui-sans-serif,system-ui,sans-serif;margin:0;background:#121212;color:#dcdcdc}}
 header{{padding:18px 24px;background:#1b1b1b;border-bottom:1px solid #2e2e2e}}
 h1{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;letter-spacing:-0.01em;font-size:18px;margin:0}}
 .sub{{color:#979797;font-size:12px;margin-top:4px}}
 .layout{{display:flex;min-height:100vh}}
 .col{{flex:1;display:flex;flex-direction:column;min-width:0}}
 nav.side{{width:200px;flex:0 0 200px;display:flex;flex-direction:column;gap:2px;background:#1b1b1b;border-right:1px solid #2e2e2e;padding:14px 10px;position:sticky;top:0;height:100vh;overflow-y:auto;box-sizing:border-box}}
 nav.side .tab{{display:flex;align-items:center;gap:9px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:14px;padding:8px 10px;cursor:pointer;color:#b4b4b4;border-radius:6px}}
 nav.side .tab svg{{flex:0 0 auto}}
 nav.side .tab:hover{{background:#262626}}
 nav.side .tab.active{{color:#dfdfdf;background:#262626}}
 main.main{{flex:1;padding:22px 26px}}
 .panel{{display:none}} .panel.active{{display:block}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #2b2b2b}}
 th{{color:#979797;font-weight:600}}
 .chip{{display:inline-block;background:#262626;border:1px solid #353535;border-radius:4px;padding:2px 8px;margin:2px;font-size:12px}}
 .cap-tbl-head{{display:flex;justify-content:flex-end;align-items:center;gap:8px;margin:14px 0 20px}}
 #cap-sort{{background:#121519;color:#aeb5bf;border:1px solid #343a42;border-radius:6px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;padding:4px 24px 4px 10px;cursor:pointer}}
 #cap-sort option{{background:#121519;color:#b6bdc7}}
 .cap-sort-lbl{{color:#7d848e;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
 .cap-more-wrap{{display:flex;justify-content:center;margin:8px 0 2px}}
 #cap-more{{background:transparent;border:1px solid #3a3a3a;color:#cfd6e0;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;padding:5px 16px;border-radius:6px;cursor:pointer}}
 #cap-more:hover{{border-color:#6b6b6b;color:#fff}}
 #cap-filter-btn{{background:transparent;border:1px solid #3a3a3a;color:#cfd6e0;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;padding:4px 12px;border-radius:6px;cursor:pointer}}
 #cap-filter-btn:hover{{border-color:#6b6b6b;color:#fff}}
 #cap-fcount{{color:#979797;margin-left:6px;font-size:11px}}
 .cap-tray-bd{{position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:59;opacity:0;pointer-events:none;transition:opacity .2s}}
 .cap-tray-bd.show{{opacity:1;pointer-events:auto}}
 .cap-tray{{position:fixed;top:0;right:0;height:100vh;width:280px;background:#101316;border-left:1px solid #3a3a3a;z-index:60;transform:translateX(105%);transition:transform .22s ease;display:flex;flex-direction:column;box-shadow:-10px 0 34px rgba(0,0,0,0.5)}}
 .cap-tray.open{{transform:translateX(0)}}
 .cap-tray-head{{display:flex;justify-content:space-between;align-items:center;padding:14px;border-bottom:1px solid #2e2e2e;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:13px;color:#e9edf3}}
 .cap-tray-head button{{background:none;border:none;color:#979797;font-size:18px;cursor:pointer;line-height:1}}
 .cap-tray-head button:hover{{color:#fff}}
 .cap-tray-body{{overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:10px}}
 .cap-tray-body label{{display:flex;justify-content:space-between;align-items:center;color:#cfd6e0;font-size:12.5px;cursor:pointer;gap:8px}}
 .cap-tray-body input[type=checkbox]{{accent-color:#cfd6e0;width:14px;height:14px}}
 .green{{color:#cdcdcd}} .amber{{color:#999999}} .red{{color:#5b5b5b}}
 .note{{color:#979797;font-size:12px}}
 .panel h3{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:13px;color:#b4b4b4;margin:18px 0 6px}}
 .navsec{{color:#676767;font-size:10.5px;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;margin:14px 10px 4px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}}
 .navlink{{display:block;margin:14px 10px 0;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;color:#dfdfdf;text-decoration:none;padding:6px 0;border-top:1px solid #2e2e2e}}
 .navlink:hover{{color:#f2f2f2}}
 .stat{{display:inline-block;background:#262626;border:1px solid #353535;border-radius:6px;padding:10px 14px;margin:4px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}}
 .stat .k{{color:#979797;font-size:11px;display:block}} .stat .v{{font-size:18px;color:#dcdcdc}}
 .cap3d-wrap{{position:relative;display:flex;gap:12px;align-items:stretch;border:1px solid #2e2e2e;border-radius:8px;padding:10px;
   background:#080808;
   background-image:linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
   background-size:48px 48px,48px 48px}}
 .cap3d-canvas{{position:relative;flex:1;min-width:0}}
 canvas#cap3d{{width:100%;aspect-ratio:680/420;height:auto;display:block;touch-action:none;cursor:default;border-radius:6px;background:transparent}}
 .cap3d-controls{{position:absolute;top:6px;left:6px;z-index:3;display:flex;align-items:center;gap:4px;background:rgba(14,14,14,0.7);border:1px solid #444444;border-radius:6px;padding:3px 4px}}
 .cap3d-controls button{{width:19px;height:19px;display:inline-flex;align-items:center;justify-content:center;padding:0;color:#d7d7d7;background:#171717;border:1px solid #444444;border-radius:4px;cursor:pointer}}
 .cap3d-controls button:hover{{border-color:#646464;color:#fff;background:#202020}}
 .cap3d-controls .cspread{{display:flex;align-items:center;gap:4px;color:#979797;font-size:9.5px;letter-spacing:.03em;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;margin:0 2px}}
 .cap3d-controls input[type=range]{{-webkit-appearance:none;appearance:none;width:90px;height:14px;background:transparent;cursor:pointer}}
 .cap3d-controls input[type=range]::-webkit-slider-runnable-track{{height:1px;background:#444444;border-radius:1px}}
 .cap3d-controls input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;appearance:none;width:8px;height:8px;border-radius:50%;background:#d7d7d7;border:none;margin-top:-3.5px}}
 .cap3d-controls input[type=range]::-moz-range-track{{height:1px;background:#444444;border:none}}
 .cap3d-controls input[type=range]::-moz-range-thumb{{width:8px;height:8px;border:none;border-radius:50%;background:#d7d7d7}}
 .cap3d-vig{{position:absolute;inset:0;pointer-events:none;border-radius:6px;background:radial-gradient(ellipse 72% 68% at 50% 48%, transparent 42%, rgba(8,8,8,0.5) 74%, #080808 100%)}}
 canvas#cap3d.dragging{{cursor:grabbing}}
 .cap3d-panel{{flex:0 0 300px;align-self:stretch;background:#0d0d0d;border-left:1px solid #2e2e2e;padding-left:12px;overflow-y:auto;font-size:12px;color:#d7d7d7}}
 .cap3d-panel .p-head{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12.5px;color:#eeeeee;margin:2px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
 .cap3d-panel .p-sub{{color:#676767;font-size:10.5px;margin:2px 0 8px}}
 .cap3d-panel .p-grp{{color:#dfdfdf;font-size:11px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;margin:8px 0 2px;cursor:pointer;display:flex;align-items:center;gap:6px}}
 .cap3d-panel .p-grp:hover{{color:#f2f2f2;text-decoration:underline}}
 .cap3d-panel .p-grp-caret{{font-size:9px;opacity:.8;transition:transform .1s}}
 .cap3d-panel .p-back{{color:#dfdfdf;cursor:pointer;text-decoration:none;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}}
 .cap3d-panel .p-back:hover{{text-decoration:underline}}
 .cap3d-panel .p-filt{{display:flex;align-items:center;gap:4px;border-bottom:1px solid #2e2e2e;padding-bottom:8px;margin-bottom:8px}}
 .cap3d-panel .p-filt-l{{color:#676767;font-size:10px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;margin-right:2px;text-transform:uppercase;letter-spacing:.06em}}
 .cap3d-panel .p-chip{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:10.5px;color:#979797;background:#171717;border:1px solid #353535;border-radius:999px;padding:2px 9px;cursor:pointer}}
 .cap3d-panel .p-chip.on{{color:#0d0d0d;background:#eeeeee;border-color:#eeeeee}}
 .cap3d-panel .p-chip:hover{{border-color:#dfdfdf;color:#dfdfdf}}
 .cap3d-panel .p-row{{display:flex;align-items:center;gap:8px;padding:3px 4px;border-radius:4px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:11px}}
 .cap3d-panel .p-row:hover{{background:#1b1b1b}}
 .cap3d-panel .p-row.sel{{background:#262626;color:#fff}}
 .cap3d-panel .p-bar{{flex:0 0 42px;height:5px;background:#202020;border-radius:3px;overflow:hidden}}
 .cap3d-panel .p-bar i{{display:block;height:100%;background:#eeeeee}}
 .cap3d-panel .p-tier{{flex:0 0 auto;margin-left:auto;font-size:9.5px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;text-transform:uppercase;letter-spacing:.04em;padding:1px 5px;border-radius:3px;border:1px solid currentColor;opacity:.9}}
 .cap3d-panel .p-tier.strong{{color:#e2b45c}}
 .cap3d-panel .p-tier.good{{color:#c6cdd8}}
 .cap3d-panel .p-tier.moderate{{color:#b08e6b}}
 .cap3d-panel .p-tier.weak{{color:#d0686b}}
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
   <p class="note">Experts × layers saliency map: one dithered cube per scored <code>(layer, expert)</code> cell, capability labels run along the depth axis; brightness = measured saliency (per-label normalised).</p>
   <div class="cap3d-wrap">
     <div class="cap3d-canvas">
       <canvas id="cap3d"
         role="img"
         aria-label="3D voxel saliency map: x-axis = expert, y-axis = layer, depth = capability label; voxel brightness (grayscale ordered dither) = measured saliency. Interact or read the panel and tables for exact values."></canvas>
       <div class="cap3d-controls">
         <button id="czoomout" title="zoom out"><svg viewBox="0 0 12 12" width="9" height="9" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"><path d="M2.5 6h7"/></svg></button>
         <button id="czoomin" title="zoom in"><svg viewBox="0 0 12 12" width="9" height="9" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"><path d="M6 2.5v7M2.5 6h7"/></svg></button>
         <label class="cspread">layers
           <input id="cspread" type="range" min="1" max="4" step="0.1" value="1" title="spread the layers apart to hover each expert">
         </label>
       </div>
       <div class="cap3d-vig"></div>
     </div>
     <aside class="cap3d-panel" id="cap3d-panel" aria-live="polite">
       <p class="mut">Hover a voxel to inspect · click to pin.</p>
     </aside>
   </div>
   <script type="application/json" id="cap3d-json">{cap3d_json}</script>
   <div class="cap-tbl-head">
     <label class="cap-sort-lbl" for="cap-sort">sort</label>
     <select id="cap-sort" title="sort"><option value="strength">strongest → weakest</option><option value="score">score</option><option value="name">category</option></select>
     <button id="cap-filter-btn" title="filter domains">Filter <span id="cap-fcount"></span></button>
   </div>
   <div class="cap-tray-bd" id="cap-tray-bd"></div>
   <div class="cap-tray" id="cap-tray"><div class="cap-tray-head"><span>Domains</span><button id="cap-tray-close" title="close">&#215;</button></div><div class="cap-tray-body" id="cap-domain-filters"></div></div>
   <table id="t-capability"></table>
   <div class="cap-more-wrap"><button id="cap-more">Load more</button></div>
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

 function tierOf(s){{return s>=0.75?'strong':s>=0.5?'good':s>=0.25?'moderate':'weak';}}
 function sc(s){{return s>=0.75?'#e2b45c':s>=0.5?'#c6cdd8':s>=0.25?'#b08e6b':'#d0686b';}}
 var CAP_COLS = ['category','layer','expert','score','strength'];
 var CAP_ROWS = DATA.capability.flatMap(r=>r.top.map(x=>({{category:r.label, layer:'L'+x.layer, expert:'E'+x.expert, scoreNum:x.score, score:Math.round(x.score*100)+'%', strength:'<span style="color:'+sc(x.score)+'">'+tierOf(x.score)+'</span>'}})));
 var capOn = {{}}; DATA.capability.forEach(function(r){{ capOn[r.label]=true; }});
 function tierRank(s){{ return s>=0.75?0:s>=0.5?1:s>=0.25?2:3; }}
 var sortMode = 'strength';
 var capLimit = 20, capStep = 20;
 function renderCap(){{
   var rows = CAP_ROWS.filter(function(r){{ return capOn[r.category]; }});
   if (sortMode !== 'name') {{
     rows = rows.slice().sort(function(a,b){{
       if (sortMode === 'score') return b.scoreNum - a.scoreNum;
       return (tierRank(a.scoreNum)-tierRank(b.scoreNum)) || (b.scoreNum-a.scoreNum);
     }});
   }}
   var total = rows.length, shown = rows.slice(0, capLimit);
   document.getElementById('t-capability').innerHTML = "<table>"+cols(CAP_COLS)+el(null,shown)+"</table>";
   var mo = document.getElementById('cap-more');
   if (mo) mo.style.display = (shown.length < total && total > capStep) ? '' : 'none';
 }}
 (function wireDom(){{
   var btn = document.getElementById('cap-filter-btn'), tray = document.getElementById('cap-tray'), bd = document.getElementById('cap-tray-bd'), close = document.getElementById('cap-tray-close'), fcount = document.getElementById('cap-fcount');
   var boxf = document.getElementById('cap-domain-filters');
   var sortSel = document.getElementById('cap-sort');
   if (!boxf) return;
   if (sortSel) sortSel.addEventListener('change', function(){{ sortMode = sortSel.value; capLimit = capStep; renderCap(); }});
   var more = document.getElementById('cap-more');
   if (more) more.addEventListener('click', function(){{ capLimit += capStep; renderCap(); }});
   function countSel(){{ var n=0,tot=0,k; for(k in capOn){{ tot++; if(capOn[k]) n++; }} if (fcount) fcount.textContent=n+'/'+tot; }}
   boxf.innerHTML = DATA.capability.map(function(r){{ return '<label><span>'+r.label+'</span><input type="checkbox" '+(capOn[r.label]?'checked':'')+' data-d="'+r.label+'"></label>'; }}).join('');
   boxf.querySelectorAll('input[type=checkbox]').forEach(function(ch){{
     ch.addEventListener('change', function(){{ capOn[ch.dataset.d]=ch.checked; capLimit=capStep; renderCap(); countSel(); }});
   }});
   countSel();
   function openTray(o){{ tray.classList.toggle('open',o); bd.classList.toggle('show',o); }}
   if (btn) btn.addEventListener('click', function(){{ openTray(true); }});
   if (close) close.addEventListener('click', function(){{ openTray(false); }});
   if (bd) bd.addEventListener('click', function(){{ openTray(false); }});
   countSel();
 }})();
 renderCap();
 fill('t-contrast', ['label','top success−failure (delta)'],
   DATA.contrast.map(r=>({{label:r.label, top:r.top.map(x=>`L${{x.layer}}E${{x.expert}} (${{x.delta}})`).join(' · ')}})));
 fill('t-coalition', ['pair','coactivity'], DATA.coalitions.map(r=>({{pair:`L0E${{r.pair[0]}} / L0E${{r.pair[1]}}`, coactivity:r.coactivity}})));
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
