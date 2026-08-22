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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_atlas.atlas.hierarchy import build_hierarchy
from model_atlas.atlas.pathways import path_stats
from model_atlas.atlas.reap import (
    CalibrationSample,
    SaliencyAccumulator,
    make_synthetic_corpus,
    run_calibration,
    run_contrast,
)
from model_atlas.atlas.runtime import MiniMoE, build_mini_moe, forward
from model_atlas.builder import build_derivative
from model_atlas.compression import expert_response_curve, get_backend_registry
from model_atlas.evaluation import detect_leakage, evaluate_heldout, promote_allowed
from model_atlas.ops.maintenance_watch import read_events, render_maintenance_status
from model_atlas.planning import SearchInputs, generate_candidates
from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.evidence import EvidenceKind
from model_atlas.schemas.ontology import CapabilityLabel, SuccessState

SEED = 0
ARCH = get_registry().get("k3-mini")


def _latest_maintenance_journal() -> Path | None:
    """Resolve the most recent maintenance run's journal dir (optional).

    Env ``ATLAS_MAINTENANCE_JOURNAL_DIR`` overrides; otherwise take the newest
    ``controlplane_maintenance/*/`` run dir. Returns None when no run exists.
    """
    env = os.environ.get("ATLAS_MAINTENANCE_JOURNAL_DIR")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    base = Path("controlplane_maintenance")
    if base.is_dir():
        hits = sorted(
            base.rglob("maintenance-events.jsonl"),
            key=lambda pth: pth.stat().st_mtime,
            reverse=True,
        )
        if hits:
            return hits[0].parent
    return None


def _maintenance_payload() -> dict[str, Any]:
    """Maintenance lifecycle status for the GUI (from the live event stream)."""
    journal = _latest_maintenance_journal()
    if journal is None:
        return {"present": False}
    path = journal / "maintenance-events.jsonl"
    if not path.exists():
        return {"present": False, "journal": str(journal)}
    with open(path, encoding="utf-8") as fh:
        raw = [ln for ln in fh]
    events = read_events(path)
    phases = ("drain", "produce", "restore", "maintenance")
    current = next(
        (p for p in reversed(phases) if any(e["phase"] == p for e in events)),
        "idle",
    )
    released = sorted(
        {e["service"] for e in events if e["phase"] == "drain" and e["status"] == "release"}
    )
    loaded = sorted(
        {e["service"] for e in events if e["phase"] == "restore" and e["status"] == "load"}
    )
    cur = 0
    total = 0
    for e in events:
        sc = e.get("shard_current")
        st = e.get("shard_total")
        if isinstance(sc, int):
            cur = max(cur, sc)
        if isinstance(st, int):
            total = max(total, st)
    produce = next(
        (e.get("method") for e in events if e["phase"] == "produce" and e["status"] == "start"),
        None,
    )
    done = next(
        (e.get("detail") for e in events if e["phase"] == "maintenance" and e["status"] == "complete"),
        None,
    )

    # Timing affordances: run/phase elapsed (live) + an honest remaining estimate.
    now_epoch = datetime.now(UTC).timestamp()

    def _ts_epoch(ts: object) -> float | None:
        if not isinstance(ts, str):
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    started: list[float] = []
    for e in events:
        if (ts := _ts_epoch(e.get("ts"))) is not None:
            started.append(ts)
    run_started_epoch = min(started, default=now_epoch)
    phase_starts: list[float] = []
    for e in events:
        if e.get("phase") == current and (ts := _ts_epoch(e.get("ts"))) is not None:
            phase_starts.append(ts)
    phase_start_epoch = min(phase_starts, default=run_started_epoch)
    elapsed = max(0.0, now_epoch - run_started_epoch)
    phase_elapsed = max(0.0, now_epoch - phase_start_epoch)
    # Honest conservative total split (labeled estimate in the UI): drain ~1m,
    # produce ~20m (width-slice/quant can vary), restore ~7m => ~28m.
    phase_duration_s = {"drain": 60, "produce": 1200, "restore": 420, "maintenance": 60}
    estimated_total = sum(phase_duration_s[p] for p in phases if p != "maintenance")
    remaining = max(0.0, float(estimated_total) - elapsed)

    return {
        "present": True,
        "journal": str(journal),
        "phase": current,
        "status": render_maintenance_status(raw),
        "released": released,
        "loaded": loaded,
        "shard_current": cur,
        "shard_total": total,
        "produce_method": produce,
        "result": done,
        "run_started_epoch": round(run_started_epoch, 3),
        "elapsed_seconds": int(elapsed),
        "phase_elapsed_seconds": int(phase_elapsed),
        "phase_remaining_seconds": int(max(0.0, phase_duration_s.get(current, 120) - phase_elapsed)),
        "estimated_total_seconds": estimated_total,
        "eta_remaining_seconds": int(remaining),
        "phase_duration_s": phase_duration_s,
    }


# Lucide icon inner-SVG (viewBox 0 0 24 24) for each side-nav tab.
_LUCIDE = "fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\""
_ICONS: dict[str, str] = {
    "summary": '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
    "capability": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "contrast": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "coalition": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>',
    "path": '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
    "structure": '<path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h10"/>',
    "compression": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "candidate": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "heldout": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1 1 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "maps": '<polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21 3 6"/><polyline points="3 6 9 9 15 6 21 9"/><line x1="9" x2="9" y1="9" y2="18"/>',
    "pareto": '<path d="M3 3v18h18"/><path d="M3 17 9 11 13 15 21 7"/>',
    "v3": '<circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/><path d="M2 12h20"/>',
    "candidates": '<path d="M6 3h12l-2 2H8z"/><path d="M6 21h12l-2-2H8z"/><path d="M5 7v10M19 7v10"/><circle cx="12" cy="12" r="3"/>',
    "corpus": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    "hierarchy": '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="2"/><circle cx="4" cy="12" r="2"/><circle cx="4" cy="18" r="2"/>',
    "reality": '<path d="M12 2v4M12 18v4M2 12h4M18 12h4"/><circle cx="12" cy="12" r="6"/>',
    "maintenance": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
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


def _contrast_voxels(model: MiniMoE, contrast: Any) -> dict[str, Any]:
    """Signed success−failure saliency per (layer, expert), per capability label.

    delta = pos_saliency − neg_saliency; positive favours success, negative
    favours failure. Normalised per label for a diverging display. ``pos`` /
    ``neg`` are the raw mean saliencies on successful vs failed runs (the two
    components Δ is derived from).
    """
    labels = [lbl.value for lbl in list(CapabilityLabel)[:12]]
    n_layers = len(model.layers)
    n_exp = model.n_exp
    voxels: list[dict[str, Any]] = []
    for li, label in enumerate(list(CapabilityLabel)[:12]):
        rows = contrast.contrast(
            label, SuccessState.SUCCESS, SuccessState.FAILURE, topk=n_layers * n_exp
        )
        by_cell = {(lay, e): d for lay, e, d in rows}
        mx = max((abs(d) for d in by_cell.values()), default=0.0) or 1.0
        for lay in range(n_layers):
            for e in range(n_exp):
                d = by_cell.get((lay, e), 0.0)
                pos, neg = contrast.cell_saliency(
                    label, lay, e, SuccessState.SUCCESS, SuccessState.FAILURE
                )
                voxels.append(
                    {
                        "label": li,
                        "layer": lay,
                        "expert": e,
                        "delta": round(d / mx, 3),
                        "pos": round(pos, 4),
                        "neg": round(neg, 4),
                    }
                )
    return {"labels": labels, "n_layers": n_layers, "n_experts": n_exp, "voxels": voxels}


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
    coalitions: list[dict[str, Any]] = [
        {"pair": [a, b], "coactivity": c}
        for (a, b), c in sorted(cmap.pair_counts.items(), key=lambda x: -x[1])[:12]
    ]
    try:
        # how often each expert appears in ANY route across the corpus (solo presence)
        from collections import Counter

        from model_atlas.atlas.coalition import pairwise_causal

        active = Counter[int]()
        for s in corpus:
            tr = forward(model, s.tokens, top_k=2)
            for t in tr.traces[0].topk_ids:
                active.update(set(t))
        coals_with_synergy: list[dict[str, Any]] = []
        for base in coalitions:
            a, b = base["pair"]
            ana = pairwise_causal(model, corpus, layer=0, a=a, b=b)
            syn = ana.synergy_ab
            # gate causal tags on a magnitude well above float noise (~1e-7):
            # on this synthetic miniature removals have ~zero effect, so nothing
            # crosses even a 1e-3 bar yet — no misleading HOT/redundant badges.
            meaningful = abs(syn) > 1e-3
            coals_with_synergy.append(
                {
                    "pair": [a, b],
                    "coactivity": base["coactivity"],
                    "activeA": int(active[a]),
                    "activeB": int(active[b]),
                    "EA": round(float(ana.effect_a), 5) if meaningful else 0.0,
                    "EB": round(float(ana.effect_b), 5) if meaningful else 0.0,
                    "EAb": round(float(ana.effect_ab), 5) if meaningful else 0.0,
                    "synergy": round(syn, 5) if meaningful else 0.0,
                    "catastrophic": bool(ana.catastrophic) and meaningful,
                    "redundant": bool(ana.redundant) and meaningful,
                    "causal": meaningful,
                }
            )
        coalitions = coals_with_synergy
    except Exception:  # never break the dashboard over coalition extras
        pass

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
    from model_atlas.planning.maps_build import (
        build_planning_maps,
        build_real_planning_maps,
    )

    _REAL = "/media/glm52/models/nvidia/GLM-5.2-NVFP4"
    maps_real = None
    if os.path.isfile(os.path.join(_REAL, "config.json")):
        try:
            maps_real = build_real_planning_maps(_REAL)
        except Exception:  # never break the dashboard over the maps section
            maps_real = None
    maps = build_planning_maps(model, saliency) if maps_real is None else maps_real
    maps_payload = {
        "source": maps.model,
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
        "contrast3d": _contrast_voxels(model, contrast),
        "coalitions": coalitions,
        "paths": paths,
        "compression": compression,
        "candidates": candidates,
        "heldout": heldout_rows or [],
        "maps": maps_payload,
        "hierarchy": hierarchy_payload,
        "reality": reality_payload,
        "ecosystem": ecosystem_payload,
        "pareto": '<path d="M3 3v18h18"/><path d="M3 17 9 11 13 15 21 7"/>',
    "v3": _v3_pipeline_payload(model, corpus, seed),
        "candidates_graph": _candidate_graph_payload(model, corpus, seed),
        "corpus": _corpus_payload(model, corpus, seed),
    }

def _candidate_graph_payload(model: MiniMoE, corpus: list[CalibrationSample], seed: int) -> dict[str, Any]:
    """Demonstration candidate graph with predicted-vs-measured discipline."""
    from model_atlas.candidates import (
        CandidateGraph,
        CandidateMetricSet,
        CandidateNode,
        CandidateStage,
        OperatorKind,
    )

    g = CandidateGraph(source_teacher_id=f"teacher-{model.arch.name}")
    g.add(
        CandidateNode(
            candidate_id="teacher",
            name="BF16 teacher",
            stage=CandidateStage.P0_REFERENCE,
            predicted=False,
            deployed=True,
            quality_vector=CandidateMetricSet(quality_retention=1.0, evidence_kind=EvidenceKind.MEASURED),
        )
    )
    g.add(
        CandidateNode(
            candidate_id="mk-exl3",
            name="EXL3 global allocation",
            parent_ids=["teacher"],
            stage=CandidateStage.P4_EXL3,
            predicted=True,
            operators=[OperatorKind.GLOBAL_BIT_BUDGET, OperatorKind.EXL3_EXPRESS],
            quality_vector=CandidateMetricSet(quality_retention=0.98, evidence_kind=EvidenceKind.PREDICTED),
        )
    )
    g.add(
        CandidateNode(
            candidate_id="mk-exl3-nvfp4",
            name="+NVFP4 substitution",
            parent_ids=["mk-exl3"],
            stage=CandidateStage.P6_SM121_ALLOCATION,
            predicted=True,
            operators=[OperatorKind.NVFP4_SUITABILITY],
            quality_vector=CandidateMetricSet(quality_retention=0.95, evidence_kind=EvidenceKind.PREDICTED),
        )
    )
    return g.model_dump(mode="json")

def _corpus_payload(model: MiniMoE, corpus: list[CalibrationSample], seed: int) -> dict[str, Any]:
    from model_atlas.analysis import build_corpus_semantic_map, project_corpus_delta
    from model_atlas.schemas.coverage import EvidenceGate

    report = build_corpus_semantic_map(model, corpus, top_k=2, gate=EvidenceGate())
    project_corpus_delta(report, candidate_id="mk-exl3", per_sample_delta={0: -0.05, 1: -0.03})
    return report.model_dump(mode="json")

def _v3_pipeline_payload(model: MiniMoE, corpus: list[CalibrationSample], seed: int) -> dict[str, Any]:
    """Run the canonical v3 pipeline and emit a JSON-safe payload for the
    V3 dashboard surface. Predictions are never styled as measured."""
    from model_atlas.atlas.v3_pipeline import run_v3_pipeline, v3_run_to_jsonable

    run = run_v3_pipeline(model, corpus, seed=seed)
    return v3_run_to_jsonable(run)


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
    var floorPad = Math.min(46, Math.max(18, H * 0.09)) * zoom;
    var midY = (y0 + y1) / 2;
    var oy = H / 2 - midY;
    if (y1 + oy > H - floorPad) oy = H - floorPad - y1;   // push up so stack bottoms sit above the floor line
    OX = W / 2 - (x0 + x1) / 2; OY = oy;
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

  function polyPath(poly) { ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]); for (var i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]); ctx.closePath(); }
  function centroid(cell) { var sx=0, sy=0; for (var i=0;i<cell.pts.length;i++){sx+=cell.pts[i][0];sy+=cell.pts[i][1];} return [sx/cell.pts.length, sy/cell.pts.length]; }

  function drawFloorAndAxes() {
    if (!cells.length) return;
    // ground the scene: a soft floor quad under everything + faint axis rails
    var minY = 1e9; for (var i = 0; i < cells.length; i++) { for (var j = 0; j < 8; j++) if (cells[i].pts[j][1] > minY) minY = cells[i].pts[j][1]; }
    var xs = [], zs = [];
    for (i = 0; i < ne; i++) xs.push((i - (ne - 1) / 2) * SX);
    for (i = 0; i < labels.length; i++) zs.push((i - (labels.length - 1) / 2) * SZ);
    function P(x, z) { var r = rot3(x, (nl - 1) / 2 * SY * spread * -1, z); return [OX + r[0] * sc(), OY + r[1] * sc()]; }
    // NOTE: y of floor uses bottom layer level so the plate hugs the stack bases
    var corners = [P(xs[0] - HFE, zs[0] - HFE), P(xs[xs.length-1] + HFE, zs[0] - HFE), P(xs[xs.length-1] + HFE, zs[zs.length-1] + HFE), P(xs[0] - HFE, zs[zs.length-1] + HFE)];
    polyPath(corners);
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1;
    ctx.stroke();
    // axis rails along the two diagonals with end ticks
    var a1 = P(xs[0] - HFE, zs[0] - HFE), b1 = P(xs[xs.length-1] + HFE, zs[0] - HFE);
    var a2 = P(xs[0] - HFE, zs[0] - HFE), b2 = P(xs[0] - HFE, zs[zs.length-1] + HFE);
    ctx.strokeStyle = 'rgba(230,230,230,0.22)';
    ctx.beginPath(); ctx.moveTo(a1[0], a1[1]); ctx.lineTo(b1[0], b1[1]); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(a2[0], a2[1]); ctx.lineTo(b2[0], b2[1]); ctx.stroke();
    ctx.font = '9.5px system-ui'; ctx.fillStyle = 'rgba(160,160,160,0.8)';
    ctx.textAlign = 'left';
    ctx.fillText('experts \u2192', (b1[0]+6), b1[1]);
    ctx.fillText('capabilities \u2192', (b2[0]-4), b2[1]+12);
  }

  function sc() { return Math.min((W - 70) / (Math.max(ne * SX, nl * SY, labels.length * SZ) + 4), (H - 70) / (nl * SY + labels.length * SZ + 3)) * 0.95 * zoom; }


  function drawRisers() {
    // vertical hairlines from each stack base to the floor: anchors columns in 3D
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    var seen = {};
    for (var i = 0; i < cells.length; i++) {
      var v = cells[i].v, key = v.expert + ':' + v.label;
      if (seen[key]) continue;
      seen[key] = true;
      var topY = null;
      for (var l = 0; l < nl; l++) { if (!layerOn[l]) continue; break; }
      var xw = (v.expert - (ne - 1) / 2) * SX, zw = (v.label - (labels.length - 1) / 2) * SZ;
      var yTop = -((nl - 1) / 2) * SY * spread + HFE;
      var yBot = ((nl - 1) / 2) * SY * spread + HFE;
      var p1r = rot3(xw, yTop, zw), p2r = rot3(xw, yBot, zw);
      ctx.beginPath();
      ctx.moveTo(OX + p1r[0] * sc(), OY + p1r[1] * sc());
      ctx.lineTo(OX + p2r[0] * sc(), OY + p2r[1] * sc());
      ctx.stroke();
    }
  }

  function draw() {
    layout();
    ctx.clearRect(0, 0, W, H);   // transparent canvas -> grid + vignette behind
    drawFloorAndAxes();
    drawRisers();
    var order = cells.slice();
    order.sort(function (a, b) { return a.depth - b.depth; });
    drawLayerPlatesUnder(order);   // paints each layer's plate then its cubes, back to front
    drawLayerLabels(order);
  }

  function drawLayerPlatesUnder(order) {
    // interleave: for each layer (back to front), its plate then its cubes
    var byLayer = {};
    order.forEach(function (c) { (byLayer[c.v.layer] = byLayer[c.v.layer] || []).push(c); });
    var layersSorted = Object.keys(byLayer).map(Number).sort(function (a, b) { return b - a; });
    layersSorted.forEach(function (l) {
      // plate for this layer
      if (layerOn[l]) {
        var yLevel = -((l - (nl - 1) / 2) * SY * spread);
        var xs0 = -(ne / 2) * SX, xs1 = ((ne - 1) - (ne - 1) / 2) * SX;
        var zs0 = -(labels.length / 2) * SZ, zs1 = ((labels.length - 1) - (labels.length - 1) / 2) * SZ;
        var pts = [];
        [[xs0 - HFE*1.6, zs0 - HFE*1.6],[xs1 + HFE*1.6, zs0 - HFE*1.6],[xs1 + HFE*1.6, zs1 + HFE*1.6],[xs0 - HFE*1.6, zs1 + HFE*1.6]].forEach(function (c) {
          var r = rot3(c[0], yLevel + HFE, c[1]); pts.push([OX + r[0] * sc(), OY + r[1] * sc()]);
        });
        polyPath(pts);
        var dimmed = (selLayer && selLayer.layer !== l);
        ctx.fillStyle = dimmed ? 'rgba(125,211,252,0.015)' : 'rgba(125,211,252,0.032)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(125,211,252,' + ((selLayer && selLayer.layer === l) ? '0.38' : '0.13') + ')';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      byLayer[l].forEach(drawCube);
    });
  }

  function drawLayerLabels(order) {
    // crisp L-tags pinned to the LEFT edge of each layer's plate
    ctx.font = 'bold 10px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (var l = 0; l < nl; l++) {
      if (!layerOn[l]) continue;
      var yLevel = -((l - (nl - 1) / 2) * SY * spread);
      var r = rot3(-(ne / 2) * SX - HFE * 2.4, yLevel + HFE, -(labels.length / 2) * SZ);
      var x = OX + r[0] * sc(), y = OY + r[1] * sc();
      var activeSel = selLayer && selLayer.layer === l;
      ctx.fillStyle = activeSel ? 'rgba(235,235,235,0.95)' : 'rgba(150,160,172,0.75)';
      ctx.fillText('L' + l, x, y);
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
  (function drawLegend() {
    var host = cv.closest('.cap3d-canvas') || cv.parentNode;
    if (!host || document.getElementById('cap3d-legend')) return;
    var lg = document.createElement('div');
    lg.id = 'cap3d-legend';
    lg.style.cssText = 'display:flex;align-items:center;gap:10px;margin:6px 2px 0;font-family:system-ui;font-size:9.5px;color:#8b93a1;';
    var grad = 'linear-gradient(90deg,rgba(235,235,235,0.08),rgba(235,235,235,0.95))';
    lg.innerHTML = '<span style="letter-spacing:.04em">dim</span>'
      + '<span style="flex:0 0 90px;height:6px;border-radius:3px;background:' + grad + '"></span>'
      + '<span>bright&nbsp;= saliency share</span>'
      + '<span style="margin-left:auto;color:#5c6672" class="cap3d-hint">drag to orbit \u00B7 wheel to zoom</span>';
    host.appendChild(lg);
  })();
  (function wireControls() {
    var zi = document.getElementById('czoomin'), zo = document.getElementById('czoomout'), sp = document.getElementById('cspread');
    if (zi) zi.addEventListener('click', function () { zoom = Math.min(3, zoom * 1.25); draw(); });
    if (zo) zo.addEventListener('click', function () { zoom = Math.max(0.4, zoom / 1.25); draw(); });
    if (sp) sp.addEventListener('input', function () { spread = parseFloat(sp.value); draw(); });
  })();
})();
"""


_CONTRAST_JS = r"""
// Success−Failure view — green/red hashed tiles, click-to-pin, no empty state.
// Each (label, layer, expert) cell encodes success−failure saliency:
//   * GREEN tile  -> success-favoured (routing pulls for successes)
//   * RED tile    -> failure-favoured (routing matters for failures)
//   * grey        -> neutral; colour depth = |delta|
// Hover details + click-to-pin; the right panel always shows a default cell
// (strongest |delta| in the focused capability) so there is never an empty state.
(function () {
  var cv = document.getElementById('cap3d-contrast');
  if (!cv || !cv.getContext || !DATA.contrast3d || !DATA.contrast3d.voxels) return;
  var panelEl = document.getElementById('cap3d-panel-contrast');
  var V = DATA.contrast3d, labels = V.labels, nl = V.n_layers, ne = V.n_experts, vox = V.voxels;
  var dpr = (window.devicePixelRatio || 1);
  var W = cv.clientWidth || 680, H = cv.clientHeight || 420;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  var ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  var hover = null, pin = null, focus = 0, cells = [], colHeads = [], PAD = 16, CELLH = 32, LOOM = 8, TOP0 = 120, COLPER = 4, COLPAD = 150, GAPW = 18, ROWGAP = 14;

  function layout() {
    var cw = cv.clientWidth || W, chh = cv.clientHeight || H;
    if (cw !== W || chh !== H) { W = cw; H = chh; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
    var pad = 16, gap = 18, rowGap = 28, loom = 8;
    var ncolPer = 4, nrows = Math.ceil(labels.length / ncolPer);
    var colW = (W - 2 * pad - (ncolPer - 1) * gap) / ncolPer;
    var cellW = colW / ne;
    var cellH = Math.min(46, cellW * 1.15);
    var stackH = nl * cellH + (nl - 1) * loom;
    var totalH = nrows * stackH + (nrows - 1) * rowGap;
    var top0 = (H - totalH) / 2;
    PAD = pad; CELLH = cellH; LOOM = loom; TOP0 = top0; COLPER = ncolPer; COLPAD = colW; GAPW = gap; ROWGAP = rowGap;
    cells = [];
    colHeads = [];
    for (var li = 0; li < labels.length; li++) {
      var colIdx = li % ncolPer, rowIdx = Math.floor(li / ncolPer);
      var cx = pad + colIdx * (colW + gap);
      var cy = top0 + rowIdx * (stackH + rowGap);
      colHeads.push({ x: cx, w: colW, y: cy, label: labels[li] });
      for (var ly = 0; ly < nl; ly++) {
        var cellY = cy + ly * (cellH + loom);
        for (var e = 0; e < ne; e++) {
          var v = vox[0];
          for (var k = 0; k < vox.length; k++) if (vox[k].label === li && vox[k].layer === ly && vox[k].expert === e) { v = vox[k]; break; }
          cells.push({ v: v, x: cx + e * cellW + 2, y: cellY + 2, w: cellW - 4, h: cellH - 4 });
        }
      }
    }
  }
  function isOn(a, b) { return a.label === b.label && a.layer === b.layer && a.expert === b.expert; }
  function draw() {
    layout();
    ctx.clearRect(0, 0, W, H);
    for (var i = 0; i < cells.length; i++) {
      var c = cells[i], d = c.v.delta, s = Math.min(1, Math.abs(d)), active = hover && isOn(hover, c.v);
      var x = c.x, y = c.y, w = c.w, h = c.h;
      var txtCol = 'rgba(235,240,246,0.95)';
      var fillA = 0.08 + 0.65 * s;
      var isPinned = pin && isOn(pin, c.v);
      if (d > 0.12) {
        ctx.fillStyle = 'rgba(74,222,128,' + fillA.toFixed(3) + ')';
        ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
        if (active || isPinned) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = isPinned?2.2:1.8; ctx.strokeRect(x, y, w, h); }
        txtCol = 'rgba(245,249,252,0.98)';
      } else if (d < -0.12) {
        ctx.fillStyle = 'rgba(248,113,113,' + fillA.toFixed(3) + ')';
        ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
        if (active || isPinned) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = isPinned?2.2:1.8; ctx.strokeRect(x, y, w, h); }
        txtCol = 'rgba(250,250,252,0.98)';
      } else {
        ctx.fillStyle = 'rgba(200,210,224,' + (0.10 + 0.20 * s).toFixed(3) + ')';
        ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
        if (active || isPinned) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = isPinned?2.2:1.8; ctx.strokeRect(x, y, w, h); }
        txtCol = 'rgba(205,213,223,0.95)';
      }
      if (w >= 22 && h >= 16) {
        ctx.fillStyle = txtCol; ctx.font = '9px "JetBrains Mono",ui-monospace,monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText((d >= 0 ? '+' : '') + (Math.round(Math.abs(d) * 100) / 100).toFixed(2), x + w / 2, y + h / 2 + 0.5);
        ctx.textBaseline = 'alphabetic';
      }
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    for (var hh = 0; hh < colHeads.length; hh++) {
      var ch = colHeads[hh];
      ctx.fillStyle = 'rgba(205,213,223,0.95)'; ctx.font = '500 10px system-ui';
      ctx.fillText(ch.label, ch.x + ch.w / 2, ch.y - 7);
    }
    ctx.textAlign = 'right';
    for (var L = 0; L < nl; L++) {
      ctx.fillStyle = 'rgba(150,160,180,0.9)'; ctx.font = '11px system-ui';
      ctx.fillText('L' + L, PAD - 8, TOP0 + L * (CELLH + LOOM) + CELLH / 2);
    }
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(150,160,180,0.9)'; ctx.font = '10px system-ui';
    ctx.fillText('green \u25A0 leans SUCCESS, red \u25A0 leans FAILURE, grey \u2014 neutral; number = \u0394 (colour depth = |\u0394|)', PAD, H - 6);
  }
  function pick(e) {
    var r = cv.getBoundingClientRect ? cv.getBoundingClientRect() : { left: 0, top: 0, width: W, height: H };
    var bx = (e.clientX - r.left) * (W / r.width), by = (e.clientY - r.top) * (H / r.height);
    for (var i = 0; i < cells.length; i++) { var c = cells[i]; if (bx >= c.x && bx <= c.x + c.w && by >= c.y && by <= c.y + c.h) return c.v; }
    return null;
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
  function renderPanel() {
    var v = hover || pin;
    var html = '';
    if (!v) {
      var all = []; for (var ai = 0; ai < vox.length; ai++) if (vox[ai].label === focus) all.push(vox[ai]);
      all.sort(function(a,b){ return Math.abs(b.delta)-Math.abs(a.delta); });
      v = all[0] || vox[0];
    }
    var d = v.delta;
    var kind = d > 0.12 ? 'success-favoured' : d < -0.12 ? 'failure-favoured' : 'neutral';
    var sign = d >= 0 ? '+' : '';
    var dec;
    if (d > 0.12) dec = 'This expert lights up mainly on SUCCESSFUL runs of this capability \u2014 its routing is tied to good outcomes.';
    else if (d < -0.12) dec = 'This expert lights up mainly on FAILED runs \u2014 routing here goes with failures.';
    else dec = 'About as involved in successes as failures \u2014 no strong lean.';
    var pinnedNow = pin && isOn(pin, v);
    var hdrIcon = pinnedNow ? '\u25CF ' : '\u25CB ';
    html += '<div class="p-head" title="' + esc(labels[v.label]) + '">' + hdrIcon + esc(labels[v.label]) + ' <span class="mut">· L' + v.layer + ' / E' + v.expert + (pinnedNow ? ' · pinned' : '') + '</span></div>';
    html += '<div class="p-sub">' + (labels[v.label] || '') + ' · layer L' + v.layer + ' · expert E' + v.expert + '</div>';
    html += '<div style="margin:10px 0;font-size:20px;font-weight:700;font-family:JetBrains Mono,monospace">\u0394 ' + sign + (Math.round(d * 1000) / 1000) + ' <span class="mut" style="font-size:11px;font-weight:400">(' + kind + ')</span></div>';
    function cell(v, key) { return (v && v[key] != null) ? v[key].toFixed(3) : '\u2013'; }
    html += '<table style="width:100%;border-collapse:collapse;margin:4px 0 8px">'
      + '<tr><td class="p-sub" style="padding:2px 0;color:#b9c0ca">success saliency</td><td class="p-sub" style="text-align:right;font-family:JetBrains Mono,monospace;color:#e9edf3">' + cell(v,'pos') + '</td></tr>'
      + '<tr><td class="p-sub" style="padding:2px 0;color:#b9c0ca">failure saliency</td><td class="p-sub" style="text-align:right;font-family:JetBrains Mono,monospace;color:#e9edf3">' + cell(v,'neg') + '</td></tr>'
      + '<tr><td class="p-sub" style="padding:2px 0;color:#b9c0ca">\u0394 = success \u2212 failure</td><td class="p-sub" style="text-align:right;font-family:JetBrains Mono,monospace;color:#e9edf3">' + sign + (Math.round(d * 1000) / 1000) + '</td></tr>'
      + '</table>';
    html += '<div class="p-sub" style="color:#d7d7d7;line-height:1.5">' + dec + '</div>';
    html += '<div class="p-sub" style="color:#838383">Saliency = this expert\u2019s average routing strength for that capability. \u0394 is normalised per capability (+1.00 = strongest success lean in the capability).</div>';
    if (panelEl) panelEl.innerHTML = html + '<div class="p-sub" style="color:#60666e;margin-top:-2px">' + (pin ? 'Pinned cell \u2014 click any tile to re-pin, click the same tile or the ring to unpin.' : 'Default cell (strongest |\u0394| in this capability) \u2014 hover or click a tile to inspect it.') + '</div>';
  }
  function redraw() { draw(); renderPanel(); }
  cv.addEventListener('pointermove', function (e) { hover = pick(e); redraw(); });
  cv.addEventListener('click', function (e) { var p = pick(e); pin = (p && pin && isOn(pin, p)) ? null : (p || pin); if (!p && !pin) pin = null; redraw(); });
  if (cv.style) cv.style.cursor = 'default';
  draw(); renderPanel();
})();
"""

_TAB_TEMPLATE = """<div class="tab" data-tab="{id}"><svg viewBox="0 0 24 24" width="15" height="15" {_LUCIDE} style="flex:0 0 auto">{icon}</svg><span>{title}</span></div>"""



def render_dashboard(data: dict[str, Any]) -> str:
    """Return a self-contained interactive HTML page embedding measured data."""
    data = {**data, "maintenance": _maintenance_payload()}
    payload = json.dumps(data).replace("</", "<\\/")
    cap3d_json = json.dumps(data.get("capability3d", {})).replace("</", "<\\/")
    nav: list[dict[str, Any]] = [
        {
            "section": "Overview",
            "tabs": [
                {"id": "summary", "title": "Summary"},
                {"id": "maintenance", "title": "Maintenance"},
            ],
        },
        {
            "section": "Profiling",
            "tabs": [
                {"id": "capability", "title": "Experts"},
                {"id": "contrast", "title": "Success\u2212Failure"},
                {"id": "coalition", "title": "Expert Pairings"},
                {"id": "structure", "title": "Structure"},
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
        {
            "section": "Researcher",
            "tabs": [
                {"id": "pareto", "title": "Pareto Explorer"},
                {"id": "v3", "title": "V3 Analyzers"},
                {"id": "candidates", "title": "Candidate Graph"},
                {"id": "corpus", "title": "Corpus Evidence"},
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&amp;display=swap">
<style>
:root{{
  --bg:#0e1013; --bg2:#12151a; --card:#151a21; --card2:#1a202a; --line:#262d38; --line2:#333c4a;
  --txt:#e6eaf0; --mut:#98a2b3; --dim:#626c79;
  --gold:#e2b45c; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --info:#7dd3fc; --vio:#c4b5fd;
  --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}}
body{{font-family:'Inter',ui-sans-serif,system-ui,sans-serif;font-size:12.6px;line-height:1.45;margin:0;background:var(--bg);color:var(--txt)}}
h1{{font-family:var(--mono);letter-spacing:-0.01em;font-size:16.2px;margin:0}}
.sub{{color:var(--mut);font-size:10.8px;margin-top:4px}}
.layout{{display:flex;min-height:100vh}}
.col{{flex:1;display:flex;flex-direction:column;min-width:0}}
nav.side{{width:212px;flex:0 0 212px;display:flex;flex-direction:column;gap:2px;background:#101318;border-right:1px solid var(--line);padding:14px 10px;position:sticky;top:0;height:100vh;overflow-y:auto;box-sizing:border-box}}
nav.side .tab{{display:flex;align-items:center;gap:9px;font-family:var(--mono);font-size:12.15px;padding:8px 10px;cursor:pointer;color:var(--mut);border-radius:6px;border:1px solid transparent}}
nav.side .tab svg{{flex:0 0 auto;opacity:.75}}
nav.side .tab:hover{{background:var(--card);color:var(--txt)}}
nav.side .tab.active{{color:#fff;background:var(--card2);border-color:var(--line2)}}
main.main{{flex:1;padding:22px 28px 60px;max-width:1460px}}
.panel{{display:none}}.panel.active{{display:block}}
.note{{color:var(--mut);font-size:11.25px;line-height:1.55;max-width:110ch}}
.mute{{color:var(--dim)}}
code{{font-family:var(--mono);font-size:.92em;background:var(--card2);border:1px solid var(--line);border-radius:4px;padding:0 4px}}
/* ---- tables ---- */
table{{border-collapse:collapse;width:100%;font-size:11.25px;margin-top:8px}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--dim);font-weight:600;font-family:var(--mono);font-size:9.45px;text-transform:uppercase;letter-spacing:.06em}}
tbody tr:hover{{background:rgba(255,255,255,.025)}}
td{{color:#cdd4de}}
/* ---- chips / pills / stats ---- */
.chip{{display:inline-block;background:var(--card2);border:1px solid var(--line2);border-radius:5px;padding:2px 8px;margin:2px;font-size:10.35px;font-family:var(--mono);color:#cdd4de}}
.pill{{display:inline-block;font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:.05em;padding:2px 8px;border-radius:999px;border:1px solid currentColor}}
.pill.ok{{color:var(--ok)}} .pill.bad{{color:var(--bad)}} .pill.warn{{color:var(--warn)}} .pill.gold{{color:var(--gold)}} .pill.dim{{color:var(--dim)}} .pill.info{{color:var(--info)}}
.green{{color:var(--ok)}} .amber{{color:var(--warn)}} .red{{color:var(--bad)}}
.stat{{display:inline-flex;flex-direction:column;gap:3px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:4px;font-family:var(--mono)}}
.stat .k{{color:var(--dim);font-size:9px;text-transform:uppercase;letter-spacing:.07em;display:block}}
.stat .v{{font-size:15.3px;color:var(--txt)}}
/* ---- KPI cards + section cards ---- */
.kgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:10px;margin:12px 0}}
.kcard{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.kcard .k{{color:var(--dim);font-size:9px;font-family:var(--mono);text-transform:uppercase;letter-spacing:.07em;display:block;margin-bottom:5px}}
.kcard .v{{font-family:var(--mono);font-size:18.9px;color:var(--txt);line-height:1.1}}
.kcard .s{{color:var(--mut);font-size:9.9px;margin-top:4px;display:block}}
.kcard.hl{{border-color:rgba(226,180,92,.45);background:linear-gradient(180deg,rgba(226,180,92,.07),var(--card))}}
.viz{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}}
.viz>h3{{font-family:var(--mono);font-size:11.7px;color:var(--txt);margin:0 0 2px;letter-spacing:-.01em}}
.viz .g-note{{color:var(--mut);font-size:10.35px;margin:0 0 12px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}}
.grid3{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}}
/* ---- labeled bars ---- */
.bar-row{{display:flex;align-items:center;gap:10px;padding:5px 0}}
.bar-row .bl{{font-family:var(--mono);font-size:10.35px;color:#cdd4de;min-width:86px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bar-row .bar{{flex:1;height:9px;background:var(--bg2);border:1px solid var(--line);border-radius:5px;overflow:hidden}}
.bar-row .bar i{{display:block;height:100%;border-radius:4px}}
.bar-row .bv{{font-family:var(--mono);font-size:10.35px;color:var(--mut);min-width:52px;text-align:right}}
.fbar{{height:7px;border-radius:4px;background:var(--bg2);overflow:hidden}}
.fbar i{{display:block;height:100%;border-radius:4px}}
/* ---- hero / stepper / flow ---- */
.hero{{display:flex;align-items:center;gap:14px;background:linear-gradient(120deg,var(--card2),var(--card));border:1px solid var(--line2);border-radius:12px;padding:14px 18px;margin:0 0 6px}}
.hero .ht{{font-family:var(--mono);font-size:14.4px;color:#fff}}
.hero .hs{{color:var(--mut);font-size:10.8px;margin-top:3px}}
.stepper{{display:flex;gap:0;margin:14px 0;flex-wrap:wrap}}
.step{{flex:1 1 130px;min-width:130px;background:var(--card);border:1px solid var(--line);padding:10px 12px;position:relative}}
.step:first-child{{border-radius:10px 0 0 10px}}
.step:last-child{{border-radius:0 10px 10px 0}}
.step+.step{{border-left:none}}
.step .sn{{font-family:var(--mono);font-size:8.55px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}}
.step .st{{font-size:11.7px;color:var(--txt);margin:3px 0 2px;font-weight:600}}
.step .sv{{font-family:var(--mono);font-size:9.9px;color:var(--gold)}}
.flow{{display:flex;align-items:center;flex-wrap:wrap;gap:4px;font-family:var(--mono);font-size:9.9px}}
.flow .fnode{{background:var(--card2);border:1px solid var(--line2);border-radius:6px;padding:2px 7px;color:#dfe5ee}}
.flow .farrow{{color:var(--dim)}}
/* ---- raw-table disclosures ---- */
details.raw{{margin-top:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg2)}}
details.raw>summary{{cursor:pointer;padding:7px 12px;font-family:var(--mono);font-size:9.9px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;list-style:none}}
details.raw>summary:before{{content:'+ ';color:var(--dim)}}
details.raw[open]>summary:before{{content:'− '}}
details.raw>summary:hover{{color:var(--txt)}}
details.raw .raw-body{{padding:2px 12px 10px}}
/* ---- maintenance tab ---- */
.mt-hero{{display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
.mt-phase{{font-family:var(--mono);font-size:23.4px;font-weight:700;letter-spacing:-.01em}}
.mt-steps{{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}}
.mt-step{{flex:1 1 180px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;display:flex;align-items:center;gap:10px}}
.mt-dot{{width:11px;height:11px;border-radius:50%;background:var(--line2);flex:0 0 auto}}
.mt-dot.on{{background:#fff;box-shadow:0 0 10px rgba(255,255,255,.8)}}
.mt-dot.done{{background:var(--ok)}}
.mt-step .tl{{font-size:11.7px}}
.mt-step .tl small{{display:block;color:var(--dim);font-size:9.9px}}
.mt-shardbar{{height:12px;border-radius:6px;background:var(--bg2);border:1px solid var(--line);overflow:hidden;margin:8px 0 4px}}
.mt-shardbar i{{display:block;height:100%;background:linear-gradient(90deg,#7dd3fc,#4ade80);transition:width .5s}}
.mt-time{{margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11.7px;color:var(--mut)}}
.mt-time b{{color:var(--txt)}}
.svc-chip{{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:9.9px;border:1px solid var(--line2);border-radius:999px;padding:3px 10px;margin:2px;color:#cdd4de}}
.svc-chip .d{{width:7px;height:7px;border-radius:50%}}
/* ---- capability extras ---- */
.cap-champs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:8px;margin:4px 0 6px}}
.cap-champ{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:9px 12px}}
.cap-champ .cl{{font-size:10.8px;color:var(--txt);font-weight:600;margin-bottom:4px}}
.cap-champ .cv{{font-family:var(--mono);font-size:10.35px;color:var(--mut);display:flex;justify-content:space-between;align-items:center;gap:8px}}
.cap-champ .cv b{{color:var(--gold);font-size:11.7px}}
/* ---- summary verdicts ---- */
.verdicts{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin:12px 0}}
.verdict{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.verdict .vt{{font-family:var(--mono);font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px}}
.verdict .vm{{font-size:12.6px;color:var(--txt)}}
.verdict .vm b{{font-family:var(--mono)}}
/* ---- candidate / compression cards ---- */
.cand-card{{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:12px 14px;display:flex;flex-direction:column;gap:8px}}
.cand-card.best{{border-color:rgba(226,180,92,.5);box-shadow:0 0 0 1px rgba(226,180,92,.25) inset}}
.cand-top{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
.cand-name{{font-family:var(--mono);font-size:12.15px;color:#fff}}
.cand-ret{{display:flex;align-items:baseline;gap:8px}}
.cand-ret .big{{font-family:var(--mono);font-size:21.6px;color:var(--ok)}}
.cand-ret .lbl{{font-size:9.45px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}}
.cand-meta{{display:flex;flex-wrap:wrap;gap:6px;font-family:var(--mono);font-size:9.9px;color:var(--mut)}}
.cand-meta b{{color:#cdd4de}}
.comp-card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
.comp-head{{font-family:var(--mono);font-size:11.25px;color:#fff;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center}}
.comp-row{{display:flex;align-items:center;gap:8px;padding:3px 0;font-family:var(--mono);font-size:9.9px;color:var(--mut)}}
.comp-row .fmt{{min-width:34px;color:#cdd4de}}
.comp-row .m{{flex:1;height:6px;background:var(--bg2);border-radius:3px;overflow:hidden}}
.comp-row .m i{{display:block;height:100%}}
/* ---- level bars (hierarchy) ---- */
.lvlbar{{display:flex;align-items:center;gap:10px;padding:4px 0}}
.lvlbar .ll{{font-family:var(--mono);font-size:10.35px;color:#cdd4de;min-width:92px}}
.lvlbar .lb{{flex:1;height:14px;background:var(--bg2);border-radius:4px;overflow:hidden}}
.lvlbar .lb i{{display:block;height:100%;background:linear-gradient(90deg,#5eead4,#7dd3fc,#c4b5fd);border-radius:3px}}
.lvlbar .lv{{font-family:var(--mono);font-size:10.35px;color:var(--mut);min-width:56px;text-align:right}}
/* ---- corpus cluster cards ---- */
.cluster-card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
.cluster-card .cd{{font-size:11.25px;color:#fff;margin-bottom:3px}}
.cluster-card .cm{{font-family:var(--mono);font-size:9.9px;color:var(--mut)}}
.delta-pos{{color:var(--ok);font-family:var(--mono)}} .delta-neg{{color:var(--bad);font-family:var(--mono)}}
/* ---- legacy kept component styles ---- */
.cap-tbl-head{{display:flex;justify-content:flex-end;align-items:center;gap:8px;margin:14px 0 20px}}
#cap-sort{{appearance:none;-webkit-appearance:none;-moz-appearance:none;background:#121519 url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='10'%20height='6'%20viewBox='0%200%2010%206'%3E%3Cpath%20d='M1%201l4%204%204-4'%20stroke='%2386909e'%20stroke-width='1.6'%20fill='none'%20stroke-linecap='round'%20stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 12px center;color:#aeb5bf;border:1px solid #343a42;border-radius:6px;font-family:var(--mono);font-size:10.8px;padding:4px 26px 4px 10px;cursor:pointer}}
#cap-sort option{{background:#121519;color:#b6bdc7}}
.cap-sort-lbl{{color:var(--dim);font-family:var(--mono);font-size:9.9px;text-transform:uppercase;letter-spacing:.06em}}
.cap-more-wrap{{display:flex;justify-content:center;margin:8px 0 2px}}
#cap-more{{background:transparent;border:1px solid var(--line2);color:#cfd6e0;font-family:var(--mono);font-size:10.8px;padding:5px 16px;border-radius:6px;cursor:pointer}}
#cap-more:hover{{border-color:#6b6b6b;color:#fff}}
#cap-filter-btn{{background:transparent;border:1px solid var(--line2);color:#cfd6e0;font-family:var(--mono);font-size:10.8px;padding:4px 12px;border-radius:6px;cursor:pointer}}
#cap-filter-btn:hover{{border-color:#6b6b6b;color:#fff}}
#cap-fcount{{color:var(--mut);margin-left:6px;font-size:9.9px}}
.cap-tray-bd{{position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:59;opacity:0;pointer-events:none;transition:opacity .2s}}
.cap-tray-bd.show{{opacity:1;pointer-events:auto}}
.cap-tray{{position:fixed;top:0;right:0;height:100vh;width:280px;background:#101316;border-left:1px solid var(--line2);z-index:60;transform:translateX(105%);transition:transform .22s ease;display:flex;flex-direction:column;box-shadow:-10px 0 34px rgba(0,0,0,0.5)}}
.cap-tray.open{{transform:translateX(0)}}
.cap-tray-head{{display:flex;justify-content:space-between;align-items:center;padding:14px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:11.7px;color:#e9edf3}}
.cap-tray-head button{{background:none;border:none;color:var(--mut);font-size:16.2px;cursor:pointer;line-height:1}}
.cap-tray-head button:hover{{color:#fff}}
.cap-tray-body{{overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:10px}}
.cap-tray-body label{{display:flex;justify-content:space-between;align-items:center;color:#cfd6e0;font-size:11.25px;cursor:pointer;gap:8px}}
.cap-tray-body input[type=checkbox]{{accent-color:#cfd6e0;width:14px;height:14px}}
.cap3d-wrap{{position:relative;display:flex;gap:12px;align-items:stretch;border:1px solid var(--line);border-radius:10px;padding:10px;background:#08090b;background-image:linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);background-size:48px 48px,48px 48px}}
.cap3d-canvas{{position:relative;flex:1 1 auto;min-width:280px}}
.cap3d-panel{{flex:0 0 300px;align-self:stretch;background:#0d0f12;border-left:1px solid var(--line);padding-left:12px;overflow:auto;max-height:460px}}
@media (max-width:1100px){{
  .cap3d-wrap{{flex-direction:column}}
  .cap3d-panel{{flex:1 1 auto;border-left:none;border-top:1px solid var(--line);padding-left:0;padding-top:10px;max-height:none}}
}}
@media (max-width:720px){{
  canvas#cap3d{{aspect-ratio:680/520}}
  .cap3d-controls{{opacity:.9}}
}}
.cap3d-canvas{{position:relative;flex:1;min-width:0}}
canvas#cap3d{{width:100%;aspect-ratio:680/420;height:auto;display:block;touch-action:none;cursor:default;border-radius:6px;background:transparent}}
canvas#cap3d-contrast{{width:100%;aspect-ratio:680/420;height:auto;display:block;touch-action:none;cursor:default;border-radius:6px;background:transparent}}
#contrast-list-wrap{{margin-top:18px}}
.c-panel{{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:10px;overflow:hidden}}
.c-panel .c-head{{display:flex;align-items:center;gap:10px;padding:10px 14px;cursor:pointer;user-select:none}}
.c-panel .c-head:hover{{background:var(--card2)}}
.c-panel.open .c-caret{{transform:rotate(90deg)}}
.c-caret{{color:var(--dim);font-size:9px;transition:transform .12s;width:10px;text-align:center}}
.c-cap{{font-family:var(--mono);font-size:11.7px;color:#e9edf3}}
.c-body{{display:none;padding:2px 14px 12px}}
.c-panel.open .c-body{{display:block}}
.c-row{{display:flex;align-items:center;gap:10px;padding:6px 0;border-top:1px dashed var(--line)}}
.c-row:first-of-type{{border-top:none}}
.c-cell{{font-family:var(--mono);font-size:10.8px;color:#c6cdd8;min-width:52px}}
.c-bar{{flex:1;height:8px;background:var(--bg2);border-radius:4px;overflow:hidden}}
.c-fill{{display:block;height:100%}}
.success .c-fill{{background:var(--ok)}} .failure .c-fill{{background:var(--bad)}} .neutral .c-fill{{background:#64748b}}
.c-d{{font-family:var(--mono);font-size:10.8px;font-weight:600;width:64px;text-align:right}}
.success .c-d{{color:var(--ok)}} .failure .c-d{{color:var(--bad)}} .neutral .c-d{{color:#cbd5e1}}
.c-tag{{font-family:var(--mono);font-size:8.55px;text-transform:uppercase;letter-spacing:.05em;padding:2px 7px;border-radius:999px;border:1px solid currentColor}}
.success .c-tag{{color:var(--ok)}} .failure .c-tag{{color:var(--bad)}} .neutral .c-tag{{color:#94a3b8}}
.c-more{{border-top:1px solid var(--line);padding-top:8px;text-align:center}}
.c-more button{{background:transparent;border:1px solid var(--line2);color:#cfd6e0;font-family:var(--mono);font-size:9.9px;padding:4px 14px;border-radius:6px;cursor:pointer}}
.c-more button:hover{{border-color:#6b6b6b;color:#fff}}
.con-counts{{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 2px}}
.coal-wrap{{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin-top:10px}}
.coal-g{{flex:1 1 460px;min-width:340px}}
.coal-g h3{{font-family:var(--mono);font-size:12.15px;color:#eee;margin:0 0 4px;letter-spacing:-0.01em}}
.coal-g .g-note{{color:var(--mut);font-size:10.35px;margin:0 0 10px}}
span.coal-node{{display:inline-flex;align-items:center;justify-content:center;min-width:22px;height:22px;padding:0 5px;margin:2px;border-radius:999px;font-family:var(--mono);font-size:9.45px;font-weight:600;background:#3a3a3a;border:1px solid #555;color:#f2f2f2}}
span.coal-node.on{{background:#4b4b4b;border-color:#6e6e6e;color:#ffffff}}
span.coal-node.mid{{background:#303030;border-color:#4a4a4a;color:#dcdcdc}}
span.coal-node.dim{{background:#1d1d1d;border-color:#2c2c2c;color:#787878}}
.coal-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin-top:2px}}
.coal-card{{background:var(--card);border:1px solid #282828;border-radius:10px;padding:10px 12px;cursor:pointer;display:flex;flex-direction:column;gap:8px;transition:border-color .15s,box-shadow .15s,opacity .15s}}
.coal-card:hover{{border-color:#3a3a3a}}
.coal-card.cs{{border-color:#5ecfff;box-shadow:0 0 0 1px #5ecfff inset}}
.coal-card.min{{opacity:.72}}
.cc-top{{display:flex;align-items:center;justify-content:space-between;gap:8px}}
.cc-pills{{display:flex;align-items:center;gap:5px}}
.cc-pills .plus{{color:var(--dim);font-family:var(--mono);font-size:11.7px}}
.cc-rank{{font-family:var(--mono);font-size:9.9px;color:#8a8a8a;border:1px solid var(--line2);border-radius:999px;padding:1px 8px;background:var(--bg2)}}
.cc-meter{{height:6px;border-radius:3px;background:var(--bg2);overflow:hidden}}
.cc-meter i{{display:block;height:100%;border-radius:3px;width:0}}
.coal-card.lo .cc-meter i{{background:#3f6212}}
.coal-card.md .cc-meter i{{background:#94cc1c}}
.coal-card.hi .cc-meter i{{background:#22c55e}}
.coal-card.cr .cc-meter i{{background:#f97316}}
.cc-count{{font-family:var(--mono);font-size:13.5px;color:#eee;line-height:1;margin-top:-2px}}
.cc-count small{{color:#8a8a8a;font-size:9.9px}}
.cc-solo{{display:flex;flex-direction:column;gap:4px}}
.cc-s{{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:9px;color:var(--mut)}}
.cc-s .en{{color:#dcdcdc;min-width:18px}}
.cc-s .sb{{flex:1;height:4px;border-radius:2px;background:var(--bg2);overflow:hidden}}
.cc-s .sb i{{display:block;height:100%;border-radius:2px;width:0}}
.cc-s .sb i.on{{background:#6e6e6e}} .cc-s .sb i.mid{{background:#4a4a4a}} .cc-s .sb i.dim{{background:#2a2a2a}}
.cc-s .v{{min-width:16px;text-align:right;color:#a1a1aa}}
.cc-badges{{display:flex;gap:4px;margin-top:-2px}}
.coal-legend{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:8px 2px 4px;font-size:9.9px;color:var(--mut)}}
.coal-legend .lg{{display:inline-flex;align-items:center;gap:6px}}
.coal-legend .sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}
.coal-legend .sw.on{{background:#4b4b4b;border:1px solid #6e6e6e}} .coal-legend .sw.mid{{background:#303030;border:1px solid #4a4a4a}} .coal-legend .sw.dim{{background:#1d1d1d;border:1px solid #2c2c2c}}
.coal-legend .ln{{height:14px;width:6px;border-radius:1px;display:inline-block;vertical-align:middle}}
.coal-legend .ln.lo{{background:#3f6212}} .coal-legend .ln.md{{background:#94cc1c}} .coal-legend .ln.hi{{background:#22c55e}} .coal-legend .ln.cr{{background:#f97316}}
.struct-sec{{margin-top:18px;padding:14px 16px;background:var(--card);border:1px solid var(--line);border-radius:10px}}
.struct-sec h3{{font-family:var(--mono);font-size:12.15px;color:#eee;margin:0 0 2px;letter-spacing:-0.01em}}
.struct-sec .g-note{{color:var(--mut);font-size:10.35px;margin:0 0 10px}}
.struct-sec table{{margin-top:6px}}
table.map-heat{{border-collapse:separate;border-spacing:2px;width:auto;margin:4px 0}}
table.map-heat td{{padding:0}}
table.map-heat caption{{caption-side:top;text-align:left;color:#8a8a8a;font-size:9.9px;margin-bottom:2px}}
.oc{{width:22px;height:22px;border-radius:4px;background:var(--bg2);font-size:8.1px;color:#fff;display:flex;align-items:center;justify-content:center;font-family:var(--mono)}}
.oc.hatch{{background-image:repeating-linear-gradient(45deg,rgba(0,0,0,0.35) 0 4px,rgba(255,255,255,0.12) 4px 8px)}}
.ch-heat{{border-collapse:separate;border-spacing:1px}}
.ch-heat td{{padding:0}}
.ch-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px 44px;align-items:start}}
.ch-grid-strip{{min-width:0}}
.strip-lab{{font-family:var(--mono);font-size:9.45px;color:#a7a7a7;margin:10px 0 3px;letter-spacing:.03em;white-space:nowrap}}
.strip-note{{color:#8a8a8a;font-size:10.35px;margin:10px 0 4px;font-style:italic}}
.ch-heat .tile-edge{{box-shadow:inset 0 0 0 2px #8a8a8a;border-radius:2px}}
.ch{{width:9px;height:9px;border-radius:2px}}
.router-box,.distill-box{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:10px;background:var(--card2);border:1px solid var(--line);border-radius:8px;margin:4px 0 10px}}
.rslot{{display:inline-flex;align-items:center;justify-content:center;min-width:26px;height:26px;border-radius:50%;background:#3c3c3c;border:1px solid #565656;color:#e8e8e8;font-family:var(--mono);font-size:9.9px;font-weight:600}}
.rslot.bias{{box-shadow:0 0 0 2px #9a9a9a inset}}
.rslot.drop{{background:#232323;border-color:#8a5a5a;color:#ffc9c9}}
.rn{{width:100%;color:#8a8a8a;font-size:9.9px;font-family:var(--mono);margin-top:2px}}
.dcell,.rcell{{display:inline-flex;align-items:center;justify-content:center;border-radius:99px;background:#4a4a4a;border:1px solid #6a6a6a;color:#f0f0f0;font-family:var(--mono);font-size:9px;font-weight:600}}
.dcell.lane{{background:#383838;border-color:#585858}}
.dcell.layer{{background:#2f2f2f;border-color:#4d4d4d}}
.rcell{{background:#3f3f3f;border:1px solid #5f5f5f;color:#f0f0f0}}
.rcell.bias{{background:#333333;border-color:#555555}}
.rcell.out{{background:#4f4f4f;border-color:#6f6f6f}}
.cap3d-controls{{position:absolute;top:6px;left:6px;z-index:3;display:flex;align-items:center;gap:4px;background:rgba(14,14,14,0.7);border:1px solid #444444;border-radius:6px;padding:3px 4px}}
.cap3d-controls button{{width:19px;height:19px;display:inline-flex;align-items:center;justify-content:center;padding:0;color:#d7d7d7;background:#171717;border:1px solid #444444;border-radius:4px;cursor:pointer}}
.cap3d-controls button:hover{{border-color:#646464;color:#fff;background:#202020}}
.cap3d-controls .cspread{{display:flex;align-items:center;gap:4px;color:var(--mut);font-size:8.55px;letter-spacing:.03em;font-family:var(--mono);margin:0 2px}}
.cap3d-controls input[type=range]{{-webkit-appearance:none;appearance:none;width:90px;height:14px;background:transparent;cursor:pointer}}
.cap3d-controls input[type=range]::-webkit-slider-runnable-track{{height:1px;background:#444444;border-radius:1px}}
.cap3d-controls input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;appearance:none;width:8px;height:8px;border-radius:50%;background:#d7d7d7;border:none;margin-top:-3.5px}}
.cap3d-controls input[type=range]::-moz-range-track{{height:1px;background:#444444;border:none}}
.cap3d-controls input[type=range]::-moz-range-thumb{{width:8px;height:8px;border:none;border-radius:50%;background:#d7d7d7}}
.cap3d-vig{{position:absolute;inset:0;pointer-events:none;border-radius:6px;background:radial-gradient(ellipse 72% 68% at 50% 48%, transparent 42%, rgba(8,8,8,0.5) 74%, #080808 100%)}}
canvas#cap3d.dragging{{cursor:grabbing}}
.cap3d-panel{{flex:0 0 300px;align-self:stretch;background:#0d0f12;border-left:1px solid var(--line);padding-left:12px;overflow-y:auto;font-size:10.8px;color:#d7d7d7}}
.cap3d-panel .p-head{{font-family:var(--mono);font-size:11.25px;color:#eeeeee;margin:2px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.cap3d-panel .p-sub{{color:var(--dim);font-size:9.45px;margin:2px 0 8px}}
.cap3d-panel .p-grp{{color:#dfdfdf;font-size:9.9px;font-family:var(--mono);margin:8px 0 2px;cursor:pointer;display:flex;align-items:center;gap:6px}}
.cap3d-panel .p-grp:hover{{color:#f2f2f2;text-decoration:underline}}
.cap3d-panel .p-grp-caret{{font-size:8.1px;opacity:.8;transition:transform .1s}}
.cap3d-panel .p-back{{color:#dfdfdf;cursor:pointer;text-decoration:none;font-family:var(--mono)}}
.cap3d-panel .p-back:hover{{text-decoration:underline}}
.cap3d-panel .p-filt{{display:flex;align-items:center;gap:4px;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:8px}}
.cap3d-panel .p-filt-l{{color:var(--dim);font-size:9px;font-family:var(--mono);margin-right:2px;text-transform:uppercase;letter-spacing:.06em}}
.cap3d-panel .p-chip{{font-family:var(--mono);font-size:9.45px;color:var(--mut);background:#171717;border:1px solid var(--line2);border-radius:999px;padding:2px 9px;cursor:pointer}}
.cap3d-panel .p-chip.on{{color:#0d0d0d;background:#eeeeee;border-color:#eeeeee}}
.cap3d-panel .p-chip:hover{{border-color:#dfdfdf;color:#dfdfdf}}
.cap3d-panel .p-row{{display:flex;align-items:center;gap:8px;padding:3px 4px;border-radius:4px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:var(--mono);font-size:9.9px}}
.cap3d-panel .p-row:hover{{background:#1b1b1b}}
.cap3d-panel .p-row.sel{{background:#262626;color:#fff}}
.cap3d-panel .p-bar{{flex:0 0 42px;height:5px;background:#202020;border-radius:3px;overflow:hidden}}
.cap3d-panel .p-bar i{{display:block;height:100%;background:#eeeeee}}
.cap3d-panel .p-tier{{flex:0 0 auto;margin-left:auto;font-size:8.55px;font-family:var(--mono);text-transform:uppercase;letter-spacing:.04em;padding:1px 5px;border-radius:3px;border:1px solid currentColor;opacity:.9}}
.cap3d-panel .p-tier.strong{{color:var(--gold)}}
.cap3d-panel .p-tier.good{{color:#c6cdd8}}
.cap3d-panel .p-tier.moderate{{color:#b08e6b}}
.cap3d-panel .p-tier.weak{{color:#d0686b}}
.navsec{{color:var(--dim);font-size:9.45px;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;margin:14px 10px 4px;font-family:var(--mono)}}
.navlink{{display:block;margin:14px 10px 0;font-family:var(--mono);font-size:12.6px;color:#dfdfdf;text-decoration:none;padding:6px 0;border-top:1px solid var(--line)}}
.navlink:hover{{color:#f2f2f2}}
.panel h3{{font-family:var(--mono);font-size:11.7px;color:#b4b4b4;margin:18px 0 6px}}</style></head>    <body>
    <!-- Isolated maintenance modal (runs before heavy chart scripts) -->
    <style>
      .im-backdrop{{position:fixed;inset:0;background:rgba(18,18,18,.78);z-index:9998;display:none;align-items:center;justify-content:center;padding:22px}}
      .im-modal{{background:#1b1b1b;border:1px solid #2e2e2e;border-radius:10px;max-width:700px;width:100%;padding:24px 26px;color:#dcdcdc;font-family:'Inter',ui-sans-serif,system-ui,sans-serif;box-shadow:0 16px 48px rgba(0,0,0,.55)}}
      .im-modal h2{{margin:0 0 6px;font-size:16px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;letter-spacing:-0.01em}}
      .im-state{{font-size:13px;color:#979797;margin-bottom:8px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}}
      .im-bar{{height:6px;border-radius:3px;background:#262626;overflow:hidden;border:1px solid #353535;margin:12px 0 4px}}
      .im-bar>div{{height:100%;width:0%;background:#f4f4f5;transition:width .4s}}
      .im-meta{{display:flex;justify-content:space-between;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;color:#979797;margin-bottom:10px}}
      .im-steps{{display:flex;flex-direction:column;gap:2px;margin:4px 0}}
      .im-step{{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #2e2e2e;font-size:14px;color:#dcdcdc}}
      .im-step:last-child{{border-bottom:none}}
      .im-dot{{width:10px;height:10px;border-radius:50%;background:#353535;flex:0 0 auto}}
      .im-dot.on{{background:#ffffff;box-shadow:0 0 8px #fff}}
      .im-dot.done{{background:#3fb950}}
      .im-lbl{{flex:1}}
      .im-lbl small{{color:#7c8798;margin-left:8px;font-weight:400}}
      .im-eta{{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:12px;color:#ffb86b;margin-top:8px}}
      .im-resume{{color:#9aa4b2;font-size:13px;margin-top:8px}}
      .im-debug{{margin-top:12px}}
      .im-debug details{{background:#0e131b;border:1px solid #2e2e2e;border-radius:8px;padding:10px}}
      .im-debug summary{{cursor:pointer;color:#58a6ff;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:13px;outline:none}}
      .im-debug pre{{background:#0c1016;border:1px solid #2e2e2e;border-radius:6px;padding:10px;font-size:11px;color:#9aa4b2;max-height:200px;overflow:auto;white-space:pre-wrap;word-break:break-all}}
      .im-copy{{background:#262626;color:#dcdcdc;border:1px solid #3a3a3a;border-radius:5px;padding:3px 10px;font-size:11px;cursor:pointer;margin-top:6px}}
      .im-copy:hover{{background:#333}}
    </style>
    <div class="im-backdrop" id="im-backdrop"><div class="im-modal">
      <h2>Maintenance <span id="im-chip" style="font-size:13px;color:#7c8798"></span></h2>
      <div class="im-state" id="im-state">IDLE</div>
      <div id="im-status" style="color:#9aa4b2;font-size:13px;margin-bottom:2px">No window running.</div>
      <div class="im-bar"><div id="im-fill"></div></div>
      <div class="im-meta"><span id="im-shards"></span><span id="im-time"></span></div>
      <div class="im-steps">
        <div class="im-step"><span class="im-dot" id="im-s-drain"></span><span class="im-lbl">1 &middot; Drain<small id="im-l-drain"></small></span></div>
        <div class="im-step"><span class="im-dot" id="im-s-produce"></span><span class="im-lbl">2 &middot; Produce<small id="im-l-produce"></small></span></div>
        <div class="im-step"><span class="im-dot" id="im-s-restore"></span><span class="im-lbl">3 &middot; Restore / reload<small id="im-l-restore"></small></span></div>
      </div>
      <div id="im-eta" class="im-eta"></div>
      <div id="im-resume" class="im-resume"></div>
      <div class="im-debug" id="im-debug-wrap" style="display:none">
        <details><summary>See debugging details</summary>
          <pre id="im-debug"></pre>
          <button class="im-copy" id="im-copy">Copy error to clipboard</button></details>
      </div>
    </div></div>
    <script>
(function(){{
  var BACK=document.getElementById('im-backdrop');
  function $(id){{ return document.getElementById(id); }}
  function fmt(s){{ s=Math.max(0,Math.floor(s||0)); var m=Math.floor(s/60), r=s%60; return m+':'+(r<10?'0':'')+r; }}
  function col(st){{ return {{'DRAIN':'#58a6ff','PRODUCE':'#d29922','RESTORE':'#3fb950','COMPLETE':'#3fb950','FAILED':'#f85149','BLOCKED':'#f85149','PREFLIGHT':'#58a6ff','IDLE':'#7c8798'}}[st]||'#7c8798'; }}
  function orderOf(st){{ return {{'IDLE':0,'PREFLIGHT':1,'DRAIN':2,'PRODUCE':3,'RESTORE':4,'COMPLETE':5,'FAILED':5,'BLOCKED':1}}[st]||0; }}
  function phaseOf(M){{
    if(!M||!M.present) return 'IDLE';
    if(M.blocked) return 'BLOCKED';
    var p=M.phase;
    if(p==='preflight') return 'PREFLIGHT';
    if(p==='drain') return 'DRAIN';
    if(p==='produce') return 'PRODUCE';
    if(p==='restore') return 'RESTORE';
    if(p==='maintenance'||p==='idle') return 'IDLE';
    return p?String(p).toUpperCase():'IDLE';
  }}
  function render(M){{
    var st=phaseOf(M); if(st==='IDLE'){{ BACK.style.display='none'; return; }}
    BACK.style.display='flex'; var c=col(st);
    $('im-chip').textContent=st; $('im-chip').style.color=c;
    $('im-state').textContent=st; $('im-state').style.color=c;
    $('im-status').textContent=(M.status||M.phase_label||st);
    var tot=M.shard_total||0,pct=0;
    if(st==='RESTORE'||st==='COMPLETE'){{ pct=tot?Math.round(100*(M.shard_current||0)/tot):(st==='COMPLETE'?100:0); }}
    else if(st==='DRAIN'){{ pct=Math.min(100,Math.round(((M.released&&M.released.length)||0)/4*100)); }}
    else if(st==='PRODUCE'){{ pct=45; }}
    $('im-fill').style.width=Math.min(100,Math.max(0,pct))+'%';
    $('im-shards').textContent=(tot&&(st==='RESTORE'||st==='COMPLETE'))?('DSV4 shards '+(M.shard_current||0)+'/'+tot+' ('+pct+'%)'):'';
    $('im-time').textContent='elapsed '+fmt(M.elapsed_seconds);
    function step(k,idx,label){{ var o=orderOf(st),el=$('im-s-'+k),lco=$('im-l-'+k); el.className='im-dot'+(st===label?' on':(o>idx?' done':'')); lco.textContent=st===label?'(running)':(o>idx?'\u2713':''); }}
    step('drain',2,'DRAIN'); step('produce',3,'PRODUCE'); step('restore',4,'RESTORE');
    var eta=$('im-eta');
    if(st==='DRAIN') eta.textContent='Services going offline — the agent will drop; this overlay keeps running.';
    else if(st==='PRODUCE') eta.textContent='Capturing & evaluating — the agent stays offline until restore. Expect minutes.';
    else if(st==='RESTORE') eta.textContent='Restoring DSV4 + prior services… the agent comes back online after this.';
    else eta.textContent='';
    $('im-resume').innerHTML=(M.result||'');
    var dbg=$('im-debug-wrap');
    if(st==='BLOCKED'||st==='FAILED'){{
      dbg.style.display='block';
      var reason=(M.blockers&&M.blockers.length)?M.blockers.map(function(b){{return b.kind+': '+b.detail;}}).join('\\n'):(M.result||M.status||st);
      $('im-debug').textContent=reason;
      $('im-copy').onclick=function(){{ var ta=document.createElement('textarea'); ta.value=reason; document.body.appendChild(ta); ta.select(); try{{document.execCommand('copy');}}catch(e){{}} document.body.removeChild(ta); $('im-copy').textContent='Copied!'; }};
    }} else dbg.style.display='none';
  }}
  (function poll(){{
    if(location.protocol.indexOf('http')!==0) return;
    fetch('/api/status',{{cache:'no-store'}}).then(function(r){{return r.json();}})
      .then(render).catch(function(){{}}).then(function(){{setTimeout(poll,1500);}});
  }})();
}})();    </script>
<div class="layout">
<nav class="side">{tab_html}</nav>
<div class="col">
<main class="main">
 <div class="panel" id="panel-summary">
   <div class="hero"><div><div class="ht">Atlas Profile Platform</div>
   <div class="hs" id="hero-sub">Every value below is computed live by the same measured code paths as the test suite. Predicted values are always labelled; nothing is styled as measured unless it is.</div></div></div>
   <h3>Pipeline at a glance</h3>
   <div class="stepper" id="sum-stepper"></div>
   <h3>Headline verdicts</h3>
   <div class="verdicts" id="sum-verdicts"></div>
   <h3>Where to look</h3>
   <div class="grid2" id="sum-map"></div>
   <details class="raw"><summary>raw payload keys</summary><div class="raw-body" id="sum-rawkeys"></div></details>
 </div>
 <div class="panel" id="panel-maintenance">
   <p class="note">Maintenance lifecycle: draining services → producing the derivative → restoring/loading DSV4 (with per-shard progress). Live if a run is active.</p>
   <div id="maintenance-body"></div>
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
   <h3>Capability champions</h3>
   <div class="g-note">Top expert per measured capability label — the full ranked table is below.</div>
   <div class="cap-champs" id="cap-champs"></div>
   <div class="cap-tbl-head">
     <label class="cap-sort-lbl" for="cap-sort">sort</label>
     <select id="cap-sort" title="sort"><option value="strength">strong → weak</option><option value="score">score</option><option value="name">category</option></select>
     <button id="cap-filter-btn" title="filter domains">Filter <span id="cap-fcount"></span></button>
   </div>
   <div class="cap-tray-bd" id="cap-tray-bd"></div>
   <div class="cap-tray" id="cap-tray"><div class="cap-tray-head"><span>Domains</span><button id="cap-tray-close" title="close">&#215;</button></div><div class="cap-tray-body" id="cap-domain-filters"></div></div>
   <table id="t-capability"></table>
   <div class="cap-more-wrap"><button id="cap-more">Load more</button></div>
 </div>
 <div class="panel" id="panel-contrast">
   <p class="note">Success − Failure: how each expert cell&#8217;s <b>saliency</b> splits across successful vs failed runs. Δ = success-saliency − failure-saliency (normalised per capability): bright filled tile = mostly lights up on <b>successful</b> runs; outlined tile = mostly on <b>failed</b>; near-neutral = equally involved. Hover any tile for exact values.</p>
   <div class="con-counts" id="con-counts"></div>
   <div class="cap3d-wrap">
     <div class="cap3d-canvas">
       <canvas id="cap3d-contrast"
         role="img"
         aria-label="Success versus failure saliency map: bright filled tiles favour successes, outlined tiles favour failures;"></canvas>
       <div class="cap3d-vig"></div>
     </div>
     <aside class="cap3d-panel" id="cap3d-panel-contrast" aria-live="polite"></aside>
   </div>
   <h3>Per-capability breakdown</h3>
   <div class="g-note">Open a capability for its strongest Δ cells, sorted by |Δ|.</div>
   <div id="contrast-list-wrap"></div>
 </div>
 <div class="panel" id="panel-coalition">
   <p class="note">Expert Pairings = pairs of experts that are <b>routed together</b> (co-activated) at layer 0, measured by running the calibration corpus forwards. Cards rank every pair by firing strength: the <b>meter = co-routing count</b>, each expert pill shows how often it appears <em>alone</em>. Cascade-risk pairs get an orange meter + <b>hot</b> tag; <b>synergy</b> = joint harm exceeds the pair's individual sum; <b>redundant</b> = overlap, keeping one covers the other.</p>
   <div class="coal-wrap">
     <div class="coal-g" style="flex:1 1 100%">
       <h3>How experts team up (layer 0)</h3>
       <div class="g-note">Strongest pairings first (all info in the card &#8212; no separate detail panel).</div>
       <div class="coal-legend"><span class="lg"><span class="sw on"></span>fires a lot</span><span class="lg"><span class="sw mid"></span>moderately routed</span><span class="lg"><span class="sw dim"></span>rare / quiet</span><span class="lg">co-routing&nbsp;meter:&nbsp;<span class="ln lo"></span>&nbsp;low</span><span class="lg"><span class="ln hi"></span>&nbsp;frequent</span><span class="lg"><span class="ln cr"></span>&nbsp;cascade&nbsp;risk</span></div>
       <div id="coal-list" class="coal-grid" role="list"></div>
     </div>
   </div>
 </div>
 <div class="panel" id="panel-structure">
   <p class="note">How the model's experts are <b>organised and connected</b> — measured end-to-end. Two views: the six-level hierarchy and the most frequent cross-layer route signatures that realise behaviour.</p>
   <div class="struct-sec">
     <h3>Hierarchy — six levels, one trace-down example</h3>
     <div class="g-note">Component concepts stack: weights → units → experts → expert pairings → pathways → behaviour. Bars show measured node counts per level.</div>
     <div id="hier-bars"></div>
     <details class="raw"><summary>trace-down detail + raw table</summary><div class="raw-body"><div id="hier-stats"></div><table id="t-hierarchy"></table></div></details>
   </div>
   <div class="struct-sec">
     <h3>Paths — most frequent cross-layer routes</h3>
     <div class="g-note">Route signatures that fire most often, with the fraction ending in a successful run. Bar length = route count.</div>
     <div id="path-bars"></div>
     <details class="raw"><summary>raw path table</summary><div class="raw-body"><table id="t-path"></table></div></details>
   </div>
 </div>
 <div class="panel" id="panel-maps">
   <p class="note">§25 planning artifacts from the <b>real mounted GLM-5.2 NVFP4</b> census when present (else synthetic caret). Channel/tile importance = measured per-tensor byte weight from the safetensors manifest; node-ownership = node A/B placement split of real expert tensors. Removal-impact fields remain estimates (causal traces need inference, which isn't run here) — every displayed number is taken from the checkpoint, none synthetic.</p>
   <div class="viz"><h3>Expert residency &amp; overflow (node A / node B / NVMe tier)</h3>
    <div class="g-note">Each cell is one expert: solid = resident on this node, hatched = stored but overflowed to NVMe (non-resident).</div>
    <div id="residency-heat"></div></div>
   <div class="viz"><h3>Channel keep-map (per layer)</h3>
    <div class="g-note">One strip per layer: bright green cells = channel kept (shade = measured importance), dark = pruned; outlined cells mark each <b>tile block</b> start. Hover any cell for exact importance.</div>
    <div id="channel-heat"></div></div>
   <div class="viz"><h3>Router repair (reindex)</h3>
    <div class="g-note">Grey discs kept in place; red discs dropped (renumbered away); solid dot = route-bias must move in lockstep.</div>
    <div id="router-map"></div></div>
   <div class="viz"><h3>Distillation targets &amp; residual repair</h3>
    <div class="g-note">Cells sized by priority/severity; &#8220;residual&#8221; cells additionally tinted by repairability.</div>
    <div id="distill-map"></div></div>
   <details class="raw"><summary>all planning-map tables (channel · tile · ownership · overflow · router · residual · distill)</summary><div class="raw-body">
    <h3>Channel map</h3><table id="t-channel"></table>
    <h3>Tile map</h3><table id="t-tile"></table>
    <h3>Node ownership</h3><table id="t-ownership"></table>
    <h3>Overflow pack (NVMe)</h3><table id="t-overflow"></table>
    <h3>Router repair</h3><table id="t-router"></table>
    <h3>Residual repair</h3><table id="t-residual"></table>
    <h3>Distillation targets</h3><table id="t-distill"></table>
   </div></details>
 </div>
 <div class="panel" id="panel-compression">
   <p class="note">Per-expert compression response (int4 vs int8), reconstruction error + output drift (measured math). One card per sampled expert; bars are relative to the worst reconstruction error in the set.</p>
   <div class="kgrid" id="comp-kpis"></div>
   <div class="grid3" id="comp-grid"></div>
   <details class="raw"><summary>raw compression table</summary><div class="raw-body"><table id="t-compression"></table></div></details>
 </div>
 <div class="panel" id="panel-candidate">
   <p class="note">Derivative candidates: kept experts/layer, resident bytes per node, go/no-go fit, held-out retention. The best measured candidate is highlighted.</p>
   <div class="grid2" id="cand-grid"></div>
   <details class="raw"><summary>raw candidate table</summary><div class="raw-body"><table id="t-candidate"></table></div></details>
 </div>
 <div class="panel" id="panel-heldout">
   <p class="note">Per-capability held-out retention (derivative vs source), measured. Bars show retention per capability label.</p>
   <div class="viz"><h3>Held-out retention by capability</h3><div class="g-note">Green ≥ 90%, amber ≥ 75%, red below. The dashed guide marks the mean.</div><div id="held-bars"></div></div>
   <details class="raw"><summary>raw held-out table</summary><div class="raw-body"><table id="t-heldout"></table></div></details>
 </div>
 <div class="panel" id="panel-reality">
   <p class="note">Real-bytes derivative envelopes (§24/§25) computed from measured checkpoint bytes — the mounted GLM-5.2 NVFP4 when present, else a synthetic caret. Retention fractions are estimates (a routing census needs inference); the byte math is measured.</p>
   <div class="kgrid" id="rl-kpis"></div>
   <div class="viz"><h3>Derivative envelopes</h3><div class="g-note">Stored size vs resident size per node — bar length is relative to the largest envelope.</div><div id="rl-envbars"></div></div>
   <details class="raw"><summary>raw envelope table</summary><div class="raw-body"><table id="t-reality"></table></div></details>
 </div>
 <div class="panel" id="panel-pareto">
   <p class="note">Pareto explorer: nondominated frontier, knee as a scored <b>region</b> (never a single point), and per-candidate neighbor deltas (fidelity / compact) with marginal quality-per-GiB. <b>Predicted candidates are shown hollow; measured are solid.</b></p>
   <div class="kgrid" id="pareto-kpis"></div>
   <div class="viz"><h3>Frontier scatter (quality vs resident GiB)</h3><canvas id="pareto-canvas" width="900" height="340" style="max-width:100%"></canvas></div>
   <div class="viz"><h3>Knee region</h3><div class="g-note">The scored compromise band — never a single recommended point.</div><div id="pareto-knee"></div></div>
   <div class="viz"><h3>Neighbor deltas (move fidelity ↔ compact)</h3><div class="g-note">What you gain/lose moving one step along the frontier.</div><div id="pareto-neighbors"></div></div>
   <details class="raw"><summary>raw frontier table</summary><div class="raw-body"><table id="t-pareto-frontier"></table></div></details>
 </div>
 <div class="panel" id="panel-v3">
   <p class="note">V3 fidelity-first analyzers: spectral / shared-structure / conditional-sensitivity / routing-consistency / global EXL3 bit-budget / NVFP4 suitability / quant-interaction / KV+system ledger / structural fallback, wired by the canonical pipeline. <b>Predictions are never styled as measured.</b></p>
   <div class="flow" id="v3-flow"></div>
   <div class="kgrid" id="v3-kpis"></div>
   <div class="viz"><h3>Spectral fingerprints</h3><div class="g-note">Effective rank + energy concentration per tensor — low effective rank with high top-3 energy = highly structured weight.</div><div id="spec-strip"></div></div>
   <div class="viz"><h3>NVFP4 suitability</h3><div class="g-note">Per-cell reconstruction error vs acceptance threshold.</div><div id="nvfp4-strip"></div></div>
   <details class="raw"><summary>remaining v3 tables (shared structure · conditional sensitivity · bit budgets · fallback · frontier)</summary><div class="raw-body">
     <h3>Shared structure</h3><table id="t-shared"></table>
     <h3>Conditional sensitivity</h3><table id="t-cond"></table>
     <h3>Routing consistency</h3><div id="v3-routing"></div>
     <h3>Global EXL3 bit budget</h3><table id="t-bitmaps"></table>
     <h3>NVFP4 suitability</h3><table id="t-nvfp4"></table>
     <h3>KV / system ledger</h3><div id="t-kv"></div>
     <h3>Structural fallback</h3><table id="t-fallback"></table>
     <h3>V3 canonical pipeline stages</h3><div id="v3-stage-list"></div>
   </div></details>
 </div>
 <div class="panel" id="panel-candidates">
   <p class="note">Candidate graph: immutable lineage, predicted-vs-measured status, operators + provenance, memory breakdown, routing stability, corpus hotspots. Predictions can never be deployable.</p>
   <div class="grid2" id="cg-grid"></div>
   <details class="raw"><summary>raw candidate-graph table</summary><div class="raw-body"><table id="t-candidates"></table></div></details>
 </div>
 <div class="panel" id="panel-corpus">
   <p class="note">Corpus ↔ model bidirectional evidence: semantic clusters, per-cluster expert coverage, expert→activating-clusters, and teacher-relative quality deltas projected onto clusters. Insufficient-evidence clusters are blocked from auto-compression.</p>
   <div class="grid3" id="corpus-grid"></div>
   <div class="viz"><h3>Quality deltas projected onto clusters</h3><div class="g-note">Positive = derivative improved on that cluster's utility.</div><div id="delta-bars"></div></div>
   <details class="raw"><summary>coverage + expert-mapping tables</summary><div class="raw-body">
     <h3>Cluster × expert coverage</h3><table id="t-cluster-coverage"></table>
     <h3>Expert → activating clusters</h3><table id="t-expert-clusters"></table>
   </div></details>
 </div>
 </main>
</div>
</div>
<script>
 const DATA = {payload};
// ---- helpers ----
(function(){{
  var m = DATA.meta||{{}};
  var ht=document.querySelector('.hero .ht');
  if(ht) ht.textContent = 'Atlas Profile Platform — '+[m.arch?('arch '+m.arch):null, m.layers?('L'+m.layers):null, m.experts?('E'+m.experts):null, m.top_k?('top-'+m.top_k):null, m.seed!=null?('seed '+m.seed):null].filter(Boolean).join(' · ');
}})();
function esc(s){{ var d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}
function el(t, rows){{ if(!rows||!rows.length){{return "<tr><td class='note'>no data</td></tr>";}}
  return rows.map(r=>{{let tds = Object.entries(r).map(([k,v]) => {{let s = Array.isArray(v)?v.map(x=>`<span class='chip'>${{x}}</span>`).join(''):(typeof v==='boolean'?(v?'<span class=green>yes</span>':'<span class=red>no</span>'):v);
    return `<td>${{s}}</td>`}}).join(''); return `<tr>${{tds}}</tr>`;}}).join(""); }}
function cols(headers){{return "<thead><tr>"+headers.map(h=>`<th scope="col">${{h}}</th>`).join("")+"</tr></thead>";}}
function fill(id, headers, rows){{var t=document.getElementById(id); if(!t) return; t.innerHTML = cols(headers)+el(null,rows);}}
function barRow(label, frac, valTxt, color){{
  var w=Math.max(2,Math.min(100,Math.round(frac*100)));
  return "<div class='bar-row'><span class='bl'>"+esc(label)+"</span><span class='bar'><i style='width:"+w+"%;background:"+color+"'></i></span><span class='bv'>"+valTxt+"</span></div>";
}}
function kcard(k,v,s,hl){{ return "<div class='kcard"+(hl?' hl':'')+"'><span class='k'>"+k+"</span><span class='v'>"+v+"</span>"+(s?"<span class='s'>"+s+"</span>":"")+"</div>"; }}
function tierOf(s){{return s>=0.75?'strong':s>=0.5?'good':s>=0.25?'moderate':'weak';}}
function sc(s){{return s>=0.75?'#e2b45c':s>=0.5?'#c6cdd8':s>=0.25?'#b08e6b':'#d0686b';}}
function retCol(r){{return r>=0.9?'#4ade80':r>=0.75?'#fbbf24':'#f87171';}}

// ---- Summary: pipeline stepper + verdicts + where-to-look ----
(function(){{
  var meta = DATA.meta||{{}};
  var stages = [
    {{n:'stage 1', t:'Profile', s:'calibration corpus → saliency', v:(DATA.capability||[]).length+' capabilities'}},
    {{n:'stage 2', t:'Contrast', s:'success − failure splits', v:(DATA.contrast||[]).length+' labels'}},
    {{n:'stage 3', t:'Plan', s:'candidates under byte budgets', v:(DATA.candidates||[]).length+' plans'}},
    {{n:'stage 4', t:'Verify', s:'held-out retention gate', v:DATA.heldout&&DATA.heldout.length?DATA.heldout.length+' labels':'—'}},
    {{n:'stage 5', t:'Map', s:'planning artifacts', v:DATA.maps?(DATA.maps.channel||[]).length+' channel rows':'—'}}
  ];
  var st=document.getElementById('sum-stepper');
  if(st) st.innerHTML = stages.map(function(x){{return "<div class='step'><div class='sn'>"+x.n+"</div><div class='st'>"+x.t+"</div><div class='sv'>"+x.v+"</div><div style='color:var(--mut);font-size:11px;margin-top:2px'>"+x.s+"</div></div>";}}).join('');
  var best=null, all=[];
  (DATA.candidates||[]).forEach(function(c){{ all.push({{name:c.name, ret:c.retention}}); if(c.fitted&&!c.promotion_blocked&&(!best||c.retention>best.ret)) best={{name:c.name, ret:c.retention}}; }});
  var v=document.getElementById('sum-verdicts');
  if(v){{
    var html='';
    html+="<div class='verdict'><div class='vt'>best derivative</div><div class='vm'>"+(best?("<b>"+esc(best.name)+"</b> · "+Math.round(best.ret*100)+"% retention"):"none fitted")+"</div></div>";
    var totC=0, hotC=0; (DATA.coalitions||[]).forEach(function(p){{ totC++; if(p.catastrophic) hotC++; }});
    html+="<div class='verdict'><div class='vt'>expert pairings</div><div class='vm'><b>"+totC+"</b> co-routed pairs"+(hotC?(" · <span style='color:var(--warn)'>"+hotC+" cascade-risk</span>"):" · none hot")+"</div></div>";
    var strong=0, weak=0; (DATA.capability||[]).forEach(function(r){{ (r.top||[]).forEach(function(x){{ if(x.score>=0.75) strong++; else if(x.score<0.25) weak++; }}); }});
    html+="<div class='verdict'><div class='vt'>expert strength spread</div><div class='vm'><b>"+strong+"</b> strong cells · <b>"+weak+"</b> weak</div></div>";
    var avgRet=null; if(DATA.heldout&&DATA.heldout.length){{ avgRet=DATA.heldout.reduce(function(a,r){{return a+r.retention;}},0)/DATA.heldout.length; }}
    html+="<div class='verdict'><div class='vt'>held-out gate</div><div class='vm'>"+(avgRet==null?"no data":"mean <b>"+Math.round(avgRet*100)+"%</b> retention "+(avgRet>=0.9?"<span style='color:var(--ok)'>· pass</span>":"<span style='color:var(--warn)'>· review</span>"))+"</div></div>";
    var real = DATA.reality||{{}};
    html+="<div class='verdict'><div class='vt'>checkpoint reality</div><div class='vm'><b>"+esc(String(real.source||'—'))+"</b> · "+(real.measured_gib||'—')+" GiB measured</div></div>";
    v.innerHTML=html;
  }}
  var mp=document.getElementById('sum-map');
  if(mp){{
    var links=[["Profiling","Experts","which experts carry which capability","capability"],["Profiling","Success−Failure","who fires on wins vs losses","contrast"],["Profiling","Expert Pairings","co-routing + cascade risk","coalition"],["Profiling","Structure","hierarchy + route paths","structure"],["Quantization & Fit","Compression","int4/int8 response per expert","compression"],["Quantization & Fit","Derivatives","prune/keep plans + fit verdict","candidate"],["Researcher","Pareto Explorer","quality vs GiB frontier","pareto"],["Researcher","V3 Analyzers","fidelity-first deep dives","v3"]];
    mp.innerHTML=links.map(function(l){{return "<div class='verdict' style='cursor:pointer' onclick=&quot;document.querySelector(&apos;[data-tab="+l[3]+"]&apos;).click()&quot;><div class='vt'>"+l[0]+"</div><div class='vm'><b>"+l[1]+"</b><span style='color:var(--mut);font-size:12px'> — "+l[2]+"</span></div></div>";}}).join('');
  }}
  var rk=document.getElementById('sum-rawkeys');
  if(rk) rk.innerHTML = Object.keys(DATA).map(function(k){{return "<span class='chip'>"+esc(k)+"</span>";}}).join('');
}})();

// ---- Experts: capability champions above the existing table ----
(function(){{
  var box=document.getElementById('cap-champs'); if(!box) return;
  box.innerHTML = (DATA.capability||[]).map(function(r){{
    var t=(r.top&&r.top[0])||null; if(!t) return '';
    return "<div class='cap-champ'><div class='cl'>"+esc(r.label)+"</div><div class='cv'><span>L"+t.layer+" · E"+t.expert+"</span><b>"+Math.round(t.score*100)+"%</b></div></div>";
  }}).join('');
}})();

// ---- Success−Failure: headline counts ----
(function(){{
  var box=document.getElementById('con-counts'); if(!box) return;
  var pos=0,neg=0,neu=0,tot=0;
  (DATA.contrast||[]).forEach(function(r){{ (r.top||[]).forEach(function(x){{ tot++; if(x.delta>0.12) pos++; else if(x.delta<-0.12) neg++; else neu++; }}); }});
  function cc(cls,n,lab){{ return "<span class='stat' style='padding:7px 12px'><span class='k'>"+lab+"</span><span class='v' style='color:var(--"+cls+")'>"+n+"</span></span>"; }}
  box.innerHTML = cc('ok',pos,'success-leaning')+cc('bad',neg,'failure-leaning')+cc('info',neu,'balanced')+cc('',tot,'cells measured');
}})();

// ---- Structure: hierarchy level bars + path bars ----
(function(){{
  var hb=document.getElementById('hier-bars'); if(hb&&DATA.hierarchy&&DATA.hierarchy.levels){{
    var counts=DATA.hierarchy.counts||{{}}; var mx=Math.max.apply(null,DATA.hierarchy.levels.map(function(l){{return counts[l]||1;}}))||1;
    hb.innerHTML=DATA.hierarchy.levels.map(function(l,i){{ var c=counts[l]||0;
      return "<div class='lvlbar'><span class='ll'>L"+(i+1)+" "+l+"</span><span class='lb'><i style='width:"+Math.max(3,Math.round(c/mx*100))+"%'></i></span><span class='lv'>"+c+"</span></div>"; }}).join('');
  }}
  var pb=document.getElementById('path-bars'); if(pb&&DATA.paths&&DATA.paths.length){{
    var mxp=Math.max.apply(null,DATA.paths.map(function(r){{return r.count;}}))||1;
    pb.innerHTML=DATA.paths.map(function(r){{
      var sig=(r.signature||[]).map(function(s){{return s.join('·');}}).join(' → ');
      return "<div class='lvlbar'><span class='ll' style='min-width:170px' title='"+esc(sig)+"'>"+esc(sig.length>28?sig.slice(0,27)+'…':sig)+"</span><span class='lb'><i style='width:"+Math.max(3,Math.round(r.count/mxp*100))+"%;background:linear-gradient(90deg,#7dd3fc,#5eead4)'></i></span><span class='lv'>"+Math.round(r.success_rate*100)+"% ok</span></div>";
    }}).join('');
  }}
}})();

// ---- Compression: KPIs + per-expert cards ----
(function(){{
  var comps=DATA.compression||[];
  var k=document.getElementById('comp-kpis');
  if(k&&comps.length){{
    var worst=0, n=0; comps.forEach(function(c){{ c.points.forEach(function(p){{ n++; if(p.recon>worst) worst=p.recon; }}); }});
    k.innerHTML = kcard('experts sampled',comps.length,'layer × expert cells')+kcard('worst recon err',worst.toFixed(4),'across '+n+' measured points')+kcard('formats','int4 · int8','per-point repair flags');
  }}
  var g=document.getElementById('comp-grid'); if(!g) return;
  var mx=0; comps.forEach(function(c){{ c.points.forEach(function(p){{ if(p.recon>mx) mx=p.recon; }}); }}); mx=mx||1;
  g.innerHTML = comps.map(function(c){{
    var head="<div class='comp-head'><span>L"+c.layer+" E"+c.expert+"</span></div>";
    var rows=c.points.map(function(p){{
      var w=Math.max(4,Math.round(p.recon/mx*100));
      var col=p.recon>mx*0.66?'var(--bad)':p.recon>mx*0.33?'var(--warn)':'var(--ok)';
      return "<div class='comp-row'><span class='fmt'>"+esc(p.format)+"</span><span class='m'><i style='width:"+w+"%;background:"+col+"'></i></span><span>"+p.recon.toFixed(4)+"</span>"+(p.repair?"<span class='pill bad'>repair</span>":"")+"</div>";
    }}).join('');
    return "<div class='comp-card'>"+head+rows+"</div>";
  }}).join('');
}})();

// ---- Derivatives: candidate cards ----
(function(){{
  var g=document.getElementById('cand-grid'); if(!g) return;
  var cands=(DATA.candidates||[]).slice().sort(function(a,b){{ return b.retention-a.retention; }});
  var best=cands.length&&cands[0].fitted&&!cands[0].promotion_blocked?cands[0]:null;
  g.innerHTML=cands.map(function(c){{
    var isBest=best&&c.name===best.name;
    var keep=c.kept_per_layer?Object.values(c.kept_per_layer).join('·'):'—';
    var retPct=Math.round(c.retention*100);
    var retCol2=retCol(c.retention);
    var badges=(c.fitted?"<span class='pill ok'>fitted</span>":"<span class='pill dim'>not fitted</span>")+" "+(c.promotion_blocked?"<span class='pill bad'>promo blocked</span>":"<span class='pill ok'>promo ok</span>")+(isBest?" <span class='pill gold'>best</span>":"");
    return "<div class='cand-card"+(isBest?' best':'')+"'><div class='cand-top'><span class='cand-name'>"+esc(c.name)+"</span><span>"+badges+"</span></div>"+
      "<div class='cand-ret'><span class='big' style='color:"+retCol2+"'>"+retPct+"%</span><span class='lbl'>held-out retention</span><span style='color:var(--dim);font-size:11px'>worst drop "+Math.round(c.worst_drop*100)+"%</span></div>"+
      "<div class='cand-meta'><span>kept/layer <b>"+esc(keep)+"</b></span><span>resident A <b>"+c.resident_a+"</b></span><span>resident B <b>"+c.resident_b+"</b></span></div></div>";
  }}).join('');
}})();

// ---- Held-out: retention bars ----
(function(){{
  var b=document.getElementById('held-bars'); if(!b||!DATA.heldout||!DATA.heldout.length) return;
  var mean=DATA.heldout.reduce(function(a,r){{return a+r.retention;}},0)/DATA.heldout.length;
  b.innerHTML=DATA.heldout.map(function(r){{
    var w=Math.max(2,Math.round(r.retention*100));
    var guide=mean>=0&&mean<=1?"<span class='fbar' style='flex:1;position:relative'><i style='width:"+w+"%;background:"+retCol(r.retention)+"'></i></span>":"";
    return "<div class='bar-row'><span class='bl'>"+esc(r.label)+"</span>"+guide+"<span class='bv'>"+Math.round(r.retention*100)+"%</span></div>";
  }}).join('')+"<div class='g-note' style='margin-top:6px'>mean "+Math.round(mean*100)+"% · n per label shown in raw table</div>";
}})();

// ---- Real-bytes: KPIs + envelope bars ----
(function(){{
  var R=DATA.reality||{{}};
  var k=document.getElementById('rl-kpis');
  if(k) k.innerHTML = kcard('checkpoint',esc(String(R.source||'—')),'measured from bytes',true)+kcard('measured size',(R.measured_gib||'—')+' GiB','on-disk checkpoint');
  var b=document.getElementById('rl-envbars');
  if(b&&R.candidates&&R.candidates.length){{
    var mx=0; R.candidates.forEach(function(r){{ mx=Math.max(mx,r.stored||0,r.resident_a||0,r.resident_b||0); }}); mx=mx||1;
    b.innerHTML=R.candidates.map(function(r){{
      function row(lab,v){{ var w=Math.max(2,Math.round((v||0)/mx*100)); return "<div class='bar-row'><span class='bl'>"+lab+"</span><span class='bar'><i style='width:"+w+"%;background:#7dd3fc'></i></span><span class='bv'>"+v+" GiB</span></div>"; }}
      return "<div class='comp-card'><div class='comp-head'><span>"+esc(r.envelope)+" GiB · keep "+Math.round(r.keep*100)+"% · "+esc(r.precision)+"</span><span class='pill "+(r.risk==='low'?'ok':(r.risk==='high'?'bad':'warn'))+"'>"+esc(String(r.risk||'—'))+" risk</span></div>"+row('stored',r.stored)+row('resident A',r.resident_a)+row('resident B',r.resident_b)+"</div>";
    }}).join('');
  }}
}})();

// ---- Pareto: KPIs + scatter + knee + neighbors (frontier table id fixed to t-pareto-frontier) ----
(function(){{
  var pf = DATA.v3 && DATA.v3.pareto; if(!pf) return;
  var pts = pf.points||[]; var fids = pf.frontier_ids||[]; var knee = pf.knee_region||[];
  var k=document.getElementById('pareto-kpis');
  if(k) k.innerHTML = kcard('frontier',fids.length,'nondominated')+kcard('knee region',knee.length+' pts','scored compromise band',true)+kcard('total candidates',pts.length,'predicted + measured');
  document.getElementById('pareto-knee').innerHTML =
    (knee.length ? knee.map(id=>`<span class='chip'>${{id}}</span>`).join('') :
    '<p class="note">no knee detected</p>');
  var nd = pf.neighbor_deltas||{{}};
  var nb = [];
  Object.entries(nd).forEach(function(e){{ var from=e[0]; var list=e[1]; (list||[]).forEach(function(dn){{ nb.push({{'from':from,'to':dn.candidate_id||'-','move':dn.direction||'-','dQ':dn.dquality,'dGiB':dn.dresident_gib,'dT/s':dn.ddecode_tps,'dQ/GiB':dn.quality_per_gib}}); }}); }});
  if (!nb.length) {{
    var srt = pts.filter(p=>fids.includes(p.candidate_id)).slice().sort(function(a,b){{ return a.values.resident_gib-b.values.resident_gib; }});
    srt.forEach(function(p,i){{ if(i>0) nb.push({{'from':p.candidate_id,'to':srt[i-1].candidate_id,'move':'fidelity','dQ':(p.values.quality-srt[i-1].values.quality).toFixed(3),'dGiB':(p.values.resident_gib-srt[i-1].values.resident_gib).toFixed(2),'dT/s':(p.values.decode_tps-srt[i-1].values.decode_tps).toFixed(1),'dQ/GiB':'—'}}); if(i<srt.length-1) nb.push({{'from':p.candidate_id,'to':srt[i+1].candidate_id,'move':'compact','dQ':(p.values.quality-srt[i+1].values.quality).toFixed(3),'dGiB':(p.values.resident_gib-srt[i+1].values.resident_gib).toFixed(2),'dT/s':(p.values.decode_tps-srt[i+1].values.decode_tps).toFixed(1),'dQ/GiB':'—'}}); }});
  }}
  fill('pareto-neighbors', ['from','to','move','dQ','dGiB','dT/s','dQ/GiB'], nb);
  fill('t-pareto-frontier', ['candidate','quality','resident GiB','decode','frontier','knee'],
    pts.map(pt=>({{'candidate':pt.candidate_id,'quality':pt.values.quality,'resident GiB':pt.values.resident_gib,
      'decode':pt.values.decode_tps,'frontier':(fids.includes(pt.candidate_id)?'yes':'no'),
      'knee':(knee.includes(pt.candidate_id)?'knee':'-')}})));
  var cv = document.getElementById('pareto-canvas');
  if (cv && cv.getContext && pts.length) {{
    var ctx = cv.getContext('2d'); var W=cv.width, H=cv.height, pad=42;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='#12151a'; ctx.fillRect(0,0,W,H);
    var qs = pts.map(p=>p.values.quality), rs = pts.map(p=>p.values.resident_gib);
    var qmin=Math.min.apply(null,qs), qmax=Math.max.apply(null,qs);
    var rmin=Math.min.apply(null,rs), rmax=Math.max.apply(null,rs);
    var qr=(qmax-qmin)||1, rr=(rmax-rmin)||1;
    function X(v){{ return pad + (v-rmin)/rr*(W-2*pad); }}
    function Y(v){{ return H-pad - (v-qmin)/qr*(H-2*pad); }}
    ctx.strokeStyle='#3a4350'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
    ctx.fillStyle='#98a2b3'; ctx.font='11px system-ui';
    ctx.fillText('resident GiB →', W/2, H-8); ctx.save(); ctx.translate(14, H/2); ctx.rotate(-Math.PI/2); ctx.fillText('quality ↑',0,0); ctx.restore();
    var fpts = pts.filter(p=>fids.includes(p.candidate_id)).sort((a,b)=>a.values.resident_gib-b.values.resident_gib);
    ctx.strokeStyle='#e2b45c'; ctx.lineWidth=2; ctx.beginPath();
    fpts.forEach(function(p,i){{ var x=X(p.values.resident_gib), y=Y(p.values.quality); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }}); ctx.stroke();
    pts.forEach(function(p){{ var isKnee=knee.includes(p.candidate_id); var isFront=fids.includes(p.candidate_id);
      var x=X(p.values.resident_gib), y=Y(p.values.quality);
      ctx.beginPath(); ctx.arc(x,y, isKnee?7:5, 0, Math.PI*2);
      ctx.fillStyle = isKnee ? '#e2b45c' : (isFront ? '#c6cdd8' : '#5b5b5b'); ctx.fill();
      ctx.strokeStyle = isKnee ? '#fff' : '#12151a'; ctx.lineWidth=2; ctx.stroke();
      ctx.fillStyle='#98a2b3'; ctx.font='9px system-ui'; ctx.textAlign='center';
      ctx.fillText(p.candidate_id, x, y-9);
    }});
  }}
}})();

// ---- V3: flow + KPIs + strips ----
(function(){{
  var v = DATA.v3; if(!v) return;
  var fl=document.getElementById('v3-flow');
  if(fl) fl.innerHTML = v.stages_run.map(function(s,i){{ return (i?"<span class='farrow'>→</span>":"")+"<span class='fnode'>"+esc(s)+" <span class='mute'>("+esc(v.evidence[s]||'measured')+")</span></span>"; }}).join('');
  var k=document.getElementById('v3-kpis');
  if(k) k.innerHTML = kcard('stages run',v.stages_run.length,'canonical pipeline')+kcard('routing identity',v.routing_consistency_passed?'pass':'FAIL',v.routing_consistency_passed?'consistent routing':'mismatch detected',!v.routing_consistency_passed);
  var sp = v.spectral && v.spectral.rows || [];
  var strip=document.getElementById('spec-strip');
  if(strip&&sp.length){{
    var mxr=Math.max.apply(null,sp.map(function(r){{return r.effective_rank||1;}}))||1;
    strip.innerHTML=sp.slice(0,24).map(function(r){{
      var w=Math.max(4,Math.round((r.effective_rank||0)/mxr*100));
      var col=r.heavy_tail?'var(--gold)':'var(--info)';
      return "<div class='bar-row'><span class='bl' title='"+esc(String(r.tensor||''))+"'>L"+r.layer+" E"+r.expert+"</span><span class='bar'><i style='width:"+w+"%;background:"+col+"'></i></span><span class='bv'>rank "+(r.effective_rank||0)+"</span></div>";
    }}).join('');
  }}
  var nv = v.nvfp4 && v.nvfp4.rows || [];
  var ns=document.getElementById('nvfp4-strip');
  if(ns&&nv.length){{
    ns.innerHTML=nv.slice(0,24).map(function(r){{
      var w=Math.max(3,Math.min(100,Math.round((r.reconstruction_error||0)*400)));
      var col=r.accepted?'var(--ok)':'var(--bad)';
      return "<div class='bar-row'><span class='bl'>L"+r.layer+" E"+r.expert+"</span><span class='bar'><i style='width:"+w+"%;background:"+col+"'></i></span><span class='bv'>"+(r.accepted?'<span class=pill ok>ok</span>':'<span class=pill bad>no</span>')+"</span></div>";
    }}).join('');
  }}
  var sh = v.shared_structure && v.shared_structure.rows || [];
  var shMap = sh.slice(0,20).map(r=>({{'expert':r.expert,'shared ratio':r.shared_energy_ratio,'unique ratio':r.unique_energy_ratio,'proj savings':r.projected_storage_savings}}));
  fill('t-shared', Object.keys(shMap[0]||{{}}), shMap);
  var cs = v.conditional_sensitivity && v.conditional_sensitivity.rows || [];
  var csMap = cs.filter(r=>r.expert===0).map(r=>({{'layer':r.layer,'expert':r.expert,'upstream noise':r.upstream_noise,'recon error':r.reconstruction_error}}));
  fill('t-cond', Object.keys(csMap[0]||{{}}), csMap);
  document.getElementById('t-kv').innerHTML = v.kv_plan ? `<span class='stat'><span class='k'>recommended KV</span><span class='v'>${{v.kv_plan.recommended_format}}</span></span>` +
    `<span class='stat'><span class='k'>headroom GiB</span><span class='v'>${{(v.kv_plan.headroom_bytes/1073741824).toFixed(2)}}</span></span>` +
    `<span class='stat'><span class='k'>ctx target</span><span class='v'>${{v.kv_plan.context_target_tokens}}</span></span>` : 'no kv plan';
  var bm = [];
  Object.entries(v.bit_maps||{{}}).forEach(function(entry){{ var k2=entry[0], o=entry[1]; bm.push({{'budget GiB':k2,'assignments':(o&&o.assignments||[]).length,'mean bpw': o&&o.mean_bpw||0}}); }});
  fill('t-bitmaps', ['budget GiB','assignments','mean bpw'], bm);
  var nvMap = nv.slice(0,30).map(r=>({{'layer':r.layer,'expert':r.expert,'recon err':r.reconstruction_error,'router impact':r.routing_impact,'accepted':r.accepted?'yes':'no','recovery':r.recovery_kind}}));
  fill('t-nvfp4', Object.keys(nvMap[0]||{{}}), nvMap);
  var fb = v.structural_fallback || [];
  fill('t-fallback', ['reduction %','preserved routing'], fb.map(r=>({{'reduction %':r.reduction_percent,'preserved routing':r.preserved_routing_destinations?'yes':'no','blocked':(r.blocked_capacity||[]).length}})));
  var sl=document.getElementById('v3-stage-list');
  if(sl) sl.innerHTML = v.stages_run.map(s=>`<span class='chip'>${{s}} <span class=mute>(${{v.evidence[s]||'measured'}})</span></span>`).join('');
}})();

// ---- Candidate graph: cards ----
(function(){{
  var cg = DATA.candidates_graph || {{nodes:{{}}}};
  var nodes = Object.values(cg.nodes||{{}});
  var g=document.getElementById('cg-grid'); if(!g) return;
  g.innerHTML=nodes.map(function(n){{
    var mem = n.memory_breakdown?Object.values(n.memory_breakdown||{{}}).reduce((a,b)=>a+(b||0),0)/1073741824:0;
    var q = n.quality_vector&&n.quality_vector.quality_retention!=null?n.quality_vector.quality_retention:null;
    var status = n.deployed?"<span class='pill ok'>deployed</span>":(n.predicted?"<span class='pill warn'>predicted</span>":"<span class='pill dim'>measured</span>");
    var ops=(n.operators||[]).join(' · ')||'—';
    return "<div class='cand-card'><div class='cand-top'><span class='cand-name'>"+esc(n.candidate_id||'—')+"</span><span>"+status+"</span></div>"+
      "<div class='cand-ret'><span class='big' style='font-size:19px;color:"+(q!=null?retCol(q):'var(--dim)')+"'>"+(q!=null?Math.round(q*100)+'%':'—')+"</span><span class='lbl'>quality retention</span></div>"+
      "<div class='cand-meta'><span>stage <b>"+esc(String(n.stage||'—'))+"</b></span><span>resident <b>"+mem.toFixed(2)+" GiB</b></span><span>ops <b>"+esc(ops)+"</b></span></div></div>";
  }}).join('');
}})();

// ---- Corpus: cluster cards + delta bars ----
(function(){{
  var cs = DATA.corpus || {{clusters:[],cluster_expert_coverage:[],expert_activation:[],deltas:[]}};
  var g=document.getElementById('corpus-grid'); if(g){{
    g.innerHTML=(cs.clusters||[]).map(function(c){{
      return "<div class='cluster-card'><div class='cd'>"+esc(c.cluster_id)+" · "+esc(c.domain)+"</div><div class='cm'>"+c.sample_ids.length+" samples · "+c.observations+" observations</div></div>";
    }}).join('');
  }}
  var db=document.getElementById('delta-bars'); if(db){{
    var ds=cs.deltas||[];
    var mxd=0; ds.forEach(function(d){{ mxd=Math.max(mxd,Math.abs(d.quality_delta||0)); }}); mxd=mxd||1;
    db.innerHTML=ds.map(function(d){{
      var v=d.quality_delta||0; var neg=v<0; var w=Math.max(3,Math.round(Math.abs(v)/mxd*100));
      return "<div class='bar-row'><span class='bl'>"+esc(d.cluster_id)+" → "+esc(d.candidate_id)+"</span><span class='bar'><i style='width:"+w+"%;background:"+(neg?'var(--bad)':'var(--ok)')+"'></i></span><span class='bv "+(neg?'delta-neg':'delta-pos')+"'>"+(neg?'':'+')+v.toFixed(4)+"</span></div>";
    }}).join('');
  }}
  var cc = cs.cluster_expert_coverage||[];
  fill('t-cluster-coverage', ['cluster','layer','expert','routed','freq','status'], cc.slice(0,40).map(r=>({{'cluster':r.cluster_id,'layer':r.layer,'expert':r.expert,'routed':r.routed_count,'freq':r.activation_frequency,'status':r.status}})));
  fill('t-expert-clusters', ['layer','expert','activating clusters','unique coverage'],
    cs.expert_activation.map(e=>({{'layer':e.layer,'expert':e.expert,'activating clusters':(e.activating_clusters||[]).join(',')||'-','unique coverage':e.unique_coverage}})) );
}})();

// ---- Tab switching ----
(function(){{
  var tabs = document.querySelectorAll('.tab');
  tabs.forEach(function(t){{
    t.addEventListener('click', function(){{
      tabs.forEach(function(x){{ x.classList.remove('active'); }});
      document.querySelectorAll('.panel').forEach(function(x){{ x.classList.remove('active'); }});
      t.classList.add('active');
      var p = document.getElementById('panel-' + t.dataset.tab);
      if (p) p.classList.add('active');
    }});
  }});
}})();

// ---- Maintenance timer (elapsed / remaining / expected) ----
(function(){{
  function fmt(s){{ s=Math.max(0,Math.round(s)); var h=Math.floor(s/3600), m=Math.floor((s%3600)/60), x=s%60;
    return (h?h+'h ':'')+(m||h?m+'m ':'')+x+'s'; }}
  function tick(){{
    var M = (window.DATA && DATA.maintenance) || {{}};
    var el = document.getElementById('mt-el');
    if (!el) return;
    if (!M.present){{ el.textContent = 'idle'; return; }}
    el.textContent = fmt(M.elapsed_seconds || 0);
  }}
  function once(){{
    var M = (window.DATA && DATA.maintenance) || {{}};
    var host = document.querySelector('#panel-maintenance');
    if (!host) return;
    var body = document.getElementById('maintenance-body');
    if (!body) return;
    var left = document.createElement('span');
    left.id = 'mt-left';
    left.style.cssText = 'color:var(--mut);font-size:12px';
    body.insertBefore(left, body.firstChild);
    function paint(){{
      var M2 = (window.DATA && DATA.maintenance) || {{}};
      if (!M2.present){{ left.textContent = 'No maintenance window running.'; return; }}
      var rem = M2.eta_remaining_seconds != null ? M2.eta_remaining_seconds : null;
      var tot = M2.estimated_total_seconds;
      left.textContent = 'remaining: ' + (rem != null ? fmt(rem) : '?')
        + (tot ? (' · expected: ' + fmt(tot)) : '')
        + (M2.status ? (' · ' + M2.status) : '');
    }}
    paint();
    tick();
    setInterval(tick, 1000);
  }}
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', once);
  else once();
}})();


// ---- Capability 3D voxel engine ----
{_CAP3D_JS}

// ---- Success-Failure 3D engine ----
{_CONTRAST_JS}
    </script>
</body></html>"""


def write_dashboard(path: str, seed: int = SEED) -> str:
    data = build_dashboard_data(seed=seed)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_dashboard(data))
    return path
