"""Cebu Profiler export executor — write the canonical `profiler_runs/<id>/` run dir.

Implements §1 + §3 of the cebu-bridge contract. Runs the real mini-MoE REAP
pipeline over an eval-lab task corpus, generates candidate keep-map plans,
optionally builds + registers a derivative, and writes the frozen JSON files
(``run_manifest.json`` / ``layer_saliency.json`` / ``plans.json`` /
``derivative.json``). This realizes the existing ``output_layout.CEBU_RUN_FILES``
contract, which previously had no writer.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cebu_profiler.builder import build_derivative, register_derivative
from cebu_profiler.ecosystem import build_mini_moe_for, prompt_corpus
from cebu_profiler.planning import CandidatePlan, SearchInputs, generate_candidates
from cebu_profiler.planning.maps_build import build_planning_maps
from cebu_profiler.profiler.coalition import coactivation_map
from cebu_profiler.profiler.compress import run_compression_pipeline
from cebu_profiler.profiler.hierarchy import build_hierarchy
from cebu_profiler.profiler.reap import SaliencyAccumulator, run_calibration
from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.schemas.model_asset import AssetType, ModelAsset
from cebu_profiler.schemas.ontology import DataPartition

# A deliberately generous per-node budget so every generated plan "fits"; the
# search is over value/coverage/coalition, not a hard memory constraint.
_NODE_BUDGET_BYTES = 1e12
_ACTIVE_BYTES_PER_TOKEN = 100.0
_TOP_K = 2
_CALIBRATION_SUITE = "cebu_calibration"
_SCHEMA_VERSION = "cebu-bridge-v1"


def _short_run_id() -> str:
    return f"cebu-{secrets.token_hex(4)}"


def _saliency_rows(saliency: SaliencyAccumulator) -> list[dict[str, Any]]:
    """One row per (layer, expert, label), aggregated across stages (§3)."""
    cells: dict[tuple[int, int, str], list[float]] = {}
    for (layer, expert, label, stage), s in saliency._sum.items():  # noqa: SLF001
        key = (layer, expert, label)
        row = cells.setdefault(key, [0.0, 0.0, 0.0])
        row[0] += s
        row[1] += saliency._count[(layer, expert, label, stage)]  # noqa: SLF001
        row[2] += saliency._freq[(layer, expert, label, stage)]  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    for (layer, expert, label), (total, count, freq) in sorted(cells.items()):
        rows.append(
            {
                "layer": layer,
                "expert": expert,
                "label": label,
                "mean": round(total / count, 5) if count else 0.0,
                "frequency": round(freq / count, 5) if count else 0.0,
                "total_value": round(saliency.total_value(layer, expert), 5),
            }
        )
    return rows


def _serialize_plan(plan: CandidatePlan, strategy: str, keep_per_layer: int) -> dict[str, Any]:
    """Serialize one candidate plan into the frozen §3 plans.json entry."""
    return {
        "name": plan.name,
        "strategy": strategy,
        "keep_per_layer": keep_per_layer,
        "kept_per_layer": {str(k): v for k, v in plan.kept_per_layer.items()},
        "resident_bytes_a": plan.resident_bytes_a,
        "resident_bytes_b": plan.resident_bytes_b,
        "keep_map": {
            "source_model_id": plan.keep.source_model_id,
            "entries": [
                {
                    "layer_index": e.layer_index,
                    "source_expert_id": e.source_expert_id,
                    "keep": e.keep,
                    "reason": e.reason,
                }
                for e in plan.keep.entries
            ],
        },
        "precision": {
            "entries": [
                {
                    "layer_index": e.layer_index,
                    "source_expert_id": e.source_expert_id,
                    "precision": e.precision,
                    "bits": e.bits,
                    "reconstruction_error": e.reconstruction_error,
                }
                for e in plan.precision.entries
            ]
        },
    }


def _planning_maps_payload(
    model: MiniMoE,
    saliency: SaliencyAccumulator,
    plans: list[CandidatePlan],
) -> dict[str, Any]:
    """Consolidate the seven granular §25 maps + per-candidate fit into one file."""
    ms = build_planning_maps(model, saliency)
    map_names = (
        "channel",
        "tile",
        "node_ownership",
        "overflow_pack",
        "router_repair",
        "residual_repair",
        "distillation_target",
    )
    return {
        "schema_version": 1,
        "source_arch": model.arch.name,
        "maps": {name: [e.model_dump() for e in getattr(ms, name).entries] for name in map_names},
        "candidates": [
            {
                "name": p.name,
                "kept_per_layer": {str(k): v for k, v in p.kept_per_layer.items()},
                "resident_bytes_a": p.resident_bytes_a,
                "resident_bytes_b": p.resident_bytes_b,
                "stored_bytes": p.stored_bytes,
                "coverage": (
                    round(p.keep.kept_count() / len(p.keep.entries), 4) if p.keep.entries else 0.0
                ),
                "precision": [
                    {
                        "layer_index": e.layer_index,
                        "source_expert_id": e.source_expert_id,
                        "precision": e.precision,
                        "bits": e.bits,
                        "reconstruction_error": e.reconstruction_error,
                    }
                    for e in p.precision.entries
                ],
            }
            for p in plans
        ],
    }


def _serialize_derivative(asset: ModelAsset, result: Any) -> dict[str, Any]:
    """Serialize the registered derivative ModelAsset into the §3 schema."""
    return {
        "model_asset_id": asset.model_asset_id,
        "display_name": asset.display_name,
        "asset_type": asset.asset_type.value,
        "model_family": asset.model_family,
        "architecture": asset.architecture,
        "checkpoint_path": asset.checkpoint_path,
        "parent_model_id": asset.parent_model_id,
        "source_experiment_id": asset.source_experiment_id,
        "kept_per_layer": {str(k): v for k, v in result.plan.kept_per_layer.items()},
        "stored_size_bytes": asset.stored_size_bytes,
        "estimated_resident_bytes": asset.estimated_resident_bytes,
        "identity_source_slots": {
            k: str(v) for k, v in asset.metadata.get("identity_source_slots", {}).items()
        },
    }


def _coalition_map(model: MiniMoE, corpus: Any, top_k: int) -> dict[int, list[tuple[int, ...]]]:
    """Coactivation coalitions per layer from the calibration corpus."""
    coalitions: dict[int, list[tuple[int, ...]]] = {}
    for layer in range(model.arch.num_text_layers):
        cmap = coactivation_map(model, corpus, layer, top_k=top_k)
        pairs: list[tuple[int, ...]] = list(cmap.candidate_coalitions(min_coactivity=1))
        coalitions[layer] = pairs
    return coalitions


def export_run(
    out_root: str,
    *,
    eval_lab_root: str,
    arch_name: str = "k3-mini",
    seed: int = 0,
    keep_per_layer: int = 4,
    partition: DataPartition = DataPartition.CEBU_CALIBRATION,
    build: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the profiler pipeline and write the canonical ``profiler_runs/<run_id>/`` dir.

    Returns ``{run_dir, run_manifest, plan_names}``.
    """
    model: MiniMoE = build_mini_moe_for(arch_name, seed)
    corpus = prompt_corpus(
        eval_lab_root,
        vocab=model.arch.vocabulary_size or 1000,
        seed=seed,
        partition=partition,
    )
    if not corpus:
        raise ValueError(f"no eval-lab task prompts under {eval_lab_root}")

    saliency = run_calibration(model, corpus, top_k=_TOP_K)
    coalitions = _coalition_map(model, corpus, _TOP_K)
    inputs = SearchInputs(model=model, saliency=saliency, coalitions=coalitions)
    plans = generate_candidates(
        inputs,
        keep_budget_per_layer=keep_per_layer,
        node_budget_bytes=_NODE_BUDGET_BYTES,
        active_bytes_per_token=_ACTIVE_BYTES_PER_TOKEN,
    )

    run_id = run_id or _short_run_id()
    run_dir = Path(out_root) / "profiler_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat()
    evidence = [
        "run_manifest.json",
        "layer_saliency.json",
        "plans.json",
        "compression_manifest.json",
        "hierarchy_map.json",
    ]

    (run_dir / "layer_saliency.json").write_text(
        json.dumps(_saliency_rows(saliency), indent=2, sort_keys=True)
    )

    plan_list = [_serialize_plan(p, p.name.rsplit("-", 1)[-1], keep_per_layer) for p in plans]
    (run_dir / "plans.json").write_text(json.dumps(plan_list, indent=2, sort_keys=True))

    # §27 compression artifact: trace -> TENP -> stability -> causal -> Taylor
    # -> SM121 width-bucket planner, over the same calibration corpus.
    compression, _cv = run_compression_pipeline(model, corpus, n_stability_runs=3)
    (run_dir / "compression_manifest.json").write_text(compression.model_dump_json(indent=2))

    # §27 hierarchy artifact: the six-level profiler map (v2 §9) built from the
    # same measured calibration corpus — traceable up (weights→behaviour) and
    # down (behaviour→weights).
    hierarchy = build_hierarchy(model, corpus, top_k=_TOP_K)
    (run_dir / "hierarchy_map.json").write_text(
        json.dumps(hierarchy.to_dict(), indent=2, sort_keys=True)
    )

    if build:
        result = build_derivative(model, plans[0])
        source_asset = ModelAsset(
            model_asset_id=f"src-{arch_name}",
            display_name=f"{arch_name} source",
            asset_type=AssetType.SOURCE_CHECKPOINT,
            model_family=arch_name,
            architecture=arch_name,
            checkpoint_path=f"/models/{arch_name}",
        )
        asset = register_derivative(
            result,
            display_name=f"{arch_name} derivative ({plans[0].name})",
            source_asset=source_asset,
            model_family=arch_name,
            source_experiment_id=None,
        )
        (run_dir / "derivative.json").write_text(
            json.dumps(_serialize_derivative(asset, result), indent=2, sort_keys=True)
        )
        evidence.append("derivative.json")

    # §25 consolidated planning-maps artifact: the seven granular §25 maps plus
    # per-candidate precision/residency — surfaced natively by the eval-harness
    # Cebu Lab via the manifest bridge.
    (run_dir / "planning_maps.json").write_text(
        json.dumps(_planning_maps_payload(model, saliency, plans), indent=2, sort_keys=True)
    )
    evidence.append("planning_maps.json")

    # --- V3 fidelity-first artifacts (analyzers + candidate graph + corpus) ---
    from cebu_profiler.analysis import build_corpus_semantic_map
    from cebu_profiler.candidates import CandidateGraph, CandidateNode, CandidateStage
    from cebu_profiler.profiler.v3_pipeline import run_v3_pipeline, v3_run_to_jsonable
    from cebu_profiler.schemas.coverage import EvidenceGate

    v3run = run_v3_pipeline(model, corpus, seed=seed)
    (run_dir / "v3_run.json").write_text(
        json.dumps(v3_run_to_jsonable(v3run), indent=2, sort_keys=True)
    )
    evidence.append("v3_run.json")

    semantic = build_corpus_semantic_map(model, corpus, top_k=_TOP_K, gate=EvidenceGate())
    (run_dir / "v3_corpus_evidence.json").write_text(
        json.dumps(semantic.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    evidence.append("v3_corpus_evidence.json")

    g = CandidateGraph(source_teacher_id=f"teacher-{arch_name}")
    g.add(
        CandidateNode(
            candidate_id="teacher",
            name="BF16 teacher",
            stage=CandidateStage.P0_REFERENCE,
            predicted=False,
            deployed=True,
        )
    )
    g.add(
        CandidateNode(
            candidate_id="mk-exl3",
            name="EXL3 global allocation",
            parent_ids=["teacher"],
            stage=CandidateStage.P4_EXL3,
            predicted=True,
        )
    )
    g.add(
        CandidateNode(
            candidate_id="mk-exl3-nvfp4",
            name="+NVFP4 substitution",
            parent_ids=["mk-exl3"],
            stage=CandidateStage.P6_SM121_ALLOCATION,
            predicted=True,
        )
    )
    (run_dir / "v3_candidate_graph.json").write_text(
        json.dumps(g.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    evidence.append("v3_candidate_graph.json")

    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "profiler_run_id": run_id,
        "source_arch": arch_name,
        "calibration_suite_id": _CALIBRATION_SUITE,
        "evidence_level": "basic_saliency",
        "status": "completed",
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "n_tasks": len(corpus),
        "evidence_present": sorted(evidence),
        "software_revision": None,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return {
        "run_dir": str(run_dir),
        "run_manifest": manifest,
        "plan_names": [p.name for p in plans],
    }
