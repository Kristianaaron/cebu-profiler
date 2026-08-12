"""Atlas export executor — write the canonical `atlas_runs/<id>/` run dir.

Implements §1 + §3 of the atlas-bridge contract. Runs the real mini-MoE REAP
pipeline over an eval-lab task corpus, generates candidate keep-map plans,
optionally builds + registers a derivative, and writes the frozen JSON files
(``run_manifest.json`` / ``layer_saliency.json`` / ``plans.json`` /
``derivative.json``). This realizes the existing ``output_layout.ATLAS_RUN_FILES``
contract, which previously had no writer.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_atlas.atlas.coalition import coactivation_map
from model_atlas.atlas.compress import run_compression_pipeline
from model_atlas.atlas.hierarchy import build_hierarchy
from model_atlas.atlas.reap import SaliencyAccumulator, run_calibration
from model_atlas.atlas.runtime import MiniMoE
from model_atlas.builder import build_derivative, register_derivative
from model_atlas.ecosystem import build_mini_moe_for, prompt_corpus
from model_atlas.planning import CandidatePlan, SearchInputs, generate_candidates
from model_atlas.schemas.model_asset import AssetType, ModelAsset
from model_atlas.schemas.ontology import DataPartition

# A deliberately generous per-node budget so every generated plan "fits"; the
# search is over value/coverage/coalition, not a hard memory constraint.
_NODE_BUDGET_BYTES = 1e12
_ACTIVE_BYTES_PER_TOKEN = 100.0
_TOP_K = 2
_CALIBRATION_SUITE = "atlas_calibration"
_SCHEMA_VERSION = "atlas-bridge-v1"


def _short_run_id() -> str:
    return f"atlas-{secrets.token_hex(4)}"


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
    partition: DataPartition = DataPartition.ATLAS_CALIBRATION,
    build: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the atlas pipeline and write the canonical ``atlas_runs/<run_id>/`` dir.

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
    run_dir = Path(out_root) / "atlas_runs" / run_id
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
    compression, _cv = run_compression_pipeline(
        model, corpus, n_stability_runs=3
    )
    (run_dir / "compression_manifest.json").write_text(
        compression.model_dump_json(indent=2)
    )

    # §27 hierarchy artifact: the six-level atlas map (v2 §9) built from the
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

    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "atlas_run_id": run_id,
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
