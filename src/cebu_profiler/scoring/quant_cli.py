"""CLI wiring: cebu-profiler quant-rd + quant-plan commands.

quant-rd    — sampled rate-distortion screen over a real BF16 checkpoint.
quant-plan  — GEMQ-style global allocation to target envelopes (Fidelity/Knee).
"""

from __future__ import annotations

import json
from pathlib import Path

from cebu_profiler.checkpoint.source_manifest import load_manifest
from cebu_profiler.checkpoint.tensor_io import TensorFetcher
from cebu_profiler.scoring.quant_rd import RDReport, screen_checkpoint
from cebu_profiler.scoring.global_alloc import allocate_two_points


def run_quant_rd(
    manifest_path: str,
    checkpoint_dir: str,
    out: str,
    *,
    seed: int = 0,
    max_tensors: int = 400,
    cache_dir: str | None = None,
) -> Path:
    """Run the R-D screen and write quant_rd_report.json. Returns the out path."""
    p = Path(manifest_path)
    root = p.parent if p.is_file() else p
    manifest = load_manifest(root)
    fetcher = TensorFetcher(checkpoint_dir=checkpoint_dir, cache_dir=cache_dir)
    report = screen_checkpoint(manifest, fetcher, seed=seed, max_tensors=max_tensors)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=1))
    return out_path


def run_quant_plan(
    rd_report_path: str,
    out: str,
    *,
    fidelity_gib: float = 200.0,
    knee_gib: float = 188.0,
) -> Path:
    """Allocate the two operating points from a measured R-D report."""
    raw = json.loads(Path(rd_report_path).read_text())
    report = RDReport(checkpoint=raw["checkpoint"], seed=raw.get("seed", 0))
    from cebu_profiler.scoring.quant_rd import TensorRD

    for t in raw["tensors"]:
        report.tensors.append(
            TensorRD(
                name=t["name"],
                role=t["role"],
                layer_index=t.get("layer"),
                expert_index=t.get("expert"),
                shape=t["shape"],
                bf16_bytes=t["bf16_bytes"],
                errors={float(k): v for k, v in t["errors"].items()},
                sample_rows=t.get("sample_rows", 0),
                sample_cols=t.get("sample_cols", 0),
            )
        )
    plans = allocate_two_points(report, fidelity_gib, knee_gib)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: plan.to_dict() for name, plan in plans.items()}
    out_path.write_text(json.dumps(payload, indent=1))
    return out_path
