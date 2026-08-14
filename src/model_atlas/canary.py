"""Preflight + real-GLM canary status (Phase 2 contract, honest status).

Reports what the real GLM-5.2 two-Spark experiment can do right now from the
repository alone: a bounded metadata census + body validation plus routing/trace
readiness. Crucially it reports `forward_trace: blocked` because the rgistered
venv has no torch/transformers executor and GLM bodies are NVFP4 (need the
modelopt decoder), which is surfaced as a blocker — never mocked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from model_atlas.preflight import build_capability_report

GLM52_NVFP4 = "/media/glm52/models/nvidia/GLM-5.2-NVFP4"


@dataclass
class CanaryStatus:
    checkpoint_present: bool
    census_coverage: float = 0.0
    unclassified: int = 0
    bounded_body_validated: bool = False
    forward_trace: str = "blocked"  # no torch executor / NVFP4 decoder in this venv
    exec_backends: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canary_status(
    checkpoint_dir: str = GLM52_NVFP4,
) -> CanaryStatus:
    """Assemble a measured canary-status snapshot for the real GLM-5.2 mount."""
    from model_atlas.checkpoint.realbody import validate_real_bodies
    from model_atlas.checkpoint.source_manifest import load_manifest
    from model_atlas.checkpoint.structural_graph import build_structural_graph

    ckpt = Path(checkpoint_dir)
    present = (ckpt / "config.json").exists()

    census_coverage = 0.0
    unclassified = 0
    bounded = False
    notes: list[str] = []
    if present:
        manifest = load_manifest(checkpoint_dir)
        graph = build_structural_graph(manifest)
        census_coverage = graph.coverage
        unclassified = len(graph.unclassified)
        try:
            validate_real_bodies(checkpoint_dir, reference_max=2, nvfp4_experts=1)
            bounded = True
        except Exception as exc:  # noqa: BLE001
            bounded = False
            notes = [f"bounded body validation failed: {exc}"]

    capability = build_capability_report([checkpoint_dir])
    exec_backends = (
        capability.execution_ready() if hasattr(capability, "execution_ready") else {}
    )
    notes.append(
        "no torch/transformers/vllm/modelopt executor in venv -> real forward trace blocked"
    )

    return CanaryStatus(
        checkpoint_present=present,
        census_coverage=census_coverage,
        unclassified=unclassified,
        bounded_body_validated=bounded,
        forward_trace="blocked",
        exec_backends=exec_backends,
        notes=notes,
    )


def write_canary_status(path: str, checkpoint_dir: str = GLM52_NVFP4) -> str:
    status = canary_status(checkpoint_dir)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    return path
