"""Blueprint §17 / Milestone E: quality-size experiments (Atlas adds value?).

Matched-budget comparison of Atlas heterogeneous per-expert width allocation
against a uniform-width control, measured by held-out utility retention and
logit fidelity (blueprint §18.1). Submodules:

- ``fidelity``: quality/fidelity measurement over held-out samples.
- ``controls``: uniform vs heterogeneous matched-budget prune + compare.
- ``pareto``: quality-size Pareto sweep over retention levels.
- ``structured``: a synthetic MoE with injected channel-importance structure,
  so the differential-cost experiment has a meaningful (deterministic) case.
"""

from __future__ import annotations

from model_atlas.experiments.controls import (
    ExperimentOutcome,
    channel_importance,
    hetero_clone,
    matched_budget_compare,
    uniform_clone,
)
from model_atlas.experiments.fidelity import FidelityReport, measure_fidelity
from model_atlas.experiments.pareto import ParetoPoint, pareto_sweep
from model_atlas.experiments.structured import build_structured_model

__all__ = [
    "ExperimentOutcome",
    "FidelityReport",
    "ParetoPoint",
    "build_structured_model",
    "channel_importance",
    "hetero_clone",
    "matched_budget_compare",
    "measure_fidelity",
    "pareto_sweep",
    "uniform_clone",
]
