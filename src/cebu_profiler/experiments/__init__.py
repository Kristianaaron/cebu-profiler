"""Blueprint §17 / Milestone E: quality-size experiments (Cebu Profiler adds value?).

Matched-budget comparison of Cebu Profiler heterogeneous per-expert width allocation
against a uniform-width control, measured by held-out utility retention and
logit fidelity (blueprint §18.1). Submodules:

- ``fidelity``: quality/fidelity measurement over held-out samples.
- ``controls``: uniform vs heterogeneous matched-budget prune + compare.
- ``pareto``: quality-size Pareto sweep over retention levels.
- ``structured``: a synthetic MoE with injected channel-importance structure,
  so the differential-cost experiment has a meaningful (deterministic) case.
"""

from __future__ import annotations

from cebu_profiler.experiments.controls import (
    ExperimentOutcome,
    channel_importance,
    compare_controls,
    control_c_clone,
    hetero_clone,
    matched_budget_compare,
    uniform_clone,
)
from cebu_profiler.experiments.fidelity import FidelityReport, measure_fidelity
from cebu_profiler.experiments.pareto import ParetoPoint, pareto_sweep
from cebu_profiler.experiments.structured import build_structured_model

__all__ = [
    "ExperimentOutcome",
    "FidelityReport",
    "ParetoPoint",
    "build_structured_model",
    "channel_importance",
    "compare_controls",
    "control_c_clone",
    "hetero_clone",
    "matched_budget_compare",
    "measure_fidelity",
    "pareto_sweep",
    "uniform_clone",
]
