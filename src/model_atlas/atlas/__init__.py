"""Atlas analysis subsystem: output contract + synthetic REAP runtime."""

from model_atlas.atlas.output_layout import (
    ATLAS_RUN_FILES,
    build_run_manifest,
    expected_run_files,
    validate_evidence_present,
)
from model_atlas.atlas.reap import (
    CalibrationSample,
    SaliencyAccumulator,
    make_synthetic_corpus,
    run_calibration,
)
from model_atlas.atlas.runtime import (
    ForwardResult,
    LayerTrace,
    LayerWeights,
    MiniMoE,
    build_mini_moe,
    forward,
)

__all__ = [
    "ATLAS_RUN_FILES",
    "build_run_manifest",
    "expected_run_files",
    "validate_evidence_present",
    "CalibrationSample",
    "SaliencyAccumulator",
    "make_synthetic_corpus",
    "run_calibration",
    "ForwardResult",
    "LayerTrace",
    "LayerWeights",
    "MiniMoE",
    "build_mini_moe",
    "forward",
]
