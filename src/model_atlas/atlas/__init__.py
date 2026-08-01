"""Atlas analysis subsystem: output contract + synthetic REAP runtime."""

from model_atlas.atlas.output_layout import (
    ATLAS_RUN_FILES,
    build_run_manifest,
    expected_run_files,
    validate_evidence_present,
)
from model_atlas.atlas.reap import (
    CalibrationSample,
    ContrastAccumulator,
    SaliencyAccumulator,
    make_synthetic_corpus,
    run_calibration,
    run_contrast,
)
from model_atlas.atlas.runtime import (
    ForwardResult,
    LayerTrace,
    LayerWeights,
    MiniMoE,
    build_mini_moe,
    forward,
    representation_profile,
)

__all__ = [
    "ATLAS_RUN_FILES",
    "build_run_manifest",
    "expected_run_files",
    "validate_evidence_present",
    "CalibrationSample",
    "ContrastAccumulator",
    "SaliencyAccumulator",
    "make_synthetic_corpus",
    "run_calibration",
    "run_contrast",
    "ForwardResult",
    "LayerTrace",
    "LayerWeights",
    "MiniMoE",
    "build_mini_moe",
    "forward",
    "representation_profile",
]
