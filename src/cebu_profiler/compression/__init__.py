"""Compression backends, quantization math, and per-expert response curves."""

from model_atlas.compression.backend import (
    BackendRegistry,
    CompressionBackend,
    SupportStatus,
    _registry,
)
from model_atlas.compression.quant import QuantMeta, float_mantissa_quant, rel_l2, uniform_int_quant
from model_atlas.compression.response import (
    ResponsePoint,
    expert_response_curve,
)

__all__ = [
    "BackendRegistry",
    "CompressionBackend",
    "SupportStatus",
    "_registry",
    "QuantMeta",
    "float_mantissa_quant",
    "rel_l2",
    "uniform_int_quant",
    "ResponsePoint",
    "expert_response_curve",
]


def get_backend_registry() -> BackendRegistry:
    return BackendRegistry(_registry())
