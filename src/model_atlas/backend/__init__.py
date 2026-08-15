"""Backend plugin contract, capability registry, and truthful adapters."""

from model_atlas.backend.contract import (
    AvailabilityProbe,
    BackendAdapter,
    BackendAdapterLike,
    BackendRecord,
    BackendUnavailable,
    CommandBackedAdapter,
    ParameterSpec,
    ResourceEstimate,
    command_exists,
    module_present,
    module_version,
)
from model_atlas.backend.registry import (
    BackendRegistry,
    build_default_registry,
    load_backend_plugins,
)

__all__ = [
    "AvailabilityProbe",
    "BackendAdapter",
    "BackendAdapterLike",
    "BackendRecord",
    "BackendRegistry",
    "BackendUnavailable",
    "CommandBackedAdapter",
    "ParameterSpec",
    "ResourceEstimate",
    "build_default_registry",
    "command_exists",
    "load_backend_plugins",
    "module_present",
    "module_version",
]
