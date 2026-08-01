"""Compression backends and support-status contract (v2 §21)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SupportStatus(StrEnum):
    UNSUPPORTED = "unsupported"
    CONVERSION_ONLY = "conversion_only"
    PROBE_ONLY = "probe_only"  # math measurable; deployable inference NOT yet claimed
    INFERENCE_SUPPORTED = "inference_supported"
    TRAINING_SUPPORTED = "training_supported"
    REQUIRES_CUSTOM_KERNEL = "requires_custom_kernel"


@dataclass(frozen=True)
class CompressionBackend:
    backend_id: str
    backend_version: str
    support: SupportStatus
    note: str = ""

    @property
    def can_probe(self) -> bool:
        # Only probe-only (or better) can have measured reconstruction/output math.
        return self.support in {
            SupportStatus.PROBE_ONLY,
            SupportStatus.INFERENCE_SUPPORTED,
            SupportStatus.TRAINING_SUPPORTED,
        }


# Registry (v2 §21 backend list). Everything is probe-only here: we measure the
# quantization math, but NO deployable inference is claimed for this synthetic
# runtime (v2 §31:13 — conversion is not inference). EXL3 / AQLM are explicitly
# UNSUPPORTED (no kernel), never reported as passing (v2 §31:24).
def _registry() -> dict[str, CompressionBackend]:
    return {
        "source_mxfp4": CompressionBackend(
            "source_mxfp4", "k3-src", SupportStatus.PROBE_ONLY, "source MXFP4; probe math only"
        ),
        "bf16": CompressionBackend("bf16", "n/a", SupportStatus.PROBE_ONLY),
        "fp16": CompressionBackend("fp16", "n/a", SupportStatus.PROBE_ONLY),
        "fp8": CompressionBackend("fp8", "e4m3-sim", SupportStatus.PROBE_ONLY),
        "nvfp4": CompressionBackend("nvfp4", "block4-sim", SupportStatus.PROBE_ONLY),
        "int8": CompressionBackend("int8", "uniform", SupportStatus.PROBE_ONLY),
        "int4": CompressionBackend("int4", "uniform", SupportStatus.PROBE_ONLY),
        "exl3": CompressionBackend(
            "exl3",
            "unpinned",
            SupportStatus.UNSUPPORTED,
            "no K3 kernel; must audit pinned revision before any probe",
        ),
        "aqlm": CompressionBackend(
            "aqlm",
            "unpinned",
            SupportStatus.UNSUPPORTED,
            "no K3 kernel; must audit pinned revision before any probe",
        ),
        "structured_pruning": CompressionBackend(
            "structured_pruning",
            "n/a",
            SupportStatus.PROBE_ONLY,
            "zero a channel group (simulated)",
        ),
        "removed": CompressionBackend(
            "removed", "n/a", SupportStatus.CONVERSION_ONLY, "expert removed; not a quant probe"
        ),
        "nvme_overflow": CompressionBackend(
            "nvme_overflow", "n/a", SupportStatus.UNSUPPORTED, "no elastic/overflow runtime yet"
        ),
        "custom_research_backend": CompressionBackend(
            "custom_research_backend", "n/a", SupportStatus.PROBE_ONLY
        ),
    }


class BackendRegistry:
    def __init__(self, backends: dict[str, CompressionBackend]) -> None:
        self._b = backends

    def get(self, backend_id: str) -> CompressionBackend:
        try:
            return self._b[backend_id]
        except KeyError:
            raise KeyError(f"unknown compression backend {backend_id!r}") from None

    def names(self) -> list[str]:
        return sorted(self._b)

    def by_status(self, status: SupportStatus) -> list[str]:
        return [k for k, v in self._b.items() if v.support == status]
