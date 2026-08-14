"""Machine-readable preflight + capability report (Phase 0 contract).

Probes the runtime for the execution dependencies the real GLM-5.2 two-Spark
experiment needs, reports each as present/missing with a pinned version where
available, and records GPU/SM, topology, mounted model path, disk/headroom and
active-service constraints. Every fact is measured here, never guessed; a
missing capability is reported as absent (never mocked as complete).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_atlas import __version__

# Execution backends the real experiment needs; value = pin candidate.
_EXEC_MODULES: dict[str, str] = {
    "torch": ">=2",
    "transformers": ">=4.44",
    "vllm": ">=0.6",
    "sglang": ">=0.4",
    "modelopt": ">=0.46",
    "exllamav2": ">=0.2",
    "safetensors": ">=0.4",
}
# Pure analysis backends (optional but useful / hard-required for body probes).
_OPTIONAL_MODULES: dict[str, str] = {
    "numpy": ">=1.24",
    "jax": ">=0.4",
    "sentencepiece": ">=0.1",
    "tokenizers": ">=0.19",
}


@dataclass
class ModuleProbe:
    module: str
    present: bool
    version: str | None = None
    pin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityReport:
    """One immutable, machine-readable capability/preflight snapshot."""

    model_atlas_version: str
    python: str
    platform: str
    arch: str
    modules: list[ModuleProbe] = field(default_factory=list)
    gpu: dict[str, Any] = field(default_factory=dict)
    cuda: dict[str, Any] = field(default_factory=dict)
    services: list[dict[str, Any]] = field(default_factory=list)
    mounted_models: list[dict[str, Any]] = field(default_factory=list)
    disk: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_atlas_version": self.model_atlas_version,
            "python": self.python,
            "platform": self.platform,
            "arch": self.arch,
            "modules": [m.to_dict() for m in self.modules],
            "gpu": self.gpu,
            "cuda": self.cuda,
            "services": self.services,
            "mounted_models": self.mounted_models,
            "disk": self.disk,
        }

    def execution_ready(self) -> dict[str, bool]:
        """Which hard P0 execution deps are present right now."""
        return {
            m.module: m.present
            for m in self.modules
            if m.module in _EXEC_MODULES
        }


def _probe_module(name: str, pin: str | None = None) -> ModuleProbe:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            # some libs (torch) expose version via their own attr
            ver = getattr(mod, "__version__", None)
        return ModuleProbe(name, True, str(ver) if ver else None, pin)
    except Exception:  # noqa: BLE001
        return ModuleProbe(name, False, None, pin)


def _nvidia_smi() -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu` into structured facts (no GPU use)."""
    q = "index,name,memory.total,memory.used,compute_cap,driver_version"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mib": parts[2],
                "memory_used_mib": parts[3],
                "compute_cap": parts[4],
                "driver_version": parts[5],
            }
        )
    return rows


def _sm_counts() -> dict[str, Any]:
    """SM count via CUDA's deviceQuery if a CUDA toolkit binary exists."""
    for cand in ("/usr/local/cuda/extras/demo_suite/deviceQuery",):
        if Path(cand).exists():
            try:
                out = subprocess.run(
                    [cand], capture_output=True, text=True, timeout=30
                ).stdout
            except Exception:  # noqa: BLE001
                return {}
            facts: dict[str, Any] = {}
            for key in ("CUDA Driver Version", "GPU Name", "Multiprocessors", "Compute Capability"):
                for line in out.splitlines():
                    if line.strip().startswith(key):
                        facts[key] = line.split(":")[-1].strip()
                        break
            return facts
    return {}


def _services() -> list[dict[str, Any]]:
    """Active GPU processes (running DeepSeek-v4 / llama-server etc.) + constraints."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append(
                {
                    "pid": parts[0],
                    "process_name": parts[1],
                    "used_memory_mib": parts[2],
                }
            )
    return rows


_KNOWN_MODEL_DIRS = [
    "/media/glm52/models/nvidia/GLM-5.2-NVFP4",
    "/media/glm52/models",
]


def _mounted_models() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in _KNOWN_MODEL_DIRS:
        p = Path(d)
        if not p.exists():
            continue
        size_b = 0
        try:
            for root, _dirs, files in os.walk(p):
                size_b += sum(
                    (Path(root) / f).stat().st_size
                    for f in files
                    if not f.startswith("._")
                )
        except OSError:
            pass
        out.append(
            {
                "path": d,
                "present": True,
                "estimated_bytes": size_b,
                "config_present": (p / "config.json").exists(),
            }
        )
    return out


def _disk(paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        out[path] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    return out


def build_capability_report(
    model_paths: list[str] | None = None,
) -> CapabilityReport:
    """Snapshot the runtime capabilities / preflight state (measured)."""
    gpus = _nvidia_smi()
    gpu_summary: dict[str, Any] = {
        "count": len(gpus),
        "devices": gpus,
        "sm": _sm_counts(),
    }
    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=15
        ).stdout
        cuda_rel = next(
            (
                ln.split("release")[-1].strip()
                for ln in nvcc.splitlines()
                if "release" in ln
            ),
            None,
        )
    except Exception:  # noqa: BLE001
        cuda_rel = None

    disk_paths = model_paths or ["/media/glm52/models"]
    return CapabilityReport(
        model_atlas_version=__version__,
        python=sys.version.split()[0],
        platform=platform.platform(),
        arch=platform.machine(),
        modules=[
            _probe_module(name, pin)
            for name, pin in {**_EXEC_MODULES, **_OPTIONAL_MODULES}.items()
        ],
        gpu=gpu_summary,
        cuda={"release": cuda_rel, "env": os.environ.get("CUDA_HOME")},
        services=_services(),
        mounted_models=_mounted_models(),
        disk=_disk(disk_paths),
    )


def write_preflight(path: str, model_paths: list[str] | None = None) -> str:
    """Write a machine-readable preflight/capability JSON report to `path`."""
    report = build_capability_report(model_paths)
    Path(path).write_text(report.to_json())
    return path
