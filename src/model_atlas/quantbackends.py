"""Real quantization backend adapters (Phase 5): EXL3 + NVIDIA ModelOpt NVFP4.

Honest feature detection + actionable dependency setup. These adapters never
claim a backend works when it is not installed, and never mislabel uniform INT4
as NVFP4. Each returns a pinned, versioned support status plus the exact setup
commands to enable it.

Measured environment facts (2026-08-14, exec venvs):
- torch 2.11.0+cu130 / CUDA 13.0, compute cap (12,1) [GB10 SM121-family]
- vllm 0.21.0 (hosts GLM-5.2 via GlmMoeDsaForCausalLM -> deepseek_v2 model,
  and a compressed-tensors NVFP4 path `compressed_tensors_w4a16_nvfp4`)
- transformers 5.9.0 with native `glm_moe_dsa`
- NVIDIA ModelOpt at
  `/home/kristianaaron/MiniMax-M3-REAP-Spark/.venv-modelopt045` (0.45.0) — an
  older release than the checkpoints's `0.46.0.dev65` producer
- exllamav2: absent on both hosts
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field

from model_atlas.compression.backend import (
    CompressionBackend,
    SupportStatus,
)

# Where a working ModelOpt venv already exists on this host (measured).
MODELOPT_VENV = "/home/kristianaaron/MiniMax-M3-REAP-Spark/.venv-modelopt045"
# The ModelOpt version that produced the mounted GLM-5.2 NVFP4 checkpoint.
CHECKPOINT_MODELOPT_PRODUCER = "0.46.0.dev65+g977d34dc3"
CHECKPOINT_QUANT_ALGO = "NVFP4"
CHECKPOINT_GROUP_SIZE = 16


@dataclass
class BackendProbe:
    """Evidence-backed support status for one backend."""

    backend_id: str
    installed: bool
    version: str | None = None
    support: SupportStatus = SupportStatus.UNSUPPORTED
    note: str = ""
    setup: list[str] = field(default_factory=list)  # actionable setup commands
    location: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _probe_module(name: str) -> tuple[bool, str | None]:
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", None)
    except Exception:  # noqa: BLE001
        return False, None


def _module_has(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _probe_modelopt_venv(venv: str) -> tuple[bool, str | None]:
    py = f"{venv}/bin/python"
    if not shutil.which(py):
        return False, None
    try:
        out = subprocess.run(
            [py, "-c", "import modelopt;print(modelopt.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0:
            return True, out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return False, None


def probe_exl3() -> BackendProbe:
    installed, ver = _probe_module("exllamav2")
    note = (
        "EXL3 conversion from the installed NVFP4 checkpoint is BLOCKED: there is "
        "no BF16 parent source and no verified ModelOpt dequantization available, "
        "so an EXL3 quant cannot be produced from it. EXL3 is a distinct 4-bit "
        "format needing its own dequant source."
    )
    setup = []
    if not installed:
        setup.append(
            "# install a pinned EXL3 build (needs a compatible CUDA toolchain; "
            "SM121 support must be audited upstream, not assumed)"
        )
        setup.append(
            "uv venv .venv-exl3 && uv pip install --python .venv-exl3/bin/python "
            "exllamav2@git+https://github.com/turboderp/exllamav2@<pinned-rev>"
        )
        support = SupportStatus.REQUIRES_CUSTOM_KERNEL
    else:
        note = (
            f"EXL3 {ver} present, but conversion from this NVFP4 source is STILL "
            "BLOCKED without a BF16 parent or verified ModelOpt dequantization "
            "(report gap, do not claim)"
        )
        support = SupportStatus.UNSUPPORTED
    return BackendProbe(
        backend_id="exl3",
        installed=installed,
        version=ver,
        support=support,
        note=note,
        setup=setup,
        location=None,
    )


def probe_modelopt_nvfp4() -> BackendProbe:
    """Detect an NVIDIA ModelOpt NVFP4 path honestly.

    Checks (a) modelopt importable in this process, (b) the known host ModelOpt
    venv, (c) whether vllm exposes a compressed-tensors NVFP4 scheme. Even when
    ModelOpt is present, decoding the mounted NVFP4 requires the matching
    producer/quant scheme — never claimed when only a different version is found.
    """
    installed, ver = _probe_module("modelopt")
    location = None
    if not installed:
        installed, ver = _probe_modelopt_venv(MODELOPT_VENV)
        location = MODELOPT_VENV
    note: list[str] = []
    if installed:
        if ver and ver != CHECKPOINT_MODELOPT_PRODUCER:
            note.append(
                f"ModelOpt {ver} present but producer is {CHECKPOINT_MODELOPT_PRODUCER}; "
                "**dequantization parity is UNPROVEN** — must be verified on a "
                "sampled tensor against the checkpoint's NVFP4 layout before any "
                "conversion/reconstruction is trusted"
            )
            support = SupportStatus.UNSUPPORTED  # parity unproven -> no decode claimed
        else:
            note.append(
                "ModelOpt version matches the checkpoint producer (decode path plausible)"
            )
            support = (
                SupportStatus.PROBE_ONLY  # reference decode measurable; inference is separate
            )
    else:
        support = SupportStatus.UNSUPPORTED
        note.append("ModelOpt not importable here; use the host ModelOpt venv or pip-install")
    setup: list[str] = []
    if not (installed and location):
        setup.append(
            f"# reuse the measured host ModelOpt venv (read-only):\n"
            f"#   {MODELOPT_VENV}/bin/python -c 'import modelopt'"
        )
        setup.append(
            "# or pip-install a version matching the checkpoint producer into a "
            "repo-local .venv-exec (do NOT touch the running services):\n"
            "uv pip install --python .venv-exec/bin/python 'model-optimizer==0.46.0' "
            "or the matching git revision"
        )
    default_note = (
        "NVFP4 is a block-scaled 4-bit scheme; never call uniform INT4 NVFP4"
    )
    return BackendProbe(
        backend_id="modelopt_nvfp4",
        installed=installed,
        version=ver,
        support=support,
        note="; ".join(note) if note else default_note,
        setup=setup,
        location=location,
    )


def probe_vllm_nvfp4() -> BackendProbe:
    """Detect whether the installed vllm exposes a ModelOpt-NVFP4 decode path
    for the mounted checkpoint (quant_method=modelopt, quant_algo=NVFP4)."""
    installed, ver = _probe_module("vllm")
    has_nvfp4 = False
    detail = ""
    if installed:
        try:
            from vllm.model_executor.layers.quantization import (  # type: ignore[import-not-found]
                get_quantization_config,
            )
            cfg = get_quantization_config("modelopt_fp4")
            if cfg.__name__ == "ModelOptNvFp4Config":
                has_nvfp4 = True
                detail = "vllm registry maps modelopt_fp4 -> ModelOptNvFp4Config"
        except Exception:  # noqa: BLE001
            has_nvfp4 = False
            detail = "vllm ModelOpt-NVFP4 path probe failed"
    support = (
        SupportStatus.PROBE_ONLY if (installed and has_nvfp4) else SupportStatus.UNSUPPORTED
    )
    return BackendProbe(
        backend_id="vllm_nvfp4",
        installed=installed,
        version=ver,
        support=support,
        note=(
            "vllm exposes a ModelOpt-NVFP4 path (modelopt_fp4 -> "
            "ModelOptNvFp4Config + NVFP4 Linear/FusedMoE); a real materialized "
            "derivative load/forward is still unvalidated"
        )
        if has_nvfp4
        else ("vllm not importable here; run under the installed vLLM exec venv "
              "(probe detail: " + detail + ")"),
        setup=[],
        location=None,
    )


def all_backend_probes() -> list[BackendProbe]:
    return [probe_exl3(), probe_modelopt_nvfp4(), probe_vllm_nvfp4()]


def to_registry() -> dict[str, CompressionBackend]:
    """Fold the honest probes into the versioned backend registry."""
    out: dict[str, CompressionBackend] = {}
    for b in all_backend_probes():
        out[b.backend_id] = CompressionBackend(
            backend_id=b.backend_id,
            backend_version=b.version or "n/a",
            support=b.support,
            note=b.note,
        )
    return out
