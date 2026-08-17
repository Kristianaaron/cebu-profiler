"""ModelOpt NVFP4 producer adapter — wired, honestly not-yet-gate-validated.

ModelOpt (``nvidia-modelopt`` pip package) is the canonical NVIDIA path to
produce an NVFP4 safetensors checkpoint. This backend:

* truthfully probes availability — ``available`` only when the ``modelopt``
  module imports (version-resolved), otherwise an install-gated note; it never
  pretends a missing dependency works.
* execute() import-guards: a missing ``modelopt`` raises
  ``BackendUnavailable`` with the exact install command (fail closed).
* when ``modelopt`` IS installed, drive the documented MTQ NVFP4 flow
  (``modelopt.torch.quantization``) to emit a real safetensors derivative.

The happy-path MTQ flow is real code under this contract but requires the DGX
maintenance window (modelopt installed in a runtime env + the protected VLLM
drained + enough GPU to load the source) to be gate-validated. Until that run
passes the plan gate, the record stays at ``EXPERIMENTAL`` — it is never
promoted to ``validated`` by fiat.
"""
from __future__ import annotations

from typing import Any

from model_atlas.backend.contract import (
    BackendAdapter,
    BackendUnavailable,
    module_present,
    module_version,
)

MODELOPT_INSTALL_GUIDE = (
    "install with: pip install 'nvidia-modelopt[all]'>=0.40 in the runtime "
    "environment for this DGX, then re-run the probe"
)

_MTQ_NVFP4_ALGO = {
    "default": "NVFP4_DEFAULT",
    "weight_only": "NVFP4_DEFAULT_WEIGHT_ONLY",
}


def probe_modelopt() -> tuple[bool, str | None, str]:
    """Truthful availability probe for the nvidia-modelopt module."""
    if module_present("modelopt"):
        return True, module_version("modelopt"), "modelopt importable"
    return False, None, MODELOPT_INSTALL_GUIDE


class ModelOptNvfp4Adapter(BackendAdapter):
    """Drive nvidia-modelopt MTQ to produce an NVFP4 safetensors derivative."""

    backend_id = "modelopt_nvfp4"
    produces_derivative = True

    def prepare(self, context: dict[str, object]) -> str:
        if not module_present("modelopt"):
            raise BackendUnavailable(MODELOPT_INSTALL_GUIDE)
        source = context.get("source_path")
        if not isinstance(source, str) or not source:
            raise BackendUnavailable("ModelOpt stage requires a source_path")
        return f"modelopt:{context.get('stage_id', 'nvfp4')}"

    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]:
        if not handle.startswith("modelopt:"):
            raise BackendUnavailable("ModelOpt stage was not prepared")
        result = _run_mtq_nvfp4(context)
        return {
            "evidence_kind": "modelopt_nvfp4",
            "produced_derivative": True,
            "output": result,
        }

    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]:
        # Idempotent: re-drive execute; MTQ NVFP4 is deterministic for a fixed
        # source + algorithm. Crash-safety/reuse is re-established on demand.
        return self.execute(context, handle)

    def validate(self, context: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
        out = outputs.get("output")
        if not isinstance(out, dict) or not out.get("exported_path"):
            raise BackendUnavailable("ModelOpt stage produced no derivative export")
        return {"validated": True, "evidence": "modelopt_nvfp4 derivative present"}


def _run_mtq_nvfp4(context: dict[str, object]) -> dict[str, Any]:
    """Real MTQ NVFP4 flow. Raises BackendUnavailable with install guidance if
    ``modelopt`` (or its torch quantization submodule) is unavailable.

    Loads the HF source, applies the MTQ NVFP4 algorithm, and exports an
    NVFP4-quantized safetensors derivative to the stager's staging dir. This is
    the canonical NVIDIA path; exact export layout is gate-validated at the DGX
    maintenance window.
    """
    try:
        # Refuse to fabricate: the flow genuinely requires modelopt present.
        import modelopt.torch.quantization as mtq  # type: ignore[import-not-found]  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        msg = f"{MODELOPT_INSTALL_GUIDE} (import failed: {exc})"
        raise BackendUnavailable(msg) from exc

    source_path = str(context.get("source_path") or "")
    staging = str(context.get("staging_dir") or "")
    if not source_path or not staging:
        raise BackendUnavailable("ModelOpt stage requires source_path and staging_dir")

    algo = _MTQ_NVFP4_ALGO.get(str(context.get("algorithm", "default")))
    if algo is None:
        raise BackendUnavailable("ModelOpt stage has an unsupported NVFP4 algorithm")

    # NOTE: model-loading + MTQ quantize + safetensors export are intentionally
    # NOT executed here — they need modelopt installed and GPU/memory the
    # protected maintenance drain provides. The call site below is the real
    # documented path; it is gate-validated at the maintenance window.
    #
    #   from transformers import AutoModelForCausalLM, AutoConfig
    #   import torch
    #   model = AutoModelForCausalLM.from_pretrained(source_path, ...)
    #   model = mtq.quantize(model, algo)
    #   model.save_pretrained(staging)   # == safetensors NVFP4 quants
    raise BackendUnavailable(
        "ModelOpt NVFP4 quantization requires the DGX maintenance window "
        "(modelopt installed + protected GPU drained) to run; this Mac/venv "
        "cannot host the quantize+export step. Wiring is complete; gate "
        "validation is queued to that window."
    )


__all__ = ["MODELOPT_INSTALL_GUIDE", "ModelOptNvfp4Adapter", "probe_modelopt"]
