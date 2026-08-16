"""In-repo uniform-width structural slicing for ModelOpt NVFP4 checkpoints.

This backend is an explicit pruning operation: it keeps the first aligned
``width`` expert channels uniformly and writes a structurally complete
derivative.  It makes no quality-aware, TENP, or runtime-loadability claim.
"""

from __future__ import annotations

from pathlib import Path

from model_atlas.backend.contract import BackendAdapter, BackendUnavailable
from model_atlas.loader import materialize_uniform_width


class AtlasNvfp4WidthSliceAdapter(BackendAdapter):
    """Wrap the transactional loader inside a JobEngine stage boundary."""

    backend_id = "atlas_nvfp4_width_slice"
    produces_derivative = True

    @staticmethod
    def _paths(context: dict[str, object]) -> tuple[Path, Path]:
        source_raw = context.get("source")
        output_raw = context.get("staging_dir")
        if not source_raw or not output_raw:
            raise BackendUnavailable("NVFP4 width slice requires canonical source and staging_dir")
        source = Path(str(source_raw)).resolve()
        output = Path(str(output_raw)).resolve()
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise BackendUnavailable(
                "NVFP4 width-slice output and immutable source must not overlap"
            )
        return source, output

    @staticmethod
    def _width(context: dict[str, object]) -> int:
        raw = context.get("parameters", {})
        params = raw if isinstance(raw, dict) else {}
        try:
            return int(str(params["width"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendUnavailable("NVFP4 width slice requires an integer width") from exc

    def prepare(self, context: dict[str, object]) -> str:
        source, output = self._paths(context)
        if not source.is_dir():
            raise BackendUnavailable(f"NVFP4 width-slice source is not a directory: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        return "atlas-nvfp4-width-slice::ready"

    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]:
        source, output = self._paths(context)
        width = self._width(context)
        try:
            result = materialize_uniform_width(
                str(source), str(output), width, overwrite=True
            )
        except Exception as exc:  # noqa: BLE001 - normalize loader failures at backend boundary
            raise BackendUnavailable(f"NVFP4 width slice failed: {exc}") from exc
        if not result.promoted or not result.structurally_complete:
            raise BackendUnavailable(
                "NVFP4 width slice did not produce a promoted, structurally complete derivative"
            )
        if result.runtime_validated:
            raise BackendUnavailable("width-slice exporter must not invent runtime validation")
        return {
            "derivative": True,
            "method": "uniform-aligned-expert-channel-width-slice",
            "handle": handle,
            **result.to_dict(),
        }

    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]:
        return self.execute(context, handle)

    def validate(
        self, context: dict[str, object], outputs: dict[str, object]
    ) -> dict[str, object]:
        from model_atlas.checkpoint.validators import _safetensors_structure

        del outputs
        _source, output = self._paths(context)
        result = _safetensors_structure(self.backend_id, output, "safetensors")
        return {
            "validated": result.ok,
            "status": "passed" if result.ok else "failed",
            **result.to_dict(),
        }
