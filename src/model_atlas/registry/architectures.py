"""Model-agnostic architecture registry.

The registry is the extension point for future large sparse MoE parents: add a
new ArchitectureSpec (structural layout; real tensor sizes come from that
checkpoint's census, never fabricated here).
"""

from __future__ import annotations

from model_atlas.schemas.architecture import (
    ArchitectureSpec,
    DType,
    LayerKind,
    MoELayout,
)
from model_atlas.synthetic.mini_moe import mini_moe_spec


def k3_spec() -> ArchitectureSpec:
    """Kimi K3 layout (structural facts only — tensor sizes require measurement).

    Figures are from the Kimi K3 model card / technical report as cited in the
    project blueprint; they describe layout, not exact per-tensor shapes, which
    is why `tensor_params` is intentionally empty and vocab size is unknown.
    """
    return ArchitectureSpec(
        name="k3",
        total_params=2_800_000_000_000,  # ~2.8T (reported)
        active_params=104_000_000_000,  # ~104B active (reported)
        checkpoint_bytes=1_560_000_000_000,  # ~1.56 TB (reported)
        num_text_layers=93,
        layers_by_kind={LayerKind.KDA: 69, LayerKind.MLA: 24},
        moe=MoELayout(
            num_routed_experts=896,
            top_k=16,
            num_shared_experts=2,
            latent_dim=3584,  # Stable LatentMoE working space
            hidden_dim=7168,  # residual width
            expert_dtype=DType.MXFP4,  # routed experts are source MXFP4
            dense_dtype=DType.BF16,  # non-expert tensors primarily BF16
        ),
        hidden_dim=7168,
        vocabulary_size=None,  # requires source measurement
        tensor_params={},
    )


class ArchitectureRegistry:
    """Holds named ArchitectureSpec instances."""

    def __init__(self) -> None:
        self._specs: dict[str, ArchitectureSpec] = {}

    def register(self, spec: ArchitectureSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ArchitectureSpec:
        try:
            return self._specs[name]
        except KeyError:
            known = ", ".join(sorted(self._specs)) or "(none)"
            raise KeyError(f"unknown architecture {name!r}; known: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs


_DEFAULT: ArchitectureRegistry | None = None


def get_registry() -> ArchitectureRegistry:
    """The default registry, populated with built-in architectures."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ArchitectureRegistry()
        _DEFAULT.register(k3_spec())
        _DEFAULT.register(mini_moe_spec())
    return _DEFAULT
