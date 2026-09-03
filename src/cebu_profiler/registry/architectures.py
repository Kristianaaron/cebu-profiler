"""Model-agnostic architecture registry.

The registry is the extension point for future large sparse MoE parents: add a
new ArchitectureSpec (structural layout; real tensor sizes come from that
checkpoint's census, never fabricated here).
"""

from __future__ import annotations

from model_atlas.integrations.glm52 import glm52_spec
from model_atlas.integrations.k3 import k3_spec
from model_atlas.schemas.architecture import ArchitectureSpec
from model_atlas.synthetic.mini_moe import mini_moe_spec


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
        _DEFAULT.register(glm52_spec())
    return _DEFAULT
