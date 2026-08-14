"""Model / tensor / runtime protocols (Phase 1 decoupling).

Analysis modules can depend on these structural protocols instead of the
concrete `MiniMoE`, keeping a real-GLM adapter a drop-in. While the current
forward/trace/scoring still operate on `MiniMoE` (the deterministic fixture),
any module that only needs *shape/count/introspection* contracts targets these
interfaces, so real tensors never need to be materialized as a MiniMoE.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExpertWeights(Protocol):
    """Coupled FFN expert tensors: gate/up rows + down columns share channels."""

    @property
    def channel_dim(self) -> int: ...

    @property
    def hidden_dim(self) -> int: ...


@runtime_checkable
class LayerWeightsLike(Protocol):
    """A per-layer weight bundle (router indices + expert bank + shared info)."""

    @property
    def num_experts(self) -> int: ...

    def expert(self, index: int) -> ExpertWeights: ...


@runtime_checkable
class MoEModelLike(Protocol):
    """Structural MoE contract analysis consumes (shape/count only)."""

    @property
    def num_text_layers(self) -> int: ...

    @property
    def num_routed_experts(self) -> int: ...

    @property
    def top_k(self) -> int: ...

    @property
    def hidden_dim(self) -> int: ...

    def layer(self, index: int) -> LayerWeightsLike: ...


@runtime_checkable
class TensorLike(Protocol):
    """A single tensor body with its byte layout (from the streaming substrate)."""

    @property
    def dtype(self) -> str: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def byte_size(self) -> int: ...
