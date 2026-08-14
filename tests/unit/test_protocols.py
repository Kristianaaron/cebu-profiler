"""Phase 1: model/tensor/runtime protocols decouple analysis from MiniMoE."""

from model_atlas.atlas.runtime import build_mini_moe
from model_atlas.registry.architectures import get_registry
from model_atlas.runtimeprotocols import (
    LayerWeightsLike,
    MoEModelLike,
    TensorLike,
)


class MiniMoEAdapter:
    """Structural adapter over `MiniMoE` exposing the MoE model protocol."""

    def __init__(self, model):  # noqa: ANN001
        self._model = model

    @property
    def num_text_layers(self) -> int:
        return self._model.arch.num_text_layers

    @property
    def num_routed_experts(self) -> int:
        return self._model.n_exp

    @property
    def top_k(self) -> int:
        return self._model.arch.moe.top_k

    @property
    def hidden_dim(self) -> int:
        return self._model.hidden

    def layer(self, index: int):  # noqa: ANN201
        return _LayerAdapter(self._model.layers[index])


class _LayerAdapter:
    def __init__(self, layer_weights):  # noqa: ANN001
        self._w = layer_weights

    @property
    def num_experts(self) -> int:
        return len(self._w.experts)

    def expert(self, index: int):  # noqa: ANN201
        return _ExpertAdapter(self._w.experts[index])


class _ExpertAdapter:
    def __init__(self, exp):  # noqa: ANN001
        self._exp = exp

    @property
    def channel_dim(self) -> int:  # noqa: ANN201
        return len(self._exp["gate"])

    @property
    def hidden_dim(self) -> int:  # noqa: ANN201
        return len(self._exp["gate"][0]) if self._exp["gate"] else 0


def _build():
    return build_mini_moe(get_registry().get("k3-mini"), seed=0)


def test_adapter_satisfies_model_protocol():
    a = MiniMoEAdapter(_build())
    assert isinstance(a, MoEModelLike)
    assert a.num_text_layers == 2
    assert a.num_routed_experts == 8
    assert a.top_k == 2
    assert a.hidden_dim == 128


def test_adapter_layer_expert_protocol():
    a = MiniMoEAdapter(_build())
    layer = a.layer(0)
    assert isinstance(layer, LayerWeightsLike)
    assert layer.num_experts == 8
    exp = layer.expert(0)
    assert exp.channel_dim > 0
    assert exp.hidden_dim == a.hidden_dim


def test_tensorlike_protocol():
    class _T:  # noqa: D101
        @property
        def dtype(self) -> str:
            return "BF16"

        @property
        def shape(self) -> tuple[int, ...]:
            return (4, 4)

        @property
        def byte_size(self) -> int:
            return 32

    assert isinstance(_T(), TensorLike)


def test_protocols_available():
    # Prototypes used by the real-GLM adapter path
    assert MoEModelLike and LayerWeightsLike and TensorLike
