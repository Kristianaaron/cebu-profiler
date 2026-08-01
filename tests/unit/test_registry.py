"""Registry tests: built-in K3 + synthetic mini, model-agnostic lookup."""

import pytest

from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.architecture import DType


def test_registry_has_k3_and_mini():
    reg = get_registry()
    assert "k3" in reg
    assert "k3-mini" in reg


def test_k3_is_layout_not_measured():
    k3 = get_registry().get("k3")
    assert k3.needs_source_measurement is True
    # layout facts from the blueprint
    assert k3.num_text_layers == 93
    assert k3.moe.num_routed_experts == 896
    assert k3.moe.top_k == 16
    assert k3.moe.num_shared_experts == 2
    assert k3.moe.latent_dim == 3584
    assert k3.moe.hidden_dim == 7168
    assert k3.moe.expert_dtype == DType.MXFP4
    assert k3.moe.dense_dtype == DType.BF16
    # unknowns are None, never fabricated
    assert k3.vocabulary_size is None


def test_mini_is_deterministic():
    mini = get_registry().get("k3-mini")
    assert mini.needs_source_measurement is False
    assert mini.num_text_layers == 2
    assert mini.moe.num_routed_experts == 8
    assert mini.moe.top_k == 2


def test_unknown_architecture_raises():
    with pytest.raises(KeyError):
        get_registry().get("no-such-model")
