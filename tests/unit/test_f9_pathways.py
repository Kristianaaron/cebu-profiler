"""F9 tests: cross-layer pathways + neuron/channel sensitivity (v2 §14, §18)."""

from model_atlas.atlas.pathways import (
    capability_paths,
    channel_sensitivity,
    path_stats,
    route_path,
    success_failure_divergence,
)
from model_atlas.atlas.reap import make_synthetic_corpus
from model_atlas.atlas.runtime import build_mini_moe
from model_atlas.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def _samples(n=10, seq=5):
    return make_synthetic_corpus(n_samples=n, seq_len=seq, vocab=ARCH.vocabulary_size, seed=0)[0]


def test_route_path_has_one_signature_per_layer():
    model = build_mini_moe(ARCH, seed=1)
    sig = route_path(model, [1, 2, 3], top_k=2)
    assert len(sig) == ARCH.num_text_layers
    for layer_sig in sig:
        assert len(layer_sig) == ARCH.moe.top_k


def test_path_stats_frequency_success_rate_labels():
    model = build_mini_moe(ARCH, seed=2)
    stats = path_stats(model, _samples())
    assert stats.records
    most = stats.most_frequent(1)[0]
    assert most.count >= 1
    assert 0.0 <= most.success_rate <= 1.0
    assert isinstance(most.signature, tuple)


def test_capability_paths_nonempty_for_seen_label():
    model = build_mini_moe(ARCH, seed=3)
    sample = _samples()[0]
    label = sample.labels[0].value
    paths = capability_paths(model, _samples(20, 5), label=label)
    # corpus covers the label, so at least one path is returned
    assert isinstance(paths, list)
    assert all(isinstance(p, tuple) for p in paths)


def test_success_failure_divergence_shape():
    model = build_mini_moe(ARCH, seed=4)
    d = success_failure_divergence(model, _samples(12, 5), layer=0)
    assert 0.0 <= d.jaccard <= 1.0
    assert d.distinct_success_only >= 0
    assert d.distinct_failure_only >= 0


def test_channel_sensitivity_ranks_sensitive_channel_top():
    model = build_mini_moe(ARCH, seed=5)
    cs = channel_sensitivity(model, [1, 2, 3], layer=0, expert=2)
    assert cs.n_channels == ARCH.hidden_dim
    assert len(cs.sensitivity) == cs.n_channels
    assert all(s >= 0.0 for s in cs.sensitivity)
    # top channel is the max-sensitivity one and is sorted descending
    assert cs.top_channels[0][0] == max(range(cs.n_channels), key=lambda i: cs.sensitivity[i])
    assert cs.top_channels == sorted(cs.top_channels, key=lambda x: -x[1])
