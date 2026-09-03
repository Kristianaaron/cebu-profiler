"""Tests for the rank-trust protocol (stability/protocol)."""

from __future__ import annotations

import json
import math

from cebu_profiler.stability.protocol import (
    CONTROL_NAMES,
    run_rank_trust_protocol,
)


def _noisy_half(base: dict[tuple[int, ...], float], seed: int, noise: float = 0.05):
    rng = __import__("random").Random(seed)
    return {k: v * (1.0 + rng.uniform(-noise, noise)) for k, v in base.items()}


def test_identical_halves_measured():
    base = {(0, e): math.exp(-0.01 * e) for e in range(300)}
    res = run_rank_trust_protocol(base, dict(base))
    assert res.verdict == "measured"
    assert res.split_half.spearman >= 0.999
    assert res.jaccard.jaccard[72] == 1.0
    assert res.split_half.n_common == 300


def test_noisy_split_half_stays_measured_when_signal_strong():
    base = {(0, e): math.exp(-0.01 * e) for e in range(300)}
    a = _noisy_half(base, seed=1, noise=0.05)
    b = _noisy_half(base, seed=2, noise=0.05)
    res = run_rank_trust_protocol(a, b)
    assert res.verdict in {"measured", "proxy"}
    assert res.split_half.spearman > 0.9


def test_insufficient_when_no_common_slots():
    a = {(0, 1): 1.0}
    b = {(1, 2): 1.0}
    res = run_rank_trust_protocol(a, b)
    assert res.verdict == "insufficient"
    assert res.split_half.spearman == 1.0  # degenerate guard, not a claim


def test_proxy_controls_named_and_bounded():
    ref = {(0, e): float(300 - e) for e in range(150)}
    perfect = dict(ref)
    anti = {k: -v for k, v in ref.items()}
    noise = _noisy_half(ref, seed=3, noise=0.3)
    res = run_rank_trust_protocol(
        ref,
        _noisy_half(ref, 4, 0.1),
        proxies={"count": perfect, "mass": anti, "proxy": noise},
        reference=ref,
    )
    assert set(res.controls) == set(CONTROL_NAMES)
    assert res.controls["count"] >= 0.999
    assert res.controls["mass"] <= 0.001  # anti-correlated
    assert 0.0 <= res.controls["proxy"] <= 1.0


def test_payload_is_jsonable_and_typed():
    base = {(0, e): math.exp(-0.02 * e) for e in range(250)}
    res = run_rank_trust_protocol(base, _noisy_half(base, 5, 0.02), meta={"capture": "unit"})
    payload = res.payload()
    assert payload["verdict"] in {"measured", "proxy", "insufficient"}
    assert set(payload) == {
        "halves",
        "keep_set_jaccard",
        "controls_vs_reference",
        "verdict",
        "meta",
    }
    assert payload["meta"] == {"capture": "unit"}
    assert isinstance(json.dumps(payload), str)
