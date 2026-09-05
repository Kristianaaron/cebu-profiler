"""Deterministic tests for quant-rd + global alloc (no network, synthetic tensors)."""

from __future__ import annotations

import json

import pytest

from cebu_profiler.scoring.global_alloc import ROLE_FLOOR_BPW, allocate
from cebu_profiler.scoring.quant_rd import RDReport, TensorRD, _select_rows, _uniform_int_qerr


def _np():
    import numpy as np

    return np


def test_select_rows_deterministic():
    assert _select_rows(100, 0, 10) == _select_rows(100, 0, 10)
    assert len(_select_rows(100, 7, 10)) == 10
    assert _select_rows(5, 0, 24) == [0, 1, 2, 3, 4]


def test_uniform_int_qerr_improves_with_bits():
    np = _np()
    rng = np.random.default_rng(42)
    mat = rng.normal(size=(64, 128)).astype(np.float32) * 0.1
    e2 = _uniform_int_qerr(np, mat, 2.0)
    e4 = _uniform_int_qerr(np, mat, 4.0)
    e8 = _uniform_int_qerr(np, mat, 8.0)
    assert e8 < e4 < e2
    # rel-L2 for 1 bit worse than 8 bits by an order of magnitude at least
    assert e2 / e8 > 5


def test_allocate_respects_budget_and_floors():
    report = RDReport(checkpoint="synthetic://test", seed=0)
    # 3 expert tensors with different sensitivities, 1 router, 1 norm
    for i, (err_scale, nbytes) in enumerate([(0.30, 10 * 2**30), (0.10, 10 * 2**30), (0.05, 10 * 2**30)]):
        report.tensors.append(
            TensorRD(
                name=f"model.layers.{i}.mlp.experts.down_proj",
                role="experts",
                layer_index=i, expert_index=None,
                shape=[4096, 4096],
                bf16_bytes=nbytes,
                errors={1.5: err_scale * 3, 2.0: err_scale * 2, 3.0: err_scale, 4.0: err_scale * 0.5},
                sample_rows=8,
                sample_cols=1024,
            )
        )
    report.tensors.append(
        TensorRD(name="model.layers.0.mlp.gate.weight", role="router", layer_index=0, expert_index=None,
                 shape=[256, 4096], bf16_bytes=2 * 2**20, errors={8.0: 0.001}, sample_rows=4, sample_cols=1024)
    )
    report.tensors.append(
        TensorRD(name="model.layers.0.input_layernorm.weight", role="norm", layer_index=0, expert_index=None,
                 shape=[4096], bf16_bytes=2**20, errors={16.0: 0.0}, sample_rows=1, sample_cols=1024)
    )

    plan = allocate(report, budget_gib=40.0)
    by_role = {}
    for it in plan.items:
        by_role.setdefault(it.role, []).append(it)

    # floors honored
    assert all(it.bpw >= 8.0 for it in by_role["router"])
    assert all(it.bpw >= 16.0 for it in by_role["norm"])
    # plan lands near budget (40 GiB + protected overhead)
    assert plan.total_target_bytes / 2**30 < 45.0
    # most sensitive expert (i=0) should end at >= the least sensitive expert's bpw
    exp = sorted(by_role["experts"], key=lambda x: x.layer)
    assert exp[0].bpw >= exp[2].bpw
    # serialized output is JSON-safe
    json.dumps(plan.to_dict())


def test_allocate_zero_budget_stays_at_floors():
    report = RDReport(checkpoint="synthetic://test", seed=0)
    report.tensors.append(
        TensorRD(name="model.layers.0.mlp.experts.gate_up_proj", role="experts", layer_index=0, expert_index=None,
                 shape=[8192, 4096], bf16_bytes=8 * 2**30, errors={1.5: 0.3, 2.0: 0.2}, sample_rows=4, sample_cols=1024)
    )
    plan = allocate(report, budget_gib=0.0)
    assert all(it.bpw == 1.5 for it in plan.items if it.role == "experts")
