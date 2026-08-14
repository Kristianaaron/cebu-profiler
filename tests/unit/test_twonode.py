"""Phase 6: two-node inventory / rank ledger / launch-plan gate tests."""

import pytest

from model_atlas.twonode import (
    NcclProbe,
    TwoNodeInventory,
    build_launch_plan,
    compute_rank_ledger,
    probe_local_node,
)


@pytest.mark.integration
def test_rank_ledger_fit_and_breakdown():
    ledger = compute_rank_ledger(
        40 * 1024**3, physical_bytes=80 * 1024**3, kv_bytes=4 * 1024**3
    )
    assert ledger.fits
    # separation into weights/runtime/allocator/KV/comm/OS is exact
    components = (
        ledger.weights_bytes
        + ledger.runtime_scratch_bytes
        + ledger.allocator_reserve_bytes
        + ledger.kv_bytes
        + ledger.comm_bytes
        + ledger.os_bytes
    )
    assert abs(components - ledger.total) < 1
    assert not ledger.failures


@pytest.mark.integration
def test_rank_ledger_fails_closed_over_budget():
    ledger = compute_rank_ledger(
        180 * 1024**3, physical_bytes=100 * 1024**3
    )
    assert not ledger.fits
    assert ledger.failures  # go/no-go explained


@pytest.mark.integration
def test_launch_plan_gates_reflect_reachability():
    inv = TwoNodeInventory()
    inv.nodes = {"n0": probe_local_node()}
    # single reachable node -> repertoire gate off for two-node
    plan = build_launch_plan(
        inv,
        weights_bytes_total=190 * 1024**3,
        physical_per_rank=100 * 1024**3,
    )
    assert plan.gates["nodes_reachable"] is False  # only 1 node reachable
    assert plan.placement  # placement documented
    assert (
        "--ray-address auto" in plan.launch_command
        or "distributed-executor" in plan.launch_command
    )


@pytest.mark.integration
def test_nccl_probe_to_dict():
    p = NcclProbe(
        nccl_version="(2, 28, 9)",
        torch_cuda_avail=True,
        gpu_count=1,
        local_rank_capable=True,
    )
    d = p.to_dict()
    assert d["nccl_version"] == "(2, 28, 9)"
    assert d["local_rank_capable"] is True


@pytest.mark.integration
def test_local_node_probe_non_evasive():
    n = probe_local_node()
    assert n.reachable
    assert n.hostname == "spark-d167"
    # metadata-only: torch version read via exec venv
    assert n.torch  # present or None, never crashes the probe
