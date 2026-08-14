"""Phase 6 review-fix: two-node inventory / rank ledger / launch-plan gate tests."""

import pytest

from model_atlas.twonode import (
    NcclProbe,
    TwoNodeInventory,
    build_launch_plan,
    compute_rank_ledger,
    probe_local_node,
)


@pytest.mark.integration
def test_rank_ledger_separates_physical_occupancy_and_available():
    ledger = compute_rank_ledger(
        40 * 1024**3,
        physical_bytes=120 * 1024**3,
        production_occupied_bytes=60 * 1024**3,
        kv_bytes=4 * 1024**3,
    )
    # allocatable = physical - production - OS, distinct from physical
    assert ledger.physical_bytes == 120 * 1024**3
    assert ledger.production_occupied_bytes == 60 * 1024**3
    assert ledger.allocatable_bytes < ledger.physical_bytes
    assert ledger.allocatable_bytes > 0
    assert abs(
        ledger.weights_bytes
        + ledger.runtime_scratch_bytes
        + ledger.allocator_reserve_bytes
        + ledger.kv_bytes
        + ledger.comm_bytes
        + ledger.os_bytes
        - ledger.total
    ) < 1


@pytest.mark.integration
def test_rank_ledger_fails_closed_on_occupancy():
    # 60GiB weights, 120 physical, but 100 occupied -> not enough free
    ledger = compute_rank_ledger(
        60 * 1024**3, physical_bytes=120 * 1024**3, production_occupied_bytes=100 * 1024**3
    )
    assert not ledger.fits
    assert ledger.failures
    assert "production" in ledger.failures[0]


@pytest.mark.integration
def test_local_node_reports_host_memory_and_occupancy():
    n = probe_local_node()
    assert n.reachable
    assert n.host_mem_total_gib > 0  # host unified memory measured (GB10 N/A VRAM)
    assert n.production_occupied_gib >= 0
    # production occupancy present (DeepSeek vLLM) while services run
    assert n.host_mem_available_gib <= n.host_mem_total_gib


@pytest.mark.integration
def test_launch_plan_uses_measured_cap_and_valid_vllm_flags():
    inv = TwoNodeInventory()
    # provide a measured node so capacity comes from the probe, not a constant
    n = probe_local_node()
    inv.nodes = {"spark-d167": n}
    plan = build_launch_plan(
        inv,
        weights_bytes_total=190 * 1024**3,
        physical_per_rank=120 * 1024**3,
    )
    # gates reflect only 1 real node (not a two-node plan)
    assert plan.gates["nodes_reachable"] is False
    # valid vllm 0.21 multi-node flags (server takes --nnodes/--node-rank, NOT
    # a server --node-ip; Ray bootstrap uses Ray's --node-ip-address spelling)
    assert "--enable-expert-parallel" in plan.launch_command
    assert "ray start" in plan.launch_command
    server_section = plan.launch_command.split("# Step 1")[-1]
    assert "--node-ip" not in server_section  # not passed to api_server
    assert "--nnodes 2" in plan.launch_command
    assert "--node-ip-address 10.77.0.1" in plan.launch_command  # ray start spelling


@pytest.mark.integration
def test_nccl_probe_to_dict():
    p = NcclProbe(
        nccl_version="(2, 28, 9)", torch_cuda_avail=True, gpu_count=1, local_rank_capable=True
    )
    d = p.to_dict()
    assert d["nccl_version"] == "(2, 28, 9)"
    assert d["local_rank_capable"] is True
