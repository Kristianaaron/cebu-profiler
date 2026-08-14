"""Phase: SM121 / MTP / KV / runtime contract tests."""

from model_atlas.runtimecontracts import (
    MTPContract,
    SM121Contract,
    build_runtime_contract,
    kv_contract_plan,
)


def test_kv_contract_account_estimate_positive():
    c = kv_contract_plan(8192, scheme="fp8")
    assert c.context_tokens == 8192
    assert c.kv_bytes_per_rank > 0
    assert c.scheme == "fp8"
    assert c.layers == 78
    assert c.kv_lora_rank == 512


def test_mtp_contract():
    m = MTPContract()
    assert m.n_mtp_layers == 1
    assert m.acceptance_required == 0.9
    assert "rollback" in m.note or "reference" in m.note


def test_sm121_contract_measured():
    s = SM121Contract()
    assert s.compute_cap == (12, 1)
    assert s.sm_family == "SM121"
    assert s.nvfp4_supported is True
    assert "no custom kernel started before primitives" in s.note


def test_runtime_contract_gates():
    rc = build_runtime_contract(context_tokens=8192, kv_scheme="fp8")
    assert rc.gates["fp8_kv_baseline"] is True
    assert rc.gates["nvfp4_kv_experimental_only"] is False
    rc2 = build_runtime_contract(kv_scheme="nvfp4")
    assert rc2.gates["nvfp4_kv_experimental_only"] is True  # behind parity gate


def test_runtime_contract_serializes():
    rc = build_runtime_contract()
    d = rc.to_dict()
    assert tuple(d["sm121"]["compute_cap"]) == (12, 1)
    assert d["mtp"]["n_mtp_layers"] == 1
    assert d["kv"]["scheme"] == "fp8"
    assert d["no_per_token_weight_fetch"] is True
