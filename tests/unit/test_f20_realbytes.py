"""F20 tests: real-bytes derivative-candidate planner (blueprint §24/§25, F10).

Covers measured-byte accounting from a checkpoint manifest and envelope
planning: feasibility, per-envelope monotonicity, resident A/B split, and the
measured-vs-estimated discipline. Deterministic on a synthetic manifest — no
checkpoint or drive required; an integration test runs against the real
GLM-5.2 NVFP4 only when it is mounted.
"""

import os

import pytest

from model_atlas.checkpoint.source_manifest import CheckpointManifest, TensorEntry
from model_atlas.planning.realbytes import (
    GIB,
    account_manifest,
    plan_candidates,
    report,
)

# token stores one FP32-ish tensor so the achieved bpw ratio is exact.
EXPERT_BPW = 8.19


def _tensor(name: str, byte_size: int, bpw: float) -> TensorEntry:
    numel = int(byte_size * 8 / bpw)
    return TensorEntry(
        name=name,
        dtype="bf16",
        shape=[numel],
        numel=numel,
        byte_size=byte_size,
        shard="shard-00001",
        offset_start=0,
        offset_end=byte_size,
    )


def _synthetic_manifest(expert_gib: float = 80.0, backbone_gib: float = 10.0) -> CheckpointManifest:
    """A tiny NVFP4-shaped manifest: F32-ish routed experts + BF16 backbone."""
    expert_bytes = int(expert_gib * GIB)
    per_expert = expert_bytes // 4
    backbone_bytes = int(backbone_gib * GIB)
    tensors = [
        _tensor(f"model.layers.0.experts.{e}.gate.weight", per_expert, EXPERT_BPW)
        for e in range(4)
    ]
    tensors += [
        _tensor("model.layers.0.self_attn.q_proj.weight", backbone_bytes // 3, 16.0),
        _tensor("model.embed_tokens.weight", backbone_bytes // 3, 16.0),
        _tensor("model.lm_head.weight", backbone_bytes - 2 * (backbone_bytes // 3), 16.0),
    ]
    total = sum(t.byte_size for t in tensors)
    return CheckpointManifest(
        checkpoint_dir="(synthetic)",
        tensors=tensors,
        total_bytes=total,
        tensor_count=len(tensors),
    )


def test_account_manifest_splits_backbone_vs_experts():
    acc = account_manifest(_synthetic_manifest(expert_gib=80.0, backbone_gib=10.0))
    assert acc.expert_bytes == pytest.approx(80.0 * GIB, rel=1e-3)
    assert acc.backbone_bytes == pytest.approx(10.0 * GIB, rel=1e-2)
    assert acc.expert_achieved_bpw == pytest.approx(EXPERT_BPW, rel=1e-2)
    assert acc.backbone_achieved_bpw == 16.0


def test_plan_candidates_respect_budgets_and_split():
    acc = account_manifest(_synthetic_manifest(80.0, 10.0))
    cands = plan_candidates(acc, envelopes=(45.0, 60.0, 89.0))
    assert len(cands) == 3
    prev_stored = 0.0
    for c in cands:
        assert not c.risk, c.risk
        assert c.stored_bytes <= c.envelope_gb * GIB + 1
        assert 0.0 < c.keep_frac <= 1.0
        assert c.resident_b_bytes < c.resident_a_bytes  # backbone rides node A
        assert c.resident_a_bytes <= c.envelope_gb * GIB + 1
        assert c.estimated is True
        assert c.stored_bytes >= prev_stored  # bigger budget -> at least as much kept
        prev_stored = c.stored_bytes


def test_larger_envelope_keeps_more_bits():
    acc = account_manifest(_synthetic_manifest(80.0, 10.0))
    lo, hi = plan_candidates(acc, envelopes=(45.0, 89.0))
    # more budget means survivors keep more precision (higher bpw), not more
    # wasteful uniform low-bit
    assert hi.mean_expert_bpw >= lo.mean_expert_bpw


def test_infeasible_envelope_is_flagged():
    acc = account_manifest(_synthetic_manifest(80.0, 10.0))
    c = plan_candidates(acc, envelopes=(20.0,))[0]
    assert c.risk.startswith("infeasible")
    assert c.mean_expert_bpw < 4.0


def test_deterministic():
    acc = account_manifest(_synthetic_manifest(80.0, 10.0))
    a = plan_candidates(acc, envelopes=(45.0, 60.0))
    b = plan_candidates(acc, envelopes=(45.0, 60.0))
    assert [(c.envelope_gb, c.keep_frac, c.mean_expert_bpw) for c in a] == [
        (c.envelope_gb, c.keep_frac, c.mean_expert_bpw) for c in b
    ]


def test_report_lists_source_and_candidates():
    acc = account_manifest(_synthetic_manifest(80.0, 10.0))
    text = report(plan_candidates(acc, envelopes=(60.0,)), acc)
    assert "(synthetic)" in text
    assert "envelope 60" in text
    assert "measured total" in text


_REAL = "/media/glm52/models/nvidia/GLM-5.2-NVFP4"


def test_real_glm52_candidates_when_mounted():
    if not os.path.isfile(os.path.join(_REAL, "config.json")):
        pytest.skip("GLM-5.2 NVFP4 not mounted")
    from model_atlas.checkpoint.source_manifest import load_manifest

    manifest = load_manifest(_REAL)
    acc = account_manifest(manifest)
    assert acc.expert_achieved_bpw is not None and acc.expert_achieved_bpw > 4.0
    cands = plan_candidates(acc)
    assert len(cands) == 3
    # the true-source guard: at least the largest envelope is feasible
    assert not cands[-1].risk
