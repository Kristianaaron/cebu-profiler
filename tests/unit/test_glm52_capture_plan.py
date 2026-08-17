from __future__ import annotations

import hashlib
from pathlib import Path

from model_atlas.evaluation.llamacpp_capture import CaptureRole, build_capture_argv
from model_atlas.glm52_capture_plan import Glm52CapturePlan, build_glm52_capture_plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path: Path, **overrides: str) -> Glm52CapturePlan:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.gguf"
    tokenizer = tmp_path / "tokenizer.json"
    work = tmp_path / "capture"
    model.write_bytes(b"single-file-gguf-placeholder")
    tokenizer.write_bytes(b"tokenizer")
    work.mkdir()
    values = {
        "model_sha256": _sha(model),
        "source_manifest_sha256": "1" * 64,
        "profile_tokenizer_sha256": _sha(tokenizer),
        "producer_artifact_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "plan_id": "recipe-" + "4" * 24,
        "run_id": "run-" + "5" * 24,
        "profile_id": "profile-" + "6" * 24,
        "profile_sha256": "7" * 64,
        "recommendation_id": "rec-" + "8" * 24,
        "compression_handoff_sha256": "9" * 64,
    }
    values.update(overrides)
    return build_glm52_capture_plan(
        work_root=work,
        model_path=model,
        profile_tokenizer_path=tokenizer,
        **values,
    )


def test_plan_binds_candidate_identity_control_and_exact_lineage(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert plan.candidate.role is CaptureRole.CANDIDATE
    assert plan.identity_control.role is CaptureRole.IDENTITY_CONTROL
    assert plan.candidate.model_sha256 == plan.identity_control.model_sha256
    assert plan.candidate.source_model_sha256 == "1" * 64
    assert plan.model_evidence.producer_artifact_sha256 == "2" * 64
    assert plan.model_evidence.plan_id == "recipe-" + "4" * 24
    assert plan.model_evidence.run_id == "run-" + "5" * 24
    assert plan.quality_claim is False
    assert build_capture_argv(plan.candidate, common_argv=plan.common_argv)[0].endswith(
        "llama-atlas-capture"
    )


def test_any_producer_or_profile_drift_changes_plan_identity(tmp_path: Path) -> None:
    baseline = _plan(tmp_path / "a")
    changed = _plan(tmp_path / "b", compression_handoff_sha256="a" * 64)
    assert baseline.plan_sha256 != changed.plan_sha256
    assert baseline.model_evidence_sha256 != changed.model_evidence_sha256
