from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from model_atlas.glm52_source_profile import (
    SourceProfileError,
    build_glm52_mixed_gguf_profile,
    build_resumable_source_manifest,
)
from model_atlas.jobs.artifacts import source_manifest_digest
from model_atlas.recommend.policy import AtlasProfile


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_is_job_engine_compatible_resumable_and_prunes_stale(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "a.txt", "one")
    _write(source / "nested/b.txt", "two")
    checkpoint = tmp_path / "state.json"
    output = tmp_path / "manifest.json"

    initial = build_resumable_source_manifest(
        source, checkpoint_path=checkpoint, output_path=output
    )
    assert initial.hashed_files == 2
    assert initial.reused_files == 0
    manifest = json.loads(output.read_text())
    assert set(manifest) == {"type", "files", "file_stats"}
    assert manifest["type"] == "dir"
    assert initial.digest == source_manifest_digest(manifest)
    assert json.loads(checkpoint.read_text())["manifest_digest"] == initial.digest

    (source / "nested/b.txt").unlink()
    _write(source / "new.txt", "three")
    resumed = build_resumable_source_manifest(
        source, checkpoint_path=checkpoint, output_path=output
    )
    assert resumed.hashed_files == 1
    assert resumed.reused_files == 1
    resumed_files = resumed.manifest["files"]
    assert isinstance(resumed_files, dict)
    assert set(resumed_files) == {"a.txt", "new.txt"}


def test_manifest_rehashes_only_when_stat_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "a.txt", "one")
    _write(source / "b.txt", "two")
    checkpoint = tmp_path / "state.json"
    output = tmp_path / "manifest.json"
    build_resumable_source_manifest(source, checkpoint_path=checkpoint, output_path=output)
    _write(source / "b.txt", "changed")
    result = build_resumable_source_manifest(source, checkpoint_path=checkpoint, output_path=output)
    assert result.hashed_files == 1
    assert result.reused_files == 1
    result_files = result.manifest["files"]
    assert isinstance(result_files, dict)
    assert result_files["b.txt"] == _sha(source / "b.txt")


def test_manifest_rejects_symlink_and_never_follows_it(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write(tmp_path / "outside.txt", "secret")
    os.symlink(tmp_path / "outside.txt", source / "escape")
    with pytest.raises(SourceProfileError, match="symlink"):
        build_resumable_source_manifest(
            source, checkpoint_path=tmp_path / "state.json", output_path=tmp_path / "manifest.json"
        )


def _profile_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    source = tmp_path / "source"
    _write(source / "config.json", '{"architectures":["GlmForCausalLM"]}')
    _write(source / "model.safetensors.index.json", '{"weight_map":{}}')
    tokenizer = source / "tokenizer.json"
    _write(tokenizer, '{"version":"1.0"}')
    manifest_path = tmp_path / "manifest.json"
    build_resumable_source_manifest(
        source, checkpoint_path=tmp_path / "state.json", output_path=manifest_path
    )
    plan = tmp_path / "plan.txt"
    _write(plan, "^blk\\.5\\.ffn_gate_exps\\.weight$=NVFP4\nblk\\..*=Q1_0\n")
    risk = tmp_path / "risk.json"
    risk.write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "config_sha256": _sha(source / "config.json"),
                "index_sha256": _sha(source / "model.safetensors.index.json"),
                "tensor_type_sha256": _sha(plan),
                "tensor_type_lines": plan.read_text(encoding="utf-8").splitlines(),
                "evidence_kind": "estimated",
                "note": "weight-only",
            }
        ),
        encoding="utf-8",
    )
    return source, manifest_path, tokenizer, risk, plan, tmp_path / "profile.json"


def test_profile_binds_completed_manifest_and_never_fabricates_calibration(tmp_path: Path) -> None:
    source, manifest, tokenizer, risk, plan, output = _profile_inputs(tmp_path)
    profile = build_glm52_mixed_gguf_profile(
        manifest_path=manifest,
        source_path=source,
        source_revision="revision",
        tokenizer_path=tokenizer,
        risk_path=risk,
        tensor_plan_path=plan,
        output_path=output,
    )
    evidence = profile["evidence"]
    assert isinstance(evidence, dict)
    nvfp4 = evidence["nvfp4_suitability"]
    assert isinstance(nvfp4, dict)
    assert evidence == {
        "nvfp4_suitability": nvfp4,
        "routing": None,
    }
    assert "calibration" not in profile
    assert "quality_metrics" not in profile
    assert nvfp4["detail"] == (
        f"risk_artifact_sha256={_sha(risk)};tensor_plan_sha256={_sha(plan)}"
    )
    assert profile["execution"] == {
        "source_id": "nvidia/GLM-5.2-NVFP4",
        "checkpoint_path": str(source.resolve()),
        "checkpoint_revision": "revision",
        "source_manifest_digest": source_manifest_digest(json.loads(manifest.read_text())),
        "source_sha256": {},
        "tokenizer_hash": _sha(tokenizer),
    }
    imported = AtlasProfile.from_dict(json.loads(output.read_text()))
    assert imported.execution is not None
    assert imported.execution.source_manifest_digest == profile["execution"][
        "source_manifest_digest"
    ]
    assert imported.execution.has_calibration is False
    assert output.exists()


def test_profile_rejects_risk_tensor_plan_mismatch(tmp_path: Path) -> None:
    source, manifest, tokenizer, risk, plan, output = _profile_inputs(tmp_path)
    risk_payload = json.loads(risk.read_text())
    risk_payload["tensor_type_sha256"] = "0" * 64
    risk.write_text(json.dumps(risk_payload), encoding="utf-8")
    with pytest.raises(SourceProfileError, match="tensor plan"):
        build_glm52_mixed_gguf_profile(
            manifest_path=manifest,
            source_path=source,
            source_revision="revision",
            tokenizer_path=tokenizer,
            risk_path=risk,
            tensor_plan_path=plan,
            output_path=output,
        )
