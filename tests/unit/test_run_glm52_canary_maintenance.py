from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from model_atlas.ops.maintenance import (
    MaintenanceFailure,
    MaintenanceInterrupted,
    MaintenanceReceipt,
)


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_glm52_canary_maintenance.py"
    spec = importlib.util.spec_from_file_location("test_glm52_maintenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canary_module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_two_node_canary.py"
    spec = importlib.util.spec_from_file_location("test_two_node_canary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        execute=True,
        journal_dir=tmp_path / "journal",
        receipt=tmp_path / "receipt.json",
        lease=tmp_path / "lease.json",
        plan_sha256=None,
        compression_result=None,
        artifact="/artifacts/glm52.gguf",
        artifact_sha256="b" * 64,
        canary_args=[],
    )


def test_main_restores_signal_handlers_when_coordinator_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    calls: list[object] = []
    previous = {2: object()}
    monkeypatch.setattr(module, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(module, "install_signal_traps", lambda: previous)
    monkeypatch.setattr(module, "restore_signal_traps", lambda value: calls.append(value))

    class _Coordinator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, _payload: object, **_kwargs: object) -> MaintenanceReceipt:
            raise MaintenanceInterrupted(15)

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    with pytest.raises(MaintenanceInterrupted):
        module.main()
    assert calls == [previous]


def test_main_installs_traps_around_fake_coordinator_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    calls: list[str] = []
    monkeypatch.setattr(module, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(module, "install_signal_traps", lambda: {2: object()})
    monkeypatch.setattr(module, "restore_signal_traps", lambda _previous: calls.append("restore"))

    class _Coordinator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            calls.append("init")

        def run(self, payload: object, **kwargs: object) -> MaintenanceReceipt:
            calls.append("run")
            assert isinstance(payload, tuple)
            assert "run_bound_canary_payload.py" not in payload
            scope_factory = kwargs["payload_scope"]
            assert callable(scope_factory)
            with scope_factory():
                assert _args(tmp_path).lease.exists()
                calls.append("lease-live")
            assert not _args(tmp_path).lease.exists()
            now = datetime.now(UTC)
            return MaintenanceReceipt(
                run_id="fake", dry_run=False, started_at=now, finished_at=now, success=True
            )

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    assert module.main() == 0
    assert calls == ["init", "run", "lease-live", "restore"]


def test_canary_cli_default_uses_installed_vllm_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _canary_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_two_node_canary.py",
            "--artifact",
            "/artifacts/glm52.gguf",
            "--artifact-sha256",
            "a" * 64,
            "--evidence",
            "/tmp/evidence.jsonl",
        ],
    )
    assert module.parse_args().telemetry_python == Path(
        "/home/kristianaaron/ai-lab/venvs/vllm/bin/python"
    )


def test_deleted_direct_lease_wrapper_cannot_be_invoked() -> None:
    assert not (Path(__file__).parents[2] / "scripts" / "run_bound_canary_payload.py").exists()


def test_direct_canary_rejects_minted_lease_when_live_drain_probe_finds_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _canary_module()
    args = Namespace(
        execute=True,
        artifact=Path("/artifacts/glm52.gguf"),
        artifact_sha256="a" * 64,
        producer_run_id=None,
        producer_plan_id=None,
        producer_recipe_sha256=None,
        producer_profile_id=None,
        producer_recommendation_id=None,
        producer_handoff_sha256=None,
        evidence=tmp_path / "evidence.jsonl",
        telemetry_probe=Path("/scripts/probe.py"),
        rdma_interface="ib0",
        disk_device="nvme0n1",
        telemetry_python=Path("/venv/bin/python"),
        telemetry_python_sha256="b" * 64,
        maintenance_lease=tmp_path / "minted.json",
        worker_ssh_target="10.77.0.2",
        worker_host="169.254.200.197",
        toolchain_root=Path("/toolchain"),
        llama_server=Path("/toolchain/llama-server"),
        worker_rpc_server=Path("/toolchain/ggml-rpc-server"),
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    events: list[str] = []

    class _Coordinator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            events.append("init")

        def verify_drained(self) -> None:
            events.append("verify")
            raise MaintenanceFailure("drain verification failed")

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    monkeypatch.setattr(
        module,
        "require_active_lease",
        lambda *_args, **_kwargs: pytest.fail("lease must not be accepted before drain"),
    )
    with pytest.raises(MaintenanceFailure):
        module.main()
    assert events == ["init", "verify"]


def _compression_result(tmp_path: Path) -> tuple[Path, Path, str]:
    payload = b"bounded-runtime-gguf"
    digest = hashlib.sha256(payload).hexdigest()
    artifact = tmp_path / "runs" / "run-1" / "objects" / digest[:2] / f"{digest}.blob"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(payload)
    evidence_payload = b"{}"
    evidence_digest = hashlib.sha256(evidence_payload).hexdigest()
    evidence = (
        tmp_path / "runs" / "run-1" / "objects" / evidence_digest[:2] / f"{evidence_digest}.blob"
    )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_bytes(evidence_payload)
    result = tmp_path / "compression-result.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "method": "llamacpp-gguf-mixed",
                "runtime_claim": "artifact_only_unvalidated",
                "run_id": "run-1",
                "plan_id": "plan-1",
                "recipe_sha256": "a" * 64,
                "profile_id": "profile-1",
                "recommendation_id": "recommendation-1",
                "outputs": {
                    "run_id": "run-1",
                    "outputs": [
                        {
                            "stage": "llamacpp-gguf-mixed",
                            "name": "llamacpp-gguf-mixed.evidence.json",
                            "sha256": evidence_digest,
                            "size_bytes": len(evidence_payload),
                            "relpath": str(evidence.relative_to(evidence.parents[2])),
                        },
                        {
                            "stage": "llamacpp-gguf-mixed",
                            "name": "model.gguf",
                            "sha256": digest,
                            "size_bytes": len(payload),
                            "relpath": str(artifact.relative_to(artifact.parents[2])),
                        },
                    ],
                },
                "runtime_artifact": {
                    "path": str(artifact),
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "stage": "llamacpp-gguf-mixed",
                    "logical_name": "model.gguf",
                    "relpath": str(artifact.relative_to(artifact.parents[2])),
                    "runtime_validated": False,
                    "evidence": {
                        "stage": "llamacpp-gguf-mixed",
                        "logical_name": "llamacpp-gguf-mixed.evidence.json",
                        "sha256": evidence_digest,
                        "size_bytes": len(evidence_payload),
                        "relpath": str(evidence.relative_to(evidence.parents[2])),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return result, artifact, digest


def test_compression_result_mode_derives_exact_verified_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    result, artifact, digest = _compression_result(tmp_path)
    handoff = module._artifact_from_compression_result(result)
    result_payload = json.loads(result.read_text(encoding="utf-8"))
    evidence = result_payload["runtime_artifact"]["evidence"]
    assert handoff.evidence_sha256 == evidence["sha256"]
    assert handoff.evidence_size_bytes == evidence["size_bytes"]
    assert handoff.evidence_relpath == evidence["relpath"]
    args = _args(tmp_path)
    args.compression_result = result
    args.artifact = None
    args.artifact_sha256 = None
    seen: dict[str, object] = {}
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "install_signal_traps", lambda: {})
    monkeypatch.setattr(module, "restore_signal_traps", lambda _previous: None)

    class _Coordinator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, payload: object, **_kwargs: object) -> MaintenanceReceipt:
            assert isinstance(payload, tuple)
            seen["payload"] = payload
            now = datetime.now(UTC)
            return MaintenanceReceipt(
                run_id="fake", dry_run=False, started_at=now, finished_at=now, success=True
            )

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    assert module.main() == 0
    payload = seen["payload"]
    assert isinstance(payload, tuple)
    assert payload[payload.index("--artifact") + 1] == str(artifact.resolve())
    assert payload[payload.index("--artifact-sha256") + 1] == digest
    assert payload[payload.index("--producer-run-id") + 1] == "run-1"
    assert payload[payload.index("--producer-plan-id") + 1] == "plan-1"
    assert payload[payload.index("--producer-recipe-sha256") + 1] == "a" * 64
    assert "--producer-handoff-sha256" in payload


def test_dry_run_derives_canonical_plan_from_compression_result_without_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    result, artifact, digest = _compression_result(tmp_path)
    args = _args(tmp_path)
    args.execute = False
    args.compression_result = result
    args.artifact = None
    args.artifact_sha256 = None
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(
        module,
        "MaintenanceCoordinator",
        lambda *_a, **_k: pytest.fail("plan preview must not construct maintenance"),
    )
    assert module.main() == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["execute"] is False
    assert preview["artifact"] == str(artifact)
    assert preview["artifact_sha256"] == digest
    assert preview["plan_sha256"] == module.build_base_canary_plan(
        module.CandidateBinding.model_validate(preview["plan"]["candidate"])
    ).canonical_sha256()
    assert preview["plan"]["candidate"]["producer_run_id"] == "run-1"
    assert preview["plan"]["candidate"]["producer_handoff_sha256"]


def test_supplied_canary_plan_hash_must_match_derived_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    result, _artifact, _digest = _compression_result(tmp_path)
    args = _args(tmp_path)
    args.execute = False
    args.compression_result = result
    args.artifact = None
    args.artifact_sha256 = None
    args.plan_sha256 = "f" * 64
    monkeypatch.setattr(module, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="differs from the canonical"):
        module.main()


def test_compression_result_mode_rejects_drifted_artifact(tmp_path: Path) -> None:
    module = _module()
    result, artifact, _digest = _compression_result(tmp_path)
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bytes drifted"):
        module._artifact_from_compression_result(result)


def test_evidence_cas_identity_changes_the_bound_handoff_digest(tmp_path: Path) -> None:
    module = _module()
    result, artifact, _digest = _compression_result(tmp_path)
    original = module._artifact_from_compression_result(result)
    payload = json.loads(result.read_text(encoding="utf-8"))
    replacement_payload = b'{"replacement":true}'
    replacement_sha = hashlib.sha256(replacement_payload).hexdigest()
    replacement = artifact.parents[2] / "objects" / replacement_sha[:2] / f"{replacement_sha}.blob"
    replacement.parent.mkdir(parents=True)
    replacement.write_bytes(replacement_payload)
    replacement_relpath = str(replacement.relative_to(artifact.parents[2]))
    evidence_contract = payload["runtime_artifact"]["evidence"]
    evidence_contract.update(
        sha256=replacement_sha,
        size_bytes=len(replacement_payload),
        relpath=replacement_relpath,
    )
    evidence_output = payload["outputs"]["outputs"][0]
    evidence_output.update(
        sha256=replacement_sha,
        size_bytes=len(replacement_payload),
        relpath=replacement_relpath,
    )
    result.write_text(json.dumps(payload), encoding="utf-8")
    replaced = module._artifact_from_compression_result(result)
    assert replaced.handoff_sha256 != original.handoff_sha256


def test_canary_rejects_mixed_artifact_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    result, _artifact, _digest = _compression_result(tmp_path)
    args = _args(tmp_path)
    args.compression_result = result
    monkeypatch.setattr(module, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="exactly one artifact mode"):
        module.main()


def test_canary_passthrough_cannot_override_reviewed_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    result, _artifact, _digest = _compression_result(tmp_path)
    args = _args(tmp_path)
    args.compression_result = result
    args.artifact = None
    args.artifact_sha256 = None
    args.canary_args = ["--producer-run-id", "forged"]
    monkeypatch.setattr(module, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="may not override"):
        module.main()


@pytest.mark.parametrize("argument", ["--producer-run", "--producer-run=forged"])
def test_canary_passthrough_rejects_abbreviated_reviewed_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argument: str
) -> None:
    module = _module()
    result, _artifact, _digest = _compression_result(tmp_path)
    args = _args(tmp_path)
    args.compression_result = result
    args.artifact = None
    args.artifact_sha256 = None
    args.canary_args = [argument, "forged"] if "=" not in argument else [argument]
    monkeypatch.setattr(module, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="may not override"):
        module.main()


def test_compression_result_rejects_symlinked_cas_ancestor(tmp_path: Path) -> None:
    module = _module()
    result, artifact, _digest = _compression_result(tmp_path)
    run_root = artifact.parents[3]
    real_runs = tmp_path / "real-runs"
    real_runs.mkdir()
    artifact.parents[2].rename(real_runs / "run-1")
    run_root.rmdir()
    run_root.symlink_to(real_runs, target_is_directory=True)
    with pytest.raises(OSError):
        module._artifact_from_compression_result(result)
