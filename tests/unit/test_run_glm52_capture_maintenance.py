from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from model_atlas.glm52_capture_plan import Glm52CapturePlan, build_glm52_capture_plan
from model_atlas.runtime_artifact_handoff import CompressionHandoff


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_glm52_capture_maintenance.py"
    spec = importlib.util.spec_from_file_location("test_glm52_capture_maintenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(tmp_path: Path) -> tuple[Glm52CapturePlan, CompressionHandoff]:
    model = tmp_path / "controlplane" / "runs" / "run-id" / "objects" / "aa" / "model.gguf"
    tokenizer = tmp_path / "source" / "tokenizer.json"
    model.parent.mkdir(parents=True)
    tokenizer.parent.mkdir()
    model.write_bytes(b"model")
    tokenizer.write_bytes(b"tokenizer")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    tokenizer_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    handoff = CompressionHandoff(
        artifact_path=str(model),
        artifact_sha256=model_sha,
        artifact_size_bytes=model.stat().st_size,
        evidence_sha256="2" * 64,
        evidence_size_bytes=1,
        evidence_relpath="objects/22/" + "2" * 64 + ".blob",
        producer_run_id="run-" + "5" * 24,
        producer_plan_id="recipe-" + "4" * 24,
        producer_recipe_sha256="3" * 64,
        producer_profile_id="profile-" + "6" * 24,
        producer_profile_sha256="7" * 64,
        producer_recommendation_id="rec-" + "8" * 24,
        handoff_sha256="9" * 64,
    )
    plan = build_glm52_capture_plan(
        work_root=tmp_path / "capture",
        model_path=model,
        model_sha256=model_sha,
        source_manifest_sha256="1" * 64,
        profile_tokenizer_path=tokenizer,
        profile_tokenizer_sha256=tokenizer_sha,
        producer_artifact_sha256=handoff.evidence_sha256,
        recipe_sha256=handoff.producer_recipe_sha256,
        plan_id=handoff.producer_plan_id,
        run_id=handoff.producer_run_id,
        profile_id=handoff.producer_profile_id,
        profile_sha256=handoff.producer_profile_sha256,
        recommendation_id=handoff.producer_recommendation_id,
        compression_handoff_sha256=handoff.handoff_sha256,
    )
    return plan, handoff


def _args(tmp_path: Path) -> Namespace:
    (tmp_path / "compression.json").write_text("{}")
    (tmp_path / "profile.json").write_text("{}")
    return Namespace(
        execute=False,
        payload=False,
        compression_result=tmp_path / "compression.json",
        profile=tmp_path / "profile.json",
        work_root=tmp_path / "capture",
        journal_dir=tmp_path / "journals",
        maintenance_receipt=tmp_path / "maintenance.json",
        lease=tmp_path / "lease.json",
        result=tmp_path / "result.json",
        plan_sha256=None,
        expected_plan_sha256="",
        expected_operation_sha256="",
        worker_ssh_target="10.77.0.2",
    )


def test_dry_run_emits_exact_requests_without_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    plan, handoff = _plan(tmp_path)
    args = _args(tmp_path)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "_plan", lambda _args: (plan, handoff))
    monkeypatch.setattr(
        module,
        "MaintenanceCoordinator",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not construct maintenance"),
    )
    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execute"] is False
    assert payload["quality_claim"] is False
    assert payload["capture_plan_sha256"] == plan.plan_sha256
    assert payload["candidate_request_id"] == plan.candidate.request_id
    assert payload["identity_request_id"] == plan.identity_control.request_id


def test_payload_runs_candidate_then_identity_and_requires_identity_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    plan, handoff = _plan(tmp_path)
    args = _args(tmp_path)
    args.execute = True
    args.payload = True
    args.expected_plan_sha256 = plan.plan_sha256
    args.expected_operation_sha256 = module._operation_sha256(args, plan)
    monkeypatch.setattr(module, "_plan", lambda _args: (plan, handoff))
    monkeypatch.setattr(module, "require_active_lease", lambda *_args: None)

    class _Coordinator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def verify_drained(self) -> None:
            pass

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    events: list[str] = []
    evidence = SimpleNamespace(model_dump=lambda **_kwargs: {"measured": True})

    class _Lifecycle:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self) -> object:
            events.append("start")
            return evidence

        def remeasure_after_capture(self) -> object:
            events.append("remeasure")
            return evidence

        def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr(module, "WorkerRpcSystemdLifecycle", _Lifecycle)
    monkeypatch.setattr(
        module, "_write_model_evidence", lambda _plan, _root_fd: events.append("evidence")
    )
    monkeypatch.setattr(module, "preflight_capture_request", lambda _request: None)
    monkeypatch.setattr(module, "_prepare_runtime_libraries", lambda root_fd: os.dup(root_fd))
    monkeypatch.setattr(
        module, "_run_native", lambda _argv, **_kwargs: events.append("native")
    )
    captures = iter(
        (
            SimpleNamespace(capture_id="candidate"),
            SimpleNamespace(capture_id="identity"),
        )
    )
    monkeypatch.setattr(module, "finalize_capture", lambda _request, **_kwargs: next(captures))
    report = SimpleNamespace(
        identity_control_passed=True,
        report_id="report",
        model_dump=lambda **_kwargs: {"identity_control_passed": True},
    )
    monkeypatch.setattr(module, "evaluate_capture_pair", lambda **_kwargs: report)
    monkeypatch.setattr(module, "build_capture_argv", lambda *_args, **_kwargs: ("capture",))
    monkeypatch.setattr(
        module,
        "_bounded_sha256",
        lambda path: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "a" * 64,
    )
    assert module._execute_payload(args) == 0
    assert events == ["evidence", "start", "native", "remeasure", "native", "remeasure", "stop"]
    result = json.loads(args.result.read_text())
    assert result["identity_control_passed"] is True
    assert result["quality_claim"] is False


def test_control_path_rejects_symlinked_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    plan, handoff = _plan(tmp_path)
    args = _args(tmp_path)
    real = tmp_path / "real-result-parent"
    real.mkdir()
    alias = tmp_path / "alias-result-parent"
    alias.symlink_to(real, target_is_directory=True)
    args.result = alias / "result.json"
    with pytest.raises(RuntimeError, match="symlinked ancestor"):
        module._validate_paths(args, plan, handoff)


def test_work_root_path_replacement_is_detected(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "work"
    root.mkdir()
    descriptor = module._open_directory_chain(root)
    try:
        identity = module._directory_identity(descriptor)
        moved = tmp_path / "moved"
        root.rename(moved)
        root.mkdir()
        with pytest.raises(RuntimeError, match="path identity drifted"):
            module._assert_directory_path_identity(root, identity)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("source_kind", ["fifo", "size-drift"])
def test_runtime_library_staging_rejects_unbounded_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
) -> None:
    module = _module()
    library_root = tmp_path / "libraries"
    library_root.mkdir()
    library = library_root / "libbounded.so.1.0"
    if source_kind == "fifo":
        os.mkfifo(library)
        expected_size = 1
    else:
        library.write_bytes(b"larger-than-contract")
        expected_size = 1
    contract = tmp_path / "build.json"
    contract.write_text(
        json.dumps(
            {
                "library_root": str(library_root),
                "libraries": {library.name: "a" * 64},
                "library_sizes": {library.name: expected_size},
            }
        )
    )
    monkeypatch.setattr(module, "CAPTURE_BUILD_CONTRACT", contract)
    monkeypatch.setattr(
        module,
        "CAPTURE_BUILD_CONTRACT_SHA256",
        hashlib.sha256(contract.read_bytes()).hexdigest(),
    )
    work = tmp_path / "work"
    work.mkdir()
    root_fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="contracted regular file"):
            module._prepare_runtime_libraries(root_fd)
    finally:
        os.close(root_fd)
