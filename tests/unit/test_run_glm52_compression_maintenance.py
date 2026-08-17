from __future__ import annotations

import hashlib
import importlib.util
from argparse import Namespace
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from model_atlas.ops.maintenance import MaintenanceReceipt


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_glm52_compression_maintenance.py"
    spec = importlib.util.spec_from_file_location("test_glm52_compression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, *, execute: bool = False, payload: bool = False) -> Namespace:
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    return Namespace(
        execute=execute,
        payload=payload,
        profile=profile,
        profiles_dir=tmp_path / "profiles",
        work_root=tmp_path / "work",
        memory_target_gib=115.0,
        poll_seconds=0.001,
        journal_dir=tmp_path / "journals",
        maintenance_receipt=tmp_path / "maintenance.json",
        result=tmp_path / "result.json",
        lease=tmp_path / "lease.json",
        expected_recipe_sha256="a" * 64,
        expected_profile_sha256="",
        expected_profile_id="profile-1",
        expected_recommendation_id="rec-1",
        expected_plan_id="plan-1",
        expected_run_id="run-1",
    )


def _preview() -> dict[str, Any]:
    return {
        "preview_id": "pv-1",
        "recipe_id": "recipe-1",
        "recipe_sha256": "a" * 64,
        "plan_id": "plan-1",
        "run_id": "run-1",
        "hash": "b" * 64,
        "readiness": {
            "verified_plan": True,
            "pins_pass": True,
            "intent_satisfied": True,
            "executable": True,
        },
        "actual_families": ["quantization"],
    }


class _Service:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self.statuses = list(statuses or ["completed"])
        self.started = False
        self.shutdown_called = False
        self.plane = _Plane()

    def start_authorized(self, *_args: object, **_kwargs: object) -> dict[str, str]:
        self.started = True
        return {"run_id": "run-1", "status": "started"}

    def job_status(self, _run_id: str) -> dict[str, str]:
        return {"status": self.statuses.pop(0) if self.statuses else "completed"}

    def job_output(self, _run_id: str) -> dict[str, object]:
        return {"outputs": []}

    def run_lineage(self, _run_id: str) -> dict[str, str]:
        return {"run_id": "run-1"}

    def shutdown(self, wait: bool = True) -> None:
        assert wait
        self.shutdown_called = True


class _Compiled:
    plan_id = "plan-1"
    recipe_sha256 = "a" * 64


class _Engine:
    compiled = _Compiled()


class _Plane:
    def __init__(self) -> None:
        self.resumed = False

    @staticmethod
    def engine_for(_run_id: str) -> _Engine:
        return _Engine()

    def resume(self, _run_id: str) -> None:
        self.resumed = True


def _authorization() -> dict[str, str]:
    return {
        "token": "t",
        "profile_id": "profile-1",
        "recommendation_id": "rec-1",
    }


def _artifact_result() -> dict[str, object]:
    return {
        "path": "/verified/model.gguf",
        "sha256": "c" * 64,
        "size_bytes": 1,
        "stage": "llamacpp-gguf-mixed",
        "logical_name": "model.gguf",
        "relpath": "objects/cc/model.gguf",
        "runtime_validated": False,
    }


def _evidence_ref(run_dir: Path) -> dict[str, object]:
    payload = b"{}"
    digest = hashlib.sha256(payload).hexdigest()
    path = run_dir / "objects" / digest[:2] / f"{digest}.blob"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "stage": "llamacpp-gguf-mixed",
        "name": "llamacpp-gguf-mixed.evidence.json",
        "sha256": digest,
        "size_bytes": len(payload),
        "relpath": str(path.relative_to(run_dir)),
    }


def _stub_profile(
    module: ModuleType,
    args: Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    digest = module._sha256_file(args.profile)
    monkeypatch.setattr(module, "_profile_identity", lambda _path: (object(), digest))
    monkeypatch.setattr(module, "_validate_control_paths", lambda *_args: None)
    return digest


def test_dry_run_previews_without_maintenance_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path)
    service = _Service()
    _stub_profile(module, args, monkeypatch)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "_preview", lambda _args: (service, _authorization(), _preview()))
    monkeypatch.setattr(
        module,
        "MaintenanceCoordinator",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not construct maintenance"),
    )
    assert module.main() == 0
    assert args.result.exists()
    assert not service.started
    assert service.shutdown_called


def test_payload_rejects_before_preview_without_live_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True, payload=True)
    args.expected_profile_sha256 = _stub_profile(module, args, monkeypatch)

    def _reject(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("no lease")

    monkeypatch.setattr(module, "require_active_lease", _reject)
    monkeypatch.setattr(module, "_preview", lambda _args: pytest.fail("preview must follow lease"))
    with pytest.raises(RuntimeError, match="no lease"):
        module._run_payload(args)


def test_payload_rechecks_drain_recipe_and_reaches_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True, payload=True)
    args.expected_profile_sha256 = _stub_profile(module, args, monkeypatch)
    service = _Service(["running", "completed"])
    events: list[str] = []
    monkeypatch.setattr(module, "require_active_lease", lambda *_a, **_k: events.append("lease"))

    class _Coordinator:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def verify_drained(self) -> None:
            events.append("drained")

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    monkeypatch.setattr(module, "_preview", lambda _args: (service, _authorization(), _preview()))
    monkeypatch.setattr(
        module, "_verified_runtime_artifact", lambda *_a: (_artifact_result(), object())
    )
    monkeypatch.setattr(module, "_reverify_runtime_artifact", lambda _pin: None)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    assert module._run_payload(args) == 0
    assert events == ["lease", "drained"]
    assert service.started and service.shutdown_called
    assert '"status": "completed"' in args.result.read_text(encoding="utf-8")


def test_execute_holds_content_bound_scope_around_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True)
    service = _Service()
    _stub_profile(module, args, monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "_preview", lambda _args: (service, _authorization(), _preview()))
    monkeypatch.setattr(module, "install_signal_traps", lambda: {2: object()})
    monkeypatch.setattr(module, "restore_signal_traps", lambda _old: events.append("restore-traps"))

    @contextmanager
    def _scope(*_args: object, **_kwargs: object):
        events.append("scope-enter")
        yield
        events.append("scope-exit")

    monkeypatch.setattr(module, "active_lease_scope", _scope)

    class _Coordinator:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def run(self, payload: object, *, payload_scope: object) -> MaintenanceReceipt:
            assert isinstance(payload, tuple)
            assert "--payload" in payload
            assert callable(payload_scope)
            with payload_scope():  # type: ignore[operator]
                events.append("payload")
            now = datetime.now(UTC)
            return MaintenanceReceipt(
                run_id="m", dry_run=False, started_at=now, finished_at=now, success=True
            )

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    assert module.main() == 0
    assert events == ["scope-enter", "payload", "scope-exit", "restore-traps"]


def test_payload_profile_drift_fails_before_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True, payload=True)
    _stub_profile(module, args, monkeypatch)
    args.expected_profile_sha256 = "f" * 64
    monkeypatch.setattr(
        module,
        "require_active_lease",
        lambda *_a, **_k: pytest.fail("drift must fail before lease acceptance"),
    )
    with pytest.raises(RuntimeError, match="profile bytes drifted"):
        module._run_payload(args)


def test_preview_resolves_file_to_content_id_and_disables_auto_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path)
    args.profiles_dir.mkdir()
    args.profile = args.profiles_dir / "glm.json"
    args.profile.write_text("{}", encoding="utf-8")
    calls: list[object] = []

    class _Profile:
        @staticmethod
        def profile_id_of() -> str:
            return "profile-content-id"

    class _PreviewService:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def import_profile(self, path: Path) -> _Profile:
            calls.append(path)
            return _Profile()

        def authorize(self, profile: str, *_args: object, **_kwargs: object) -> dict[str, object]:
            calls.append(profile)
            return {"token": "t", "authorized_methods": [module.METHOD]}

        @staticmethod
        def preview_selection(_token: str, _selected: list[str]) -> dict[str, Any]:
            return _preview()

    monkeypatch.setattr(module, "RecommendationService", _PreviewService)
    _service, _authorization, preview = module._preview(args)
    assert preview["recipe_sha256"] == "a" * 64
    assert calls[0]["supervised_executor"] is False  # type: ignore[index]
    assert calls[1] == args.profile
    assert calls[2] == "profile-content-id"


def test_payload_resumes_matching_recoverable_run_instead_of_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True, payload=True)
    args.expected_profile_sha256 = _stub_profile(module, args, monkeypatch)
    job = args.work_root / "runs" / "run-1" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")
    service = _Service()
    states = ["failed_recoverable", "completed", "completed"]

    plane = _Plane()
    service.plane = plane  # type: ignore[attr-defined]
    service.job_status = lambda _run_id: {"status": states.pop(0)}  # type: ignore[method-assign]
    monkeypatch.setattr(module, "require_active_lease", lambda *_a, **_k: None)

    class _Coordinator:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        @staticmethod
        def verify_drained() -> None:
            pass

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    monkeypatch.setattr(module, "_preview", lambda _args: (service, _authorization(), _preview()))
    monkeypatch.setattr(
        module, "_verified_runtime_artifact", lambda *_a: (_artifact_result(), object())
    )
    monkeypatch.setattr(module, "_reverify_runtime_artifact", lambda _pin: None)
    assert module._run_payload(args) == 0
    assert plane.resumed
    assert not service.started


def test_completed_run_must_pass_engine_integrity_before_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True, payload=True)
    args.expected_profile_sha256 = _stub_profile(module, args, monkeypatch)
    job = args.work_root / "runs" / "run-1" / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}", encoding="utf-8")
    service = _Service(["completed"])

    class _CorruptPlane(_Plane):
        def resume(self, _run_id: str) -> None:
            raise RuntimeError("DONE output hash mismatch")

    service.plane = _CorruptPlane()  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "require_active_lease", lambda *_a, **_k: None)

    class _Coordinator:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        @staticmethod
        def verify_drained() -> None:
            pass

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    monkeypatch.setattr(module, "_preview", lambda _args: (service, _authorization(), _preview()))
    with pytest.raises(RuntimeError, match="DONE output hash mismatch"):
        module._run_payload(args)
    assert not args.result.exists()
    assert service.shutdown_called


def test_control_paths_cannot_alias_each_other_or_protected_roots(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    profile_path = profiles / "profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    work = tmp_path / "work"
    args = Namespace(
        result=tmp_path / "result.json",
        maintenance_receipt=tmp_path / "receipt.json",
        lease=tmp_path / "lease.json",
        journal_dir=tmp_path / "journals",
        profile=profile_path,
        profiles_dir=profiles,
        work_root=work,
    )
    profile = SimpleNamespace(execution=SimpleNamespace(checkpoint_path=str(source)))
    module._validate_control_paths(args, profile)

    args.maintenance_receipt = args.result
    with pytest.raises(ValueError, match="overlaps"):
        module._validate_control_paths(args, profile)
    args.maintenance_receipt = tmp_path / "receipt.json"

    args.result = work / "runs" / "run-1" / "job.json"
    with pytest.raises(ValueError, match="work root"):
        module._validate_control_paths(args, profile)
    args.result = source / "result.json"
    with pytest.raises(ValueError, match="source checkpoint"):
        module._validate_control_paths(args, profile)


def test_post_preview_identity_failure_still_shuts_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True, payload=True)
    args.expected_profile_sha256 = _stub_profile(module, args, monkeypatch)
    args.expected_plan_id = "different-plan"
    service = _Service()
    monkeypatch.setattr(module, "require_active_lease", lambda *_a, **_k: None)

    class _Coordinator:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        @staticmethod
        def verify_drained() -> None:
            pass

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    monkeypatch.setattr(module, "_preview", lambda _args: (service, _authorization(), _preview()))
    with pytest.raises(RuntimeError, match="authorization identity drifted"):
        module._run_payload(args)
    assert service.shutdown_called


def test_completed_output_handoff_resolves_and_rehashes_exact_cas_blob(tmp_path: Path) -> None:
    module = _module()
    run_dir = tmp_path / "runs" / "run-1"
    payload = b"bounded-gguf"
    digest = hashlib.sha256(payload).hexdigest()
    blob = run_dir / "objects" / digest[:2] / f"{digest}.blob"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    evidence = _evidence_ref(run_dir)
    service = SimpleNamespace(
        plane=SimpleNamespace(engine_for=lambda _run_id: SimpleNamespace(run_dir=run_dir))
    )
    outputs = {
        "run_id": "run-1",
        "outputs": [
            evidence,
            {
                "stage": "llamacpp-gguf-mixed",
                "name": "model.gguf",
                "sha256": digest,
                "size_bytes": len(payload),
                "relpath": str(blob.relative_to(run_dir)),
            },
        ],
    }
    handoff, pin = module._verified_runtime_artifact(service, "run-1", outputs)
    assert handoff == {
        "path": str(blob.resolve()),
        "sha256": digest,
        "size_bytes": len(payload),
        "stage": "llamacpp-gguf-mixed",
        "logical_name": "model.gguf",
        "relpath": str(blob.relative_to(run_dir)),
        "runtime_validated": False,
        "evidence": {
            "stage": "llamacpp-gguf-mixed",
            "logical_name": "llamacpp-gguf-mixed.evidence.json",
            "sha256": evidence["sha256"],
            "size_bytes": evidence["size_bytes"],
            "relpath": evidence["relpath"],
        },
    }
    module._reverify_runtime_artifact(pin)


@pytest.mark.parametrize(
    ("relpath", "sha256", "size", "message"),
    [
        ("../outside.gguf", "a" * 64, 1, "escapes"),
        (f"objects/00/{'0' * 64}.blob", "0" * 64, 12, "identity differs"),
    ],
)
def test_completed_output_handoff_fails_closed_on_escape_or_identity_drift(
    tmp_path: Path,
    relpath: str,
    sha256: str,
    size: int,
    message: str,
) -> None:
    module = _module()
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    blob = run_dir / relpath
    if ".." not in blob.parts:
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"bounded-gguf")
    service = SimpleNamespace(
        plane=SimpleNamespace(engine_for=lambda _run_id: SimpleNamespace(run_dir=run_dir))
    )
    outputs = {
        "run_id": "run-1",
        "outputs": [
            _evidence_ref(run_dir),
            {
                "stage": "llamacpp-gguf-mixed",
                "name": "model.gguf",
                "sha256": sha256,
                "size_bytes": size,
                "relpath": relpath,
            },
        ],
    }
    with pytest.raises(RuntimeError, match=message):
        module._verified_runtime_artifact(service, "run-1", outputs)


def test_completed_output_handoff_rejects_symlinked_cas_ancestor(tmp_path: Path) -> None:
    module = _module()
    payload = b"bounded-gguf"
    digest = hashlib.sha256(payload).hexdigest()
    real_runs = tmp_path / "real-runs"
    real_blob = real_runs / "run-1" / "objects" / digest[:2] / f"{digest}.blob"
    real_blob.parent.mkdir(parents=True)
    real_blob.write_bytes(payload)
    linked_runs = tmp_path / "runs"
    linked_runs.symlink_to(real_runs, target_is_directory=True)
    run_dir = linked_runs / "run-1"
    service = SimpleNamespace(
        plane=SimpleNamespace(engine_for=lambda _run_id: SimpleNamespace(run_dir=run_dir))
    )
    outputs = {
        "run_id": "run-1",
        "outputs": [
            _evidence_ref(real_runs / "run-1"),
            {
                "stage": "llamacpp-gguf-mixed",
                "name": "model.gguf",
                "sha256": digest,
                "size_bytes": len(payload),
                "relpath": f"objects/{digest[:2]}/{digest}.blob",
            },
        ],
    }
    with pytest.raises(OSError):
        module._verified_runtime_artifact(service, "run-1", outputs)


def test_result_is_removed_if_artifact_drifts_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path, execute=True, payload=True)
    args.expected_profile_sha256 = _stub_profile(module, args, monkeypatch)
    service = _Service(["completed"])
    monkeypatch.setattr(module, "require_active_lease", lambda *_a, **_k: None)

    class _Coordinator:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        @staticmethod
        def verify_drained() -> None:
            pass

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    monkeypatch.setattr(module, "_preview", lambda _args: (service, _authorization(), _preview()))
    monkeypatch.setattr(
        module, "_verified_runtime_artifact", lambda *_a: (_artifact_result(), object())
    )
    monkeypatch.setattr(
        module,
        "_reverify_runtime_artifact",
        lambda _pin: (_ for _ in ()).throw(RuntimeError("drifted during publication")),
    )
    with pytest.raises(RuntimeError, match="drifted during publication"):
        module._run_payload(args)
    assert not args.result.exists()
