from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_glm52_candidate_eval_maintenance.py"
    spec = importlib.util.spec_from_file_location("test_candidate_eval_maintenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> Namespace:
    operation = tmp_path / "operation"
    operation.mkdir()
    journal = tmp_path / "journal"
    journal.mkdir()
    for name in ("compression.json", "canary.json", "profile.json"):
        (tmp_path / name).write_text("{}")
    return Namespace(
        execute=False,
        compression_result=(tmp_path / "compression.json").resolve(),
        canary_result=(tmp_path / "canary.json").resolve(),
        profile=(tmp_path / "profile.json").resolve(),
        operation_root=operation.resolve(),
        journal_dir=journal.resolve(),
        receipt=(tmp_path / "receipt.json").resolve(),
        lease=(tmp_path / "lease.json").resolve(),
        result=(operation / "result.json").resolve(),
        expected_plan_sha256=None,
    )


def _bound_objects() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    compression = SimpleNamespace(
        artifact_path="/objects/model.gguf",
        artifact_sha256="a" * 64,
        handoff_sha256="b" * 64,
    )
    canary = SimpleNamespace()
    config = SimpleNamespace()
    plan = SimpleNamespace(
        plan_sha256="c" * 64,
        runtime_canary_handoff_sha256="d" * 64,
        argv=("eval-lab", "run"),
        model_dump=lambda **_kwargs: {"schema_version": 1, "plan_sha256": "c" * 64},
    )
    return compression, canary, config, plan


def test_dry_run_emits_plan_without_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    args = _args(tmp_path)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "_build_plan", lambda _args: _bound_objects())
    monkeypatch.setattr(
        module,
        "MaintenanceCoordinator",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not construct maintenance"),
    )
    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execute"] is False
    assert payload["plan_sha256"] == "c" * 64
    assert payload["artifact_sha256"] == "a" * 64


def test_paths_require_result_inside_operation_root(tmp_path: Path) -> None:
    module = _module()
    args = _args(tmp_path)
    args.result = (tmp_path / "outside.json").resolve()
    with pytest.raises(RuntimeError, match="direct child"):
        module._validate_paths(args)


def test_outer_execute_dispatches_only_through_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    args = _args(tmp_path)
    args.execute = True
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "_build_plan", lambda _args: _bound_objects())
    calls: list[object] = []

    class Coordinator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs.get("payload_action"))
            return SimpleNamespace(success=False, model_dump_json=lambda: "{}")

    monkeypatch.setattr(module, "MaintenanceCoordinator", Coordinator)
    monkeypatch.setattr(module, "install_signal_traps", lambda: {})
    monkeypatch.setattr(module, "restore_signal_traps", lambda _value: None)
    assert module.main() == 1
    assert len(calls) == 1
    assert callable(calls[0])


@pytest.mark.parametrize("field", ["receipt", "lease", "result"])
def test_existing_authority_outputs_fail_before_any_plan_work(tmp_path: Path, field: str) -> None:
    module = _module()
    args = _args(tmp_path)
    getattr(args, field).write_text("stale")
    with pytest.raises(RuntimeError, match="fresh paths"):
        module._validate_paths(args)


@pytest.mark.parametrize("field", ["receipt", "lease", "result"])
def test_broken_symlink_authority_outputs_are_not_fresh(tmp_path: Path, field: str) -> None:
    module = _module()
    args = _args(tmp_path)
    getattr(args, field).symlink_to(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="fresh paths"):
        module._validate_paths(args)


def test_symlinked_input_parent_is_rejected(tmp_path: Path) -> None:
    module = _module()
    args = _args(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    args.profile = alias / "profile.json"
    with pytest.raises(OSError):
        module._validate_paths(args)
