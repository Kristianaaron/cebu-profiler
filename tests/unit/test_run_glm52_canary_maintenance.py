from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from model_atlas.ops.maintenance import MaintenanceInterrupted, MaintenanceReceipt


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
        plan_sha256="a" * 64,
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

        def run(self, _payload: object) -> MaintenanceReceipt:
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

        def run(self, _payload: object) -> MaintenanceReceipt:
            calls.append("run")
            now = datetime.now(UTC)
            return MaintenanceReceipt(
                run_id="fake", dry_run=False, started_at=now, finished_at=now, success=True
            )

    monkeypatch.setattr(module, "MaintenanceCoordinator", _Coordinator)
    assert module.main() == 0
    assert calls == ["init", "run", "restore"]


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
