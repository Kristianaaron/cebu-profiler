import pytest

import model_atlas.backend.modelopt_adapter as ma
from model_atlas.backend.contract import BackendUnavailable
from model_atlas.backend.modelopt_adapter import (
    MODELOPT_INSTALL_GUIDE,
    ModelOptNvfp4Adapter,
    probe_modelopt,
)


def test_probe_absent_is_truthful_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ma, "module_present", lambda n: False)
    monkeypatch.setattr(ma, "module_version", lambda n: None)
    ok, ver, note = probe_modelopt()
    assert ok is False
    assert ver is None
    assert "pip install" in note


def test_probe_present_when_module_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ma, "module_present", lambda n: True)
    monkeypatch.setattr(ma, "module_version", lambda n: "0.45.0")
    ok, ver, note = probe_modelopt()
    assert ok is True
    assert ver == "0.45.0"
    assert note == "modelopt importable"


def test_prepare_fails_closed_without_modelopt() -> None:
    adapter = ModelOptNvfp4Adapter()
    with pytest.raises(BackendUnavailable, match="pip install"):
        adapter.prepare({"source_path": "/x"})


def test_execute_fails_closed_without_modelopt() -> None:
    adapter = ModelOptNvfp4Adapter()
    with pytest.raises(BackendUnavailable) as exc:
        adapter.execute(
            {"source_path": "/x", "staging_dir": "/y", "stage_id": "nvfp4"},
            "modelopt:nvfp4",
        )
    assert "pip install" in str(exc.value) or "maintenance window" in str(exc.value)


def test_adapter_is_derivative_producer() -> None:
    assert ModelOptNvfp4Adapter().produces_derivative is True


def test_record_probe_is_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ma, "module_present", lambda n: False)
    from model_atlas.backend.registry import build_default_registry

    rec = build_default_registry().get("modelopt_nvfp4")
    assert rec is not None
    ok, _ver, note = rec.availability_probe()
    assert ok is False
    assert "pip install" in note


def test_record_adapter_is_modelopt_nvfp4() -> None:
    from model_atlas.backend.registry import build_default_registry

    adapter = build_default_registry().get("modelopt_nvfp4").adapter
    assert type(adapter).__name__ == "ModelOptNvfp4Adapter"
    assert adapter.produces_derivative is True


def test_install_guide_is_bounded() -> None:
    assert isinstance(MODELOPT_INSTALL_GUIDE, str)
    assert "nvidia-modelopt" in MODELOPT_INSTALL_GUIDE
