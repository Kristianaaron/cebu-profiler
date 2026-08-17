from pathlib import Path

import pytest

import model_atlas.runtime_artifact_handoff as handoff_mod
from model_atlas.recommend.policy import method_spec
from model_atlas.runtime_artifact_handoff import (
    _WIDTH_SLICE_ARTIFACT,
    _WIDTH_SLICE_METHOD,
    load_verified_width_slice_handoff,
)


def test_width_slice_handoff_routes_to_bundle_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_base(path: Path, *, method: str, artifact_name: str) -> str:
        captured["path"] = path
        captured["method"] = method
        captured["artifact_name"] = artifact_name
        return "verified"

    monkeypatch.setattr(handoff_mod, "load_verified_compression_handoff", fake_base)
    result = load_verified_width_slice_handoff(Path("/x/result.json"))
    assert result == "verified"
    assert captured["path"] == Path("/x/result.json")
    assert captured["method"] == _WIDTH_SLICE_METHOD
    assert captured["artifact_name"] == _WIDTH_SLICE_ARTIFACT


def test_width_slice_handoff_matches_registered_catalog_method() -> None:
    spec = method_spec("atlas-nvfp4-width-slice")
    assert spec.method == _WIDTH_SLICE_METHOD
    assert _WIDTH_SLICE_ARTIFACT == "model.safetensors.atlasbundle"


def test_gguf_default_still_used_when_not_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_base(
        path: Path,
        *,
        method: str = "llamacpp-gguf-mixed",
        artifact_name: str = "model.gguf",
    ) -> str:
        captured["method"] = method
        captured["artifact_name"] = artifact_name
        return "ok"

    monkeypatch.setattr(handoff_mod, "load_verified_compression_handoff", fake_base)
    handoff_mod.load_verified_compression_handoff(Path("/x/gguf.json"))
    assert captured["method"] == handoff_mod._COMPRESSION_METHOD  # noqa: SLF001
    assert captured["artifact_name"] == "model.gguf"
