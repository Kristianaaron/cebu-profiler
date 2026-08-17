from pathlib import Path

from model_atlas.recommend.policy import (
    METHOD_CATALOG,
    _load_method_plugins,  # noqa: PLC2701
    _method_plugin_root,  # noqa: PLC2701
    method_catalog_digest,
)

_PLUGIN_SRC = '''\
from model_atlas.recommend.policy import (
    CompressionIntent,
    MethodFamily,
    MethodSpec,
    StageEffectClass,
)

def register_methods():
    return {
        "{ID}": MethodSpec(
            "{ID}", 900, MethodFamily.PRUNING, "dummy-be-dummy-do",
            ("channel_saliency",), ("width-slice",),
            (StageEffectClass.PRUNING,), (CompressionIntent.PRUNE_ONLY,),
            "down", routing_dependent=False, provenance_ids=("test-plugin",),
        ),
    }
'''


def _write_plugin(root: Path, name: str, method_id: str) -> None:
    (root / name).write_text(_PLUGIN_SRC.replace("{ID}", method_id), encoding="utf-8")


def test_method_plugin_root_resolves_env(monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_METHOD_PLUGIN_DIR", raising=False)
    assert _method_plugin_root() is None
    monkeypatch.setenv("ATLAS_METHOD_PLUGIN_DIR", "/tmp/atlas-methods")
    assert _method_plugin_root() == Path("/tmp/atlas-methods").resolve()


def test_load_method_plugins_missing_dir_returns_empty() -> None:
    assert _load_method_plugins(Path("/nonexistent/atlas-methods")) == ()


def test_load_method_plugins_deterministic_order(tmp_path) -> None:
    _write_plugin(tmp_path, "beta.py", "plugin-a")
    _write_plugin(tmp_path, "alpha.py", "plugin-z")
    specs = _load_method_plugins(tmp_path)
    assert [s.method for s in specs] == ["plugin-a", "plugin-z"]


def test_plugin_method_folds_into_digest(tmp_path) -> None:
    _write_plugin(tmp_path, "my.py", "plugin-q")
    specs = _load_method_plugins(tmp_path)
    assert any(s.method == "plugin-q" for s in specs)
    base = method_catalog_digest(METHOD_CATALOG)
    extended = method_catalog_digest(METHOD_CATALOG + specs)
    assert extended != base
