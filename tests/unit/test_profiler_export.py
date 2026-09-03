"""Atlas export executor tests: canonical `atlas_runs/<id>/` run dir (§1/§3)."""

import json
from pathlib import Path

import pytest

from model_atlas.atlas.export import export_run
from model_atlas.atlas.output_layout import ATLAS_RUN_FILES


def _write_fixture_corpus(root: Path) -> None:
    """A tiny eval-lab-shaped task tree with two prompt.md files."""
    tasks = root / "eval_lab" / "tasks"
    specs = [("coding", "refactor"), ("mathematics", "addition")]
    for i, (domain, name) in enumerate(specs):
        p = tasks / domain / name / "prompt.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"this is task {i} with several words to tokenize foo bar baz qux")


def _read_json(path: Path):
    return json.loads(path.read_text())


def test_export_writes_parseable_run_files(tmp_path):
    _write_fixture_corpus(tmp_path)
    result = export_run(
        str(tmp_path / "out"), eval_lab_root=str(tmp_path / "eval_lab"), seed=1, keep_per_layer=3
    )
    run_dir = Path(result["run_dir"])
    fnames = [
        "run_manifest.json",
        "layer_saliency.json",
        "plans.json",
        "compression_manifest.json",
        "hierarchy_map.json",
        "planning_maps.json",
        # v3 fidelity-first artifacts (always generated, no build)
        "v3_run.json",
        "v3_corpus_evidence.json",
        "v3_candidate_graph.json",
    ]
    for fname in fnames:
        assert (run_dir / fname).exists(), fname
        assert isinstance(_read_json(run_dir / fname), (list, dict))
    n_extra = len(list(run_dir.iterdir())) - len(fnames)
    # every extra file must be declared in the §27 output contract
    for p in run_dir.iterdir():
        assert p.name in ATLAS_RUN_FILES or p.name in fnames, p.name
    assert n_extra >= 0


def test_export_writes_planning_maps_artifact(tmp_path):
    _write_fixture_corpus(tmp_path)
    result = export_run(
        str(tmp_path / "out"), eval_lab_root=str(tmp_path / "eval_lab"), seed=1, keep_per_layer=3
    )
    run_dir = Path(result["run_dir"])
    pm = _read_json(run_dir / "planning_maps.json")
    assert pm["source_arch"] == "k3-mini"
    for name in (
        "channel", "tile", "node_ownership", "overflow_pack",
        "router_repair", "residual_repair", "distillation_target",
    ):
        assert pm["maps"][name], f"empty maps.{name}"
    assert pm["candidates"]
    assert all("precision" in c and "resident_bytes_a" in c for c in pm["candidates"])


def test_export_writes_parseable_hierarchy_map(tmp_path):
    _write_fixture_corpus(tmp_path)
    result = export_run(
        str(tmp_path / "out"), eval_lab_root=str(tmp_path / "eval_lab"), seed=1, keep_per_layer=3
    )
    run_dir = Path(result["run_dir"])
    hm = _read_json(run_dir / "hierarchy_map.json")
    assert hm["levels"] == ["weights", "units", "experts", "coalitions", "pathways", "behaviour"]
    for lv in hm["levels"]:
        assert hm["counts"][lv] > 0, f"empty level {lv}"
    assert hm["counts"]["units"] >= hm["counts"]["experts"]


def test_export_writes_valid_compression_manifest(tmp_path):
    _write_fixture_corpus(tmp_path)
    result = export_run(
        str(tmp_path / "out"), eval_lab_root=str(tmp_path / "eval_lab"), seed=1, keep_per_layer=3
    )
    run_dir = Path(result["run_dir"])
    cm = _read_json(run_dir / "compression_manifest.json")
    assert cm["model"] == "k3-mini"
    assert cm["source_checkpoint"]
    assert cm["allowed_widths"]
    manifest = _read_json(run_dir / "run_manifest.json")
    assert "compression_manifest.json" in manifest["evidence_present"]
    for lp in cm["layers"].values():
        for plan in lp["experts"].values():
            assert plan["target_width"] <= plan["original_width"]
            assert len(plan["keep_channels"]) == plan["target_width"]
            assert plan["target_width"] in cm["allowed_widths"]


def test_export_layer_saliency_and_keep_map(tmp_path):
    _write_fixture_corpus(tmp_path)
    result = export_run(
        str(tmp_path / "out"), eval_lab_root=str(tmp_path / "eval_lab"), seed=1, keep_per_layer=3
    )
    run_dir = Path(result["run_dir"])

    saliency = _read_json(run_dir / "layer_saliency.json")
    assert saliency, "layer_saliency must be non-empty"
    row = saliency[0]
    assert all(k in row for k in ("layer", "expert", "label", "mean", "frequency", "total_value"))

    plans = _read_json(run_dir / "plans.json")
    assert plans, "plans must be non-empty"
    plan = plans[0]
    assert plan["name"] == result["plan_names"][0]
    # keep-map entries preserve source_expert_id (plan.keep.entries — KeepMap)
    keep_map = plan["keep_map"]
    assert keep_map["entries"]
    assert all("source_expert_id" in e and "layer_index" in e for e in keep_map["entries"])


def test_export_build_writes_derivative(tmp_path):
    _write_fixture_corpus(tmp_path)
    result = export_run(
        str(tmp_path / "out"),
        eval_lab_root=str(tmp_path / "eval_lab"),
        seed=1,
        keep_per_layer=3,
        build=True,
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "derivative.json").exists()
    der = _read_json(run_dir / "derivative.json")
    assert der["asset_type"] == "derivative_checkpoint"
    assert der["parent_model_id"]
    manifest = _read_json(run_dir / "run_manifest.json")
    assert "derivative.json" in manifest["evidence_present"]


def test_export_new_run_id_each_call(tmp_path):
    _write_fixture_corpus(tmp_path)
    r1 = export_run(str(tmp_path / "out"), eval_lab_root=str(tmp_path / "eval_lab"), seed=1)
    r2 = export_run(str(tmp_path / "out"), eval_lab_root=str(tmp_path / "eval_lab"), seed=1)
    assert r1["run_dir"] != r2["run_dir"]


def test_export_empty_corpus_raises(tmp_path):
    with pytest.raises(ValueError):
        export_run(str(tmp_path / "out"), eval_lab_root=str(tmp_path / "no_such_root"))
