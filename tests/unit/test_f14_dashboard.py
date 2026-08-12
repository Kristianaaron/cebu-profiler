"""F14 tests: dashboard rendering over measured data."""

import json

from model_atlas.dashboard import build_dashboard_data, render_dashboard, write_dashboard


def test_build_dashboard_data_has_sections():
    data = build_dashboard_data(seed=0)
    assert data["meta"]["arch"] == "k3-mini"
    assert data["capability"]  # per-label expert rankings
    assert "voxels" in data["capability3d"] and data["capability3d"]["voxels"]
    assert data["capability3d"]["labels"]
    assert data["contrast"]
    assert data["coalitions"]
    assert data["paths"]
    assert data["compression"]
    assert data["candidates"]
    assert data["heldout"]
    assert set(data["hierarchy"]["levels"]) == {
        "weights", "units", "experts", "coalitions", "pathways", "behaviour"
    }
    assert all(data["hierarchy"]["counts"][lv] > 0 for lv in data["hierarchy"]["levels"])
    assert data["reality"]["candidates"], "real-bytes envelopes must be non-empty"
    assert data["reality"]["measured_gib"] > 0
    assert data["ecosystem"]["eval_host"] == 8100


def test_render_dashboard_is_self_contained_html():
    data = build_dashboard_data(seed=0)
    html = render_dashboard(data)
    assert html.startswith("<!doctype html>")
    assert "Atlas Lab" in html
    assert html.rstrip().endswith("</html>")
    # embedded JSON payload parses
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";", start)
    embedded = json.loads(html[start:end])
    assert embedded["meta"]["arch"] == "k3-mini"


def test_write_dashboard_writes_file(tmp_path):
    write_dashboard(str(tmp_path / "dash.html"), seed=0)
    assert tmp_path.joinpath("dash.html").read_text().startswith("<!doctype html>")
