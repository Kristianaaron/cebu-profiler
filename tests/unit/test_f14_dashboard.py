"""F14 tests: dashboard rendering over measured data."""

import json

from cebu_profiler.dashboard import build_dashboard_data, render_dashboard, write_dashboard


def test_build_dashboard_data_has_sections():
    data = build_dashboard_data(seed=0)
    assert data["meta"]["arch"] == "k3-mini"
    assert data["capability"]  # per-label expert rankings
    assert "voxels" in data["capability3d"] and data["capability3d"]["voxels"]
    assert data["capability3d"]["labels"]
    assert data["contrast"]
    assert "voxels" in data["contrast3d"] and data["contrast3d"]["voxels"]
    # each contrast cell exposes its raw success/failure components plus Δ
    vox = data["contrast3d"]["voxels"][0]
    assert {"delta", "pos", "neg"} <= set(vox)
    assert all(
        0.0 <= v["pos"] <= 1.0 and 0.0 <= v["neg"] <= 1.0 for v in data["contrast3d"]["voxels"]
    )
    assert data["coalitions"]
    assert data["paths"]
    assert data["compression"]
    assert data["candidates"]
    assert data["heldout"]
    assert set(data["hierarchy"]["levels"]) == {
        "weights",
        "units",
        "experts",
        "coalitions",
        "pathways",
        "behaviour",
    }
    assert all(data["hierarchy"]["counts"][lv] > 0 for lv in data["hierarchy"]["levels"])
    assert data["reality"]["candidates"], "real-bytes envelopes must be non-empty"
    assert data["reality"]["measured_gib"] > 0
    assert data["ecosystem"]["eval_host"] == 8100
    assert data["kernel_evidence"]["status"] == "unmeasured"
    assert data["kernel_evidence"]["rows"] == []


def test_render_dashboard_is_self_contained_html():
    data = build_dashboard_data(seed=0)
    html = render_dashboard(data)
    assert html.startswith("<!doctype html>")
    assert "Cebu Lab" in html
    assert html.rstrip().endswith("</html>")
    # both 3D canvases must carry an explicit responsive size rule; a bare canvas
    # collapses to its intrinsic 300x150 size and the cells render invisibly.
    assert "canvas#cap3d{width:100%" in html
    assert "canvas#cap3d-contrast{width:100%" in html
    assert 'id="panel-kernels"' in html
    assert "Runtime Kernels" in html
    # embedded JSON payload parses (const DATA = {...}; then JS follows)
    start = html.index("const DATA = ") + len("const DATA = ")
    i = start
    depth = 0
    while i < len(html):
        ch = html[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    embedded = json.loads(html[start : i + 1])
    assert embedded["meta"]["arch"] == "k3-mini"


def test_write_dashboard_writes_file(tmp_path):
    write_dashboard(str(tmp_path / "dash.html"), seed=0)
    assert tmp_path.joinpath("dash.html").read_text().startswith("<!doctype html>")
