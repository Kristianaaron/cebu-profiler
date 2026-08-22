"""Phase 4/G: size/fit plan tests (metadata census based)."""

import json

from model_atlas.experiments.legacy.fitplan import WidthFits, choose_display, fit_plan


def test_fit_plan_returns_widths_and_fit():
    fits = fit_plan("/media/glm52/models/nvidia/GLM-5.2-NVFP4")
    assert 256 in fits
    assert 2048 in fits
    f = fits[256]
    assert isinstance(f, WidthFits)
    assert f.per_rank_gib > 0
    # 2048 (full width) per-rank (234 GiB) exceeds window budget
    assert fits[2048].fits_window is False


def test_choose_display_picks_largest_fit():
    d = choose_display("/media/glm52/models/nvidia/GLM-5.2-NVFP4")
    assert d["recommended_width"] is not None
    assert d["fits"]
    assert "SEPARATE gate" in d["note"]
    assert d["fits"][str(d["recommended_width"])]["fits_window"] is True


def test_fit_plan_json_serializable():
    import json

    from model_atlas.experiments.legacy.fitplan import to_json

    obj = json.loads(to_json("/media/glm52/models/nvidia/GLM-5.2-NVFP4"))
    assert "recommended_width" in obj
    assert "fits" in obj


def test_valid_widths_all_multiple_of_group():
    for w in (64, 128, 256, 512, 1024, 2048):
        assert w % 16 == 0  # NVFP4 group alignment invariant


def test_fit_default_semantics_no_double_reserve():
    """Round-5 #9: DEFAULT_WINDOW_GIB is already usable (no extra headroom
    subtraction); measured physical capacity subtracts headroom once."""
    from model_atlas.experiments.legacy.fitplan import DEFAULT_WINDOW_GIB, OS_HEADROOM_GIB, fit_plan

    # explicit default => budget == DEFAULT_WINDOW_GIB
    fits = fit_plan("/media/glm52/models/nvidia/GLM-5.2-NVFP4")
    # per-rank W=256 must be well under DEFAULT_WINDOW (115)
    assert fits[256].per_rank_gib < DEFAULT_WINDOW_GIB
    # measured physical 115 => usable 110 (=115-5), distinct from default 115
    fits_meas = fit_plan(
        "/media/glm52/models/nvidia/GLM-5.2-NVFP4", measured_per_node_gib=115.0
    )
    # usables differ by exactly OS_HEADROOM
    head = fits[256].window_headroom_bytes - fits_meas[256].window_headroom_bytes
    assert abs(head - OS_HEADROOM_GIB * (1024**3)) < 1024


def test_size_plan_int_bytes_exact():
    """Round-5 #9: SizePlan uses integer bytes; total compares exactly, not via
    GiB round-trip."""
    from model_atlas.checkpoint.source_manifest import load_manifest
    from model_atlas.loader import _build_keep_map, _infer_geometry, plan_exact_sizes

    manifest = load_manifest("/media/glm52/models/nvidia/GLM-5.2-NVFP4")
    with open("/media/glm52/models/nvidia/GLM-5.2-NVFP4/config.json") as f:
        cfg = json.load(f)
    full, n_exp, sl = _infer_geometry(manifest, cfg)
    keep = _build_keep_map(None, 256, full, n_exp, sl)
    sp = plan_exact_sizes(manifest, cfg, keep)
    assert isinstance(sp.total_bytes, int)
    assert isinstance(sp.per_rank_bytes, int)
    assert sp.total_bytes == sp.replicated_bytes + sp.sharded_expert_bytes
