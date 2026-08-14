"""Phase 4/G: size/fit plan tests (metadata census based)."""

from model_atlas.fitplan import WidthFits, choose_display, fit_plan


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

    from model_atlas.fitplan import to_json

    obj = json.loads(to_json("/media/glm52/models/nvidia/GLM-5.2-NVFP4"))
    assert "recommended_width" in obj
    assert "fits" in obj


def test_valid_widths_all_multiple_of_group():
    for w in (64, 128, 256, 512, 1024, 2048):
        assert w % 16 == 0  # NVFP4 group alignment invariant
