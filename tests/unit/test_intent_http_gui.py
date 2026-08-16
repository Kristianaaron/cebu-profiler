from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from model_atlas.cli import app
from model_atlas.recommend import RecommendationService
from model_atlas.recommend.gui import _GUI_PAGE, render_gui
from model_atlas.recommend.policy import CompressionIntent


def test_recommend_cli_strategy_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_authorize(
        _service: RecommendationService,
        profile: object,
        target: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        seen["profile"] = profile
        seen["intent"] = kwargs["intent"]
        return {
            "token": "token",
            "recommendation_id": "rec-test",
            "profile_id": "profile-test",
            "no_pruning": True,
            "intent": CompressionIntent.PRUNE_ONLY.value,
            "authorized_methods": [],
            "selection_hash": "selection",
            "recommendation": {"intent": CompressionIntent.PRUNE_ONLY.value},
        }

    monkeypatch.setattr(RecommendationService, "authorize", fake_authorize)

    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "--profile",
            "profile-test",
            "--profiles-dir",
            str(tmp_path / "profiles"),
            "--strategy",
            "prune_only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"intent": "prune_only"' in result.output
    assert seen == {
        "profile": "profile-test",
        "intent": CompressionIntent.PRUNE_ONLY,
    }


def test_gui_renders_strategy_families_and_blocker_detail() -> None:
    for strategy in ("quantize_only", "prune_only", "hybrid", "custom"):
        assert f'value="{strategy}"' in _GUI_PAGE
    assert "required_families" in _GUI_PAGE
    assert "available_families" in _GUI_PAGE
    assert "missing_families" in _GUI_PAGE
    assert "intent_satisfied" in _GUI_PAGE
    assert "x.message || x.code" in _GUI_PAGE
    assert "x.stage_id ? ' [stage '" in _GUI_PAGE
    assert "intent_blockers" in _GUI_PAGE
    assert "intent: intent" in _GUI_PAGE


def test_gui_strategy_changes_invalidate_authorization_and_gate_compress() -> None:
    page = render_gui()

    assert 'id="allowPrune" disabled' in page
    assert "pruningCapable" in page
    assert "clearBinding('strategy changed')" in page
    assert "clearBinding('allow_pruning changed')" in page
    assert "preview.readiness.intent_satisfied" in page
    assert "preview compiled effects do not satisfy strategy" in page
    assert "btn.disabled = !gate.ready" in page
