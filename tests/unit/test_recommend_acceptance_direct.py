from __future__ import annotations

import time
from pathlib import Path

import pytest

from model_atlas.recommend import RecommendationService
from model_atlas.recommend.api import AuthError
from model_atlas.recommend.policy import RecTarget
from tests.unit.test_recommend_auth_close import _analysis_profile, _seed_executable


def _service(tmp_path: Path, **kwargs: object) -> RecommendationService:
    return RecommendationService(
        profile_root=tmp_path / "profiles",
        work_root=tmp_path / "runs",
        **kwargs,
    )


def test_configured_token_ttl_is_enforced(tmp_path: Path) -> None:
    svc = _service(tmp_path, token_ttl_seconds=0.01)
    svc.save_profile(_analysis_profile())
    auth = svc.authorize("k3-mini", RecTarget(memory_target_gib=115.0))
    time.sleep(0.03)
    with pytest.raises(AuthError, match="expired") as exc:
        svc.preview_selection(auth["token"], auth["authorized_methods"][:1])
    assert exc.value.code == "token_expired"
    svc.shutdown()


def test_preview_digest_binds_inputs_and_persists_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from model_atlas.recommend import api as api_mod

    svc = _service(tmp_path)
    svc.save_profile(_analysis_profile())
    auth = svc.authorize("k3-mini", RecTarget(memory_target_gib=115.0))
    selected = auth["authorized_methods"][:1]
    real_write = api_mod.atomic_write_json
    calls = 0

    def fail_first(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("preview disk unavailable")
        real_write(*args, **kwargs)

    monkeypatch.setattr(api_mod, "atomic_write_json", fail_first)
    with pytest.raises(OSError, match="preview disk unavailable"):
        svc.preview_selection(auth["token"], selected, inputs={"source": "a"})
    assert svc.pending_previews == {}
    assert not list((svc.store_root / "previews").glob("pv-*"))

    monkeypatch.setattr(api_mod, "atomic_write_json", real_write)
    first = svc.preview_selection(auth["token"], selected, inputs={"source": "a"})
    second = svc.preview_selection(auth["token"], selected, inputs={"source": "b"})
    assert first["preview_id"] != second["preview_id"]
    assert first["hash"] != second["hash"]
    svc.shutdown()


def test_registry_failure_preserves_preexisting_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _service(tmp_path)
    digest, package = _seed_executable(svc, tmp_path)
    assert svc._persist_run(package, package.run_id) is True
    run_dir = svc.plane.work_root / "runs" / package.run_id
    job_before = (run_dir / "job.json").read_bytes()
    plan_before = (run_dir / "plan.json").read_bytes()
    marker = run_dir / "operator-marker"
    marker.write_text("preserve", encoding="utf-8")

    def fail_registry() -> None:
        raise OSError("registry disk unavailable")

    monkeypatch.setattr(svc, "_save_dispatch_registry", fail_registry)
    with pytest.raises(OSError, match="registry disk unavailable"):
        svc.start_authorized(
            package.token,
            package.preview_id,
            digest,
            package.selected,
            plan_id=package.plan_id,
            recipe_sha256=package.recipe_sha256,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert (run_dir / "job.json").read_bytes() == job_before
    assert (run_dir / "plan.json").read_bytes() == plan_before
    assert package.run_id not in svc.dispatched
    svc.shutdown()


def test_executor_is_lazy_and_start_handles_are_mandatory(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    assert svc._worker is None
    digest, package = _seed_executable(svc, tmp_path)
    with pytest.raises(AuthError) as missing_plan:
        svc.start_authorized(
            package.token,
            package.preview_id,
            digest,
            package.selected,
            recipe_sha256=package.recipe_sha256,
        )
    assert missing_plan.value.code == "plan_required"
    with pytest.raises(AuthError) as missing_recipe:
        svc.start_authorized(
            package.token,
            package.preview_id,
            digest,
            package.selected,
            plan_id=package.plan_id,
        )
    assert missing_recipe.value.code == "recipe_required"
    assert svc._worker is None
    svc.shutdown()
