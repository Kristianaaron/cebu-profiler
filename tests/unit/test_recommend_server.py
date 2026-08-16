"""Recommendation HTTP server tests.

Covers the stdlib ThreadingHTTPServer endpoint surface against an ephemeral
port: profiles / recommend round-trips, oversized-body 413, path-traversal 403
on import, fail-closed start 4xx, and read-only job/validate/lineage/outputs
endpoints (job reads via monkeypatched service so no real run is needed).
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from model_atlas.recommend import RecommendationService
from model_atlas.recommend.api import _profile_to_dict
from model_atlas.recommend.policy import RecTarget
from model_atlas.recommend.server import (
    MAX_BODY_BYTES,
    RecommendationServer,
    _require_loopback,
    start_server,
)
from tests.unit.test_recommend import _full_profile


@pytest.fixture
def service(tmp_path: Path) -> RecommendationService:
    return RecommendationService(
        profile_root=str(tmp_path / "profiles"), work_root=str(tmp_path / "runs")
    )


@pytest.fixture
def server(service: RecommendationService) -> RecommendationServer:
    srv = start_server(service, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _conn(server: RecommendationServer) -> HTTPConnection:
    host, port = server.server_address[:2]
    return HTTPConnection(host, port)


def _request(server: RecommendationServer, method: str, path: str, body=None, headers=None):
    conn = _conn(server)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def _json(resp) -> dict:
    status, data = resp
    decoded = json.loads(data.decode("utf-8")) if data else {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _post(server, path, payload):
    return _request(
        server, "POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"}
    )


def _authorize(server, service, profile_id="k3-mini"):
    """recommend -> token (the browser/GUI flow). Returns the auth payload."""
    service.save_profile(_full_profile())
    status, data = _post(
        server,
        "/api/recommend",
        {
            "profile_id": profile_id,
            "memory_target_gib": 115.0,
            "constraints": {"allow_pruning": False},
        },
    )
    assert status == 200
    return _json((status, data))


# --------------------------------------------------------------------------- profiles / recommend
def test_profiles_and_recommend_round_trip(server: RecommendationServer, service):
    service.save_profile(_full_profile())
    status, data = _request(server, "GET", "/api/profiles")
    assert status == 200
    profiles = _json((status, data))["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["profile_id"] == _full_profile().profile_id_of()

    status, data = _post(
        server,
        "/api/recommend",
        {"profile_id": "k3-mini", "memory_target_gib": 115.0, "constraints": {}},
    )
    assert status == 200
    rec = _json((status, data))["recommendation"]
    assert rec["recommendation_id"].startswith("rec-")
    assert rec["confidence"] in {"high", "medium", "low", "insufficient"}


def test_recommend_requires_profile_id(server):
    status, _ = _post(server, "/api/recommend", {})
    assert status == 400


def test_recommend_unknown_profile_404(server):
    status, data = _post(server, "/api/recommend", {"profile_id": "nope"})
    assert status == 404


def test_import_profile(server: RecommendationServer, service, tmp_path):
    src = service.profile_root / "ext.json"
    src.write_text(json.dumps(_profile_to_dict(_full_profile())))
    status, data = _post(server, "/api/profiles/import", {"path": "ext.json"})
    assert status == 200
    assert _json((status, data))["imported"] is True


# path traversal + body limits
def test_import_path_traversal_blocked(server: RecommendationServer, service, tmp_path):
    # a file OUTSIDE profile_root
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text(json.dumps(_profile_to_dict(_full_profile())))
    status, data = _post(server, "/api/profiles/import", {"path": "../outside/secret.json"})
    assert status == 403
    assert "error" in _json((status, data))


def test_import_absolute_path_escapes_blocked(server: RecommendationServer, service, tmp_path):
    outside = tmp_path / "outside2"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text(json.dumps(_profile_to_dict(_full_profile())))
    status, _ = _post(server, "/api/profiles/import", {"path": str(secret)})
    assert status == 403


def test_oversized_body_413(server: RecommendationServer):
    big = {"path": "x" * (MAX_BODY_BYTES + 1)}
    status, data = _request(
        server,
        "POST",
        "/api/profiles/import",
        body=json.dumps(big),
        headers={"Content-Type": "application/json"},
    )
    assert status == 413


# --------------------------------------------------------------------------- start fails closed
def test_start_blocked_fail_closed(server: RecommendationServer):
    # an arbitrary full-recipe start is REMOVED: /api/start only accepts the
    # token/preview/hash/selection binding. A recipe-only body is rejected.
    from model_atlas.recipes.builtin import glm52_no_pruning_recipe

    recipe = glm52_no_pruning_recipe()
    status, data = _post(
        server, "/api/start", {"recipe": recipe.model_dump(mode="json"), "inputs": {}}
    )
    assert status == 400  # 'token' required — never an arbitrary start
    assert "error" in _json((status, data))


def test_start_invalid_recipe_schema_400(server):
    status, _ = _post(server, "/api/start", {"recipe": {"name": "no-fields"}, "inputs": {}})
    assert status == 400


# ---------------------------------------------------------------------------- static GUI page
def test_gui_page_served_no_embedded_payload(server: RecommendationServer, service):
    service.save_profile(_full_profile())
    for path in ("/", "/index.html", "/gui"):
        status, data = _request(server, "GET", path)
        assert status == 200
        text = data.decode("utf-8")
        assert "<script>" in text
        # static only: no embedded profile/recommendation JSON
        assert "rec-" not in text
        assert "k3-mini" not in text
        assert "innerHTML" not in text
        assert "fetch('/api/profiles')" in text


# --- preview-from-selection (token-gated) ---
def test_preview_selection_endpoint(server: RecommendationServer, service):
    from model_atlas.recommend.api import _selection_hash

    auth = _authorize(server, service)
    token = auth["token"]
    authorized = auth["authorized_methods"]
    # NONEMPTY PROPER SUBSET is allowed, and the returned hash is the subset's
    # own full authorization digest — not the selection-only hash.
    subset = authorized[:2]
    subset_hash = _selection_hash(subset)
    assert subset_hash != auth["selection_hash"]  # the subsets genuinely differ
    status, data = _post(
        server, "/api/preview-selection", {"token": token, "selected": subset}
    )
    assert status == 200
    body = _json((status, data))
    assert body["preview_id"].startswith("pv-")
    assert body["hash"] != subset_hash
    assert len(body["hash"]) == 64
    assert body["plan_id"] is None  # placeholder adapters -> no verified plan
    assert body["readiness"]["verified_plan"] is False
    assert body["selected_methods"] == subset


def test_preview_selection_requires_token(server: RecommendationServer, service):
    _authorize(server, service)
    status, data = _post(
        server, "/api/preview-selection", {"selected": ["calibration", "sensitivity"]}
    )
    assert status == 400  # missing token


def test_preview_selection_rejects_not_authorized(server: RecommendationServer, service):
    auth = _authorize(server, service)
    # a method OUTSIDE the authorized set — a subset is fine but this is not
    # authorized at all.
    status, data = _post(
        server,
        "/api/preview-selection",
        {"token": auth["token"], "selected": ["calibration", "exl3-primary"]},
    )
    assert status == 403
    assert _json((status, data))["code"] == "selection_not_authorized"


def test_preview_selection_rejects_unknown_token(server: RecommendationServer, service):
    _authorize(server, service)
    status, data = _post(
        server, "/api/preview-selection", {"token": "bogus", "selected": ["calibration"]}
    )
    assert status == 401
    assert _json((status, data))["code"] == "token_unknown"


def test_preview_selection_full_set_allowed(server: RecommendationServer, service):
    """The FULL authorized set is a valid (non-empty, all-authorized) subset and
    still previews — its authorization digest is stronger than selection-only."""

    auth = _authorize(server, service)
    status, data = _post(
        server,
        "/api/preview-selection",
        {"token": auth["token"], "selected": auth["authorized_methods"]},
    )
    assert status == 200
    body = _json((status, data))
    assert body["hash"] != auth["selection_hash"]
    assert len(body["hash"]) == 64
    assert body["selected_methods"] == auth["authorized_methods"]


def test_preview_selection_rejects_empty(server: RecommendationServer, service):
    auth = _authorize(server, service)
    status, data = _post(server, "/api/preview-selection", {"token": auth["token"], "selected": []})
    assert status == 400
    assert _json((status, data))["code"] == "selection_empty"


def test_preview_selection_requires_list_of_strings(server: RecommendationServer):
    status, _ = _post(server, "/api/preview-selection", {"selected": "not-a-list"})
    assert status == 400


def test_preview_selection_omitted_all_requires_token(server: RecommendationServer, service):
    service.save_profile(_full_profile())
    status, data = _post(server, "/api/preview-selection", {})
    assert status == 400  # no token, no selected


# --- start (token + preview bound, fail closed) ---
def test_start_requires_full_token_binding(server: RecommendationServer, service):
    """start accepts ONLY token+preview_id+hash+exact selection+inputs. Any
    missing handle is a 4xx — no arbitrary-recipe or raw-selection start."""
    _authorize(server, service)
    for payload in (
        {"selected": ["calibration"], "inputs": {}},  # no token
        {"token": "x", "selected": ["calibration"], "inputs": {}},  # no preview
    ):
        status, data = _post(server, "/api/start", payload)
        assert status in (400, 401, 404)
        assert "error" in _json((status, data))
    # arbitrary full-recipe start is REMOVED
    from model_atlas.recipes.builtin import glm52_no_pruning_recipe

    status, _ = _post(
        server, "/api/start", {"recipe": glm52_no_pruning_recipe().model_dump(mode="json")}
    )
    assert status == 400  # 'token' required


def test_start_rejects_mismatched_hash(server: RecommendationServer, service):
    auth = _authorize(server, service)
    sel = auth["authorized_methods"]
    preview = _json(_post(
        server, "/api/preview-selection", {"token": auth["token"], "selected": sel}
    ))
    status, data = _post(
        server,
        "/api/start",
        {"token": auth["token"], "preview_id": preview["preview_id"],
         "hash": "wrong", "plan_id": "wrong-plan", "recipe_sha256": "wrong-recipe",
         "selected": sel, "inputs": {}},
    )
    assert status == 409
    assert _json((status, data))["code"] == "preview_mismatch"


def test_start_rejects_unknown_preview(server: RecommendationServer, service):
    auth = _authorize(server, service)
    sel = auth["authorized_methods"]
    status, data = _post(
        server,
        "/api/start",
        {"token": auth["token"], "preview_id": "pv-never",
         "hash": auth["selection_hash"], "plan_id": "none", "recipe_sha256": "none",
         "selected": sel, "inputs": {}},
    )
    assert status == 410
    assert _json((status, data))["code"] == "preview_unknown"


def test_start_rejects_empty_selection(server: RecommendationServer, service):
    auth = _authorize(server, service)
    sel = auth["authorized_methods"]
    preview = _json(_post(
        server, "/api/preview-selection", {"token": auth["token"], "selected": sel}
    ))
    status, data = _post(
        server,
        "/api/start",
        {"token": auth["token"], "preview_id": preview["preview_id"],
         "hash": preview["hash"], "plan_id": "none", "recipe_sha256": "none",
         "selected": [], "inputs": {}},
    )
    assert status == 400
    assert _json((status, data))["code"] == "selection_empty"


def test_start_rejects_unknown_token(server: RecommendationServer, service):
    _authorize(server, service)
    status, data = _post(
        server,
        "/api/start",
        {"token": "bogus", "preview_id": "pv-x", "hash": "h",
         "plan_id": "none", "recipe_sha256": "none", "selected": ["a"], "inputs": {}},
    )
    assert status == 401
    assert _json((status, data))["code"] == "token_unknown"


def test_start_rejects_not_executable_preview(server: RecommendationServer, service):
    """Placeholder adapters -> preview stored non-executable -> start refused
    (never fakes quantization)."""
    auth = _authorize(server, service)
    sel = auth["authorized_methods"]
    preview = _json(_post(
        server, "/api/preview-selection", {"token": auth["token"], "selected": sel}
    ))
    assert preview["readiness"]["executable"] is False
    status, data = _post(
        server,
        "/api/start",
        {"token": auth["token"], "preview_id": preview["preview_id"],
         "hash": preview["hash"], "plan_id": "none",
         "recipe_sha256": preview["recipe_sha256"], "selected": sel, "inputs": {}},
    )
    assert status == 409
    assert _json((status, data))["code"] == "preview_not_executable"


# --- browser-like HTTP sequence ---
def test_browser_http_sequence(server: RecommendationServer, service):
    """A browser session: serve GUI -> list profiles -> recommend (token) ->
    preview selection (token) -> start (token+preview bound, fail closed on
    placeholder adapters). This is the exact HTTP sequence the static GUI
    drives."""
    service.save_profile(_full_profile())

    # 1. static GUI (no embedded data)
    status, data = _request(server, "GET", "/")
    assert status == 200
    assert "innerHTML" not in data.decode("utf-8")

    # 2. /api/profiles
    status, data = _request(server, "GET", "/api/profiles")
    assert status == 200
    profiles = _json((status, data))["profiles"]
    assert profiles and profiles[0]["profile_id"]

    # 3. /api/recommend -> token
    status, data = _post(
        server,
        "/api/recommend",
        {
            "profile_id": profiles[0]["profile_id"],
            "memory_target_gib": 115.0,
            "constraints": {"allow_pruning": False},
        },
    )
    assert status == 200
    auth = _json((status, data))
    assert auth["token"]
    blocked = auth["recommendation"]["blocked_methods"]
    assert blocked  # placeholder adapters are fatally blocked
    authorized = auth["authorized_methods"]
    assert authorized

    # 4. preview the authorized set (token-bound)
    status, data = _post(
        server, "/api/preview-selection", {"token": auth["token"], "selected": authorized}
    )
    assert status == 200
    preview = _json((status, data))
    assert preview["preview_id"].startswith("pv-")
    assert preview["hash"] != auth["selection_hash"]
    assert preview["readiness"]["verified_plan"] is False

    # 5. start without immutable plan handles fails closed at the HTTP boundary
    status, data = _post(
        server,
        "/api/start",
        {"token": auth["token"], "preview_id": preview["preview_id"],
         "hash": preview["hash"], "selected": authorized, "inputs": {}},
    )
    assert status == 400


# job read endpoints (fixture/monkeypatch)
def test_job_status_and_events(server: RecommendationServer, service, monkeypatch):
    monkeypatch.setattr(
        service, "job_status", lambda run_id: {"run_id": run_id, "status": "DONE", "stages": {}}
    )
    monkeypatch.setattr(service, "job_events", lambda run_id: [{"event": "stage.done"}])

    status, data = _request(server, "GET", "/api/jobs/abc123")
    assert status == 200
    assert _json((status, data))["status"] == "DONE"

    status, data = _request(server, "GET", "/api/jobs/abc123/events")
    assert status == 200
    evs = _json((status, data))["events"]
    assert evs == [{"event": "stage.done"}]


def test_job_validate(server: RecommendationServer, service, monkeypatch):
    monkeypatch.setattr(
        service,
        "job_validate",
        lambda run_id, stage: {"run_id": run_id, "stage": stage, "validated": True},
    )
    status, data = _request(server, "GET", "/validate?run_id=r1&stage=s1")
    assert status == 200
    body = _json((status, data))
    assert body["run_id"] == "r1"
    assert body["stage"] == "s1"


def test_lineage_requires_run_id(server):
    """recipe={} lineage is REMOVED — /lineage now requires an actual run_id."""
    status, data = _request(server, "GET", "/lineage?recipe={}")
    assert status == 400
    status, _ = _request(server, "GET", "/lineage")
    assert status == 400


def test_job_lineage_run_id_bound(server: RecommendationServer, service, monkeypatch):
    """/lineage?run_id=… delegates to svc.run_lineage for an actual run."""
    monkeypatch.setattr(
        service,
        "run_lineage",
        lambda run_id: {"run_id": run_id, "plan_id": "plan-x", "lineage": True},
    )
    status, data = _request(server, "GET", "/lineage?run_id=abc")
    assert status == 200
    body = _json((status, data))
    assert body["run_id"] == "abc"
    assert body["lineage"] is True

    # unknown run -> run_lineage raises (fail closed)
    monkeypatch.setattr(
        service, "run_lineage", lambda run_id: (_ for _ in ()).throw(KeyError("no"))
    )
    status, data = _request(server, "GET", "/lineage?run_id=nope")
    assert status == 404


def test_outputs_list(server: RecommendationServer, service, monkeypatch):
    refs = {
        "run_id": "abc",
        "outputs": [
            {
                "stage": "s1",
                "name": "plan.json",
                "sha256": "a" * 64,
                "size_bytes": 3,
                "format": "",
                "relpath": "",
            }
        ],
    }
    monkeypatch.setattr(service, "job_output", lambda run_id, stage_id=None, name=None: refs)
    status, data = _request(server, "GET", "/outputs?run_id=abc")
    assert status == 200
    assert _json((status, data))["run_id"] == "abc"


def test_outputs_by_name_binary(server: RecommendationServer, service, monkeypatch):
    monkeypatch.setattr(service, "job_output", lambda run_id, stage_id=None, name=None: b"BLOB")
    status, data = _request(server, "GET", "/outputs?run_id=abc&name=plan.json")
    assert status == 200
    assert data == b"BLOB"


def test_outputs_requires_run_id(server):
    status, _ = _request(server, "GET", "/outputs")
    assert status == 400


def test_unknown_route_404(server):
    status, _ = _request(server, "GET", "/api/nope")
    assert status == 404


# --------------------------------------------------------------------------- host binding guard
def test_reject_non_loopback_host():
    from model_atlas.recommend.server import ServerError

    with pytest.raises(ServerError):
        _require_loopback("0.0.0.0", unsafe=False)


def test_allow_non_loopback_with_unsafe_flag():
    _require_loopback("0.0.0.0", unsafe=True)


def test_loopback_allowed():
    _require_loopback("127.0.0.1", unsafe=False)


# --- endpoint-level async start via persisted job engine (feasible seeding) ---
def test_start_async_immediate_return_monkeypatched(
    server: RecommendationServer, service, tmp_path
):
    """Endpoint /api/start with a seeded executable preview returns run_id
    immediately (background worker), the job is durable (status via
    /api/jobs/<id>), and a duplicate start is rejected as replay — all through
    the real HTTP server + persisted job engine."""
    from model_atlas.recipe.compiler import RecipeCompiler
    from model_atlas.recipe.schema import (
        CalibrationIdentity,
        CompressionRecipe,
        RecipeConstraints,
        SourceIdentity,
    )
    from model_atlas.recipes import CompiledPlanArtifact
    from model_atlas.recommend.api import (
        _AuthorizationSession,
        _PendingPreview,
        _selection_hash,
    )
    # executable single-stage recipe (in-repo quant probe, pinned immutable-ish)
    src = tmp_path / "src"
    src.mkdir()
    (src / "w.bin").write_bytes(b"stable")
    from model_atlas.jobs.artifacts import source_manifest

    files = {
        k: v
        for k, v in source_manifest(str(src)).get("files", {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }
    from model_atlas.recipe.schema import RecipeStage, StageBackendPin, StageEffectClass

    recipe = CompressionRecipe(
        name="endpoint-exe",
        source=SourceIdentity(source_id="s", checkpoint_path=str(src), sha256=files),
        calibration=CalibrationIdentity(calibration_id="c", corpus_name="corp"),
        constraints=RecipeConstraints(
            no_pruning=True, allow_pruning_capability=False,
            preserve_non_expert_backbone=True, immutable_source=True,
            allow_hybrid_precision=False, max_resident_gib=115.0,
        ),
        stages=[RecipeStage(
            id="s1", name="s1", effect_class=StageEffectClass.PROFILING,
            backend=StageBackendPin(backend_id="atlas_quant_probe", version="1.0.0"),
            produces_format=["manifest.json"], evidence_policy=__import__(
                "model_atlas.schemas.evidence", fromlist=["EvidenceKind"]
            ).EvidenceKind.ESTIMATED,
        )],
    )
    comp = RecipeCompiler(service.registry).compile(recipe)
    artifact = CompiledPlanArtifact.from_compiled(comp, inputs={}, registry=service.registry)
    artifact.verify()
    artifact.verify_pins_against(service.registry)

    tok = "t-ep"
    h = _selection_hash(["m1"])
    service.sessions[tok] = _AuthorizationSession(
        token=tok, recommendation_id="rec-ep", profile_id="p",
        target=RecTarget(), no_pruning=True, constraints_snapshot={},
        authorized_methods=["m1"],
    )
    service.pending_previews["pv-ep"] = _PendingPreview(
        token=tok, preview_id="pv-ep", selection_hash=h, selected=["m1"],
        recipe=recipe, artifact=artifact, inputs={}, run_id=artifact.run_id,
    )

    status, data = _post(
        server, "/api/start",
        {"token": tok, "preview_id": "pv-ep", "hash": h,
         "plan_id": artifact.plan_id, "recipe_sha256": artifact.recipe_sha256,
         "selected": ["m1"], "inputs": {}},
    )
    assert status == 200
    body = _json((status, data))
    assert body["status"] == "started"
    run_id = body["run_id"]
    assert run_id == artifact.run_id

    # durable + observable
    status, data = _request(server, "GET", "/api/jobs/" + run_id)
    assert status == 200
    assert _json((status, data))["run_id"] == run_id

    # replay rejected deterministically
    status, data = _post(
        server, "/api/start",
        {"token": tok, "preview_id": "pv-ep", "hash": h,
         "plan_id": artifact.plan_id, "recipe_sha256": artifact.recipe_sha256,
         "selected": ["m1"], "inputs": {}},
    )
    assert status == 409
    assert _json((status, data))["code"] == "replay"
