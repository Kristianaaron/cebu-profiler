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
from urllib.parse import urlencode

import pytest

from model_atlas.recommend import RecommendationService
from model_atlas.recommend.api import _profile_to_dict
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
    from model_atlas.recipes.builtin import glm52_no_pruning_recipe

    recipe = glm52_no_pruning_recipe()
    status, data = _post(
        server, "/api/start", {"recipe": recipe.model_dump(mode="json"), "inputs": {}}
    )
    # unavailable backends -> compile/verify error, never a 2xx start
    assert status in (400, 404)
    assert "error" in _json((status, data))


def test_start_invalid_recipe_schema_400(server):
    status, _ = _post(server, "/api/start", {"recipe": {"name": "no-fields"}, "inputs": {}})
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


def test_job_lineage(server: RecommendationServer, service, monkeypatch):
    from model_atlas.recipes.builtin import glm52_no_pruning_recipe

    recipe = glm52_no_pruning_recipe()
    monkeypatch.setattr(
        service, "job_lineage", lambda r: {"recipe_id": "recipe-x", "lineage": True}
    )
    q = urlencode({"recipe": recipe.model_dump_json()})
    status, data = _request(server, "GET", f"/lineage?{q}")
    assert status == 200
    assert _json((status, data))["lineage"] is True


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
