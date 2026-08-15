"""Local HTTP server for the Atlas recommendation workflow.

A dependency-light (stdlib-only) JSON API over :class:`RecommendationService`
used by agents and local tooling. It deliberately exposes only safe,
deterministic operations — recommendations, previews, and read-only job/lineage
lookups — and never performs mutation outside a verified start.

Security posture:

* Default binds loopback only; a non-loopback host requires the explicit
  ``unsafe_allow_non_loopback`` flag.
* Request bodies are capped at 1 MiB (``413``).
* Profile import is server-side only: the given path MUST resolve under the
  configured ``profile_root`` (``403`` otherwise). ``..`` traversal is refused.
* Job start fails closed: the recipe + inputs are compiled through
  :meth:`RecommendationService.start` and any compile/verify error surfaces as a
  ``4xx`` (never a partial start).

All responses are JSON; errors use ``{"error": …}``.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus, urlparse

from model_atlas.recipe.schema import CompressionRecipe
from model_atlas.recommend.api import RecommendationService
from model_atlas.recommend.policy import RecTarget

# --------------------------------------------------------------------------- config
MAX_BODY_BYTES = 1 << 20  # 1 MiB
_OK = 200
_BAD_REQUEST = 400
_FORBIDDEN = 403
_NOT_FOUND = 404
_TOO_LARGE = 413

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


class ServerError(Exception):
    """Raised to produce an ``error`` JSON response with the given status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- helpers
_HOST_ALLOWED = {"127.0.0.1", "::1", "localhost"}


def _require_loopback(host: str, unsafe: bool) -> None:
    """Reject non-loopback binds unless the caller explicitly opts into unsafe."""
    if unsafe or host in _HOST_ALLOWED:
        return
    raise ServerError(
        _FORBIDDEN, f"refusing to bind host {host!r}; pass unsafe_allow_non_loopback to allow"
    )


def _resolve_under_root(root: Path, path: str) -> Path:
    """Resolve ``path`` and reject anything escaping ``root`` (incl. symlinks)."""
    root = root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ServerError(_FORBIDDEN, f"path {path!r} escapes profile_root") from exc
    return candidate


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except (TypeError, ValueError) as exc:
        raise ServerError(_BAD_REQUEST, "invalid Content-Length") from exc
    if length <= 0:
        raise ServerError(_BAD_REQUEST, "missing request body")
    if length > MAX_BODY_BYTES:
        raise ServerError(_TOO_LARGE, f"request body exceeds {MAX_BODY_BYTES} bytes")
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServerError(_BAD_REQUEST, f"invalid JSON body: {exc}") from exc
    if not isinstance(data, dict):
        raise ServerError(_BAD_REQUEST, "JSON body must be an object")
    return data


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


# --------------------------------------------------------------------------- request handler
class RecommendationHandler(BaseHTTPRequestHandler):
    """Serve the recommendation HTTP API bound to one service instance."""

    service: RecommendationService  # set by the factory

    # stdlib's default logging is noisy; silence per-request stderr.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    # ------------------------------------------------------------ dispatch
    def _dispatch_get(self, path: str, query: dict[str, str]) -> None:
        self._route_get(path, query)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            self._dispatch_get(parsed.path, self._query(parsed.query))
        except ServerError as exc:
            _send_json(self, exc.status, {"error": exc.message})
        except KeyError as exc:
            _send_json(self, _NOT_FOUND, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — fail closed on unexpected errors
            _send_json(self, _BAD_REQUEST, {"error": f"internal error: {exc}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            self._route_post(parsed.path, _read_json_body(self))
        except ServerError as exc:
            _send_json(self, exc.status, {"error": exc.message})
        except KeyError as exc:
            _send_json(self, _NOT_FOUND, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — fail closed
            _send_json(self, _BAD_REQUEST, {"error": f"internal error: {exc}"})

    @staticmethod
    def _query(raw: str) -> dict[str, str]:
        out: dict[str, str] = {}
        if not raw:
            return out
        for part in raw.split("&"):
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            out[unquote_plus(k)] = unquote_plus(v)
        return out

    # ------------------------------------------------------------- GET routes
    def _route_get(self, path: str, query: dict[str, str]) -> None:
        svc = self.service
        if path in {"/", "/index.html", "/gui"}:
            self._serve_gui_page()
        elif path == "/api/profiles":
            _send_json(self, _OK, {"profiles": svc.list_profiles()})
        elif path.startswith("/api/jobs/") and path.endswith("/events"):
            run_id = path[len("/api/jobs/") : -len("/events")]
            _send_json(self, _OK, {"run_id": run_id, "events": svc.job_events(run_id)})
        elif path.startswith("/api/jobs/"):
            run_id = path[len("/api/jobs/") :]
            if not run_id:
                raise ServerError(_NOT_FOUND, "missing run_id")
            _send_json(self, _OK, svc.job_status(run_id))
        elif path == "/lineage":
            recipe = self._recipe_from_json(query.get("recipe") or "{}")
            _send_json(self, _OK, svc.job_lineage(recipe))
        elif path == "/outputs":
            run_id = query.get("run_id", "")
            stage_id = query.get("stage_id")
            name = query.get("name")
            if not run_id:
                raise ServerError(_BAD_REQUEST, "outputs requires run_id")
            result = svc.job_output(run_id, stage_id=stage_id, name=name)
            if name is not None:
                blob = result if isinstance(result, bytes) else b""
                self.send_response(_OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                return
            _send_json(self, _OK, result)
        elif path.startswith("/validate"):
            run_id = query.get("run_id", "")
            stage_id = query.get("stage", "")
            if not run_id or not stage_id:
                raise ServerError(_BAD_REQUEST, "validate requires run_id and stage")
            _send_json(self, _OK, svc.job_validate(run_id, stage_id))
        else:
            raise ServerError(_NOT_FOUND, f"no such route: {path}")

    def _serve_gui_page(self) -> None:
        from model_atlas.recommend.gui import render_gui

        body = render_gui().encode("utf-8")
        self.send_response(_OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------ POST routes
    def _route_post(self, path: str, body: dict[str, Any]) -> None:
        svc = self.service
        if path == "/api/profiles/import":
            self._post_import(svc, body)
        elif path == "/api/recommend":
            self._post_recommend(svc, body)
        elif path == "/api/preview":
            recipe = self._recipe_from_body(body, "recipe")
            _send_json(self, _OK, svc.preview_recipe(recipe))
        elif path == "/api/preview-selection":
            raw = body.get("selected")
            selected: list[str] | None = None
            if raw is not None:
                if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                    raise ServerError(_BAD_REQUEST, "'selected' must be a list of method strings")
                selected = raw
            _send_json(self, _OK, svc.recipe_preview(selected))
        elif path == "/api/start":
            if "selected" in body and "recipe" not in body:
                raw = body.get("selected")
                if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                    raise ServerError(_BAD_REQUEST, "'selected' must be a list of method strings")
                inputs = body.get("inputs", {})
                if not isinstance(inputs, dict):
                    raise ServerError(_BAD_REQUEST, "'inputs' must be an object")
                # Rebuild the deterministic editable draft server-side — the
                # browser never holds an embedded recipe (XSS/no payload).
                recipe = svc._selected_recipe(raw)
                try:
                    engine = svc.start(recipe, inputs=inputs)
                except KeyError as exc:
                    raise ServerError(_NOT_FOUND, f"start failed: {exc}") from exc
                except Exception as exc:  # noqa: BLE001 — fail closed
                    raise ServerError(_BAD_REQUEST, f"start refused: {exc}") from exc
                _send_json(self, _OK, {"run_id": engine.run_dir.name, "status": "started"})
                return
            recipe = self._recipe_from_body(body, "recipe")
            inputs = body.get("inputs", {})
            if not isinstance(inputs, dict):
                raise ServerError(_BAD_REQUEST, "'inputs' must be an object")
            try:
                engine = svc.start(recipe, inputs=inputs)
            except KeyError as exc:
                raise ServerError(_NOT_FOUND, f"start failed: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 — fail closed
                raise ServerError(_BAD_REQUEST, f"start refused: {exc}") from exc
            _send_json(self, _OK, {"run_id": engine.run_dir.name, "status": "started"})
        else:
            raise ServerError(_NOT_FOUND, f"no such route: {path}")

    def _post_import(self, svc: RecommendationService, body: dict[str, Any]) -> None:
        profile_path = body.get("path")
        if not isinstance(profile_path, str) or not profile_path:
            raise ServerError(_BAD_REQUEST, "import requires a server-side 'path' string")
        candidate = _resolve_under_root(svc.profile_root, profile_path)
        if not candidate.is_file():
            raise ServerError(_NOT_FOUND, f"profile file not found: {profile_path}")
        profile = svc.import_profile(candidate)
        _send_json(
            self,
            _OK,
            {
                "profile_id": profile.profile_id_of(),
                "path": str(candidate),
                "imported": True,
            },
        )

    def _post_recommend(self, svc: RecommendationService, body: dict[str, Any]) -> None:
        profile_id = body.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ServerError(_BAD_REQUEST, "recommend requires 'profile_id'")
        raw_mem = body.get("memory_target_gib")
        default_mem = RecTarget().memory_target_gib
        try:
            memory_target_gib = float(raw_mem) if raw_mem is not None else default_mem
        except (TypeError, ValueError):
            memory_target_gib = default_mem
        raw_constraints = body.get("constraints")
        constraints: dict[str, Any] = raw_constraints if isinstance(raw_constraints, dict) else {}
        allow_pruning = bool(constraints.get("allow_pruning", False))
        rec = svc.recommend(
            profile_id,
            RecTarget(memory_target_gib=memory_target_gib),
            allow_pruning=allow_pruning,
        )
        payload = rec.to_dict() if hasattr(rec, "to_dict") else rec
        _send_json(self, _OK, {"recommendation": payload})

    @staticmethod
    def _recipe_from_json(payload: str) -> CompressionRecipe:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ServerError(_BAD_REQUEST, f"invalid recipe JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ServerError(_BAD_REQUEST, "recipe must be a JSON object")
        return RecommendationHandler._recipe_from_body({"recipe": data}, "recipe")

    @staticmethod
    def _recipe_from_body(body: dict[str, Any], key: str) -> CompressionRecipe:
        payload = body.get(key)
        if not isinstance(payload, dict):
            raise ServerError(_BAD_REQUEST, f"'{key}' must be a JSON object")
        try:
            return CompressionRecipe.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — schema/validation errors
            raise ServerError(_BAD_REQUEST, f"invalid {key}: {exc}") from exc


# --------------------------------------------------------------------------- server factory
class RecommendationServer(ThreadingHTTPServer):
    """Threaded JSON server exposing the recommendation API."""

    daemon_threads = True


def start_server(
    service: RecommendationService,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    unsafe_allow_non_loopback: bool = False,
) -> RecommendationServer:
    """Start the threaded HTTP server bound to ``service``.

    Non-loopback binds are refused unless ``unsafe_allow_non_loopback`` is set.
    """
    _require_loopback(host, unsafe_allow_non_loopback)

    # Bind the service to a fresh subclass: BaseHTTPRequestHandler's __init__
    # runs the ENTIRE request lifecycle, so the attribute must exist before
    # construction, not be assigned on the returned handler.
    handler_class = type(
        "BoundRecommendationHandler",
        (RecommendationHandler,),
        {"service": service},
    )

    server = RecommendationServer((host, port), handler_class)
    server.daemon_threads = True
    return server


def serve_forever(
    service: RecommendationService,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    unsafe_allow_non_loopback: bool = False,
) -> None:
    """Serve the recommendation API until interrupted."""
    server = start_server(
        service,
        host=host,
        port=port,
        unsafe_allow_non_loopback=unsafe_allow_non_loopback,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
