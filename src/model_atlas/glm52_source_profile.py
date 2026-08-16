"""Resumable source identity and honest GLM-5.2 profile artifacts.

These helpers deliberately do not use the normal JobEngine snapshot function:
that function is correct for small sources, but a several-hundred-GB mounted
checkpoint needs a crash-resumable, bounded-memory walk.  The final manifest is
still exactly the JobEngine manifest shape and its digest is calculated by the
authoritative :func:`source_manifest_digest` helper.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_atlas.jobs.artifacts import source_manifest_digest

_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_SHA256_HEX_LENGTH = 64


class SourceProfileError(RuntimeError):
    """A source identity or profile evidence invariant was not satisfied."""


@dataclass(frozen=True)
class ManifestBuildResult:
    """Result of a completed resumable directory-manifest build."""

    manifest: dict[str, object]
    digest: str
    hashed_files: int
    reused_files: int


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _stat_record(st: os.stat_result) -> dict[str, int]:
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _is_same_stat(expected: Mapping[str, object], actual: os.stat_result) -> bool:
    return expected == _stat_record(actual)


def _safe_source_root(source_path: Path) -> Path:
    try:
        source_lstat = source_path.lstat()
    except FileNotFoundError as exc:
        raise SourceProfileError(f"source does not exist: {source_path}") from exc
    if stat.S_ISLNK(source_lstat.st_mode):
        raise SourceProfileError(f"source root may not be a symlink: {source_path}")
    if not stat.S_ISDIR(source_lstat.st_mode):
        raise SourceProfileError(f"source root must be a directory: {source_path}")
    return source_path.resolve(strict=True)


def _enumerate_regular_files(root: Path) -> dict[str, dict[str, int]]:
    """Enumerate regular files without following symlinks or special nodes."""
    files: dict[str, dict[str, int]] = {}

    def visit(directory: Path, prefix: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SourceProfileError(f"cannot enumerate {directory}: {exc}") from exc
        for entry in entries:
            relative = prefix / entry.name
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SourceProfileError(f"cannot stat source entry {relative}: {exc}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise SourceProfileError(f"source contains forbidden symlink: {relative}")
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(Path(entry.path), relative)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise SourceProfileError(f"source contains non-regular entry: {relative}")
            relative_text = relative.as_posix()
            if relative_text.startswith("../") or relative_text == "..":
                raise SourceProfileError(f"source path escape: {relative_text}")
            files[relative_text] = _stat_record(entry_stat)

    visit(root, Path())
    return files


def _hash_checked_file(path: Path, expected: dict[str, int]) -> str:
    """Hash one fixed regular file using bounded reads and race checks."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise SourceProfileError(f"cannot safely open {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not _is_same_stat(expected, before):
            raise SourceProfileError(f"source changed or became non-regular before hashing: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if not _is_same_stat(expected, after):
            raise SourceProfileError(f"source changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceProfileError(f"invalid manifest checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceProfileError(f"manifest checkpoint must be an object: {path}")
    return value


def _checkpoint_payload(
    *, root: Path, manifest: dict[str, object], complete: bool, digest: str | None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_root": str(root),
        "complete": complete,
        "manifest": manifest,
    }
    if digest is not None:
        payload["manifest_digest"] = digest
    return payload


def build_resumable_source_manifest(
    source_path: str | Path,
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> ManifestBuildResult:
    """Build a resumable, path-bound directory manifest.

    Each successfully hashed file is atomically checkpointed.  A prior digest
    is reused only when that exact relative path still has the same size and
    nanosecond mtime.  Entries for deleted files are removed before work starts.
    """
    root = _safe_source_root(Path(source_path))
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    current_stats = _enumerate_regular_files(root)
    previous = _load_checkpoint(checkpoint)
    previous_root = previous.get("source_root")
    if previous_root is not None and previous_root != str(root):
        raise SourceProfileError("manifest checkpoint belongs to a different source root")
    old_manifest = previous.get("manifest", {})
    if not isinstance(old_manifest, dict):
        raise SourceProfileError("manifest checkpoint has non-object manifest")
    old_files = old_manifest.get("files", {})
    old_stats = old_manifest.get("file_stats", {})
    if not isinstance(old_files, dict) or not isinstance(old_stats, dict):
        raise SourceProfileError("manifest checkpoint files/file_stats must be objects")

    files: dict[str, str] = {}
    file_stats: dict[str, dict[str, int]] = {}
    for relative, record in current_stats.items():
        old_digest = old_files.get(relative)
        old_record = old_stats.get(relative)
        if isinstance(old_digest, str) and _is_sha256(old_digest) and old_record == record:
            files[relative] = old_digest
            file_stats[relative] = record

    manifest: dict[str, object] = {"type": "dir", "files": files, "file_stats": file_stats}
    # Persist stale-entry removal before any potentially long hashing work.
    _atomic_json(
        checkpoint,
        _checkpoint_payload(root=root, manifest=manifest, complete=False, digest=None),
    )

    hashed_files = 0
    reused_files = len(files)
    for relative, expected in current_stats.items():
        if relative in files:
            continue
        digest = _hash_checked_file(root / relative, expected)
        files[relative] = digest
        file_stats[relative] = expected
        hashed_files += 1
        _atomic_json(
            checkpoint,
            _checkpoint_payload(root=root, manifest=manifest, complete=False, digest=None),
        )

    # Re-enumerate to prove the completed manifest describes the current exact
    # path set.  Mutation after a reused entry is a hard failure, never silently
    # a mixed-time snapshot.
    if _enumerate_regular_files(root) != current_stats:
        raise SourceProfileError("source path set or file stats changed during manifest build")
    digest = source_manifest_digest(manifest)
    _atomic_json(output, manifest)
    _atomic_json(
        checkpoint,
        _checkpoint_payload(root=root, manifest=manifest, complete=True, digest=digest),
    )
    return ManifestBuildResult(
        manifest=manifest,
        digest=digest,
        hashed_files=hashed_files,
        reused_files=reused_files,
    )


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceProfileError(f"invalid {description} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceProfileError(f"{description} JSON must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise SourceProfileError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_manifest(path: Path) -> dict[str, object]:
    manifest = _read_json_object(path, "source manifest")
    if manifest.get("type") != "dir":
        raise SourceProfileError("source manifest must describe a directory")
    files = manifest.get("files")
    stats = manifest.get("file_stats")
    if not isinstance(files, dict) or not isinstance(stats, dict):
        raise SourceProfileError("source manifest requires files and file_stats objects")
    if set(files) != set(stats) or not all(_is_sha256(value) for value in files.values()):
        raise SourceProfileError("source manifest is incomplete or malformed")
    return manifest


def build_glm52_mixed_gguf_profile(
    *,
    manifest_path: str | Path,
    source_path: str | Path,
    source_revision: str,
    tokenizer_path: str | Path,
    risk_path: str | Path,
    tensor_plan_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Emit an honest, hash-bound GLM-5.2 mixed-GGUF profile artifact.

    The artifact contains only bounded weight evidence.  It intentionally has
    no calibration, activation, Hessian, routing, KLD, or CKA claims.
    """
    manifest = _completed_manifest(Path(manifest_path))
    files = manifest["files"]
    assert isinstance(files, dict)  # narrowed by _completed_manifest
    source = _safe_source_root(Path(source_path))
    risk_file = Path(risk_path)
    tensor_plan = Path(tensor_plan_path)
    tokenizer = Path(tokenizer_path)
    risk = _read_json_object(risk_file, "NVFP4 risk")
    risk_sha256 = _sha256_file(risk_file)
    tensor_plan_sha256 = _sha256_file(tensor_plan)
    tokenizer_sha256 = _sha256_file(tokenizer)

    expected_source = str(source)
    if risk.get("source") != expected_source:
        raise SourceProfileError("risk source does not match profile source")
    config_sha256 = _sha256_file(source / "config.json")
    index_sha256 = _sha256_file(source / "model.safetensors.index.json")
    if risk.get("config_sha256") != config_sha256 or risk.get("index_sha256") != index_sha256:
        raise SourceProfileError("risk config/index hashes do not match source")
    if (
        files.get("config.json") != config_sha256
        or files.get("model.safetensors.index.json") != index_sha256
    ):
        raise SourceProfileError("manifest config/index hashes do not match source")
    lines = tensor_plan.read_text(encoding="utf-8").splitlines()
    risk_lines = risk.get("tensor_type_lines")
    if risk.get("tensor_type_sha256") != tensor_plan_sha256 or risk_lines != lines:
        raise SourceProfileError("risk tensor plan does not match supplied tensor plan")
    if risk.get("evidence_kind") != "estimated":
        raise SourceProfileError("risk evidence kind must be estimated")

    manifest_digest = source_manifest_digest(manifest)
    profile: dict[str, object] = {
        "schema_version": 1,
        "profile_kind": "glm52_mixed_gguf_quantize_only",
        "execution": {
            "source": {
                "path": expected_source,
                "revision": source_revision,
                "manifest_digest": manifest_digest,
            },
            "tokenizer": {"path": str(tokenizer.resolve()), "sha256": tokenizer_sha256},
        },
        "evidence": {
            "nvfp4_suitability": {
                "kind": "estimated",
                "present": True,
                "detail": {
                    "risk_sha256": risk_sha256,
                    "tensor_plan_sha256": tensor_plan_sha256,
                    "config_sha256": config_sha256,
                    "index_sha256": index_sha256,
                    "note": risk.get("note"),
                },
            },
            "routing": None,
        },
        "calibration": None,
        "quality_metrics": {"kld": None, "cka": None},
    }
    _atomic_json(Path(output_path), profile)
    return profile
