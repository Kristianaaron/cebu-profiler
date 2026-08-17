"""Resumable source identity and honest GLM-5.2 profile artifacts.

These helpers deliberately do not use the normal JobEngine snapshot function:
that function is correct for small sources, but a several-hundred-GB mounted
checkpoint needs a crash-resumable, bounded-memory walk.  The final manifest is
still exactly the JobEngine manifest shape and its digest is calculated by the
authoritative :func:`source_manifest_digest` helper.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from model_atlas.jobs.artifacts import (
    is_huggingface_source_cache,
    is_model_source_payload,
    source_manifest_digest,
)
from model_atlas.recipes.builtin import GLM52_GGUF_TENSOR_PLAN_SHA256

_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_SHA256_HEX_LENGTH = 64
_GLM52_SOURCE_ID = "nvidia/GLM-5.2-NVFP4"


class SourceProfileError(RuntimeError):
    """A source identity or profile evidence invariant was not satisfied."""


def _advise_sequential(descriptor: int) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_SEQUENTIAL)


def _drop_clean_cache(descriptor: int, offset: int, length: int) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)


@dataclass(frozen=True)
class ManifestBuildResult:
    """Result of a completed resumable directory-manifest build."""

    manifest: dict[str, object]
    digest: str
    hashed_files: int
    reused_files: int


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(8))
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _stat_record(st: os.stat_result) -> dict[str, int]:
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _reuse_stat_record(st: os.stat_result) -> dict[str, int]:
    return {
        **_stat_record(st),
        "ctime_ns": st.st_ctime_ns,
        "device": st.st_dev,
        "inode": st.st_ino,
    }


def _is_same_stat(expected: Mapping[str, object], actual: os.stat_result) -> bool:
    return expected == _reuse_stat_record(actual)


def _read_bounded_regular(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceProfileError(f"cannot safely open {description} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceProfileError(f"{description} must be a regular file: {path}")
        payload = b""
        while len(payload) <= _MAX_METADATA_BYTES:
            chunk = os.read(descriptor, min(64 * 1024, _MAX_METADATA_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
        if len(payload) > _MAX_METADATA_BYTES:
            raise SourceProfileError(f"{description} exceeds metadata size bound")
        after = os.fstat(descriptor)
        if _reuse_stat_record(before) != _reuse_stat_record(after):
            raise SourceProfileError(f"{description} changed while being read")
        return payload
    finally:
        os.close(descriptor)


def _outside_source(root: Path, candidate: Path, description: str) -> None:
    resolved = candidate.resolve(strict=False)
    if resolved == root or root in resolved.parents:
        raise SourceProfileError(f"{description} must be outside the source tree")


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


def _open_root_directory(root: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        return os.open(root, flags)
    except OSError as exc:
        raise SourceProfileError(f"cannot pin source root {root}: {exc}") from exc


def _enumerate_regular_files(root_descriptor: int) -> dict[str, dict[str, int]]:
    """Enumerate regular files without following symlinks or special nodes."""
    files: dict[str, dict[str, int]] = {}

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def visit(directory_descriptor: int, prefix: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory_descriptor), key=lambda entry: entry.name)
        except OSError as exc:
            raise SourceProfileError(f"cannot enumerate source directory: {exc}") from exc
        for entry in entries:
            relative = prefix / entry.name
            try:
                entry_stat = os.stat(
                    entry.name, dir_fd=directory_descriptor, follow_symlinks=False
                )
            except OSError as exc:
                raise SourceProfileError(f"cannot stat source entry {relative}: {exc}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise SourceProfileError(f"source contains forbidden symlink: {relative}")
            if stat.S_ISDIR(entry_stat.st_mode):
                if is_huggingface_source_cache(str(relative)):
                    continue
                try:
                    child_descriptor = os.open(
                        entry.name, directory_flags, dir_fd=directory_descriptor
                    )
                except OSError as exc:
                    raise SourceProfileError(
                        f"cannot safely open source directory {relative}: {exc}"
                    ) from exc
                try:
                    child_stat = os.fstat(child_descriptor)
                    if (child_stat.st_dev, child_stat.st_ino) != (
                        entry_stat.st_dev,
                        entry_stat.st_ino,
                    ):
                        raise SourceProfileError(
                            f"source directory changed while opening: {relative}"
                        )
                    visit(child_descriptor, relative)
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise SourceProfileError(f"source contains non-regular entry: {relative}")
            relative_text = str(relative)
            if not is_model_source_payload(relative_text):
                continue
            if relative_text.startswith("../") or relative_text == "..":
                raise SourceProfileError(f"source path escape: {relative_text}")
            files[relative_text] = _reuse_stat_record(entry_stat)

    visit(root_descriptor, PurePosixPath())
    return files


def _open_parent_directory(root_descriptor: int, relative: str) -> tuple[int, str]:
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceProfileError(f"invalid relative source path: {relative}")
    descriptor = os.dup(root_descriptor)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for component in path.parts[:-1]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, path.name
    except Exception:
        os.close(descriptor)
        raise


def _hash_checked_file(
    root_descriptor: int, relative: str, expected: dict[str, int]
) -> str:
    """Hash one fixed regular file using bounded reads and race checks."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor, name = _open_parent_directory(root_descriptor, relative)
        descriptor = os.open(name, os.O_RDONLY | nofollow, dir_fd=parent_descriptor)
    except OSError as exc:
        raise SourceProfileError(f"cannot safely open {relative}: {exc}") from exc
    finally:
        if "parent_descriptor" in locals():
            os.close(parent_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not _is_same_stat(expected, before):
            raise SourceProfileError(
                f"source changed or became non-regular before hashing: {relative}"
            )
        digest = hashlib.sha256()
        _advise_sequential(descriptor)
        offset = 0
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
                _drop_clean_cache(descriptor, offset, len(chunk))
                offset += len(chunk)
        after = os.fstat(descriptor)
        if not _is_same_stat(expected, after):
            raise SourceProfileError(f"source changed while hashing: {relative}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value: object = json.loads(_read_bounded_regular(path, "manifest checkpoint"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceProfileError(f"invalid manifest checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceProfileError(f"manifest checkpoint must be an object: {path}")
    return value


def _checkpoint_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        read_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            existing_descriptor = os.open(path, read_flags)
        except OSError as exc:
            raise SourceProfileError(f"cannot safely open checkpoint key: {exc}") from exc
        try:
            key_stat = os.fstat(existing_descriptor)
            if (
                not stat.S_ISREG(key_stat.st_mode)
                or key_stat.st_mode & 0o077
                or key_stat.st_uid != os.getuid()
            ):
                raise SourceProfileError(
                    "manifest checkpoint key must be owned by the user and mode 0600"
                ) from None
            payload = os.read(existing_descriptor, 33)
            if len(payload) != 32 or os.read(existing_descriptor, 1):
                raise SourceProfileError(
                    "manifest checkpoint key has invalid length"
                ) from None
            key_stat_after = os.fstat(existing_descriptor)
            if _reuse_stat_record(key_stat) != _reuse_stat_record(key_stat_after):
                raise SourceProfileError("manifest checkpoint key changed while read")
            return payload
        finally:
            os.close(existing_descriptor)
    except OSError as exc:
        raise SourceProfileError(f"cannot create manifest checkpoint key: {exc}") from exc
    payload = secrets.token_bytes(32)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _checkpoint_mac(payload: Mapping[str, object], key: bytes) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _checkpoint_payload(
    *,
    root: Path,
    root_identity: Mapping[str, int],
    manifest: dict[str, object],
    reuse_stats: Mapping[str, Mapping[str, int]],
    complete: bool,
    digest: str | None,
    key: bytes,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "source_root": str(root),
        "root_identity": dict(root_identity),
        "complete": complete,
        "manifest": manifest,
        "reuse_stats": dict(reuse_stats),
    }
    if digest is not None:
        payload["manifest_digest"] = digest
    payload["hmac_sha256"] = _checkpoint_mac(payload, key)
    return payload


def build_resumable_source_manifest(
    source_path: str | Path,
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    checkpoint_key_path: str | Path | None = None,
) -> ManifestBuildResult:
    """Build a resumable, path-bound directory manifest.

    Each successfully hashed file is atomically checkpointed.  A prior digest
    is reused only when that exact relative path still has the same size and
    nanosecond mtime.  Entries for deleted files are removed before work starts.
    """
    root = _safe_source_root(Path(source_path))
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    key_path = (
        Path(checkpoint_key_path)
        if checkpoint_key_path is not None
        else checkpoint.with_name(checkpoint.name + ".key")
    )
    for candidate, description in (
        (checkpoint, "checkpoint"),
        (key_path, "checkpoint key"),
        (output, "manifest output"),
    ):
        _outside_source(root, candidate, description)
    resolved_controls = {
        checkpoint.resolve(strict=False),
        key_path.resolve(strict=False),
        output.resolve(strict=False),
    }
    if len(resolved_controls) != 3:
        raise SourceProfileError("checkpoint, checkpoint key, and manifest must be distinct")
    key = _checkpoint_key(key_path)
    root_descriptor = _open_root_directory(root)
    root_stat = os.fstat(root_descriptor)
    root_identity = {"device": root_stat.st_dev, "inode": root_stat.st_ino}
    current_stats = _enumerate_regular_files(root_descriptor)
    previous = _load_checkpoint(checkpoint)
    if previous:
        supplied_mac = previous.pop("hmac_sha256", None)
        if (
            previous.get("schema_version") != 2
            or previous.get("source_root") != str(root)
            or previous.get("root_identity") != root_identity
            or not isinstance(supplied_mac, str)
            or not hmac.compare_digest(supplied_mac, _checkpoint_mac(previous, key))
        ):
            raise SourceProfileError("manifest checkpoint authentication failed")
    old_manifest = previous.get("manifest", {})
    if not isinstance(old_manifest, dict):
        raise SourceProfileError("manifest checkpoint has non-object manifest")
    old_files = old_manifest.get("files", {})
    old_stats = old_manifest.get("file_stats", {})
    old_reuse_stats = previous.get("reuse_stats", {})
    if (
        not isinstance(old_files, dict)
        or not isinstance(old_stats, dict)
        or not isinstance(old_reuse_stats, dict)
    ):
        raise SourceProfileError("manifest checkpoint files/file_stats must be objects")
    if previous.get("complete") is True and previous.get(
        "manifest_digest"
    ) != source_manifest_digest(old_manifest):
        raise SourceProfileError("completed manifest checkpoint digest mismatch")

    files: dict[str, str] = {}
    file_stats: dict[str, dict[str, int]] = {}
    for relative, record in current_stats.items():
        old_digest = old_files.get(relative)
        old_record = old_stats.get(relative)
        old_reuse_record = old_reuse_stats.get(relative)
        if (
            isinstance(old_digest, str)
            and _is_sha256(old_digest)
            and old_record == {"size": record["size"], "mtime_ns": record["mtime_ns"]}
            and old_reuse_record == record
        ):
            files[relative] = old_digest
            file_stats[relative] = {
                "size": record["size"],
                "mtime_ns": record["mtime_ns"],
            }

    manifest: dict[str, object] = {"type": "dir", "files": files, "file_stats": file_stats}
    # Persist stale-entry removal before any potentially long hashing work.
    _atomic_json(
        checkpoint,
        _checkpoint_payload(
            root=root,
            root_identity=root_identity,
            manifest=manifest,
            reuse_stats={path: current_stats[path] for path in files},
            complete=False,
            digest=None,
            key=key,
        ),
    )

    hashed_files = 0
    reused_files = len(files)
    for relative, expected in current_stats.items():
        if relative in files:
            continue
        digest = _hash_checked_file(root_descriptor, relative, expected)
        files[relative] = digest
        file_stats[relative] = {
            "size": expected["size"],
            "mtime_ns": expected["mtime_ns"],
        }
        hashed_files += 1
        _atomic_json(
            checkpoint,
            _checkpoint_payload(
                root=root,
                root_identity=root_identity,
                manifest=manifest,
                reuse_stats={path: current_stats[path] for path in files},
                complete=False,
                digest=None,
                key=key,
            ),
        )

    # Re-enumerate to prove the completed manifest describes the current exact
    # path set.  Mutation after a reused entry is a hard failure, never silently
    # a mixed-time snapshot.
    if _enumerate_regular_files(root_descriptor) != current_stats:
        raise SourceProfileError("source path set or file stats changed during manifest build")
    final_root_stat = os.fstat(root_descriptor)
    if root_identity != {"device": final_root_stat.st_dev, "inode": final_root_stat.st_ino}:
        raise SourceProfileError("source root changed during manifest build")
    digest = source_manifest_digest(manifest)
    _atomic_json(output, manifest)
    _atomic_json(
        checkpoint,
        _checkpoint_payload(
            root=root,
            root_identity=root_identity,
            manifest=manifest,
            reuse_stats=current_stats,
            complete=True,
            digest=digest,
            key=key,
        ),
    )
    os.close(root_descriptor)
    return ManifestBuildResult(
        manifest=manifest,
        digest=digest,
        hashed_files=hashed_files,
        reused_files=reused_files,
    )


def _read_json_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_bounded_regular(path, description)
    try:
        value: object = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceProfileError(f"invalid {description} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceProfileError(f"{description} JSON must be an object: {path}")
    return value, payload


def _sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceProfileError(f"cannot safely open file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceProfileError(f"expected regular file: {path}")
        digest = hashlib.sha256()
        _advise_sequential(descriptor)
        offset = 0
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            digest.update(chunk)
            _drop_clean_cache(descriptor, offset, len(chunk))
            offset += len(chunk)
        after = os.fstat(descriptor)
        if _reuse_stat_record(before) != _reuse_stat_record(after):
            raise SourceProfileError(f"file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _completed_manifest(path: Path) -> dict[str, object]:
    manifest, _ = _read_json_object(path, "source manifest")
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
    output = Path(output_path)
    _outside_source(source, output, "profile output")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise SourceProfileError("source revision must be nonempty")
    risk_file = Path(risk_path)
    tensor_plan = Path(tensor_plan_path)
    tokenizer = Path(tokenizer_path)
    input_paths = {
        Path(manifest_path).resolve(strict=False),
        risk_file.resolve(strict=False),
        tensor_plan.resolve(strict=False),
        tokenizer.resolve(strict=False),
    }
    if output.resolve(strict=False) in input_paths:
        raise SourceProfileError("profile output must be distinct from all profile inputs")
    risk, risk_payload = _read_json_object(risk_file, "NVFP4 risk")
    risk_sha256 = hashlib.sha256(risk_payload).hexdigest()
    tensor_plan_payload = _read_bounded_regular(tensor_plan, "tensor plan")
    tensor_plan_sha256 = hashlib.sha256(tensor_plan_payload).hexdigest()
    if tensor_plan_sha256 != GLM52_GGUF_TENSOR_PLAN_SHA256:
        raise SourceProfileError("tensor plan does not match the canonical executable plan")
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
    try:
        lines = tensor_plan_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SourceProfileError("tensor plan is not UTF-8") from exc
    risk_lines = risk.get("tensor_type_lines")
    if risk.get("tensor_type_sha256") != tensor_plan_sha256 or risk_lines != lines:
        raise SourceProfileError("risk tensor plan does not match supplied tensor plan")
    if risk.get("evidence_kind") != "estimated":
        raise SourceProfileError("risk evidence kind must be estimated")

    manifest_digest = source_manifest_digest(manifest)
    profile: dict[str, object] = {
        "schema_version": 1,
        "profile_kind": "glm52_mixed_gguf_quantize_only",
        "model": _GLM52_SOURCE_ID,
        "execution": {
            "source_id": _GLM52_SOURCE_ID,
            "checkpoint_path": expected_source,
            "checkpoint_revision": source_revision,
            "source_manifest_digest": manifest_digest,
            "source_sha256": {},
            "tokenizer_hash": tokenizer_sha256,
        },
        "evidence": {
            "nvfp4_suitability": {
                "kind": "estimated",
                "present": True,
                "detail": (
                    f"risk_artifact_sha256={risk_sha256};"
                    f"tensor_plan_sha256={tensor_plan_sha256}"
                ),
            },
            "routing": None,
        },
    }
    _atomic_json(output, profile)
    return profile
