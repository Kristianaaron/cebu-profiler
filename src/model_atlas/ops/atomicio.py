"""Shared crash-safe file primitives (single source of truth).

Consolidates the previously divergent copies of temp-file + rename writing
(jobs/artifacts.py, ops/maintenance.py, jobs/engine.py journal). Semantics:

- atomic_write_bytes/text/json: write to a sibling temp file, fsync the file,
  rename over the destination, then fsync the parent directory. A crash leaves
  either the old file or the new one -- never a torn mix.
- publish_json_exclusive: hard-link-based creation that never clobbers an
  existing destination and never follows symlinks on any path component
  (maintenance receipts must not be redirectable by a local attacker).
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path


def canonical_json(obj: object) -> str:
    """Deterministic JSON encoding (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # best effort: some filesystems refuse directory fsync
        pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{secrets.token_hex(6)}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: object, *, newline: bool = True) -> None:
    atomic_write_text(path, canonical_json(obj) + ("\n" if newline else ""))


def publish_json_exclusive(path: Path, encoded: bytes) -> None:
    """Create ``path`` holding ``encoded`` only if it does not already exist.

    Symlink-hardened: walks every parent component with O_NOFOLLOW, creates a
    0600 temp file beside the destination, fsyncs, then hard-links into place
    and removes the temp. Raises FileExistsError if the destination exists.
    """
    path = Path(path)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError("destination path must be absolute and non-trivial")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent = os.open("/", flags)
    temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = -1
    try:
        for component in path.parent.parts[1:]:
            following = os.open(component, flags, dir_fd=parent)
            os.close(parent)
            parent = following
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.fsync(parent)
        os.unlink(temporary, dir_fd=parent)
        os.fsync(parent)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent)
            os.fsync(parent)
        except OSError:
            pass
        raise
    finally:
        if descriptor != -1:
            os.close(descriptor)
        os.close(parent)
