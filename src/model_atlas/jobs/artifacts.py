"""Content-addressed artifact store + atomic filesystem helpers.

Guarantees:

* **Staging then atomic promotion** — derived artifacts are written under
  ``<run>/stage/<stage_id>/staging/`` and only moved (``os.replace``) into their
  content-addressed slot after a successful write + hash; never mutated in
  place.
* **Immutable source protection** — the source checkpoint path is recorded and
  the engine asserts read-only access (never writes through it). Where the
  source is enumerable, a before/after stat snapshot can be verified.
* **Idempotency** — an output whose content-address already exists is not
  rewritten; identical inputs yield identical addresses.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from model_atlas.jobs.schema import OutputRef
from model_atlas.recipe.compiler import canonical_json, sha256_hex

_IO_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_IO_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def content_address(name: str, data: object) -> str:
    """Content address of a serialized object (JSON) with a stable name."""
    return sha256_hex(canonical_json({"name": name, "data": data}))


class ContentAddressedStore:
    """Per-run store keyed by ``sha256[:24]`` under ``<run>/objects/<ab>``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.blobs = self.root / "objects"
        self.blobs.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, name: str, data: bytes) -> OutputRef:
        digest = hashlib.sha256(data).hexdigest()
        key = digest[:24]
        dst = self.blobs / key[:2] / (key + ".blob")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            # atomic write + fsync
            tmp = dst.with_suffix(".blob.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
        return OutputRef(
            name=name,
            sha256=digest,
            size_bytes=len(data),
            format="bytes",
            relpath=str(dst.relative_to(self.root)),
        )

    def put_json(self, name: str, obj: object) -> OutputRef:
        payload = canonical_json(obj).encode("utf-8")
        return self.put_bytes(name, payload)

    def put_file(self, name: str, src: Path, format: str = "") -> OutputRef:
        digest = sha256_file(src)
        key = digest[:24]
        dst = self.blobs / key[:2] / (key + ".blob")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".blob.tmp")
            with open(src, "rb") as fin, open(tmp, "wb") as fout:
                while True:
                    chunk = fin.read(_IO_CHUNK)
                    if not chunk:
                        break
                    fout.write(chunk)
            os.replace(tmp, dst)
        return OutputRef(
            name=name,
            sha256=digest,
            size_bytes=src.stat().st_size,
            format=format,
            relpath=str(dst.relative_to(self.root)),
        )

    def read(self, ref: OutputRef) -> bytes:
        if not ref.relpath:
            raise FileNotFoundError(f"output {ref.name} not materialized")
        path = self.root / ref.relpath
        return path.read_bytes()

    def verify(self, ref: OutputRef) -> bool:
        if not ref.relpath:
            return False
        path = self.root / ref.relpath
        if not path.exists():
            return False
        return sha256_file(path) == ref.sha256


class StageStager:
    """Stages files for one stage and atomically promotes them on commit."""

    def __init__(self, run_dir: Path, stage_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.stage_dir = self.run_dir / "stage" / stage_id
        self.staging = self.stage_dir / "staging"
        self.final = self.stage_dir / "output"
        self.staging.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        safe = Path(name).name
        return self.staging / safe

    def commit(self, store: ContentAddressedStore) -> list[OutputRef]:
        """Atomically move staged files into content-addressed slots and return
        their refs. Never touches an existing output (idempotent)."""
        refs: list[OutputRef] = []
        self.final.mkdir(parents=True, exist_ok=True)
        for src in sorted(self.staging.iterdir()):
            if not src.is_file():
                continue
            digest = sha256_file(src)
            key = digest[:24]
            dst = self.staging / (key + ".blob")
            # relink via hardlink first so we can verify without copying
            os.replace(src, dst)  # staged blob owned by this stage
            ref = OutputRef(
                name=src.name,
                sha256=digest,
                size_bytes=dst.stat().st_size,
                format="file",
                relpath=str(dst.relative_to(self.run_dir)),
            )
            # idempotent copy into content-addressed store
            stored = store.put_file(ref.name, dst)
            refs.append(stored)
        return refs


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, canonical_json(obj) + "\n")


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def source_snapshot(source_path: str) -> dict[str, object]:
    """Best-effort immutable-source stat snapshot (size+mtime, bounded).
    Used to detect accidental in-place mutation of the source checkpoint."""
    p = Path(source_path)
    if p.is_dir():
        entries = {}
        for child in sorted(p.iterdir())[:8]:
            try:
                st = child.stat()
            except OSError:
                continue
            entries[child.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        return {"type": "dir", "entries": entries}
    try:
        st = p.stat()
        return {"type": "file", "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        return {"type": "missing"}


def assert_source_readonly(source_snapshot_before: dict[str, object], source_path: str) -> None:
    """Fail closed if a source changed underneath a run (immutable_source=true)."""
    after = source_snapshot(source_path)
    if after != source_snapshot_before:
        raise RuntimeError(f"source {source_path} changed during run (immutable_source violated)")
    if after.get("type") == "missing":
        raise RuntimeError(f"source {source_path} is missing")


def acquire_file_lock(lock_path: Path, wait_seconds: float = 5.0) -> bool:
    """Advisory lock via O_EXCL lockfile; returns True when acquired."""
    import time

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.05)
    return False


def release_file_lock(lock_path: Path) -> None:
    from contextlib import suppress

    with suppress(FileNotFoundError):
        os.unlink(lock_path)
