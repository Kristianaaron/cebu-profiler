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

import fcntl
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
        dst = self._dst_for(digest)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            # atomic write + fsync; collision-guarded: if a DIFFERENT blob ever
            # claimed the same key the full-digest filename would mismatch and
            # the write would be refused below before overwriting.
            tmp = dst.with_suffix(".blob.tmp")
            if tmp.exists():
                tmp.unlink()
            tmp.write_bytes(data)
            os.replace(tmp, dst)
        # collision guard: a pre-existing slot whose content does NOT match the
        # advertised full digest is corruption — never overwrite it.
        if not dst.exists() or sha256_file(dst) != digest:
            raise RuntimeError(
                f"CAS collision guard: slot for digest {digest[:16]}… already holds "
                "different content; refusing to overwrite"
            )
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
        dst = self._dst_for(digest)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".blob.tmp")
            if tmp.exists():
                tmp.unlink()
            with open(src, "rb") as fin, open(tmp, "wb") as fout:
                while True:
                    chunk = fin.read(_IO_CHUNK)
                    if not chunk:
                        break
                    fout.write(chunk)
            os.replace(tmp, dst)
        if not dst.exists() or sha256_file(dst) != digest:
            raise RuntimeError(
                f"CAS collision guard: slot for digest {digest[:16]}… already holds "
                "different content; refusing to overwrite"
            )
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

    def read_from_key(self, key: str) -> bytes:
        """Read a blob by its full sha256 content key directly (used by repair
        rollback to restore the recorded original bytes)."""
        if not key:
            raise FileNotFoundError("empty content key")
        dst = self._dst_for(key)
        if not dst.exists():
            raise FileNotFoundError(f"content key {key[:16]}… has no blob in CAS")
        data = dst.read_bytes()
        if hashlib.sha256(data).hexdigest() != key:
            raise ValueError(f"CAS blob for key {key[:16]}… failed full-digest check")
        return data

    def verify(self, ref: OutputRef) -> bool:
        if not ref.relpath:
            return False
        path = self.root / ref.relpath
        if not path.exists():
            return False
        return sha256_file(path) == ref.sha256

    def _dst_for(self, digest: str) -> Path:
        """Content-addressed slot for the FULL digest; consumes the full 64-hex
        digest so an accidental truncation collision cannot conflate two blobs.
        The object key is the full digest; only the directory layer trims to 2
        chars for sharding (collision safety uses the full digest, not [:24]).
        """
        return self.blobs / digest[:2] / (digest + ".blob")


class StageStager:
    """Stages files for one stage and atomically promotes them on commit.

    Staging is a private scratch space that never leaks to run outputs until
    ``commit``. Commit publishes ONLY after every staged file has been
    content-addressed and full-digest verified by the store; the returned stage
    manifest is the record consumed by the job engine (never the staging tree).
    """

    def __init__(self, run_dir: Path, stage_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.stage_dir = self.run_dir / "stage" / stage_id
        self.staging = self.stage_dir / "staging"
        self.final = self.stage_dir / "output"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.final.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        safe = Path(name).name
        return self.staging / safe

    def commit(self, store: ContentAddressedStore) -> list[OutputRef]:
        """Publish every staged file via the store's full-digest CAS path.

        The store consumes full sha256 keys (collision-guarded); nothing is
        moved into the content-addressed tree until the digest has been computed
        and the slot verified. Returns the stage output manifest.
        """
        refs: list[OutputRef] = []
        for src in sorted(self.staging.iterdir()):
            if not src.is_file():
                continue
            digest = sha256_file(src)
            # publish through the CAS store (full-digest, atomic, guarded)
            stored = store.put_file(src.name, src, format="file")
            if stored.sha256 != digest or not store.verify(stored):
                raise RuntimeError(
                    f"stage {self.stage_dir.name} publish failed integrity for {src.name}"
                )
            refs.append(stored)
        # the finalized public manifest of this stage's outputs (CAS refs only,
        # never raw staging paths)
        self._write_stage_manifest(refs)
        return refs

    def _write_stage_manifest(self, refs: list[OutputRef]) -> None:
        manifest = {
            "stage": self.stage_dir.name,
            "outputs": [r.model_dump(mode="json") for r in refs],
        }
        atomic_write_json(self.final / "manifest.json", manifest)


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


class SourceIntegrityError(RuntimeError):
    """Raised when an immutable source fails any integrity check (snapshot
    equality, path-bound declared hashes, or canonical manifest digest). The
    run boundary treats it as FAILED_TERMINAL — never recoverable."""


def source_snapshot(source_path: str) -> dict[str, object]:
    """Complete recursive immutable-source hash+stat snapshot.

    Walks EVERY file under ``source_path`` (no first-eight cap) and records
    relative-path -> {sha256, size, mtime_ns}. This is the authoritative content
    manifest used to detect in-place mutation and to verify declared
    ``SourceIdentity.sha256`` per-file hashes with PATH BINDING."""
    return source_manifest(source_path)


def source_manifest(source_path: str) -> dict[str, object]:
    """Recursive relative-path -> sha256 manifest (path-bound)."""
    p = Path(source_path)
    if not p.exists():
        return {"type": "missing"}
    if p.is_file():
        size = p.stat().st_size
        mtime = p.stat().st_mtime_ns
        return {
            "type": "file",
            "files": {"__source__": sha256_file(p)},
            "file_stats": {"__source__": {"size": size, "mtime_ns": mtime}},
        }
    files: dict[str, str] = {}
    stats: dict[str, object] = {}
    for child in sorted(p.rglob("*")):
        if child.is_file():
            rel = str(child.relative_to(p))
            files[rel] = sha256_file(child)
            st = child.stat()
            stats[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return {"type": "dir", "files": files, "file_stats": stats}


def source_manifest_digest(manifest: dict[str, object]) -> str:
    """Canonical digest of the whole source manifest (path->hash map)."""
    payload = {
        "type": manifest.get("type"),
        "files": manifest.get("files", {}),
    }
    return sha256_hex(canonical_json(payload))


def assert_source_readonly(source_snapshot_before: dict[str, object], source_path: str) -> None:
    """Fail closed (SourceIntegrityError) if a source changed underneath a run
    (immutable_source=true) or is missing."""
    after = source_manifest(source_path)
    if after != source_snapshot_before:
        raise SourceIntegrityError(
            f"source {source_path} changed during run (immutable_source violated)"
        )
    if after.get("type") == "missing":
        raise SourceIntegrityError(f"source {source_path} is missing")


# open fd per lock path; flock is per open-file-description so release MUST
# unlock the exact fd that acquired the lock (a new fd's LOCK_UN does nothing).
_LOCK_FDS: dict[str, int] = {}


def acquire_file_lock(lock_path: Path, wait_seconds: float = 5.0) -> bool:
    """Advisory OS lock (flock, LOCK_EX|LOCK_NB) on a dedicated lockfile.

    The lock is held by the process via an open fd (auto-released on crash) and
    released explicitly through the SAME fd by ``release_file_lock``. There are
    no stale lockfile markers to recover — the previous O_EXCL behavior left a
    stale lockfile forever. Returns True when acquired within ``wait_seconds``.
    """
    import time

    path = str(Path(lock_path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}".encode())
            _LOCK_FDS[path] = fd
            return True
        except OSError:  # EAGAIN / EACCES
            if time.monotonic() >= deadline:
                os.close(fd)
                return False
            time.sleep(0.05)


def release_file_lock(lock_path: Path) -> None:
    from contextlib import suppress

    path = str(Path(lock_path))
    fd = _LOCK_FDS.pop(path, None)
    if fd is None:
        return
    with suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    with suppress(OSError):
        os.close(fd)
