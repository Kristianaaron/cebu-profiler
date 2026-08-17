"""Deterministic, verifiable single-file bundle for a width-sliced derivatives.

A width-sliced NVFP4 derivative is a *directory* (config.json, index, one or
more safetensors shards). The single-blob CAS handoff contract verifies one
artifact by SHA-256, so the derivative is packed into one deterministic
``.atlasbundle`` that can be unpacked to a directory vLLM can load.

Reproducibility guarantees:
- entries are traversed in sorted (name) order;
- tar member metadata is normalized (uid/gid=0, mtime=0, fixed mode);
- every pack is byte-identical for the same derivative tree (no gzip mtime).
The bundle embeds a ``.atlasbundle.manifest.json`` with per-file sha256+size
so unpack re-verifies extraction integrity.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

BUNDLE_SUFFIX = ".atlasbundle"
_MANIFEST_NAME = ".atlasbundle.manifest.json"
_BUFFER = 4 * 1024 * 1024


class BundleError(RuntimeError):
    """Bundle pack/unpack failed without weakening evidence."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_BUFFER):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise BundleError(f"bundle source is not a directory: {root}")
    files: list[Path] = []
    for current, dirs, names in os.walk(root):
        dirs.sort()
        for name in sorted(names):
            candidate = Path(current) / name
            if candidate.is_symlink():
                raise BundleError(f"bundle must not contain symlinks: {candidate}")
            if candidate.is_file():
                files.append(candidate.relative_to(root))
            else:
                raise BundleError(f"bundle source contains non-regular entry: {candidate}")
    files.sort(key=lambda rel: str(rel).encode("utf-8"))
    if not files:
        raise BundleError("bundle source contains no files")
    if _MANIFEST_NAME in {str(p) for p in files}:
        raise BundleError(f"source contains reserved bundle manifest name {_MANIFEST_NAME}")
    return files


def _manifest_member(data: bytes) -> tarfile.TarInfo:
    member = tarfile.TarInfo(_MANIFEST_NAME)
    member.size = len(data)
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.mode = 0o644
    return member


def pack_derivative_bundle(source_dir: Path, bundle_path: Path) -> tuple[str, int]:
    """Pack ``source_dir`` into a deterministic single-file bundle.

    Returns ``(sha256, size_bytes)`` of the produced bundle file.
    """
    root = source_dir.resolve()
    files = _walk_regular_files(root)
    manifest: dict[str, dict[str, int | str]] = {}
    for rel in files:
        absolute = root / rel
        manifest[str(rel)] = {
            "size": absolute.stat().st_size,
            "sha256": _sha256_file(absolute),
        }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(_manifest_member(manifest_bytes), io.BytesIO(manifest_bytes))
        for rel in files:
            absolute = root / rel
            member = tar.gettarinfo(str(absolute), arcname=str(rel))
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            member.mode = 0o644
            with absolute.open("rb") as reader:
                tar.addfile(member, reader)

    digest = hashlib.sha256()
    with bundle_path.open("rb") as handle:
        while chunk := handle.read(_BUFFER):
            digest.update(chunk)
    return digest.hexdigest(), bundle_path.stat().st_size


def unpack_derivative_bundle(bundle_path: Path, out_dir: Path) -> list[str]:
    """Unpack ``bundle_path`` into ``out_dir`` and verify every file sha+size.

    The leading member must be the manifest; each following regular member is
    checked against it. Any missing/extra/mismatched file raises ``BundleError``
    and the output is left in a failed state (caller owns cleanup).
    """
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise BundleError(f"bundle is not a regular file: {bundle_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_root = out_dir.resolve()
    written: list[str] = []
    try:
        with tarfile.open(bundle_path, "r:*") as tar:
            members = [m for m in tar if m.isfile()]
            if not members or members[0].name != _MANIFEST_NAME:
                raise BundleError("bundle lacks a leading manifest member")
            manifest_member = members[0]
            raw = tar.extractfile(manifest_member)
            if raw is None:
                raise BundleError("cannot read bundle manifest")
            manifest = json.loads(raw.read())
            if not isinstance(manifest, dict):
                raise BundleError("bundle manifest is not an object")

            for member in members[1:]:
                if member.name not in manifest:
                    raise BundleError(f"bundle member not in manifest: {member.name}")
                expected = manifest[member.name]
                expected_size = int(expected["size"])
                expected_sha = str(expected["sha256"])
                source = tar.extractfile(member)
                if source is None:
                    raise BundleError(f"cannot read bundle member: {member.name}")
                data = source.read()
                if len(data) != expected_size:
                    raise BundleError(f"{member.name} size mismatch")
                candidate = (out_dir / member.name).resolve()
                if not candidate.is_relative_to(out_root):
                    raise BundleError(f"bundle member escapes output dir: {member.name}")
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(data)
                actual_sha = _sha256_file(candidate)
                if actual_sha != expected_sha:
                    raise BundleError(f"{member.name} sha256 mismatch")
                written.append(member.name)
    except tarfile.ReadError as exc:
        raise BundleError(f"bundle is not a readable archive: {exc}") from exc

    if sorted(written) != sorted(manifest):
        raise BundleError("bundle extraction does not match the manifest")
    return sorted(written)


__all__ = [
    "BUNDLE_SUFFIX",
    "BundleError",
    "pack_derivative_bundle",
    "unpack_derivative_bundle",
]
