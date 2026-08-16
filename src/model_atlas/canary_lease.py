"""Content-bound, ephemeral authority token for maintenance-bracketed canaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CanaryLeaseError(RuntimeError):
    """Fail-closed maintenance authority error."""


class CanaryLeaseBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_path: str = Field(pattern=r"^/")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    head_unit: str = Field(min_length=1)
    worker_unit: str = Field(min_length=1)


class ActiveMaintenanceLease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    active: Literal[True] = True
    binding: CanaryLeaseBinding


def write_active_lease(path: Path, binding: CanaryLeaseBinding) -> None:
    """Atomically create a private, exact lease after maintenance acquisition."""
    if not path.is_absolute():
        raise ValueError("maintenance lease path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        encoded = ActiveMaintenanceLease(binding=binding).model_dump_json().encode()
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise CanaryLeaseError("incomplete maintenance lease write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def require_active_lease(path: Path, binding: CanaryLeaseBinding) -> ActiveMaintenanceLease:
    """Verify exact plan/artifact/transient-unit authority before GPU execution."""
    if not path.is_absolute() or path.is_symlink():
        raise CanaryLeaseError("maintenance lease path is invalid")
    try:
        stat = path.stat()
        if stat.st_mode & 0o077:
            raise CanaryLeaseError("maintenance lease permissions are unsafe")
        lease = ActiveMaintenanceLease.model_validate_json(path.read_bytes())
    except CanaryLeaseError:
        raise
    except (OSError, ValueError) as exc:
        raise CanaryLeaseError("maintenance lease unavailable") from exc
    if lease.binding != binding:
        raise CanaryLeaseError("maintenance lease binding mismatch")
    return lease


def remove_active_lease(path: Path) -> None:
    """Remove only the exact per-run lease after payload completion."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise CanaryLeaseError("maintenance lease cleanup failed") from exc
