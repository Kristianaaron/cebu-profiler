"""Content-bound, ephemeral authority token for maintenance-bracketed canaries."""

from __future__ import annotations

import fcntl
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

    schema_version: Literal[2] = 2
    active: Literal[True] = True
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    coordinator_pid: int = Field(gt=0)
    coordinator_start_ticks: int = Field(gt=0)
    issued_at: datetime
    expires_at: datetime
    binding: CanaryLeaseBinding


@dataclass
class ActiveLeaseHandle:
    path: Path
    descriptor: int
    device: int
    inode: int


def _process_start_ticks(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        value = int(fields[21])
    except (OSError, ValueError, IndexError) as exc:
        raise CanaryLeaseError("maintenance coordinator identity unavailable") from exc
    if value <= 0:
        raise CanaryLeaseError("maintenance coordinator identity unavailable")
    return value


def write_active_lease(
    path: Path,
    binding: CanaryLeaseBinding,
    *,
    lifetime: timedelta = timedelta(hours=6),
) -> ActiveLeaseHandle:
    """Create and exclusively lock authority for the live coordinator lifetime."""
    if not path.is_absolute():
        raise ValueError("maintenance lease path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = datetime.now(UTC)
        encoded = (
            ActiveMaintenanceLease(
                nonce=secrets.token_hex(32),
                coordinator_pid=os.getpid(),
                coordinator_start_ticks=_process_start_ticks(os.getpid()),
                issued_at=now,
                expires_at=now + lifetime,
                binding=binding,
            )
            .model_dump_json()
            .encode()
        )
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise CanaryLeaseError("incomplete maintenance lease write")
        os.fsync(descriptor)
        stat = os.fstat(descriptor)
        return ActiveLeaseHandle(path, descriptor, stat.st_dev, stat.st_ino)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def require_active_lease(
    path: Path,
    binding: CanaryLeaseBinding,
    *,
    expected_coordinator_pid: int | None = None,
) -> ActiveMaintenanceLease:
    """Verify exact plan/artifact/transient-unit authority before GPU execution."""
    if not path.is_absolute() or path.is_symlink():
        raise CanaryLeaseError("maintenance lease path is invalid")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        stat = os.fstat(descriptor)
        if stat.st_mode & 0o077:
            raise CanaryLeaseError("maintenance lease permissions are unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise CanaryLeaseError("maintenance coordinator is not live")
        encoded = os.read(descriptor, 16385)
        if len(encoded) > 16384:
            raise CanaryLeaseError("maintenance lease is oversized")
        lease = ActiveMaintenanceLease.model_validate_json(encoded)
    except CanaryLeaseError:
        raise
    except (OSError, ValueError) as exc:
        raise CanaryLeaseError("maintenance lease unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if lease.binding != binding:
        raise CanaryLeaseError("maintenance lease binding mismatch")
    coordinator = os.getppid() if expected_coordinator_pid is None else expected_coordinator_pid
    if (
        lease.coordinator_pid != coordinator
        or lease.coordinator_start_ticks != _process_start_ticks(coordinator)
    ):
        raise CanaryLeaseError("maintenance coordinator identity mismatch")
    now = datetime.now(UTC)
    if lease.expires_at <= now or lease.issued_at > now:
        raise CanaryLeaseError("maintenance lease is expired")
    return lease


def remove_active_lease(handle: ActiveLeaseHandle) -> None:
    """Remove only the same locked inode created by this coordinator."""
    try:
        stat = handle.path.stat()
        if (stat.st_dev, stat.st_ino) != (handle.device, handle.inode):
            raise CanaryLeaseError("maintenance lease inode changed")
        handle.path.unlink()
    except OSError as exc:
        raise CanaryLeaseError("maintenance lease cleanup failed") from exc
    finally:
        os.close(handle.descriptor)


@contextmanager
def active_lease_scope(path: Path, binding: CanaryLeaseBinding) -> Iterator[None]:
    """Issue and remove a lease only while an acquired coordinator runs payload."""
    handle = write_active_lease(path, binding)
    try:
        yield
    finally:
        remove_active_lease(handle)
