"""Pinned interpreter identity for local and remote telemetry probes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_CHUNK = 1 << 20


class TelemetryPythonIdentityError(RuntimeError):
    """Fail-closed interpreter identity error with no command output."""


class ArgvRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str: ...


class FileHasher(Protocol):
    def __call__(self, path: Path) -> str: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TelemetryPythonConfig:
    interpreter: Path
    interpreter_sha256: str
    worker_ssh_target: str
    ssh_argv: tuple[str, ...] = ("ssh", "-o", "BatchMode=yes")

    def __post_init__(self) -> None:
        if not self.interpreter.is_absolute():
            raise ValueError("telemetry interpreter must be absolute")
        if len(self.interpreter_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.interpreter_sha256
        ):
            raise ValueError("telemetry interpreter SHA-256 must be lowercase")
        if (
            not self.worker_ssh_target
            or not self.ssh_argv
            or any(not value for value in self.ssh_argv)
        ):
            raise ValueError("worker target and SSH argv are required")

    def probe_argv(self, script: Path, *script_args: str) -> tuple[str, ...]:
        if not script.is_absolute():
            raise ValueError("telemetry script must be absolute")
        if any(not value for value in script_args):
            raise ValueError("telemetry arguments must be non-empty")
        return (str(self.interpreter), str(script), *script_args)


@dataclass(frozen=True)
class TelemetryPythonAttestation:
    interpreter_path: str
    expected_sha256: str
    head_sha256: str
    worker_sha256: str

    @property
    def verified(self) -> bool:
        return self.head_sha256 == self.expected_sha256 == self.worker_sha256


def verify_telemetry_python(
    config: TelemetryPythonConfig,
    *,
    head_hasher: FileHasher = sha256_file,
    runner: ArgvRunner,
) -> TelemetryPythonAttestation:
    """Hash exact interpreter bytes locally and over authenticated SSH."""
    try:
        head_sha = head_hasher(config.interpreter)
        output = runner(
            config.ssh_argv + ("--", config.worker_ssh_target, "sha256sum", str(config.interpreter))
        )
    except Exception as exc:  # noqa: BLE001 - external boundary is sanitized
        raise TelemetryPythonIdentityError("telemetry interpreter identity unavailable") from exc
    fields = output.strip().split()
    if (
        len(fields) < 2
        or fields[1] != str(config.interpreter)
        or len(fields[0]) != 64
        or any(value not in "0123456789abcdef" for value in fields[0])
    ):
        raise TelemetryPythonIdentityError("worker telemetry interpreter hash malformed")
    attestation = TelemetryPythonAttestation(
        interpreter_path=str(config.interpreter),
        expected_sha256=config.interpreter_sha256,
        head_sha256=head_sha,
        worker_sha256=fields[0],
    )
    if not attestation.verified:
        raise TelemetryPythonIdentityError("telemetry interpreter hash mismatch")
    return attestation
