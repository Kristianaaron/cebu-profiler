from __future__ import annotations

from pathlib import Path

import pytest

from model_atlas.telemetry_python import (
    TelemetryPythonConfig,
    TelemetryPythonIdentityError,
    verify_telemetry_python,
)


def _config() -> TelemetryPythonConfig:
    return TelemetryPythonConfig(
        interpreter=Path("/venvs/atlas/bin/python"),
        interpreter_sha256="a" * 64,
        worker_ssh_target="10.77.0.2",
    )


def test_verifies_exact_head_and_worker_interpreter_and_builds_python_argv() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: object) -> str:
        calls.append(tuple(argv))  # type: ignore[arg-type]
        return "a" * 64 + "  /venvs/atlas/bin/python\n"

    attestation = verify_telemetry_python(
        _config(), head_hasher=lambda _path: "a" * 64, runner=runner
    )
    assert attestation.verified
    assert calls == [
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            "--",
            "10.77.0.2",
            "sha256sum",
            "/venvs/atlas/bin/python",
        )
    ]
    assert _config().probe_argv(Path("/repo/collect.py"), "--rdma-interface", "ib0") == (
        "/venvs/atlas/bin/python",
        "/repo/collect.py",
        "--rdma-interface",
        "ib0",
    )


def test_refuses_hash_drift_or_malformed_worker_output() -> None:
    with pytest.raises(TelemetryPythonIdentityError, match="hash mismatch"):
        verify_telemetry_python(
            _config(),
            head_hasher=lambda _path: "b" * 64,
            runner=lambda _argv: "a" * 64 + "  /venvs/atlas/bin/python\n",
        )
    with pytest.raises(TelemetryPythonIdentityError, match="malformed"):
        verify_telemetry_python(
            _config(), head_hasher=lambda _path: "a" * 64, runner=lambda _argv: "not-a-hash\n"
        )


def test_config_rejects_relative_or_unpinned_interpreter() -> None:
    with pytest.raises(ValueError, match="absolute"):
        TelemetryPythonConfig(Path("python"), "a" * 64, "10.77.0.2")
    with pytest.raises(ValueError, match="SHA"):
        TelemetryPythonConfig(Path("/python"), "bad", "10.77.0.2")
