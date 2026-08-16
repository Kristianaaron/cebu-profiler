from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_atlas.canary_lease import (
    CanaryLeaseBinding,
    CanaryLeaseError,
    remove_active_lease,
    require_active_lease,
    write_active_lease,
)


def _binding() -> CanaryLeaseBinding:
    return CanaryLeaseBinding(
        plan_sha256="a" * 64,
        artifact_path="/artifacts/glm52.gguf",
        artifact_sha256="b" * 64,
        head_unit="atlas-glm52-rpc-head",
        worker_unit="atlas-glm52-rpc-worker",
    )


def test_private_lease_binds_exact_plan_artifact_and_unit_names(tmp_path: Path) -> None:
    path = tmp_path / "active.json"
    handle = write_active_lease(path, _binding())
    assert require_active_lease(
        path, _binding(), expected_coordinator_pid=os.getpid()
    ).binding == _binding()
    with pytest.raises(CanaryLeaseError, match="binding"):
        require_active_lease(
            path,
            _binding().model_copy(update={"head_unit": "other"}),
            expected_coordinator_pid=os.getpid(),
        )
    remove_active_lease(handle)
    assert not path.exists()


def test_lease_rejects_unsafe_permissions_or_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "active.json"
    with pytest.raises(CanaryLeaseError, match="unavailable"):
        require_active_lease(path, _binding())
    handle = write_active_lease(path, _binding())
    path.chmod(0o644)
    with pytest.raises(CanaryLeaseError, match="permissions"):
        require_active_lease(path, _binding(), expected_coordinator_pid=os.getpid())
    path.chmod(0o600)
    remove_active_lease(handle)


def test_stale_or_replayed_file_cannot_authorize_execution(tmp_path: Path) -> None:
    path = tmp_path / "active.json"
    handle = write_active_lease(path, _binding())
    os.close(handle.descriptor)  # simulate coordinator death without cleanup
    with pytest.raises(CanaryLeaseError, match="not live"):
        require_active_lease(path, _binding(), expected_coordinator_pid=os.getpid())
    with pytest.raises(FileExistsError):
        write_active_lease(path, _binding())
    path.unlink()


def test_live_lease_rejects_wrong_coordinator_pid(tmp_path: Path) -> None:
    path = tmp_path / "active.json"
    handle = write_active_lease(path, _binding())
    with pytest.raises(CanaryLeaseError, match="identity"):
        require_active_lease(path, _binding(), expected_coordinator_pid=os.getppid())
    remove_active_lease(handle)
