from __future__ import annotations

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
    write_active_lease(path, _binding())
    assert require_active_lease(path, _binding()).binding == _binding()
    with pytest.raises(CanaryLeaseError, match="binding"):
        require_active_lease(path, _binding().model_copy(update={"head_unit": "other"}))
    remove_active_lease(path)
    assert not path.exists()


def test_lease_rejects_unsafe_permissions_or_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "active.json"
    with pytest.raises(CanaryLeaseError, match="unavailable"):
        require_active_lease(path, _binding())
    write_active_lease(path, _binding())
    path.chmod(0o644)
    with pytest.raises(CanaryLeaseError, match="permissions"):
        require_active_lease(path, _binding())
