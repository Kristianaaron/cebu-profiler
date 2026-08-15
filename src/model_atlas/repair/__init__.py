"""Deterministic repair framework: typed proposals, registered transforms,
real verification, atomic CAS rollback."""

from model_atlas.repair.gate import (
    ALLOWLIST,
    DETERMINISTIC_REPAIRS,
    CompiledRepair,
    RepairGate,
    RepairProposal,
    RepairTransform,
    RepairValidation,
    known_repairs,
    register_transform,
    sha256_hex,
)

__all__ = [
    "ALLOWLIST",
    "CompiledRepair",
    "DETERMINISTIC_REPAIRS",
    "RepairGate",
    "RepairProposal",
    "RepairTransform",
    "RepairValidation",
    "known_repairs",
    "register_transform",
    "sha256_hex",
]
