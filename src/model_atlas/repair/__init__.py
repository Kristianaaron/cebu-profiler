"""Deterministic repair framework: typed versioned transforms, real
CAS-verified apply, atomic target-ref publish, byte-restoring rollback."""

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
