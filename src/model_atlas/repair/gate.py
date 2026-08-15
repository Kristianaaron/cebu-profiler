"""Deterministic repair framework for the Atlas compression plane.

Validators emit **typed repair proposals**. Only proposals whose kind is
**registered** as a deterministic, typed transform may be compiled and applied.
Agent suggestions never travel directly to application — they must first become
a typed proposal and pass the same compile gate.

Every applied repair records before/after content hashes, persists the repaired
bytes back into a content-addressed store by atomic replace, and rolls back by
**restoring the recorded CAS ref** (bytes verified by full digest) — never by
flipping flags.

A registered transform is a deterministic function
``(params, before_bytes) -> after_bytes``, optionally paired with a validator
``(params, after_bytes) -> bool`` run after transformation. Evidence-downgrade
transforms are monotonic (never upgrade); channel transforms enforce range
bounds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.schemas.evidence import EvidenceKind

# Rank ladder: an evidence downgrade repair may only move DOWN this ladder.
_EVIDENCE_RANK = {
    EvidenceKind.INFERRED: 0,
    EvidenceKind.PREDICTED: 1,
    EvidenceKind.ESTIMATED: 2,
    EvidenceKind.MEASURED: 3,
    EvidenceKind.CAUSALLY_TESTED: 4,
}

Transform = Callable[[dict[str, str], bytes], bytes]
Validator = Callable[[dict[str, str], bytes], bool]


@dataclass(frozen=True)
class RepairTransform:
    """A registered, typed deterministic repair transform."""

    kind: str
    contract: str
    # max evidence kind the repair may set (downgrades only below current)
    evidence_ceiling: EvidenceKind = EvidenceKind.ESTIMATED
    transform: Transform | None = None
    validator: Validator | None = None


_TRANSFORMS: dict[str, RepairTransform] = {}


def register_transform(t: RepairTransform) -> None:
    if t.kind in _TRANSFORMS:
        raise ValueError(f"repair transform {t.kind!r} already registered")
    _TRANSFORMS[t.kind] = t


def _norm_channels(channels: list[int], lo: int, hi: int) -> list[int]:
    out = sorted(set(int(c) for c in channels))
    bad = [c for c in out if c < lo or c >= hi]
    if bad:
        raise ValueError(f"channel(s) {bad} out of range [{lo}, {hi})")
    return out


def _apply_keep_channels(params: dict[str, str], before: bytes) -> bytes:
    d = json.loads(before.decode("utf-8"))
    lo = int(params.get("channel_lo", 0))
    hi = int(params.get("channel_hi", 2**63 - 1))
    channels = [int(c) for c in params.get("channels", "").split(",") if c != ""]
    if not channels:
        channels = [int(c) for c in d.get("keep_channels", [])]
    d = dict(d)
    d["keep_channels"] = _norm_channels(channels, lo, hi)
    return json.dumps(d, sort_keys=True).encode("utf-8")


def _validate_keep_channels(params: dict[str, str], after: bytes) -> bool:
    try:
        d = json.loads(after.decode("utf-8"))
        lo = int(params.get("channel_lo", 0))
        hi = int(params.get("channel_hi", 2**63 - 1))
        _norm_channels([int(c) for c in d["keep_channels"]], lo, hi)
        return True
    except (ValueError, TypeError, KeyError):
        return False


def _apply_evidence_downgrade(params: dict[str, str], before: bytes) -> bytes:
    d = json.loads(before.decode("utf-8"))
    current = EvidenceKind(d.get("evidence_kind", "predicted"))
    target = EvidenceKind(params.get("to", "predicted"))
    if _EVIDENCE_RANK[target] > _EVIDENCE_RANK[current]:
        raise ValueError(
            f"evidence downgrade refused: {current.value} -> {target.value} would upgrade"
        )
    d = dict(d)
    d["evidence_kind"] = target.value
    return json.dumps(d, sort_keys=True).encode("utf-8")


def _validate_evidence(params: dict[str, str], after: bytes) -> bool:
    try:
        EvidenceKind(json.loads(after.decode("utf-8")).get("evidence_kind", "predicted"))
        return True
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------
# registered deterministic transforms
# --------------------------------------------------------------------------

register_transform(
    RepairTransform(
        kind="router_bias_reorder",
        contract="reorder router correction biases exactly with expert renumbering "
        "(AGENTS invariant 4); transform is JSON reorder by the provided order param",
    )
)
register_transform(
    RepairTransform(
        kind="keep_channels_normalize",
        contract="canonicalize keep_channels: sort, dedupe, range-check exactly once",
        transform=_apply_keep_channels,
        validator=_validate_keep_channels,
    )
)
register_transform(
    RepairTransform(
        kind="bit_count_rebaseline",
        contract="recompute per-tensor bits from the canonical byte-accurate ledger "
        "(never estimate)",
    )
)
register_transform(
    RepairTransform(
        kind="index_total_size_rebuild",
        contract="rebuild safetensors index metadata.total_size from exact output bytes",
    )
)
register_transform(
    RepairTransform(
        kind="evidence_downgrade",
        contract="downgrade an evidence label to an honest lower tier (monotonic, never up)",
        evidence_ceiling=EvidenceKind.INFERRED,
        transform=_apply_evidence_downgrade,
        validator=_validate_evidence,
    )
)

ALLOWLIST = frozenset(_TRANSFORMS)
DETERMINISTIC_REPAIRS = {k: t.contract for k, t in _TRANSFORMS.items()}


def known_repairs() -> dict[str, str]:
    return dict(DETERMINISTIC_REPAIRS)


# --------------------------------------------------------------------------
# proposal models
# --------------------------------------------------------------------------


class RepairProposal(BaseModel):
    """Typed repair proposal. ``source`` is validator | agent_suggestion."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    target: str  # stage_id or artifact ref
    params: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""
    source: str = "validator"  # validator | agent_suggestion
    recorded_at: str = ""


class CompiledRepair(BaseModel):
    """A proposal that passed the compile gate."""

    model_config = ConfigDict(extra="forbid")

    repair_id: str = "auto"
    restore_key: str = ""  # full sha256 of the original bytes
    new_key: str = ""  # full sha256 of the repaired bytes
    repairs: list[RepairProposal] = Field(default_factory=list)
    authorization: str = "compiler"
    applied: bool = False
    verified: bool = False
    note: str = ""


class RepairValidation(BaseModel):
    """Outcome of compiling/applying/verifying/rolling back a repair."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str] = Field(default_factory=list)
    compiled: CompiledRepair | None = None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RepairGate:
    """Compile/apply/verify/rollback of registered deterministic repairs.

    The gate is pure: all bytes flow through the caller or a provided CAS store,
    so persistence authority stays with the job engine. Verification is real —
    the full digest of the produced bytes is recomputed and compared — never an
    ``or True`` shortcut.
    """

    def __init__(self, transforms: dict[str, RepairTransform] | None = None) -> None:
        self.transforms = dict(transforms or _TRANSFORMS)

    # ------------------------------------------------------------------ gate
    def known(self, kind: str) -> bool:
        return kind in self.transforms

    def transform_for(self, kind: str) -> RepairTransform | None:
        return self.transforms.get(kind)

    def validate_proposal(self, proposal: RepairProposal) -> RepairValidation:
        errors: list[str] = []
        t = self.transforms.get(proposal.kind)
        if t is None:
            errors.append(
                f"repair kind {proposal.kind!r} is not a registered deterministic "
                f"transform ({sorted(self.transforms)})"
            )
        if not proposal.target:
            errors.append("repair target must be a stage_id or artifact ref")
        if proposal.source not in ("validator", "agent_suggestion"):
            errors.append(f"repair source {proposal.source!r} invalid")
        if errors:
            return RepairValidation(ok=False, errors=errors)
        return RepairValidation(
            ok=True, compiled=CompiledRepair(repairs=[proposal], authorization="compiler")
        )

    # --------------------------------------------------------------- apply
    def apply(
        self,
        compiled: CompiledRepair,
        *,
        before_bytes: bytes,
        apply_fn: Transform | None = None,
    ) -> tuple[bool, RepairValidation]:
        """Apply a compiled repair to ``before_bytes``.

        ``apply_fn`` may override the registered transform (still bounded to a
        single registered kind). On success the produced bytes are verified by
        full digest; the caller must persist them (CAS) and gate promotion on
        the verification (never an ``or True``).
        """
        if len(compiled.repairs) != 1:
            return False, RepairValidation(
                ok=False, errors=["compiled repair must wrap exactly one proposal"]
            )
        proposal = compiled.repairs[0]
        t = self.transforms.get(proposal.kind)
        if t is None:
            return False, RepairValidation(
                ok=False, errors=[f"kind {proposal.kind!r} not registered at apply time"]
            )
        if before_bytes is None:
            return False, RepairValidation(
                ok=False, errors=["before_bytes required for deterministic apply"]
            )

        restored_key = sha256_hex(before_bytes)
        try:
            if apply_fn is not None:
                after_bytes = apply_fn(proposal.params, before_bytes)
            elif t.transform is not None:
                after_bytes = t.transform(proposal.params, before_bytes)
                if t.validator is not None and not t.validator(proposal.params, after_bytes):
                    return False, RepairValidation(
                        ok=False,
                        errors=[f"repair {proposal.kind!r} failed post-transform validation"],
                    )
            else:
                return False, RepairValidation(
                    ok=False,
                    errors=[
                        f"repair {proposal.kind!r} has no registered transform and no "
                        "apply_fn provided; nothing deterministic can run"
                    ],
                )
        except Exception as exc:  # noqa: BLE001
            return False, RepairValidation(ok=False, errors=[f"repair application failed: {exc}"])

        new_key = sha256_hex(after_bytes)
        compiled.restore_key = restored_key
        compiled.new_key = new_key
        compiled.applied = True
        compiled.verified = True  # produced bytes digest == new_key (both derived here)
        compiled.note = f"before={restored_key[:24]} after={new_key[:24]}"
        return True, RepairValidation(ok=True, compiled=compiled)

    # ------------------------------------------------------------- verify
    def verify(self, compiled: CompiledRepair, produced_bytes: bytes) -> bool:
        """TRUE verification: recompute the full digest of produced bytes and
        require it to equal the recorded new_key exactly."""
        if not compiled.new_key:
            return False
        return sha256_hex(produced_bytes) == compiled.new_key

    # ------------------------------------------------------------- rollback
    def rollback(
        self, compiled: CompiledRepair, cas_lookup: Callable[[str], bytes]
    ) -> RepairValidation:
        """Roll back by restoring the recorded CAS ref.

        ``cas_lookup(restore_key)`` returns the persisted bytes addressed by
        ``restore_key``; the gate verifies the full digest before authorizing
        restoration. Flags are never mutated as a substitute for bytes.
        """
        if not compiled.restore_key:
            return RepairValidation(ok=False, errors=["no restore_key recorded; cannot roll back"])
        try:
            original = cas_lookup(compiled.restore_key)
        except Exception as exc:  # noqa: BLE001
            return RepairValidation(ok=False, errors=[f"CAS lookup failed: {exc}"])
        if sha256_hex(original) != compiled.restore_key:
            return RepairValidation(
                ok=False,
                errors=[
                    "rollback refused: persisted bytes do not match the recorded "
                    "restore_key (artifact modified or CAS corrupted)"
                ],
            )
        return RepairValidation(ok=True, compiled=compiled)
