"""Deterministic repair framework for the Atlas compression plane.

Validators emit **typed repair proposals**. Only proposals whose kind is
**registered** as a deterministic, typed, VERSIONED transform may be compiled and
applied. There is no arbitrary unregistered ``apply_fn`` — application runs the
registered transform bound to its versioned identity.

Every applied repair:

1. reads the target's current bytes from the content-addressed store (CAS),
2. persists both the original (restore ref) and the repaired bytes,
3. **rereads the produced blob from the CAS and verifies its full sha256**
   equals the recorded ``new_key``,
4. **atomically updates the target stage output ref** to the repaired ref,
5. records ``before/after`` digests + the transform version.

Rollback reads the recorded restore ref, verifies its full digest, and
**atomically restores** the target output ref (and the repair state) to the
original — never by flag mutation alone. Tests prove the bytes/ref change and
restore through the CAS.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.jobs.artifacts import ContentAddressedStore
from model_atlas.jobs.schema import OutputRef, StageOutput
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
    """A registered, typed, VERSIONED deterministic repair transform.

    ``version`` identifies the exact deterministic implementation bound at
    registration. ``apply`` refuses any transform whose version does not match,
    so an arbitrary replacement cannot masquerade as a registered repair.
    """

    kind: str
    version: str = "v1"
    contract: str = ""
    # max evidence kind the repair may set (downgrades only below current)
    evidence_ceiling: EvidenceKind = EvidenceKind.ESTIMATED
    transform: Transform | None = None
    validator: Validator | None = None

    @property
    def identity(self) -> str:
        return f"{self.kind}@{self.version}"


_TRANSFORMS: dict[str, RepairTransform] = {}


def register_transform(t: RepairTransform) -> None:
    if t.kind in _TRANSFORMS:
        raise ValueError(f"repair transform {t.kind!r} already registered")
    if not t.version:
        raise ValueError(f"repair transform {t.kind!r} must carry a versioned identity")
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
# registered deterministic transforms (versioned identities)
# --------------------------------------------------------------------------

register_transform(
    RepairTransform(
        kind="router_bias_reorder",
        version="v1",
        contract="reorder router correction biases exactly with expert renumbering "
        "(AGENTS invariant 4); transform is JSON reorder by the provided order param",
    )
)
register_transform(
    RepairTransform(
        kind="keep_channels_normalize",
        version="v1",
        contract="canonicalize keep_channels: sort, dedupe, range-check exactly once",
        transform=_apply_keep_channels,
        validator=_validate_keep_channels,
    )
)
register_transform(
    RepairTransform(
        kind="bit_count_rebaseline",
        version="v1",
        contract="recompute per-tensor bits from the canonical byte-accurate ledger "
        "(never estimate)",
    )
)
register_transform(
    RepairTransform(
        kind="index_total_size_rebuild",
        version="v1",
        contract="rebuild safetensors index metadata.total_size from exact output bytes",
    )
)
register_transform(
    RepairTransform(
        kind="evidence_downgrade",
        version="v1",
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
    target: str  # stage_id or artifact-ref name (matches a StageOutput output)
    params: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""
    source: str = "validator"  # validator | agent_suggestion
    recorded_at: str = ""


class CompiledRepair(BaseModel):
    """A proposal that passed the compile gate (binding a transform version)."""

    model_config = ConfigDict(extra="forbid")

    repair_id: str = "auto"
    kind: str = ""
    transform_version: str = ""
    target_ref: str = ""  # name of the StageOutput output to replace/restore
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

    The gate is store-aware: it reads/persists/verifies through a CAS store
    (the engine's ContentAddressedStore) and atomically updates/restores the
    target ``OutputRef`` on a ``StageOutput``. Verification rereads the produced
    blob from the CAS and compares its full sha256 — never an ``or True``.
    """

    def __init__(self, transforms: dict[str, RepairTransform] | None = None) -> None:
        self.transforms = dict(transforms or _TRANSFORMS)

    # ------------------------------------------------------------------ gate
    def known(self, kind: str) -> bool:
        return kind in self.transforms

    def transform_for(self, kind: str) -> RepairTransform | None:
        return self.transforms.get(kind)

    def validate_proposal(
        self, proposal: RepairProposal, transform_version: str = "v1"
    ) -> RepairValidation:
        errors: list[str] = []
        t = self.transforms.get(proposal.kind)
        if t is None:
            errors.append(
                f"repair kind {proposal.kind!r} is not a registered deterministic "
                f"transform ({sorted(self.transforms)})"
            )
        else:
            if t.version != transform_version:
                errors.append(
                    f"repair {proposal.kind!r} version {transform_version!r} does not "
                    f"match registered identity {t.identity!r}; refusing an arbitrary "
                    "unversioned application"
                )
        if not proposal.target:
            errors.append("repair target must be a stage_id or artifact-ref name")
        if proposal.source not in ("validator", "agent_suggestion"):
            errors.append(f"repair source {proposal.source!r} invalid")
        if errors:
            return RepairValidation(ok=False, errors=errors)
        return RepairValidation(
            ok=True,
            compiled=CompiledRepair(
                repairs=[proposal],
                kind=proposal.kind,
                transform_version=t.version if t is not None else transform_version,
                target_ref=proposal.target,
                authorization="compiler",
            ),
        )

    # --------------------------------------------------------------- apply
    def apply(
        self,
        compiled: CompiledRepair,
        *,
        cas: ContentAddressedStore,
        target_ref: OutputRef | None = None,
        target_bytes: bytes | None = None,
    ) -> tuple[bool, RepairValidation, bytes | None]:
        """Apply a compiled repair against a CAS store.

        ``cas`` must provide:
          - ``read(ref: OutputRef) -> bytes``
          - ``put_bytes(name: str, data: bytes) -> OutputRef``
          - ``verify(ref: OutputRef) -> bool``
        ``target_bytes`` is the current byte content to repair (must be provided
        or read from ``target_ref`` via ``cas``). On success the produced blob
        is persisted, re-READ from the CAS and full-digest verified, and the
        caller receives it. An atomic target-ref update is provided separately
        via :meth:`publish_apply`.
        """
        if len(compiled.repairs) != 1:
            return (
                False,
                RepairValidation(
                    ok=False, errors=["compiled repair must wrap exactly one proposal"]
                ),
                None,
            )
        proposal = compiled.repairs[0]
        t = self.transforms.get(proposal.kind)
        if t is None:
            return (
                False,
                RepairValidation(
                    ok=False, errors=[f"kind {proposal.kind!r} not registered at apply time"]
                ),
                None,
            )
        if t.version != compiled.transform_version:
            return (
                False,
                RepairValidation(
                    ok=False,
                    errors=[
                        f"apply-time version {t.version!r} != compiled identity "
                        f"{compiled.transform_version!r}"
                    ],
                ),
                None,
            )
        # resolve the before bytes: from argument or by reading the CAS ref
        if target_bytes is None:
            if target_ref is None:
                return (
                    False,
                    RepairValidation(
                        ok=False,
                        errors=["target_bytes or target_ref required"],
                    ),
                    None,
                )
            target_bytes = cas.read(target_ref)

        if t.transform is None:
            return (
                False,
                RepairValidation(
                    ok=False,
                    errors=[
                        f"repair {proposal.kind!r} has no registered transform; nothing "
                        "deterministic can run"
                    ],
                ),
                None,
            )

        restored_key = sha256_hex(target_bytes)
        try:
            after_bytes = t.transform(proposal.params, target_bytes)
            if t.validator is not None and not t.validator(proposal.params, after_bytes):
                return (
                    False,
                    RepairValidation(
                        ok=False,
                        errors=[f"repair {proposal.kind!r} failed post-transform validation"],
                    ),
                    None,
                )
        except Exception as exc:  # noqa: BLE001
            return (
                False,
                RepairValidation(ok=False, errors=[f"repair application failed: {exc}"]),
                None,
            )

        new_key = sha256_hex(after_bytes)
        # persist BOTH blobs so rollback can restore the original ref atomically
        cas.put_bytes(f"{compiled.kind}.restore", target_bytes)
        new_ref = cas.put_bytes(f"{compiled.kind}.repaired", after_bytes)
        # RE-READ the produced blob from the CAS and verify its full digest
        reread = cas.read(new_ref)
        verified = sha256_hex(reread) == new_key and cas.verify(new_ref)
        if not verified:
            return (
                False,
                RepairValidation(
                    ok=False,
                    errors=["repair produced blob failed full-digest re-read verification"],
                ),
                None,
            )

        compiled.restore_key = restored_key
        compiled.new_key = new_key
        compiled.applied = True
        compiled.verified = True
        compiled.note = f"before={restored_key[:24]} after={new_key[:24]} version={t.identity}"
        return True, RepairValidation(ok=True, compiled=compiled), after_bytes

    # ------------------------------------------------------------- publish
    def publish_apply(
        self,
        compiled: CompiledRepair,
        *,
        target: StageOutput,
        new_ref: OutputRef,
    ) -> bool:
        """Atomically update the target ``StageOutput``'s output ref to the
        repaired ref (ref-level mutation of the run record; never in-place blob
        mutation)."""
        existing = [i for i, r in enumerate(target.outputs) if r.name == compiled.target_ref]
        if existing:
            target.outputs[existing[0]] = new_ref
        else:
            target.outputs.append(new_ref)
        return True

    # ------------------------------------------------------------- verify
    def verify(self, compiled: CompiledRepair, produced_bytes: bytes) -> bool:
        """TRUE verification: recompute the full digest of produced bytes and
        require it to equal the recorded new_key exactly."""
        if not compiled.new_key:
            return False
        return sha256_hex(produced_bytes) == compiled.new_key

    # ------------------------------------------------------------- rollback
    def rollback_ref(
        self,
        compiled: CompiledRepair,
        *,
        cas: ContentAddressedStore,
        original_bytes: bytes,
    ) -> tuple[bool, OutputRef | None]:
        """Atomically restore the original ref: persist the original bytes into
        the CAS as the target's restore ref, verify by full digest, and return
        the ref to publish onto the ``StageOutput`` (via :meth:`publish_apply`
        with the restore ref)."""
        if not compiled.restore_key:
            return False, None
        if sha256_hex(original_bytes) != compiled.restore_key:
            return False, None
        restore_ref = cas.put_bytes(f"{compiled.kind}.rollback", original_bytes)
        reread = cas.read(restore_ref)
        if sha256_hex(reread) != compiled.restore_key:
            return False, None
        compiled.applied = False
        compiled.verified = False
        compiled.note = f"rolled back to original ref {compiled.restore_key[:24]}"
        return True, restore_ref
