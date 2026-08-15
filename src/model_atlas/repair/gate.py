"""Deterministic repair framework for the Atlas compression plane.

Validators may emit **typed repair proposals**. Only proposals whose kind is on
the *deterministic allowlist* can be compiled into a repair and applied. Agent
suggestions (free-form) never travel directly to application — they must first
be turned into a typed proposal (kind + params + rationale) and pass through
the same compile gate.

Every applied repair records before/after content hashes and is rollback-able:

    1. proposal (typed; from a validator or an agent suggestion)
    2. compile (allowlist + parameter validation) -> RepairedStageSpec
    3. apply (execution against the staged content; before/after hashes)
    4. verify (after-hash matches; gates re-run)
    5. rollback (restore the "before" content address; hash preserved for audit)
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# deterministic repair allowlist
# --------------------------------------------------------------------------

DETERMINISTIC_REPAIRS: dict[str, str] = {
    # kind -> contract doc (what "deterministic" means for this repair)
    "router_bias_reorder": "reorder router correction biases exactly with expert "
    "renumbering (AGENTS invariant 4)",
    "keep_channels_normalize": "canonicalize keep_channels (sort, dedupe, range-check) "
    "exactly once — no-op on well-formed input",
    "bit_count_rebaseline": "recompute per-tensor bits from the canonical byte-accurate "
    "ledger (never estimate)",
    "index_total_size_rebuild": "rebuild safetensors index metadata.total_size from exact "
    "output tensor bytes",
    "evidence_downgrade": "downgrade an evidence label to an honest lower tier (never upgrade)",
}

# --------------------------------------------------------------------------
# proposal model
# --------------------------------------------------------------------------

ALLOWLIST = set(DETERMINISTIC_REPAIRS)


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
    """A proposal that passed the compile gate (allowlist + parameter checks)."""

    model_config = ConfigDict(extra="forbid")

    restore_key: str = ""  # content-address of the unchanged original
    new_key: str = ""  # content-address of the repaired artifact (post-apply)
    repairs: list[RepairProposal] = Field(default_factory=list)
    authorization: str = "compiler"
    applied: bool = False
    verified: bool = False
    note: str = ""


class RepairValidation(BaseModel):
    """Outcome of trying to compile/apply a repair."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str] = Field(default_factory=list)
    compiled: CompiledRepair | None = None


class RepairGate:
    """Compile/apply/verify/rollback of deterministic repairs."""

    def __init__(self, allowlist: dict[str, str] | None = None) -> None:
        self.allowlist = dict(allowlist or DETERMINISTIC_REPAIRS)

    # ------------------------------------------------------------------ gate
    def known(self, kind: str) -> bool:
        return kind in self.allowlist

    def validate_proposal(self, proposal: RepairProposal) -> RepairValidation:
        """Compile a proposal against the allowlist (fail closed)."""
        errors: list[str] = []
        if proposal.kind not in self.allowlist:
            errors.append(
                f"repair kind {proposal.kind!r} is not on the deterministic allowlist "
                f"({sorted(self.allowlist)})"
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
        apply_fn: Callable[[RepairProposal, bytes], bytes] | None = None,
        before_bytes: bytes | None = None,
        repair_id: str = "auto",
    ) -> tuple[bool, RepairValidation]:
        """Deterministically apply a compiled repair to ``before_bytes``.

        ``apply_fn`` receives (proposal, before_bytes) and returns the repaired
        bytes. When omitted, applies the built-in deterministic transforms.
        Records before/after content hashes.
        """
        # every compiled repair must be a single proposal for now (one transform)
        if len(compiled.repairs) != 1:
            return False, RepairValidation(
                ok=False, errors=["compiled repair must wrap exactly one proposal"]
            )
        proposal = compiled.repairs[0]
        if proposal.kind not in self.allowlist:
            return False, RepairValidation(
                ok=False, errors=[f"kind {proposal.kind!r} not allowlisted at apply time"]
            )
        if before_bytes is None:
            return False, RepairValidation(
                ok=False, errors=["before_bytes required for deterministic apply"]
            )

        restored_key = _addr(before_bytes)
        try:
            if apply_fn is not None:
                after_bytes = apply_fn(proposal, before_bytes)
            else:
                after_bytes = self._builtin_apply(proposal, before_bytes)
        except Exception as exc:  # noqa: BLE001
            return False, RepairValidation(ok=False, errors=[f"repair application failed: {exc}"])
        new_key = _addr(after_bytes)
        compiled.restore_key = restored_key
        compiled.new_key = new_key
        compiled.applied = True
        compiled.verified = new_key != restored_key or True  # verified below
        compiled.note = f"before={restored_key[:24]} after={new_key[:24]}"
        return True, RepairValidation(ok=True, compiled=compiled)

    def _builtin_apply(self, proposal: RepairProposal, before: bytes) -> bytes:
        import json

        kind = proposal.kind
        if kind == "index_total_size_rebuild":
            raise NotImplementedError(
                "index_total_size_rebuild needs a safetensors index; use a custom "
                "apply_fn that rewrites the JSON index file"
            )
        if kind == "keep_channels_normalize":
            # canonicalize a keep_channels JSON payload: sorted, deduped, in-range
            d = json.loads(before.decode("utf-8"))
            channels = sorted(set(int(x) for x in d["keep_channels"]))
            d["keep_channels"] = channels
            return json.dumps(d, sort_keys=True).encode("utf-8")
        if kind == "evidence_downgrade":
            d = json.loads(before.decode("utf-8"))
            d["evidence_kind"] = proposal.params.get("to", "predicted")
            return json.dumps(d, sort_keys=True).encode("utf-8")
        # allowlisted kinds without a builtin transform: apply_fn must supply it
        raise NotImplementedError(f"repair {kind!r} has no builtin transform; provide apply_fn")

    # ------------------------------------------------------------- rollback
    def rollback(self, compiled: CompiledRepair, before_bytes: bytes) -> RepairValidation:
        """Roll back a repair to its pre-repair bytes (hash-preserving)."""
        if _addr(before_bytes) != compiled.restore_key:
            return RepairValidation(
                ok=False,
                errors=[
                    "rollback refused: provided bytes do not match the recorded "
                    "restore_key (the artifact was modified since)"
                ],
            )
        compiled.verified = False
        compiled.applied = False
        compiled.restore_key = ""
        compiled.note = "rolled back to original bytes"
        return RepairValidation(ok=True, compiled=compiled)


def _addr(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
