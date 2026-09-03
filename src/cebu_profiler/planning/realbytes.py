"""Real-bytes derivative-candidate planner (blueprint §24/§25, F10).

Turns a *measured* checkpoint manifest (metadata-first census; no tensor bodies,
no torch, no GPU) into derivative candidate plans at the target resident
envelopes (default 190 / 210 / 225 GB per the blueprint §3).

Every byte figure is scaled from the checkpoint's **measured per-tensor
``byte_size``** — never from ``numel × dtype_bytes`` — because the source is
NVFP4, whose routed experts are stored at ~8.19 bpw, so dtype-derived
accounting would overstate them. Measured facts (current per-role bytes and
achieved bpw) are kept distinct from projected foot-prints at lower target
bpw / retained fractions, which are tagged ``estimated`` (v2 §31:20: never
present a projection as a measurement).

Key structural fact this produces: for a fixed envelope, pruning retained
experts (lower ``keep_frac``) buys *higher* mean expert bpw for the survivors —
compression via pruning costs less fidelity per retained weight than uniform
low-bit of every expert. That is the measured-side echo of the Milestone E /
FlexMoE heterogeneous-width thesis, now grounded in real bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_atlas.checkpoint.classifier import classify_tensor
from model_atlas.checkpoint.source_manifest import CheckpointManifest

GIB = 1024**3

# Supported (low-bit) formats this planner will realocate to, in bpw.
# Labels are the deployment format names; 8.19 is the source NVFP4 expert bpw
# (kept so an "unchanged experts / only prune" arm is comparable).
SUPPORTED_BPW: tuple[tuple[float, str], ...] = (
    (4.0, "int4"),
    (4.5, "nvfp4"),
    (5.5, "fp8"),
    (8.19, "source-nvfp4"),
)


@dataclass
class RoleAccount:
    role: str
    tensor_count: int
    stored_bytes: float
    achieved_bpw: float | None


@dataclass
class RealAccount:
    """Measured (metadata-only) byte account of a real checkpoint manifest."""

    source_dir: str
    total_bytes: float
    backbone_bytes: float  # measured 16 bpw roles (attention/embed/head/shared/norm/latent)
    backbone_achieved_bpw: float
    expert_bytes: float  # measured routed-expert bytes
    expert_achieved_bpw: float
    n_experts: int
    by_role: dict[str, RoleAccount] = field(default_factory=dict)

    def expert_gib(self) -> float:
        return self.expert_bytes / GIB

    def backbone_gib(self) -> float:
        return self.backbone_bytes / GIB


def account_manifest(manifest: CheckpointManifest) -> RealAccount:
    """Aggregate measured per-tensor bytes into backbone / routed-expert buckets.

    A role is "the routed expert bank" when it classifies as EXPERTS; a role is
    "backbone" when its achieved bpw is clearly BF16 (>12, i.e. ~16 bpw). Any
    residual role is reported in ``by_role`` but kept out of both buckets so
    nothing is silently miscounted.
    """
    per_role: dict[str, dict[str, float | int]] = {}
    for t in manifest.tensors:
        role = classify_tensor(t.name)
        key = role.role.value if role and role.role else "unclassified"
        row = per_role.setdefault(key, {"bytes": 0.0, "numel": 0, "count": 0})
        row["bytes"] += t.byte_size
        row["numel"] += t.numel
        row["count"] += 1

    by_role: dict[str, RoleAccount] = {}
    backbone = 0.0
    expert = 0.0
    for key, row in per_role.items():
        b = float(row["bytes"])
        numel = int(row["numel"])
        bpw = (b * 8) / numel if numel else None
        by_role[key] = RoleAccount(
            role=key,
            tensor_count=int(row["count"]),
            stored_bytes=b,
            achieved_bpw=bpw,
        )
        if key == "experts":
            expert += b
        elif bpw is not None and bpw > 12.0:
            backbone += b

    expert_bpw = by_role["experts"].achieved_bpw if "experts" in by_role else None
    return RealAccount(
        source_dir=manifest.checkpoint_dir,
        total_bytes=float(manifest.total_bytes),
        backbone_bytes=backbone,
        backbone_achieved_bpw=16.0,
        expert_bytes=expert,
        expert_achieved_bpw=expert_bpw or 0.0,
        n_experts=int(by_role.get("experts", RoleAccount("experts", 0, 0.0, None)).tensor_count),
        by_role=by_role,
    )


@dataclass
class RealCandidate:
    envelope_gb: float
    keep_frac: float  # fraction of experts retained (ESTIMATED — needs routing census)
    backbone_bpw: float
    mean_expert_bpw: float
    expert_precision: str
    stored_bytes: float
    resident_a_bytes: float
    resident_b_bytes: float
    risk: str
    estimated: bool = True  # bpw / coverage are projections, not measurements

    def stored_gib(self) -> float:
        return self.stored_bytes / GIB

    def resident_a_gib(self) -> float:
        return self.resident_a_bytes / GIB

    def resident_b_gib(self) -> float:
        return self.resident_b_bytes / GIB


def _best_expert_point(
    account: RealAccount,
    env_gib: float,
    backbone_bpw: float,
    keep_frac: float,
) -> tuple[float, float, str] | None:
    """Best (stored_expert_bytes, mean_expert_bpw, precision) fitting the envelope.

    Experts are scaled by ``keep_frac`` and by the chosen bpw relative to the
    measured source bpw. Returns None when even the smallest supported bpw at
    this keep fraction exceeds the envelope (needs deeper pruning or <4 bpw).
    """
    env_bytes = env_gib * GIB
    backbone_after = account.backbone_bytes * (backbone_bpw / account.backbone_achieved_bpw)
    budget_experts = env_bytes - backbone_after
    if budget_experts <= 0:
        return None
    for bpw, label in sorted(SUPPORTED_BPW, reverse=True):
        exp_bytes = account.expert_bytes * (bpw / account.expert_achieved_bpw) * keep_frac
        if budget_experts - exp_bytes >= -1e-6:  # highest bpw that fits (least damage)
            return exp_bytes, bpw, label
    return None


def plan_candidates(
    account: RealAccount,
    envelopes: tuple[float, ...] = (190.0, 210.0, 225.0),
    keep_fracs: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    backbone_bpw: float = 8.19,
) -> list[RealCandidate]:
    """One recommended candidate per envelope (least precision loss that fits).

    For each envelope every (backbone_bpw, keep_frac) combo is scored; the
    chosen operating point is the one with the *highest* mean expert bpw (least
    precision damage), tie-broken by higher keep_frac. Envelopes with no
    feasible combo among the supported bpw / keep fractions produce an
    infeasible candidate whose ``risk`` explains the shortfall.
    """
    candidates: list[RealCandidate] = []
    for env_gib in envelopes:
        best: RealCandidate | None = None
        infeasible: RealCandidate | None = None
        for k in keep_fracs:
            pt = _best_expert_point(account, env_gib, backbone_bpw, k)
            if pt is None:
                # how far below the smallest supported bpw we would still need to
                # go at full retention, to saturate the envelope
                env_bytes = env_gib * GIB
                backbone_after = account.backbone_bytes * (
                    backbone_bpw / account.backbone_achieved_bpw
                )
                shortest = (
                    (env_bytes - backbone_after) * account.expert_achieved_bpw
                    / account.expert_bytes
                )
                if infeasible is None:
                    infeasible = RealCandidate(
                        envelope_gb=env_gib,
                        keep_frac=1.0,
                        backbone_bpw=backbone_bpw,
                        mean_expert_bpw=round(shortest, 3),
                        expert_precision="<int4",
                        stored_bytes=env_bytes,
                        resident_a_bytes=env_bytes,
                        resident_b_bytes=0.0,
                        risk=(
                            f"infeasible even at 4.0 bpw and full retention: needs "
                            f"mean expert ~{shortest:.2f} bpw -> more pruning or <4-bit"
                        ),
                    )
                continue
            exp_bytes, bpw, label = pt
            backbone_after = account.backbone_bytes * (backbone_bpw / account.backbone_achieved_bpw)
            total = backbone_after + exp_bytes
            resident_a = backbone_after + exp_bytes / 2.0
            resident_b = exp_bytes / 2.0
            cand = RealCandidate(
                envelope_gb=env_gib,
                keep_frac=k,
                backbone_bpw=backbone_bpw,
                mean_expert_bpw=bpw,
                expert_precision=label,
                stored_bytes=total,
                resident_a_bytes=resident_a,
                resident_b_bytes=resident_b,
                risk="",
            )
            if best is None or (
                cand.mean_expert_bpw > best.mean_expert_bpw
                or (
                    cand.mean_expert_bpw == best.mean_expert_bpw
                    and cand.keep_frac > best.keep_frac
                )
            ):
                best = cand
        chosen = best if best is not None else infeasible
        if chosen is None:  # pragma: no cover - unreachable, defensive
            chosen = RealCandidate(
                envelope_gb=env_gib,
                keep_frac=1.0,
                backbone_bpw=backbone_bpw,
                mean_expert_bpw=account.expert_achieved_bpw,
                expert_precision="source-nvfp4",
                stored_bytes=0.0,
                resident_a_bytes=0.0,
                resident_b_bytes=0.0,
                risk="planning produced no feasible point",
            )
        candidates.append(chosen)
    return candidates


def report(candidates: list[RealCandidate], account: RealAccount) -> str:
    """Human-readable per-candidate report (§25 per-candidate report)."""
    lines = [
        f"source: {account.source_dir}",
        f"measured total: {account.total_bytes / GIB:.1f} GiB  "
        f"(backbone {account.backbone_gib():.1f} GiB @ {account.backbone_achieved_bpw} bpw, "
        f"experts {account.expert_gib():.1f} GiB @ {account.expert_achieved_bpw:.2f} bpw)",
        "",
    ]
    for c in candidates:
        lines.append(
            f"envelope {c.envelope_gb:.0f} GiB | keep {c.keep_frac:.0%} of experts | "
            f"experts -> {c.expert_precision} {c.mean_expert_bpw:.2f} bpw | "
            f"backbone -> {c.backbone_bpw:.2f} bpw | "
            f"stored {c.stored_gib():.1f} GiB | "
            f"resident A {c.resident_a_gib():.1f} / B {c.resident_b_gib():.1f} GiB"
        )
        if c.risk:
            lines.append(f"    RISK: {c.risk}")
        elif c.estimated:
            lines.append(
                "    estimated: bpw + retention are projections; real keep-map "
                "needs a routing census (offline inference is not available)"
            )
    return "\n".join(lines)
