"""KV memory optimizer (v3 %10 / blueprint §3.2).

Memory fit = weights + correction payload + runtime + graphs + MTP + comm + OS
reserve + KV + safety headroom. This module evaluates KV precision alternatives
(FP8 first, then experimental NVFP4/transform/VQ) against a per-rank memory
budget and decides how much safe context a given KV format can buy. It treats KV
as part of the SAME global budget: never buy context by damaging weights if a
safer KV method yields a better global Pareto point.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from model_atlas.schemas.evidence import EvidenceKind

GIB = 1024**3

# KV bytes per token per layer (hidden dim * 2 key+value * bytes-per-value)
KV_FORMATS: tuple[tuple[str, float], ...] = (
    ("fp8", 1.0),
    ("bf16", 2.0),
    ("nvfp4", 0.5),
    ("int8", 1.0),
    ("vq", 0.5),  # vector-quantized (experimental)
)


class KvOption(BaseModel):
    """One KV-precision option and its context capacity."""

    model_config = ConfigDict(extra="forbid")

    format: str
    bytes_per_value: float = Field(ge=0.0)
    max_safe_context_tokens: int = Field(ge=0)
    kv_bytes: float = Field(ge=0.0)
    evidence_kind: EvidenceKind = EvidenceKind.ESTIMATED


class KvBudgetResult(BaseModel):
    """Recommended KV format under a per-rank memory ledger."""

    model_config = ConfigDict(extra="forbid")

    headroom_bytes: float = Field(ge=0.0)
    context_target_tokens: int = Field(ge=0)
    kv_bytes_available: float = Field(ge=0.0)
    options: list[KvOption] = Field(default_factory=list)
    recommended_format: str = "fp8"
    reason: str = ""


class MemoryLedger(BaseModel):
    """Per-rank memory ledger (v3 %H / blueprint §8)."""

    model_config = ConfigDict(extra="forbid")

    rank: str  # node_a | node_b
    physical_bytes: float = Field(ge=0.0)
    os_reserve_bytes: float = Field(default=0.0, ge=0.0)
    weights_bytes: float = Field(default=0.0, ge=0.0)
    runtime_workspace_bytes: float = Field(default=0.0, ge=0.0)
    cuda_graphs_bytes: float = Field(default=0.0, ge=0.0)
    mtp_bytes: float = Field(default=0.0, ge=0.0)
    communication_bytes: float = Field(default=0.0, ge=0.0)
    correction_payload_bytes: float = Field(default=0.0, ge=0.0)
    libre_kv_bytes: float = Field(default=0.0, ge=0.0)  # free for KV
    safety_reserve_bytes: float = Field(default=0.0, ge=0.0)

    @property
    def free_bytes(self) -> float:
        used = (
            self.os_reserve_bytes
            + self.weights_bytes
            + self.runtime_workspace_bytes
            + self.cuda_graphs_bytes
            + self.mtp_bytes
            + self.communication_bytes
            + self.correction_payload_bytes
            + self.safety_reserve_bytes
        )
        return max(0.0, self.physical_bytes - used)

    @property
    def safe(self) -> bool:
        return self.free_bytes > 0


def kv_bytes_per_token(arch_hidden: int, n_layers: int, bytes_per_value: float) -> float:
    """Approx KV bytes per generated token (2 tensors per layer)."""
    return arch_hidden * 2 * n_layers * bytes_per_value


def plan_kv_budget(
    ledger: MemoryLedger,
    *,
    arch_hidden: int,
    n_layers: int,
    context_target_tokens: int = 32000,
) -> KvBudgetResult:
    """Find the KV format whose context capacity best meets the target."""
    headroom = ledger.free_bytes
    options: list[KvOption] = []
    for fmt, bpv in KV_FORMATS:
        per_tok = kv_bytes_per_token(arch_hidden, n_layers, bpv)
        max_ctx = int(headroom / per_tok) if per_tok > 0 else 0
        options.append(
            KvOption(
                format=fmt,
                bytes_per_value=bpv,
                max_safe_context_tokens=max_ctx,
                kv_bytes=headroom,
            )
        )
    # recommendation: format that hits the context target with the SMALLEST
    # bytes per value (fp8 first as stable baseline, then experimental)
    options_meeting = [o for o in options if o.max_safe_context_tokens >= context_target_tokens]
    if options_meeting:
        recommended = min(options_meeting, key=lambda o: o.bytes_per_value)
    else:
        recommended = max(options, key=lambda o: o.max_safe_context_tokens)
    return KvBudgetResult(
        headroom_bytes=round(headroom, 2),
        context_target_tokens=context_target_tokens,
        kv_bytes_available=round(headroom, 2),
        options=options,
        recommended_format=recommended.format,
        reason=(
            f"{recommended.format}: {recommended.max_safe_context_tokens:_} ctx tokens "
            f"fits headroom within target {context_target_tokens:_}"
            if recommended.max_safe_context_tokens >= context_target_tokens
            else (
                f"{recommended.format}: best available "
                f"{recommended.max_safe_context_tokens:_} ctx tokens (< target)"
            )
        ),
    )
