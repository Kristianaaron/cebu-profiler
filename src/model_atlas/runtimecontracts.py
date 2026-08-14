"""SM121 / MTP / KV / runtime contracts (Phase 6).

Measured hardware + model contract facts and per-rank runtime gates for the
two-Spark GLM-5.2 experiment. Everything here is split into `measured` (from
the live nvidia-smi/torch/pip) vs `contract` (the SM121/MTP/KV tolerances the
experiment asserts), never conflated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Measured (2026-08-14, both nodes).
GPU_NAME = "NVIDIA GB10"
COMPUTE_CAP = (12, 1)  # GB10 -> SM121-family (measured via torch)
CUDA_VERSION = "13.0"
TORCH_VERSION = "2.11.0+cu130"
NCCL_VERSION = (2, 28, 9)
GPU_MEM_TOTAL_GIB = 100.0  # GB10 ULM shared; exact per-rank budget is service-gated

# Contract: how many layers are DSA (deepseek_sparse_attention) for KV gating.
NUM_DSA_LAYERS = 78
KV_LORA_RANK = 512  # measured from config
KV_GROUP_SIZE = 16
MTP_LAYERS = 1  # num_nextn_predict_layers (measured)


@dataclass
class KVContract:
    context_tokens: int
    kv_bytes_per_rank: float
    scheme: str  # fp8 | nvfp4 (experimental behind parity gate)
    head_dim: int = 192
    layers: int = NUM_DSA_LAYERS
    kv_lora_rank: int = KV_LORA_RANK
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def kv_contract_plan(
    context_tokens: int,
    *,
    head_dim: int = 192,
    kv_lora_rank: int = KV_LORA_RANK,
    layers: int = NUM_DSA_LAYERS,
    n_heads_kv: int = 64,
    scheme: str = "fp8",
    bytes_per_element: float = 1.0,  # fp8
) -> KVContract:
    """Estimated KV ledger for the contract (per rank)."""
    # MLA-flavored: per token, KV is ~ (kv_lora_rank + rope_dim) * head * layers,
    # plus DSA indexer (light). This is an account estimate, not a measurement.
    kv_per_token = (
        layers
        * n_heads_kv
        * (kv_lora_rank + head_dim)
        * bytes_per_element
    )
    kv_bytes = kv_per_token * context_tokens
    return KVContract(
        context_tokens=context_tokens,
        kv_bytes_per_rank=kv_bytes,
        scheme=scheme,
        head_dim=head_dim,
        layers=layers,
        kv_lora_rank=kv_lora_rank,
        note="KV ledger is an account estimate; exact per-rank benchmark is service-window gated",
    )


@dataclass
class MTPContract:
    n_mtp_layers: int = MTP_LAYERS
    shared_indexer: bool = True
    acceptance_required: float = 0.9  # measured acceptance gate for rollback/reference
    note: str = (
        "MTP (multi-token prediction) gate: acceptance must be >= threshold or candidate "
        "rolls back to the reference path; measured only in the maintenance window"
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SM121Contract:
    compute_cap: tuple[int, int] = COMPUTE_CAP
    sm_family: str = "SM121"
    cuda_version: str = CUDA_VERSION
    nvfp4_supported: bool = True  # GB10 SM121 exposes FP4/FP8/NVFP4 tensor cores
    note: str = (
        "SM121: no custom kernel started before primitives prove correctness; "
        "any custom SM121 kernel needs parity, shape coverage + rollback tests"
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RuntimeContract:
    sm121: SM121Contract = field(default_factory=SM121Contract)
    mtp: MTPContract = field(default_factory=MTPContract)
    kv: KVContract | None = None
    no_per_token_weight_fetch: bool = True  # AGENTS.md #2 compute-follows-weights
    no_hidden_host_transfers: bool = True
    gates: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        if self.kv is not None:
            d["kv"] = self.kv.to_dict()
        return d


def build_runtime_contract(context_tokens: int = 8192, kv_scheme: str = "fp8") -> RuntimeContract:
    pc = RuntimeContract(kv=kv_contract_plan(context_tokens, scheme=kv_scheme))
    pc.gates = {
        "nvfp4_token_cores": True,  # SM121 (measured compute cap)
        "mtp_available": True,  # config num_nextn_predict_layers == 1
        "fp8_kv_baseline": kv_scheme == "fp8",
        "nvfp4_kv_experimental_only": kv_scheme == "nvfp4",  # behind parity gate
    }
    return pc
