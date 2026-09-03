"""Census engine: enumerate every tensor of an architecture into an ownership
manifest, layerwise, with source identity preserved. Reports
`needs_source_measurement` when the architecture carries no deterministic
tensor sizes (real checkpoints) rather than fabricating numbers.

A role is emitted only when the architecture declares a numel for it, so the
census stays model-agnostic (a dense model simply omits the MoE/MLA/KDA roles).
"""

from __future__ import annotations

from cebu_profiler.census.tensor_ownership import (
    OwnershipManifest,
    PhysicalLocation,
    PlacementPolicy,
    TensorOwnership,
)
from cebu_profiler.schemas.architecture import (
    ArchitectureSpec,
    DType,
    LayerKind,
    TensorRole,
)

# Single-tensor per-layer roles (emitted only if declared in tensor_params).
_PER_LAYER_ROLES: tuple[TensorRole, ...] = (
    TensorRole.ROUTER,
    TensorRole.ROUTER_BIAS,
    TensorRole.LATENT_PROJ,
    TensorRole.ATTENTION,
    TensorRole.MLA_STATE,
    TensorRole.KDA_DECAY,
    TensorRole.NORM,
)

# Single-tensor global roles (emitted once, if declared).
_GLOBAL_ROLES: tuple[TensorRole, ...] = (TensorRole.EMBEDDING, TensorRole.LM_HEAD)


def _layer_kind_index(arch: ArchitectureSpec) -> list[LayerKind]:
    """Ordered per-layer kind list (first N of a kind then the next, as declared)."""
    kinds: list[LayerKind] = []
    for kind, count in arch.layers_by_kind.items():
        kinds.extend([kind] * count)
    if not kinds:
        kinds = [LayerKind.KDA] * arch.num_text_layers
    return (kinds + [LayerKind.KDA] * arch.num_text_layers)[: arch.num_text_layers]


def _place(
    policy: PlacementPolicy, *, layer_index: int, expert_index: int | None
) -> PhysicalLocation:
    """Assign a physical home under the given placement policy."""
    if expert_index is not None:
        # Split routed experts across the two nodes (expert parallel).
        return PhysicalLocation.NODE_A if expert_index % 2 == 0 else PhysicalLocation.NODE_B
    if policy == PlacementPolicy.REPLICATE_SHARED_ON_A:
        # Non-expert per-layer + shared tensors live on node A under default policy.
        return PhysicalLocation.NODE_A
    return PhysicalLocation.NODE_A if layer_index % 2 == 0 else PhysicalLocation.NODE_B


def _make_tensor(
    arch: ArchitectureSpec,
    role: TensorRole,
    *,
    dtype: DType,
    layer_index: int,
    expert_index: int | None,
    policy: PlacementPolicy,
) -> TensorOwnership:
    numel = arch.tensor_params[role]
    suffix = f".{expert_index}" if expert_index is not None else ""
    key = f"{arch.name}.{role.value}.layer{layer_index}{suffix}"
    return TensorOwnership(
        key=key,
        role=role,
        dtype=dtype,
        numel=numel,
        layer_index=layer_index,
        expert_index=expert_index,
        location=_place(policy, layer_index=layer_index, expert_index=expert_index),
    )


def build_manifest(
    arch: ArchitectureSpec,
    *,
    policy: PlacementPolicy = PlacementPolicy.REPLICATE_SHARED_ON_A,
) -> OwnershipManifest:
    """Enumerate all tensors of `arch` into an ownership manifest."""
    if arch.needs_source_measurement:
        return OwnershipManifest(
            architecture=arch.name,
            records=[],
            status="needs_source_measurement",
        )

    kinds = _layer_kind_index(arch)
    records: list[TensorOwnership] = []
    dense_dtype, expert_dtype = arch.moe.dense_dtype, arch.moe.expert_dtype

    for i, _kind in enumerate(kinds):
        for role in _PER_LAYER_ROLES:
            if role not in arch.tensor_params:
                continue
            dtype = expert_dtype if role == TensorRole.LATENT_PROJ else dense_dtype
            records.append(
                _make_tensor(
                    arch, role, dtype=dtype, layer_index=i, expert_index=None, policy=policy
                )
            )
        if arch.moe.num_routed_experts > 0 and TensorRole.EXPERTS in arch.tensor_params:
            for e in range(arch.moe.num_routed_experts):
                records.append(
                    _make_tensor(
                        arch,
                        TensorRole.EXPERTS,
                        dtype=expert_dtype,
                        layer_index=i,
                        expert_index=e,
                        policy=policy,
                    )
                )
        if arch.moe.num_shared_experts > 0 and TensorRole.SHARED_EXPERT in arch.tensor_params:
            records.append(
                _make_tensor(
                    arch,
                    TensorRole.SHARED_EXPERT,
                    dtype=dense_dtype,
                    layer_index=i,
                    expert_index=None,
                    policy=policy,
                )
            )

    for role in _GLOBAL_ROLES:
        if role not in arch.tensor_params:
            continue
        records.append(
            _make_tensor(
                arch, role, dtype=dense_dtype, layer_index=0, expert_index=None, policy=policy
            )
        )

    return OwnershipManifest(
        architecture=arch.name,
        records=records,
        status="synthetic",
    )
