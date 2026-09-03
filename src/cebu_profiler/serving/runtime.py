"""Two-node serving execution model (v2 §14, §5).

Simulated distributed execution over the two DGX-Spark nodes: kept derivative
experts are assigned node_a / node_b (expert parallel), each node's resident
bytes and routed-expert count are tracked from a real forward, cross-node
activation volume per token is estimated, and a go/no-go fit check enforces per
node budgets + runtime reserve. Throughput figures are ESTIMATES from the
active-bytes model, never claimed as measured tokens/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_atlas.atlas.runtime import MiniMoE, forward
from model_atlas.schemas.architecture import DTYPE_BYTES, TensorRole

BYTES_PER_ACTIVATION = 2.0  # BF16-style activation bytes for the estimate


@dataclass
class NodeAssignment:
    node_a: list[tuple[int, int]] = field(default_factory=list)  # (layer, slot)
    node_b: list[tuple[int, int]] = field(default_factory=list)

    def node_of(self, layer: int, slot: int) -> str:
        if (layer, slot) in self.node_a:
            return "node_a"
        return "node_b"


def assign_nodes(
    derivative: MiniMoE, residency: list[tuple[int, int, str]] | None = None
) -> NodeAssignment:
    """Assign each kept expert to a node (expert-parallel parity by default)."""
    a: list[tuple[int, int]] = []
    b: list[tuple[int, int]] = []
    if residency:
        for layer, slot, loc in residency:
            (a if loc == "node_a" else b).append((layer, slot))
        return NodeAssignment(node_a=sorted(a), node_b=sorted(b))
    for layer in range(derivative.arch.num_text_layers):
        for slot in range(len(derivative.layers[layer].experts)):
            (a if slot % 2 == 0 else b).append((layer, slot))
    return NodeAssignment(node_a=sorted(a), node_b=sorted(b))


def per_expert_bytes(derivative: MiniMoE) -> float:
    numel = derivative.arch.tensor_params.get(TensorRole.EXPERTS)
    return (numel or 0) * DTYPE_BYTES[derivative.arch.moe.expert_dtype]


def resident_bytes_by_node(assignment: NodeAssignment, derivative: MiniMoE) -> tuple[float, float]:
    b = per_expert_bytes(derivative)
    return len(assignment.node_a) * b, len(assignment.node_b) * b


def cross_node_activation_bytes_per_token(derivative: MiniMoE) -> float:
    """Per generated token, bytes moved between nodes (estimate)."""
    return derivative.arch.num_text_layers * derivative.hidden * BYTES_PER_ACTIVATION


@dataclass
class FitResult:
    node_a_resident_bytes: float
    node_b_resident_bytes: float
    reserve_bytes: float
    fitted: bool
    failures: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.fitted


def fit(
    derivative: MiniMoE,
    assignment: NodeAssignment,
    *,
    node_budget_bytes: float,
    runtime_reserve_bytes: float = 0.0,
) -> FitResult:
    """go/no-go: each node's resident + reserve must fit its budget."""
    a, b = resident_bytes_by_node(assignment, derivative)
    failures: list[str] = []
    if a + runtime_reserve_bytes > node_budget_bytes:
        failures.append(f"node A resident {a:.0f}B + reserve > budget {node_budget_bytes:.0f}B")
    if b + runtime_reserve_bytes > node_budget_bytes:
        failures.append(f"node B resident {b:.0f}B + reserve > budget {node_budget_bytes:.0f}B")
    return FitResult(
        node_a_resident_bytes=a,
        node_b_resident_bytes=b,
        reserve_bytes=runtime_reserve_bytes,
        fitted=not failures,
        failures=failures,
    )


@dataclass
class DistributedRun:
    node_a_routed: int  # routed expert executions on node A
    node_b_routed: int
    cross_node_bytes_per_token: float


def run_distributed(
    derivative: MiniMoE, tokens: list[int], assignment: NodeAssignment
) -> DistributedRun:
    """Execute a real forward and attribute each routed expert to its node."""
    result = forward(derivative, tokens)
    na = nb = 0
    for trace in result.traces:
        for ids in trace.topk_ids:
            for slot in ids:
                if assignment.node_of(trace.layer, slot) == "node_a":
                    na += 1
                else:
                    nb += 1
    return DistributedRun(
        node_a_routed=na,
        node_b_routed=nb,
        cross_node_bytes_per_token=cross_node_activation_bytes_per_token(derivative),
    )
