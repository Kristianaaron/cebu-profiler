"""Structural model graph: the anatomical foundation for later traces.

Groups a source checkpoint's tensors into nodes by (role, layer, expert),
draws layer and global edges, and validates 100% tensor-key coverage with no
unclassified tensors (v2 §6 / §10.1; no-unclassified invariant).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cebu_profiler.checkpoint.classifier import Classification, classify_tensor
from cebu_profiler.checkpoint.source_manifest import CheckpointManifest
from cebu_profiler.schemas.architecture import TensorRole


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: TensorRole
    layer_index: int | None = None
    expert_index: int | None = None


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: str


class StructuralGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    coverage: float = 0.0  # classified tensor entries / total entries
    unclassified: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.coverage == 1.0 and not self.unclassified


def _node_id(role: TensorRole, layer: int | None, expert: int | None) -> str:
    parts = [role.value]
    if layer is not None:
        parts.append(f"l{layer}")
    if expert is not None:
        parts.append(f"e{expert}")
    return ".".join(parts)


def build_structural_graph(manifest: CheckpointManifest) -> StructuralGraph:
    """Build the structural graph and validate full coverage."""
    classifications: list[Classification] = [classify_tensor(t.name) for t in manifest.tensors]
    unclassified = [c.name for c in classifications if c.unclassified]

    covered = len(manifest.tensors) - len(unclassified)
    coverage = covered / len(manifest.tensors) if manifest.tensors else 0.0

    # Nodes: one per distinct (role, layer, expert).
    nodes: dict[str, GraphNode] = {}
    per_layer: dict[int, set[str]] = {}
    global_nodes: set[str] = set()
    for c in classifications:
        role = c.role
        if role is None:
            continue
        nid = _node_id(role, c.layer_index, c.expert_index)
        node = GraphNode(id=nid, role=role, layer_index=c.layer_index, expert_index=c.expert_index)
        nodes.setdefault(nid, node)
        if c.layer_index is not None:
            per_layer.setdefault(c.layer_index, set()).add(nid)
        else:
            global_nodes.add(nid)

    edges: list[GraphEdge] = []
    # Each layer container -> its role nodes.
    for layer in sorted(per_layer):
        container = f"layer.{layer}"
        for nid in per_layer[layer]:
            edges.append(GraphEdge(source=container, target=nid, kind="contains"))
            edges.append(GraphEdge(source=nid, target=container, kind="belongs_to"))
    # Global pathway: embedding -> lm_head.
    if "embedding" in global_nodes and "lm_head" in global_nodes:
        edges.append(GraphEdge(source="embedding", target="lm_head", kind="output"))

    return StructuralGraph(
        model_name=manifest.checkpoint_dir,
        nodes=sorted(nodes.values(), key=lambda n: n.id),
        edges=edges,
        coverage=coverage,
        unclassified=sorted(unclassified),
    )
