"""Six-level profiler hierarchy (v2 §9): L1 weights → L2 units → L3 experts →
L4 coalitions → L5 pathways → L6 behaviour.

The six levels are linked bottom-up (``children`` point DOWN toward weights,
``parents`` point UP toward behaviour), so a hierarchy is traceable in both
directions:

- **up** (``ancestors`` / ``behaviours_of``): from a weight / unit / expert /
  coalition / path to the behaviours it supports — "what does this component
  carry?";
- **down** (``descendants`` / ``project_down``): from a behaviour to the
  experts / units / weights that realise it — "what must stay for this
  behaviour to survive?"

Every node is built from **measured** forwards (routing traces, expert/channel
aggregates, co-routed coalitions, cross-layer paths, per-label success) and is
tagged with an evidence label (all ``measured`` on the synthetic runtime) — the
hierarchy never invents a connection that was not observed. This closes the §9
gap where only the L1 ownership layer existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from cebu_profiler.profiler.pathways import path_stats
from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.profiler.traces import trace_records
from cebu_profiler.schemas.ontology import SuccessState


class ProfilerLevel(StrEnum):
    WEIGHTS = "weights"  # L1: tensors
    UNITS = "units"  # L2: channels / neurons
    EXPERTS = "experts"  # L3
    COALITIONS = "coalitions"  # L4
    PATHWAYS = "pathways"  # L5
    BEHAVIOUR = "behaviour"  # L6


LEVEL_ORDER: list[ProfilerLevel] = [
    ProfilerLevel.WEIGHTS,
    ProfilerLevel.UNITS,
    ProfilerLevel.EXPERTS,
    ProfilerLevel.COALITIONS,
    ProfilerLevel.PATHWAYS,
    ProfilerLevel.BEHAVIOUR,
]
_LEVEL_INDEX = {lv: i for i, lv in enumerate(LEVEL_ORDER)}


def next_up(level: ProfilerLevel) -> ProfilerLevel | None:
    """The more-abstract level directly above ``level`` (toward behaviour)."""
    i = _LEVEL_INDEX[level]
    return LEVEL_ORDER[i + 1] if i + 1 < len(LEVEL_ORDER) else None


def next_down(level: ProfilerLevel) -> ProfilerLevel | None:
    """The more-granular level directly below ``level`` (toward weights)."""
    i = _LEVEL_INDEX[level]
    return LEVEL_ORDER[i - 1] if i - 1 >= 0 else None


@dataclass
class HierarchyNode:
    level: ProfilerLevel
    key: str
    label: str
    evidence: str = "measured"  # measured | estimated | predicted | inferred | causally_tested
    parents: list[str] = field(default_factory=list)  # up -> behaviour
    children: list[str] = field(default_factory=list)  # down -> weights
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "key": self.key,
            "label": self.label,
            "evidence": self.evidence,
            "parents": sorted(self.parents),
            "children": sorted(self.children),
            "metrics": dict(self.metrics),
        }


@dataclass
class HierarchyMap:
    model_id: str
    nodes: dict[str, HierarchyNode] = field(default_factory=dict)

    def nodes_at(self, level: ProfilerLevel) -> list[HierarchyNode]:
        return [n for n in self.nodes.values() if n.level is level]

    def counts(self) -> dict[str, int]:
        return {lv.value: len(self.nodes_at(lv)) for lv in LEVEL_ORDER}

    def parents_of(self, key: str) -> list[HierarchyNode]:
        n = self.nodes[key]
        return [self.nodes[p] for p in n.parents if p in self.nodes]

    def children_of(self, key: str) -> list[HierarchyNode]:
        n = self.nodes[key]
        return [self.nodes[c] for c in n.children if c in self.nodes]

    def ancestors(self, key: str) -> list[HierarchyNode]:
        """Everything reachable by walking up (parents), nearest-above first."""
        seen: dict[str, HierarchyNode] = {}
        stack = list(self.nodes[key].parents)
        while stack:
            k = stack.pop()
            if k in seen or k == key:
                continue
            seen[k] = self.nodes[k]
            stack.extend(self.nodes[k].parents)
        return sorted(seen.values(), key=lambda n: _LEVEL_INDEX[n.level], reverse=True)

    def descendants(self, key: str) -> list[HierarchyNode]:
        """Everything reachable by walking down (children), nearest-below first."""
        seen: dict[str, HierarchyNode] = {}
        stack = list(self.nodes[key].children)
        while stack:
            k = stack.pop()
            if k in seen or k == key:
                continue
            seen[k] = self.nodes[k]
            stack.extend(self.nodes[k].children)
        return sorted(seen.values(), key=lambda n: _LEVEL_INDEX[n.level])

    def behaviours_of(self, key: str) -> list[str]:
        """Distinct L6 behaviours a node (tensor/unit/expert/…) supports (trace up)."""
        return sorted({n.key for n in self.ancestors(key) if n.level is ProfilerLevel.BEHAVIOUR})

    def project_down(self, key: str) -> dict[str, list[dict[str, Any]]]:
        """Decompose a node into its per-level contributors (trace down).

        Each contributor carries a ``prevalence`` = number of *distinct* L6
        behaviours that also include it — a measured "how shared / how
        load-bearing" signal (AGENTS invariant 1: never cut on routing
        frequency alone).
        """
        level_out: dict[ProfilerLevel, dict[str, dict[str, Any]]] = {}
        for n in self.descendants(key):
            behav = self.behaviours_of(n.key)
            level_out.setdefault(n.level, {})[n.key] = {
                "key": n.key,
                "prevalence": len(behav),
            }
        return {
            lv.value: sorted(rows.values(), key=lambda r: -r["prevalence"])
            for lv, rows in sorted(level_out.items(), key=lambda kv: _LEVEL_INDEX[kv[0]])
        }

    def validate(self) -> list[str]:
        """Structural invariants: level adjacency, no dangling refs, non-empty."""
        warnings: list[str] = []
        keys = set(self.nodes)
        if not keys:
            return ["empty hierarchy"]
        for n in self.nodes.values():
            idx = _LEVEL_INDEX[n.level]
            for p in n.parents:
                if p not in keys:
                    warnings.append(f"dangling parent {p!r} on {n.key!r}")
                elif _LEVEL_INDEX[self.nodes[p].level] != idx + 1:
                    warnings.append(f"non-adjacent parent {p} on {n.key}")
            for c in n.children:
                if c not in keys:
                    warnings.append(f"dangling child {c!r} on {n.key!r}")
                elif _LEVEL_INDEX[self.nodes[c].level] != idx - 1:
                    warnings.append(f"non-adjacent child {c} on {n.key}")
        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable §27 ``hierarchy_map.json`` payload."""
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "levels": [lv.value for lv in LEVEL_ORDER],
            "counts": self.counts(),
            "nodes": {
                lv.value: [
                    self.nodes[k].to_dict()
                    for k in sorted(n.key for n in self.nodes.values() if n.level is lv)
                ]
                for lv in LEVEL_ORDER
            },
        }


def _success_value(state: SuccessState) -> float:
    if state in (SuccessState.SUCCESS, SuccessState.RECOVERED):
        return 1.0
    if state == SuccessState.PARTIALLY_RECOVERED:
        return 0.5
    return 0.0


def _coalition_key(model_id: str, layer: int, ids: tuple[int, ...]) -> str:
    return f"{model_id}:coalition:L{layer}:{'-'.join(map(str, ids))}"


def _path_key(model_id: str, sig: tuple[tuple[int, ...], ...]) -> str:
    parts = ["-".join(str(x) for x in sorted(tuple(elt))) for elt in sig]
    return f"{model_id}:path:{'|'.join(parts) if parts else 'empty'}"


def build_hierarchy(
    model: MiniMoE,
    samples: list[CalibrationSample],
    top_k: int | None = None,
    model_id: str | None = None,
) -> HierarchyMap:
    """Build the six-level hierarchy for ``model`` over the calibration corpus.

    All links and metrics come from real forwards: routing traces →
    per-layer coalitions, expert/channel aggregates, cross-layer path_stats,
    and per-label success rates from the samples.
    """
    model_id = model_id or model.arch.name
    hm = HierarchyMap(model_id=model_id)
    n_layers = len(model.layers)
    hidden = model.hidden

    # ---- measured aggregates (each runs real forwards) -------------------- #
    recs = trace_records(model, samples, top_k=top_k)
    stats = path_stats(model, samples, top_k=top_k)
    expert_agg = {(r.layer_id, r.expert_id): r for r in recs.experts}
    channel_agg = {(c.layer_id, c.expert_id, c.channel_id): c for c in recs.channels}

    beh_success: dict[str, list[float]] = {}
    for s in samples:
        for lab in s.labels:
            beh_success.setdefault(str(lab), []).append(_success_value(s.success_state))

    # ---- L1: expert tensors (gate / up / down) ---------------------------- #
    for lay in range(n_layers):
        for e in range(model.n_exp):
            for part in ("gate", "up", "down"):
                w = model.layers[lay].experts[e][part]
                numel = len(w) * (len(w[0]) if w else 0)
                k = f"{model_id}:tensor:L{lay}.e{e}.{part}"
                hm.nodes[k] = HierarchyNode(
                    level=ProfilerLevel.WEIGHTS,
                    key=k,
                    label=f"L{lay}·exp{e}·{part}",
                    metrics={"numel": float(numel), "bytes": float(numel * 4.0)},
                )

    # ---- L3: experts (children = their L2 channels, added below) ---------- #
    for lay in range(n_layers):
        for e in range(model.n_exp):
            ek = f"{model_id}:expert:L{lay}.e{e}"
            agg = expert_agg.get((lay, e))
            hm.nodes[ek] = HierarchyNode(
                level=ProfilerLevel.EXPERTS,
                key=ek,
                label=f"L{lay}·exp{e}",
                metrics={
                    "activation_count": float(agg.activation_count) if agg else 0.0,
                    "gate_weight_mean": agg.gate_weight_mean if agg else 0.0,
                    "output_norm_mean": agg.output_norm_mean if agg else 0.0,
                },
            )

    # ---- L2: channels, linked down to their down-tensor + up to the expert - #
    for lay in range(n_layers):
        for e in range(model.n_exp):
            ek = f"{model_id}:expert:L{lay}.e{e}"
            dk = f"{model_id}:tensor:L{lay}.e{e}.down"
            for c in range(hidden):
                ck = f"{model_id}:unit:L{lay}.e{e}.c{c}"
                ch = channel_agg.get((lay, e, c))
                hm.nodes[ck] = HierarchyNode(
                    level=ProfilerLevel.UNITS,
                    key=ck,
                    label=f"L{lay}·exp{e}·ch{c}",
                    parents=[ek],
                    children=[dk],
                    metrics={
                        "activation_frequency": ch.activation_frequency if ch else 0.0,
                        "activation_rms": ch.activation_rms if ch else 0.0,
                    },
                )
                hm.nodes[ek].children.append(ck)
                hm.nodes[dk].parents.append(ck)

    # ---- L5: cross-layer pathways; L6: behaviours ------------------------- #
    # (L4 coalitions are the per-layer slices of these paths — built below so
    #  every coalition sits under a pathway and trace-up holds for exactly the
    #  experts that are actually routed)
    # (behaviour nodes first so paths can link up to them)
    for beh_key, vals in beh_success.items():
        bk = f"{model_id}:behaviour:{beh_key}"
        hm.nodes[bk] = HierarchyNode(
            level=ProfilerLevel.BEHAVIOUR,
            key=bk,
            label=f"behaviour {beh_key}",
            metrics={
                "success_rate": round(sum(vals) / len(vals), 5),
                "samples": float(len(vals)),
            },
        )

    for rec in stats.records:
        sig = tuple(tuple(sorted(tuple(elt))) for elt in rec.signature)
        pk = _path_key(model_id, sig)
        if pk not in hm.nodes:
            hm.nodes[pk] = HierarchyNode(
                level=ProfilerLevel.PATHWAYS,
                key=pk,
                label=f"path {pk}",
                metrics={
                    "count": float(rec.count),
                    "success_rate": round(rec.success_rate, 5),
                },
            )
            # down: path -> the per-layer coalitions it routes through
            for lay, elt in enumerate(sig):
                tup = tuple(elt)
                ck = _coalition_key(model_id, lay, tup)
                if ck not in hm.nodes:
                    hm.nodes[ck] = HierarchyNode(
                        level=ProfilerLevel.COALITIONS,
                        key=ck,
                        label=f"L{lay} coalition ({','.join(map(str, tup))})",
                        metrics={"size": float(len(tup))},
                    )
                    for e in tup:
                        ek = f"{model_id}:expert:L{lay}.e{e}"
                        if ek in hm.nodes:
                            hm.nodes[ck].children.append(ek)
                            hm.nodes[ek].parents.append(ck)
                hm.nodes[pk].children.append(ck)
                hm.nodes[ck].parents.append(pk)
        # up: path -> behaviours that observed it
        for lab_str in rec.labels or []:
            bk = f"{model_id}:behaviour:{str(lab_str)}"
            if bk in hm.nodes and bk not in hm.nodes[pk].parents:
                hm.nodes[pk].parents.append(bk)
                hm.nodes[bk].children.append(pk)

    return hm
