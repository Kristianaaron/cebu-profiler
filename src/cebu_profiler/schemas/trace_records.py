"""Normalized trace/score data contracts (blueprint §10).

Pure data containers decoupled from Python object identity and from the runtime
(builder lives in ``profiler/traces.py``): ``RouterRecord`` (per token+layer),
``ExpertAggregate`` (per layer+expert), ``ChannelAggregate`` (per
layer+expert+channel). All fields are *measured* from a real forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RouterRecord:
    sample_id: int
    token_index: int
    layer_id: int
    selected_experts: list[int]
    gate_weights: list[float]
    routing_entropy: float


@dataclass
class ExpertAggregate:
    layer_id: int
    expert_id: int
    activation_count: int
    gate_weight_mean: float
    output_norm_mean: float
    direction_change_mean: float


@dataclass
class ChannelAggregate:
    layer_id: int
    expert_id: int
    channel_id: int
    activation_rms: float
    activation_frequency: float
    tenp_score: float


@dataclass
class TraceRecords:
    router: list[RouterRecord] = field(default_factory=list)
    experts: list[ExpertAggregate] = field(default_factory=list)
    channels: list[ChannelAggregate] = field(default_factory=list)
