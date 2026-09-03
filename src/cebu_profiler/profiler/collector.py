"""Streaming channel-level trace collector (blueprint §7-Module A).

The collector accumulates FFN intermediate activation statistics at channel
granularity *online*, per (layer, expert, channel), without persisting per-token
activation tensors. This is what feeds every channel-importance scorer (TENP,
grouped Taylor surrogate, causal boundary).

A "channel" is the structured FFN unit of an expert: the coupled `gate[j,:]`,
`up[j,:]` rows and `down[:,j]` column (blueprint §12.1). Here we only measure
the activation side (`gate*up` intermediate value); the structural weight side
is read from the model by the scorers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChannelStat:
    """Finalized per-channel statistics for one (layer, expert, channel)."""

    layer: int
    expert: int
    channel: int
    rms: float  # sqrt(mean z^2) of the intermediate activation
    mean_abs: float  # mean |z|
    frequency: float  # fraction of tokens where |z| above epsilon
    peak: float  # max |z| observed
    samples: int


@dataclass
class ChannelStatsAccumulator:
    """Online aggregator keyed by (layer, expert, channel). """

    _sum_abs: dict[tuple[int, int, int], float] = field(default_factory=dict)
    _sum_sq: dict[tuple[int, int, int], float] = field(default_factory=dict)
    _n: dict[tuple[int, int, int], int] = field(default_factory=dict)
    _peak: dict[tuple[int, int, int], float] = field(default_factory=dict)
    _n_active: dict[tuple[int, int, int], int] = field(default_factory=dict)
    _eps: float = 1e-6

    def observe_expert(self, layer: int, expert: int, gate: list[float], up: list[float]) -> None:
        """Accumulate one token's intermediate activation per channel.

        `gate` and `up` are the [mid] per-channel values of one expert for one
        token; the intermediate activation is `z[c] = gate[c] * up[c]`.
        """
        for c, (g, u) in enumerate(zip(gate, up, strict=True)):
            z = g * u
            key = (layer, expert, c)
            az = abs(z)
            if key in self._n:
                self._sum_abs[key] += az
                self._sum_sq[key] += z * z
                self._n[key] += 1
                if az > self._peak[key]:
                    self._peak[key] = az
                if az > self._eps:
                    self._n_active[key] += 1
            else:
                self._sum_abs[key] = az
                self._sum_sq[key] = z * z
                self._n[key] = 1
                self._peak[key] = az
                self._n_active[key] = 1 if az > self._eps else 0

    def finalize(self) -> list[ChannelStat]:
        """Emit sorted finalized per-channel statistics (layer, expert, channel)."""
        rows: list[ChannelStat] = []
        for key, n in self._n.items():
            layer, expert, channel = key
            rows.append(
                ChannelStat(
                    layer=layer,
                    expert=expert,
                    channel=channel,
                    rms=(self._sum_sq[key] / n) ** 0.5,
                    mean_abs=self._sum_abs[key] / n,
                    frequency=self._n_active[key] / n,
                    peak=self._peak[key],
                    samples=n,
                )
            )
        rows.sort(key=lambda r: (r.layer, r.expert, r.channel))
        return rows
