"""Streaming channel-level trace collector (blueprint §7-Module A).

The collector accumulates FFN intermediate activation statistics at channel
granularity *online*, per (layer, expert, channel), without persisting per-token
activation tensors. This is what feeds every channel-importance scorer (TENP,
grouped Taylor surrogate, causal boundary).

A "channel" is the structured FFN unit of an expert: the coupled `gate[j,:]`,
`up[j,:]` rows and `down[:,j]` column (blueprint §12.1). Here we only measure
the activation side (`gate*up` intermediate value); the structural weight side
is read from the model by the scorers.

The accumulator stores per-(layer, expert) [mid]-vectors instead of per-channel
dict entries; `observe_expert` also accepts a [T, mid] block straight from the
vectorized runtime. Finalized stats are numerically identical to the previous
per-scalar implementation.
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


np = None


def _ensure_np():
    global np
    if np is None:
        import numpy as _numpy

        np = _numpy
    return np


@dataclass
class ChannelStatsAccumulator:
    """Online aggregator keyed by (layer, expert); per-channel vectors inside."""

    _sum_abs: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    _sum_sq: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    _n: dict[tuple[int, int], int] = field(default_factory=dict)
    _peak: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    _n_active: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    _eps: float = 1e-6

    def observe_expert(
        self,
        layer: int,
        expert: int,
        gate: list[float] | "np.ndarray",
        up: list[float] | "np.ndarray" | None = None,
    ) -> None:
        """Accumulate intermediate activations per channel.

        Vectorized form (used by the NumPy runtime): `gate` is the [T, mid]
        intermediate block `z = gate*up` for one expert across tokens and `up`
        is ignored. Scalar form (kept for direct callers): pass the per-token
        `gate` and `up` channel vectors; the intermediate is `z[c] = gate[c]*up[c]`.
        """
        _ensure_np()
        if up is not None:
            z = np.asarray([g * u for g, u in zip(gate, up, strict=True)], dtype=np.float64)
        else:
            z = np.asarray(gate, dtype=np.float64)
            if z.ndim == 1:  # a bare gate vector without `up` is scalar-form misuse
                raise TypeError("scalar form requires both gate and up")
        az = np.abs(z)

        key = (layer, expert)
        if key in self._n:
            n = self._n[key]
            tot = n + z.shape[0]
            self._sum_abs[key] += az.sum(axis=0)
            self._sum_sq[key] += (z * z).sum(axis=0)
            self._peak[key] = np.maximum(self._peak[key], az.max(axis=0))
            self._n_active[key] += (az > self._eps).sum(axis=0)
            self._n[key] = tot
        else:
            self._sum_abs[key] = az.sum(axis=0).copy()
            self._sum_sq[key] = (z * z).sum(axis=0).copy()
            self._peak[key] = az.max(axis=0).copy()
            self._n_active[key] = (az > self._eps).sum(axis=0)
            self._n[key] = z.shape[0]

    def finalize(self) -> list[ChannelStat]:
        """Emit sorted finalized per-channel statistics (layer, expert, channel)."""
        rows: list[ChannelStat] = []
        for key in sorted(self._n):
            layer, expert = key
            n = self._n[key]
            sum_abs = self._sum_abs[key]
            sum_sq = self._sum_sq[key]
            peak = self._peak[key]
            n_active = self._n_active[key]
            for c in range(sum_abs.shape[0]):
                rows.append(
                    ChannelStat(
                        layer=layer,
                        expert=expert,
                        channel=c,
                        rms=(float(sum_sq[c]) / n) ** 0.5,
                        mean_abs=float(sum_abs[c]) / n,
                        frequency=float(n_active[c]) / n,
                        peak=float(peak[c]),
                        samples=n,
                    )
                )
        return rows
