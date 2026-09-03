"""Quantization sensitivity per expert (blueprint §8.4).

Records how each expert responds to a candidate lossy representation, so the
final planner can choose both *width* and *precision* rather than one global
setting. Sensitivity is measured as the reconstruction error when the expert's
gate/up/down are quantized to a sample format; the most sensitive experts keep
more bits (higher bpw), robust ones can take a lower bpw.
"""

from __future__ import annotations

from dataclasses import dataclass

from cebu_profiler.compression.quant import rel_l2
from cebu_profiler.compression.response import quantize_expert_tensor
from cebu_profiler.profiler.runtime import MiniMoE

_EXPERT_MATS = ("gate", "up", "down")
_BPW_LEVELS: tuple[float, ...] = (4.0, 3.5, 3.25, 3.0)


@dataclass
class SensitivityReport:
    """Per-expert measured quantization sensitivity + recommended bpw."""

    sensitivity: dict[tuple[int, int], float]
    bpw: dict[tuple[int, int], float]
    levels: tuple[float, ...] = _BPW_LEVELS


def expert_quant_sensitivity(model: MiniMoE, fmt: str = "int8") -> dict[tuple[int, int], float]:
    """RMS reconstruction error (rel L2) of quantizing each expert's mats.

    Higher = the expert's raw weights survive quantization worse -> more
    sensitive -> keep more bits. Deterministic; uses raw expert tensors only.
    """
    sens: dict[tuple[int, int], float] = {}
    for layer, layer_w in enumerate(model.layers):
        for e, w in enumerate(layer_w.experts):
            rels: list[float] = []
            for key in _EXPERT_MATS:
                q, _ = quantize_expert_tensor(w[key], fmt)
                rels.append(rel_l2(w[key], q))
            sens[(layer, e)] = (sum(r * r for r in rels) / len(rels)) ** 0.5
    return sens


def recommend_bpw(
    sensitivity: dict[tuple[int, int], float],
    levels: tuple[float, ...] = _BPW_LEVELS,
) -> dict[tuple[int, int], float]:
    """Map measured sensitivity to a bpw level (more sensitive -> more bits).

    Deterministic: assigns bpw by the expert's sensitivity rank within the run.
    """
    if not sensitivity:
        return {}
    order = sorted(sensitivity.values())
    n = len(order)

    def _frac(s: float) -> float:
        # fraction of experts strictly less sensitive than this one
        return order.index(s) / (n - 1) if n > 1 else 0.0

    # levels sorted descending; cutpoints split [0,1) into equal ranks
    desc = sorted(levels, reverse=True)
    cut = [(i + 1) / len(desc) for i in range(len(desc))]
    out: dict[tuple[int, int], float] = {}
    for key, s in sensitivity.items():
        f = _frac(s)
        for _i, (bpw, c) in enumerate(zip(desc, cut, strict=True)):
            if f < c:
                out[key] = bpw
                break
        else:
            out[key] = desc[-1]
    return out


def sensitivity_report(model: MiniMoE, fmt: str = "int8") -> SensitivityReport:
    sens = expert_quant_sensitivity(model, fmt)
    return SensitivityReport(sensitivity=sens, bpw=recommend_bpw(sens))
