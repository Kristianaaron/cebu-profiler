"""Per-expert compression-response curves (v2 §21–§23).

For a priority expert, quantize its real gate/up/down weights with a format's
quantizer, rebuild a clone, and measure: effective bits, stored bytes,
reconstruction error, output drift (quantized vs original expert direction on a
fixed input), downstream logit KL, and whether repair is indicated.

Formats that are UNSUPPORTED (EXL3, AQLM, …) are recorded with their support
status and their measured fields left `None` — they are never reported as
passing or working (v2 §31:24).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from cebu_profiler.compression.backend import BackendRegistry, SupportStatus
from cebu_profiler.compression.quant import (
    QuantMeta,
    float_mantissa_quant,
    rel_l2,
    uniform_int_quant,
)
from cebu_profiler.profiler.counterfactual import logit_kl
from cebu_profiler.profiler.runtime import MiniMoE, forward

_EXPERT_MATS = ("gate", "up", "down")


_Q = Callable[[list[list[float]], int], tuple[list[list[float]], QuantMeta]]
_Qm = Callable[[list[list[float]], int, int], tuple[list[list[float]], QuantMeta]]


# format -> (quantizer, bit_width[, mantissa])
def _quantizer_for(fmt: str) -> tuple[object, int] | tuple[object, int, int] | None:
    table: dict[str, tuple[object, int] | tuple[object, int, int]] = {
        "source_mxfp4": (uniform_int_quant, 4),
        "fp8": (uniform_int_quant, 8),
        "nvfp4": (uniform_int_quant, 4),
        "int8": (uniform_int_quant, 8),
        "int4": (uniform_int_quant, 4),
        "bf16": (float_mantissa_quant, 16, 7),  # 8 exponent + 7 mantissa
        "fp16": (float_mantissa_quant, 16, 10),
        "structured_pruning": (uniform_int_quant, 8),
    }
    if fmt not in table:
        return None
    return table[fmt]


def _apply_quantizer(rows: list[list[float]], fmt: str) -> tuple[list[list[float]], QuantMeta]:
    spec = _quantizer_for(fmt)
    assert spec is not None
    if len(spec) == 3:
        fn, bits, mantissa = spec
        return cast(_Qm, fn)(rows, bits, mantissa)
    fn, bits = spec
    return cast(_Q, fn)(rows, bits)


def _expert_mats(model: MiniMoE, layer: int, expert: int) -> dict[str, list[list[float]]]:
    return model.layers[layer].experts[expert]


def quantize_expert_tensor(
    rows: list[list[float]], fmt: str
) -> tuple[list[list[float]], QuantMeta]:
    """Quantize the weights of one expert (e.g. gate) for a format."""
    return _apply_quantizer(rows, fmt)


def _clone_with_quantized_expert(
    model: MiniMoE, layer: int, expert: int, fmt: str
) -> tuple[MiniMoE, dict[str, QuantMeta]]:
    m2 = copy.deepcopy(model)
    metas: dict[str, QuantMeta] = {}
    w = m2.layers[layer].experts[expert]
    for key in _EXPERT_MATS:
        qrows, meta = _apply_quantizer(w[key], fmt)
        w[key] = qrows
        metas[key] = meta
    return m2, metas


@dataclass
class ResponsePoint:
    format: str
    support: SupportStatus
    effective_bits: float | None = None
    stored_bytes: float | None = None
    reconstruction_error: float | None = None  # rel L2 over expert weights
    output_drift: float | None = None  # rel L2 quantized vs original expert direction
    logit_kl_impact: float | None = None  # full-output KL under forced expert route
    repair_required: bool | None = None
    note: str = ""


def expert_response_curve(
    model: MiniMoE,
    tokens: list[int],
    *,
    layer: int,
    expert: int,
    backends: BackendRegistry,
    formats: list[str] | None = None,
    token_index: int = 0,
    repair_threshold: float = 0.05,
) -> list[ResponsePoint]:
    """Measure a response curve for one expert across compression formats."""
    if formats is None:
        formats = backends.names()
    k = model.arch.moe.top_k
    override = {(layer, token_index): [expert] * k}
    base = forward(model, tokens, route_override=override)
    orig_dir = list(base.traces[layer].combined[token_index])
    orig_w = _expert_mats(model, layer, expert)

    points: list[ResponsePoint] = []
    for fmt in formats:
        backend = backends.get(fmt)
        if not backend.can_probe or _quantizer_for(fmt) is None:
            points.append(
                ResponsePoint(
                    format=fmt,
                    support=backend.support,
                    note="not probed (unsupported or no probe defined)",
                )
            )
            continue

        # proper aggregate reconstruction error (root of average of per-mat rel^2)
        rels = []
        for key in _EXPERT_MATS:
            qrows, _ = _apply_quantizer(orig_w[key], fmt)
            rels.append(rel_l2(orig_w[key], qrows))
        recon = (sum(r * r for r in rels) / len(rels)) ** 0.5

        # rebuild clone, measure output drift + full-output KL under forced route
        m2, metas = _clone_with_quantized_expert(model, layer, expert, fmt)
        qres = forward(m2, tokens, route_override=override)
        qdir = list(qres.traces[layer].combined[token_index])
        drift = rel_l2([orig_dir], [qdir])

        kl = logit_kl(base.logits, qres.logits)
        stored = sum(m.stored_bytes for m in metas.values())
        bits = max(m.effective_bits for m in metas.values())

        points.append(
            ResponsePoint(
                format=fmt,
                support=backend.support,
                effective_bits=bits,
                stored_bytes=stored,
                reconstruction_error=recon,
                output_drift=drift,
                logit_kl_impact=kl,
                repair_required=drift > repair_threshold,
            )
        )
    return points
