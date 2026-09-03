"""F7 tests: compression backends, quantization math, response curves."""

from cebu_profiler.compression import (
    SupportStatus,
    expert_response_curve,
    get_backend_registry,
)
from cebu_profiler.compression.quant import float_mantissa_quant, rel_l2, uniform_int_quant
from cebu_profiler.profiler.runtime import build_mini_moe
from cebu_profiler.registry.architectures import get_registry

ARCH = get_registry().get("k3-mini")


def _weights():
    return [[1.0, -2.0, 0.5, 3.0], [0.1, -0.4, 2.0, -1.5]]


def test_exl3_aqlm_unsupported_not_fabricated():
    reg = get_backend_registry()
    assert reg.get("exl3").support == SupportStatus.UNSUPPORTED
    assert reg.get("aqlm").support == SupportStatus.UNSUPPORTED
    assert not reg.get("exl3").can_probe


def test_int_quant_reconstruction_decreases_with_bits():
    w = _weights()
    w8, meta8 = uniform_int_quant(w, 8)
    w4, meta4 = uniform_int_quant(w, 4)
    assert meta8.stored_bytes > meta4.stored_bytes
    assert meta8.effective_bits == 8.0
    assert rel_l2(w, w8) < 0.05  # int8 reconstructs these values closely
    assert rel_l2(w, w8) < rel_l2(w, w4)


def test_rel_l2_zero_for_identical():
    w = _weights()
    assert rel_l2(w, w) == 0.0


def test_float_mantissa_quant_fp16_tighter_than_bf16():
    w = _weights()
    w16, _m16 = float_mantissa_quant(w, 16, 10)  # fp16 mantissa
    wbf, _mbf = float_mantissa_quant(w, 16, 7)  # bf16 mantissa
    assert rel_l2(w, w16) < rel_l2(w, wbf)


def test_response_curve_measures_supported_and_skips_unsupported():
    model = build_mini_moe(ARCH, seed=1)
    reg = get_backend_registry()
    points = expert_response_curve(model, [1, 2, 3], layer=0, expert=2, backends=reg)
    by_fmt = {p.format: p for p in points}
    # unsupported formats get support recorded, never fabricated measurements
    assert by_fmt["exl3"].support == SupportStatus.UNSUPPORTED
    assert by_fmt["exl3"].reconstruction_error is None
    # supported formats have measured fields
    p8 = by_fmt["int8"]
    assert p8.reconstruction_error is not None
    assert p8.effective_bits == 8.0
    assert p8.stored_bytes is not None
    assert p8.output_drift is not None
    assert p8.logit_kl_impact is not None and p8.logit_kl_impact >= 0.0


def test_response_curve_int4_worse_than_int8_for_same_expert():
    model = build_mini_moe(ARCH, seed=2)
    reg = get_backend_registry()
    points = expert_response_curve(
        model, [1, 2, 3], layer=0, expert=1, backends=reg, formats=["int4", "int8"]
    )
    by_fmt = {p.format: p for p in points}
    assert by_fmt["int4"].reconstruction_error > by_fmt["int8"].reconstruction_error
