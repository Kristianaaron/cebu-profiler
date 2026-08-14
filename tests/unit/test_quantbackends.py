"""Phase 5: real quantization backend probe tests (honest detection).

These only exercise the probe logic that must work without the exec stack
(module presence is detected, never assumed). The actual exec-env probes run
under `.venv-exec`; in the repo venv modelopt/vllm are absent and the probe
must report UNSUPPORTED/absent — not crash.
"""

import pytest

from model_atlas.compression.backend import SupportStatus
from model_atlas.quantbackends import (
    probe_exl3,
    probe_modelopt_nvfp4,
    probe_vllm_nvfp4,
    to_registry,
)


@pytest.mark.integration
def test_exl3_probe_is_honest():
    b = probe_exl3()
    assert b.backend_id == "exl3"
    # repo venv has no exllamav2 -> must not claim support
    assert b.support in {SupportStatus.UNSUPPORTED, SupportStatus.REQUIRES_CUSTOM_KERNEL}
    assert b.setup  # actionable setup instructions always present


@pytest.mark.integration
def test_modelopt_probe_never_claims_when_absent():
    b = probe_modelopt_nvfp4()
    # even if a host ModelOpt venv exists, this must not be UNSUPPORTED unless
    # nothing is importable in this process; must never be INFERENCE_SUPPORTED
    assert b.support not in {SupportStatus.INFERENCE_SUPPORTED, SupportStatus.TRAINING_SUPPORTED}
    assert "INT4" not in b.note  # never calls uniform INT4 NVFP4


@pytest.mark.integration
def test_vllm_probe_returns_status():
    b = probe_vllm_nvfp4()
    assert b.backend_id == "vllm_nvfp4"
    # repo venv lacks vllm -> honest unsupported
    assert b.support in {SupportStatus.UNSUPPORTED, SupportStatus.PROBE_ONLY}


@pytest.mark.integration
def test_to_registry_builds_backend_entries():
    reg = to_registry()
    assert {"exl3", "modelopt_nvfp4", "vllm_nvfp4"} <= set(reg)
    for name, cb in reg.items():
        assert cb.backend_id == name
        assert cb.support in set(SupportStatus)
