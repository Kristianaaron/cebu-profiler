"""Round-7 runtimeprobe tests.

Uses a fake/subprocess-adapter so repo tests run WITHOUT vLLM installed in the
repo venv; the authoritative installed-vLLM mounted-config probe is exercised by
an integration test when run under the vLLM exec venv (or via subprocess with a
given exec python).
"""

import json
from pathlib import Path

import pytest

from model_atlas.runtimeprobe import probe_installed, write_capability_report

VLLM_EXEC = "/home/kristianaaron/ai-lab/venvs/vllm/bin/python"


@pytest.fixture
def check_skip_vllm_exec():
    """Point to the installed vLLM exec python; skip the real probe if absent."""
    from pathlib import Path

    if not Path(VLLM_EXEC).exists():
        pytest.skip("vLLM exec venv not present")
    return VLLM_EXEC


def _mounted_quant_fixture(tmp_path):
    """Config with the real ModelOpt NVFP4 quantization block (parsed without
    vLLM in repo tests)."""
    root = Path(tmp_path) / "glm"
    root.mkdir(parents=True)
    cfg = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 78,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "kv_cache_scheme": {"type": "float", "dynamic": False, "num_bits": 8},
            "config_groups": {"group_0": {"weights": {"group_size": 16}}},
            "producer": {"name": "modelopt", "version": "0.46.0.dev65+g977d34dc3"},
        },
    }
    (root / "config.json").write_text(json.dumps(cfg))
    return str(root)


@pytest.mark.integration
def test_runtime_probe_schema_fixture(tmp_path):
    """Fake-adapter probe: mounted architecture + quant-config recognized,
    decoder path reported (source-scan/heuristic), env absent, unvalidated."""
    ckpt = _mounted_quant_fixture(tmp_path)
    r = probe_installed(ckpt, offline=True)
    assert r.architecture_registered is True
    assert r.quant_config_recognized is True
    assert r.quant_override == "modelopt_fp4"
    assert r.schema_supported is True
    assert r.decoder_path_present is True
    assert r.linear_method_class == "ModelOptNvFp4LinearMethod"
    assert r.fused_moe_method_class == "ModelOptNvFp4FusedMoE"
    # env absent in repo venv
    assert r.external_modelopt_installed is False
    # unvalidated + not ready
    assert r.derivative_load_validated is False
    assert r.runtime_ready is False


@pytest.mark.integration
def test_runtime_probe_missing_config_fails_closed(tmp_path):
    r = probe_installed(str(tmp_path / "missing"), offline=True)
    assert r.error
    assert r.schema_supported is False
    assert r.derivative_load_validated is False


@pytest.mark.integration
def test_capability_report_json(tmp_path):
    ckpt = _mounted_quant_fixture(tmp_path)
    r = probe_installed(ckpt, offline=True)
    out = tmp_path / "cap.json"
    write_capability_report(r, str(out))
    d = json.loads(out.read_text())
    assert d["schema_supported"] is True
    assert d["decoder_path_present"] is True
    assert d["runtime_ready"] is False
    # exact evidence present


@pytest.mark.integration
def test_runtime_probe_exec_python_real(check_skip_vllm_exec):
    """Run the authoritative installed-vLLM mounted-config probe when a vLLM exec
    venv is available. Skipped in the repo venv (no vllm)."""
    from model_atlas.runtimeprobe import probe_installed

    r = probe_installed(
        "/media/glm52/models/nvidia/GLM-5.2-NVFP4",
        exec_python=check_skip_vllm_exec,
    )
    assert r.quant_config_recognized is True
    assert r.quant_override == "modelopt_fp4"
    assert r.decoder_path_present is True
    assert r.linear_method_class == "ModelOptNvFp4LinearMethod"
    assert r.fused_moe_method_class == "ModelOptNvFp4FusedMoE"
    # not end-to-end validated
    assert r.derivative_load_validated is False
    assert r.runtime_ready is False
