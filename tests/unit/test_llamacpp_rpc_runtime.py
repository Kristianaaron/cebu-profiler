from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from model_atlas.backend.llamacpp_gguf import build_llamacpp_gguf_record
from model_atlas.llamacpp_rpc_runtime import (
    EXPECTED_LLAMA_SERVER_SHA256,
    EXPECTED_RPC_SERVER_SHA256,
    PINNED_COMMIT,
    RUNTIME_ID,
    LlamaCppRpcMtpConfig,
    LlamaCppRpcRuntimeAdapter,
    LlamaCppRpcRuntimeConfig,
    LlamaCppRpcToolProbe,
    LlamaCppRpcValidationReceipt,
    LlamaCppRpcWorkerAttestation,
    probe_llamacpp_rpc_runtime,
    validate_runtime_receipt,
)

ARTIFACT_SHA = "a" * 64


def _config(tmp_path: Path) -> LlamaCppRpcRuntimeConfig:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"artifact")
    return LlamaCppRpcRuntimeConfig(
        artifact_path=artifact,
        artifact_sha256=hashlib.sha256(b"artifact").hexdigest(),
        llama_server_path=tmp_path / "llama-server",
        worker_rpc_server_path=tmp_path / "ggml-rpc-server",
    )


def _probe(config: LlamaCppRpcRuntimeConfig, available: bool = True) -> LlamaCppRpcToolProbe:
    return LlamaCppRpcToolProbe(
        available=available,
        commit=PINNED_COMMIT,
        llama_server_path=str(config.llama_server_path),
        llama_server_sha256=EXPECTED_LLAMA_SERVER_SHA256,
        worker_rpc_server_path=str(config.worker_rpc_server_path),
        worker_rpc_server_sha256=EXPECTED_RPC_SERVER_SHA256,
        remote_worker_attested=available,
        artifact_path=str(config.artifact_path),
        artifact_sha256=config.artifact_sha256,
        artifact_verified=available,
    )


def _receipt(config: LlamaCppRpcRuntimeConfig) -> LlamaCppRpcValidationReceipt:
    return LlamaCppRpcValidationReceipt(
        runtime_id=RUNTIME_ID,
        config_sha256=config.canonical_sha256(),
        artifact_sha256=config.artifact_sha256,
        commit=PINNED_COMMIT,
        llama_server_sha256=EXPECTED_LLAMA_SERVER_SHA256,
        worker_rpc_server_sha256=EXPECTED_RPC_SERVER_SHA256,
        worker_rpc_server_path=str(config.worker_rpc_server_path),
        worker_hash_attested=True,
        worker_host="169.254.200.197",
        observed_devices=("CUDA0", "RPC0"),
        load_succeeded=True,
        generation_succeeded=True,
        evidence_kind="measured",
    )


def test_base_argv_orders_rpc_before_exact_devices_and_is_conservative(tmp_path: Path) -> None:
    config = _config(tmp_path)
    argv = config.head_argv()
    assert argv == (
        str(config.llama_server_path),
        "--model",
        str(config.artifact_path),
        "--alias",
        "glm52-mixed-gguf",
        "--rpc",
        "169.254.200.197:50052",
        "--device",
        "CUDA0,RPC0",
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
        "--flash-attn",
        "auto",
        "--load-mode",
        "auto",
        "--host",
        "127.0.0.1",
        "--port",
        "8892",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--fit",
        "off",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--no-warmup",
        "--metrics",
        "--no-ui",
        "--threads",
        "8",
        "--threads-batch",
        "8",
    )
    assert argv.index("--rpc") < argv.index("--device")
    assert argv[argv.index("--device") + 1] == "CUDA0,RPC0"
    assert argv[argv.index("--alias") + 1] == "glm52-mixed-gguf"
    assert argv[argv.index("--n-gpu-layers") + 1] == "all"
    assert argv[argv.index("--split-mode") + 1] == "layer"
    assert argv[argv.index("--tensor-split") + 1] == "1,1"
    assert argv[argv.index("--batch-size") + 1] == "512"
    assert argv[argv.index("--ubatch-size") + 1] == "128"
    assert argv[argv.index("--flash-attn") + 1] == "auto"
    assert argv[argv.index("--load-mode") + 1] == "auto"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8892"
    assert argv[argv.index("--ctx-size") + 1] == "4096"
    assert argv[argv.index("--parallel") + 1] == "1"
    assert argv[argv.index("--fit") + 1] == "off"
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
    assert "--no-warmup" in argv
    assert "--metrics" in argv
    assert "--no-ui" in argv
    assert argv[argv.index("--threads") + 1] == "8"
    assert argv[argv.index("--threads-batch") + 1] == "8"
    assert not {
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--cache-type-k-draft",
        "--cache-type-v-draft",
    }.intersection(argv)
    worker_argv = config.worker_argv()
    assert worker_argv == (
        str(config.worker_rpc_server_path),
        "--host",
        "169.254.200.197",
        "--port",
        "50052",
        "--device",
        "CUDA0",
        "--threads",
        "8",
    )


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1", "8.8.8.8", "worker.local"])
def test_worker_must_bind_a_private_non_loopback_ip(tmp_path: Path, host: str) -> None:
    with pytest.raises(ValueError, match="worker_host"):
        LlamaCppRpcRuntimeConfig(
            artifact_path=tmp_path / "model.gguf",
            artifact_sha256=ARTIFACT_SHA,
            worker_host=host,
        )


def test_api_must_remain_loopback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        LlamaCppRpcRuntimeConfig(
            artifact_path=tmp_path / "model.gguf",
            artifact_sha256=ARTIFACT_SHA,
            api_host="0.0.0.0",
        )


def test_mtp_is_a_separate_explicit_follow_up(tmp_path: Path) -> None:
    base = _config(tmp_path)
    adapter = LlamaCppRpcRuntimeAdapter(base, tmp_path / "llama.cpp")
    assert adapter.base_head_argv() == base.head_argv()
    assert "--spec-type" not in base.head_argv()
    mtp = LlamaCppRpcMtpConfig(base)
    argv = mtp.head_argv()
    assert adapter.mtp_head_argv() == argv
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "3"
    assert argv[argv.index("--spec-draft-n-min") + 1] == "0"
    assert argv[argv.index("--cache-type-k-draft") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v-draft") + 1] == "q8_0"


def _write_fake_git(root: Path) -> None:
    (root / ".git/refs/heads").mkdir(parents=True)
    (root / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / ".git/refs/heads/main").write_text(PINNED_COMMIT + "\n", encoding="utf-8")


def test_probe_is_filesystem_only_and_fails_closed_on_hash_drift(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    _write_fake_git(root)
    head = root / "llama-server"
    worker = root / "ggml-rpc-server"
    head.write_bytes(b"head")
    worker.write_bytes(b"worker")
    os.chmod(head, 0o755)
    os.chmod(worker, 0o755)
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"artifact")
    config = LlamaCppRpcRuntimeConfig(
        artifact_path=artifact,
        artifact_sha256=hashlib.sha256(b"artifact").hexdigest(),
        llama_server_path=head,
        worker_rpc_server_path=worker,
    )
    head_sha = hashlib.sha256(b"head").hexdigest()
    worker_sha = hashlib.sha256(b"worker").hexdigest()
    ok = probe_llamacpp_rpc_runtime(
        config,
        toolchain_root=root,
        expected_llama_server_sha256=head_sha,
        expected_worker_rpc_server_sha256=worker_sha,
        worker_attestation=LlamaCppRpcWorkerAttestation(
            host=config.worker_host,
            rpc_server_path=str(worker),
            rpc_server_sha256=worker_sha,
            commit=PINNED_COMMIT,
        ),
    )
    assert ok.available
    assert not ok.executed_binaries
    worker.write_bytes(b"drift")
    drift = probe_llamacpp_rpc_runtime(
        config,
        toolchain_root=root,
        expected_llama_server_sha256=head_sha,
        expected_worker_rpc_server_sha256=worker_sha,
        worker_attestation=LlamaCppRpcWorkerAttestation(
            host=config.worker_host,
            rpc_server_path=str(worker),
            rpc_server_sha256=hashlib.sha256(b"drift").hexdigest(),
            commit=PINNED_COMMIT,
        ),
    )
    assert not drift.available
    assert not drift.executed_binaries


def test_probe_rejects_missing_or_drifted_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    missing = replace(config, artifact_path=tmp_path / "missing.gguf")
    missing_probe = probe_llamacpp_rpc_runtime(missing, toolchain_root=tmp_path)
    assert not missing_probe.available
    assert not missing_probe.artifact_verified

    config.artifact_path.write_bytes(b"changed")
    drift = probe_llamacpp_rpc_runtime(config, toolchain_root=tmp_path)
    assert not drift.available
    assert drift.artifact_sha256 != config.artifact_sha256
    assert not drift.artifact_verified


def test_no_runtime_claim_before_matching_measured_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert build_llamacpp_gguf_record().runtime_compat == ()
    no_receipt = validate_runtime_receipt(config, _probe(config), None)
    assert not no_receipt.validated
    assert no_receipt.runtime_compat == ()

    unmeasured = replace(_receipt(config), evidence_kind="estimated")
    assert validate_runtime_receipt(config, _probe(config), unmeasured).runtime_compat == ()

    claim = validate_runtime_receipt(config, _probe(config), _receipt(config))
    assert claim.validated
    assert claim.runtime_compat == (RUNTIME_ID,)


def test_receipt_rejects_tool_or_artifact_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    receipt = _receipt(config)
    bad_artifact = replace(receipt, artifact_sha256="b" * 64)
    assert validate_runtime_receipt(config, _probe(config), bad_artifact).runtime_compat == ()
    assert validate_runtime_receipt(config, _probe(config, False), receipt).runtime_compat == ()


def test_adapter_never_derives_worker_attestation_from_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = LlamaCppRpcRuntimeAdapter(config, tmp_path / "llama.cpp")
    claim = adapter.validate_receipt(_receipt(config), independently_measured_worker=None)
    assert not claim.validated
    assert claim.runtime_compat == ()
