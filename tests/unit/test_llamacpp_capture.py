from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import pytest

from model_atlas.evaluation.capture_metrics import (
    CaptureMetricError,
    evaluate_capture_pair,
    full_vocab_kld_row,
)
from model_atlas.evaluation.llamacpp_capture import (
    LLAMA_CPP_CAPTURE_COMMIT,
    CaptureRequest,
    CaptureRole,
    CaptureToolIdentity,
    CaptureValidationError,
    RawCapture,
    build_capture_argv,
    canonical_capture_runtime_argv,
    capture_runtime_argv_sha256,
    finalize_capture,
    validate_capture_pair,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _common_argv(model: Path) -> tuple[str, ...]:
    return (
        "--model",
        str(model),
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
        "--ctx-size",
        "8",
        "--batch-size",
        "8",
        "--ubatch-size",
        "4",
        "--fit",
        "off",
        "--no-warmup",
        "--threads",
        "2",
        "--threads-batch",
        "2",
    )


def _request(tmp_path: Path, output: Path, tokens: Path) -> CaptureRequest:
    model = tmp_path / "model.gguf"
    model_manifest = tmp_path / "model-artifact.json"
    tool = tmp_path / "llama-atlas-capture"
    tool_contract = tmp_path / "capture-build.json"
    held_out = tmp_path / "held-out-manifest.json"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"bounded-model")
    model_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_sha256": _sha(b"bounded-model"),
                "source_model_sha256": "5" * 64,
                "profile_tokenizer_sha256": _sha(b"bounded-tokenizer"),
                "producer_artifact_sha256": "6" * 64,
                "recipe_sha256": "7" * 64,
                "plan_id": "8" * 64,
                "run_id": "9" * 64,
                "evidence_kind": "measured",
            }
        )
    )
    tool.write_bytes(b"bounded-tool")
    tool_contract.write_bytes(b"bounded-build-contract")
    held_out.write_bytes(b"bounded-held-out")
    tokenizer.write_bytes(b"bounded-tokenizer")
    runtime_argv = canonical_capture_runtime_argv(
        tool_path=str(tool),
        common_argv=_common_argv(model),
        forced_tokens_path=str(tokens),
        output_dir=str(output),
        layers=(0, 2),
    )
    return CaptureRequest(
        model_id="candidate-a",
        model_path=str(model),
        model_sha256=_sha(model.read_bytes()),
        model_artifact_manifest_path=str(model_manifest),
        model_artifact_manifest_sha256=_sha(model_manifest.read_bytes()),
        source_model_sha256="5" * 64,
        role=CaptureRole.CANDIDATE,
        reference_kind="candidate",
        forced_tokens_path=str(tokens),
        forced_tokens_sha256=_sha(tokens.read_bytes()),
        held_out_manifest_path=str(held_out),
        held_out_manifest_sha256=_sha(held_out.read_bytes()),
        ordered_sample_ids_sha256=_sha(b'["s1","s2"]'),
        profile_tokenizer_path=str(tokenizer),
        profile_tokenizer_sha256=_sha(tokenizer.read_bytes()),
        output_dir=str(output),
        layers=(0, 2),
        context_tokens=8,
        batch_tokens=8,
        ubatch_tokens=4,
        threads=2,
        runtime_argv_sha256=capture_runtime_argv_sha256(runtime_argv),
        tool=CaptureToolIdentity(
            llama_cpp_commit=LLAMA_CPP_CAPTURE_COMMIT,
            binary_path=str(tool),
            binary_sha256=_sha(tool.read_bytes()),
            build_contract_path=str(tool_contract),
            build_contract_sha256=_sha(tool_contract.read_bytes()),
        ),
    )


def _raw_capture(
    tmp_path: Path, *, role: CaptureRole = CaptureRole.CANDIDATE
) -> tuple[CaptureRequest, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tokens = tmp_path / "tokens.jsonl"
    records = (
        {"sample_id": "s1", "domain": "math", "token_ids": [0, 1, 2]},
        {"sample_id": "s2", "domain": "code", "token_ids": [2, 3]},
    )
    tokens.write_text("".join(json.dumps(row) + "\n" for row in records))
    output = tmp_path / "capture"
    output.mkdir()
    request = _request(tmp_path, output, tokens)
    if role is not CaptureRole.CANDIDATE:
        request = CaptureRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "role": role.value,
                "reference_kind": "identity_control",
                "request_id": None,
            }
        )
    alignment = (
        {
            "row": 0,
            "sample_id": "s1",
            "domain": "math",
            "input_position": 0,
            "target_position": 1,
            "target_token_id": 1,
        },
        {
            "row": 1,
            "sample_id": "s1",
            "domain": "math",
            "input_position": 1,
            "target_position": 2,
            "target_token_id": 2,
        },
        {
            "row": 2,
            "sample_id": "s2",
            "domain": "code",
            "input_position": 0,
            "target_position": 1,
            "target_token_id": 3,
        },
    )
    (output / "alignment.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in alignment)
    )
    (output / "tokenizer.tsv").write_bytes(b"0\t00\n1\t61\n2\t62\n3\tff\n")
    logits = [float(index) / 10 for index in range(3 * 4)]
    (output / "logits.f32").write_bytes(struct.pack("<12f", *logits))
    for layer in (0, 2):
        values = [float(layer + index) for index in range(3 * 3)]
        (output / f"layer-{layer:03d}.f32").write_bytes(struct.pack("<9f", *values))
    raw = {
        "schema_version": 1,
        "capture_mode": "teacher_forced",
        "vocab_size": 4,
        "hidden_size": 3,
        "n_hidden_layers": 3,
        "row_count": 3,
        "sample_count": 2,
        "layers": [0, 2],
        "receipt": {
            "request_id": request.request_id,
            "model_sha256": request.model_sha256,
            "measured_model_sha256": request.model_sha256,
            "model_artifact_manifest_sha256": request.model_artifact_manifest_sha256,
            "tool_binary_sha256": request.tool.binary_sha256,
            "measured_tool_binary_sha256": request.tool.binary_sha256,
            "tool_build_contract_sha256": request.tool.build_contract_sha256,
            "forced_tokens_sha256": request.forced_tokens_sha256,
            "measured_forced_tokens_sha256": request.forced_tokens_sha256,
            "held_out_manifest_sha256": request.held_out_manifest_sha256,
            "ordered_sample_ids_sha256": request.ordered_sample_ids_sha256,
            "profile_tokenizer_sha256": request.profile_tokenizer_sha256,
            "runtime_argv_sha256": request.runtime_argv_sha256,
            "role": request.role.value,
            "reference_kind": request.reference_kind,
            "layers": [0, 2],
            "normalized_runtime_argv": list(
                canonical_capture_runtime_argv(
                    tool_path=request.tool.binary_path,
                    common_argv=_common_argv(Path(request.model_path)),
                    forced_tokens_path=request.forced_tokens_path,
                    output_dir=request.output_dir,
                    layers=request.layers,
                )
            ),
            "runtime_params": {
                "model_path": request.model_path,
                "tokens_jsonl": request.forced_tokens_path,
                "output_dir": request.output_dir,
                "layers": [0, 2],
                "context_tokens": 8,
                "batch_tokens": 8,
                "ubatch_tokens": 4,
                "threads": 2,
                "threads_batch": 2,
                "split_mode": "layer",
                "n_gpu_layers": -1,
                "main_gpu": 0,
                "fit_params": False,
                "devices": ["CUDA0", "RPC0"],
                "warmup": False,
            },
        },
        "files": {
            "logits": "logits.f32",
            "layer_inputs": ["layer-000.f32", "layer-002.f32"],
            "alignment": "alignment.jsonl",
            "tokenizer": "tokenizer.tsv",
        },
    }
    (output / "raw-capture.json").write_text(json.dumps(raw))
    return request, output


def test_finalize_capture_and_identity_pair(tmp_path: Path) -> None:
    request, output = _raw_capture(tmp_path)
    candidate = finalize_capture(request)
    identity_request, _ = _raw_capture(tmp_path / "identity", role=CaptureRole.IDENTITY_CONTROL)
    identity = finalize_capture(identity_request)
    pair = validate_capture_pair(identity, candidate)

    assert candidate.capture_id
    assert pair.identity_control is True
    assert pair.row_count == 3
    assert {artifact.name for artifact in candidate.artifacts} == {
        "logits.f32",
        "layer-000.f32",
        "layer-002.f32",
        "alignment.jsonl",
        "tokenizer.tsv",
    }
    assert (output / "capture-manifest.json").is_file()


def test_build_capture_argv_is_exact_and_non_shell(tmp_path: Path) -> None:
    request, _ = _raw_capture(tmp_path)
    argv = build_capture_argv(request, common_argv=_common_argv(Path(request.model_path)))
    assert argv[0] == request.tool.binary_path
    assert argv.count("--request-id") == 1
    assert argv[argv.index("--request-id") + 1] == request.request_id
    assert argv.index("--rpc") < argv.index("--device")
    with pytest.raises(CaptureValidationError, match="request digest"):
        build_capture_argv(
            request,
            common_argv=(*_common_argv(Path(request.model_path)), "--flash-attn", "off"),
        )


@pytest.mark.parametrize("target", ["model", "tool", "tokens"])
def test_finalize_rejects_bound_input_drift(tmp_path: Path, target: str) -> None:
    request, _ = _raw_capture(tmp_path)
    path = {
        "model": Path(request.model_path),
        "tool": Path(request.tool.binary_path),
        "tokens": Path(request.forced_tokens_path),
    }[target]
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(CaptureValidationError, match="drift"):
        finalize_capture(request)


def test_finalize_rejects_alignment_and_tokenizer_drift(tmp_path: Path) -> None:
    request, output = _raw_capture(tmp_path)
    alignment = output / "alignment.jsonl"
    alignment.write_text(
        alignment.read_text().replace('"target_token_id":1', '"target_token_id":2')
    )
    with pytest.raises(CaptureValidationError, match="alignment row"):
        finalize_capture(request)

    request, output = _raw_capture(tmp_path / "fresh")
    (output / "tokenizer.tsv").write_bytes(b"0\t00\n1\t61\n3\t62\n2\tff\n")
    with pytest.raises(CaptureValidationError, match="tokenizer table IDs"):
        finalize_capture(request)


def test_finalize_rejects_nonfinite_extra_and_symlink(tmp_path: Path) -> None:
    request, output = _raw_capture(tmp_path)
    values = [0.0] * 11 + [math.nan]
    (output / "logits.f32").write_bytes(struct.pack("<12f", *values))
    with pytest.raises(CaptureValidationError, match="non-finite"):
        finalize_capture(request)

    request, output = _raw_capture(tmp_path / "extra-case")
    (output / "extra").write_text("unexpected")
    with pytest.raises(CaptureValidationError, match="file set"):
        finalize_capture(request)

    request, output = _raw_capture(tmp_path / "link-case")
    logits = output / "logits.f32"
    backing = tmp_path / "backing"
    backing.write_bytes(logits.read_bytes())
    logits.unlink()
    logits.symlink_to(backing)
    with pytest.raises(CaptureValidationError, match="regular files only"):
        finalize_capture(request)


def test_capture_role_and_pair_alignment_fail_closed(tmp_path: Path) -> None:
    request, _ = _raw_capture(tmp_path, role=CaptureRole.IDENTITY_CONTROL)
    with pytest.raises(ValueError, match="role and reference_kind"):
        CaptureRequest.model_validate(
            {**request.model_dump(mode="json"), "reference_kind": "bf16", "request_id": None}
        )
    with pytest.raises(ValueError, match="precision evidence"):
        CaptureRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "role": "bf16_teacher",
                "reference_kind": "bf16",
                "source_model_sha256": None,
                "request_id": None,
            }
        )

    first = finalize_capture(request)
    second_request, second_output = _raw_capture(tmp_path / "second")
    tokenizer = second_output / "tokenizer.tsv"
    tokenizer.write_bytes(b"0\t00\n1\t61\n2\t63\n3\tff\n")
    second = finalize_capture(second_request)
    with pytest.raises(CaptureValidationError, match="not token/layer aligned"):
        validate_capture_pair(first, second)


def test_native_receipt_and_role_matrix_fail_closed(tmp_path: Path) -> None:
    request, output = _raw_capture(tmp_path)
    raw_path = output / "raw-capture.json"
    raw = json.loads(raw_path.read_text())
    raw["receipt"]["model_sha256"] = "f" * 64
    raw_path.write_text(json.dumps(raw))
    with pytest.raises(CaptureValidationError, match="native capture receipt"):
        finalize_capture(request)

    candidate_request, _ = _raw_capture(tmp_path / "candidate")
    candidate = finalize_capture(candidate_request)
    with pytest.raises(CaptureValidationError, match="reference side"):
        validate_capture_pair(candidate, candidate)


def test_split_model_symlink_and_layer_boundary_rejected(tmp_path: Path) -> None:
    request, _ = _raw_capture(tmp_path)
    with pytest.raises(ValueError, match="single-file GGUF"):
        CaptureRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "model_path": str(tmp_path / "model-00001-of-00002.gguf"),
                "request_id": None,
            }
        )

    target = Path(request.model_path)
    backing = tmp_path / "model-real.gguf"
    target.rename(backing)
    target.symlink_to(backing)
    with pytest.raises(CaptureValidationError, match="non-symlinks"):
        finalize_capture(request)

    raw = {
        "schema_version": 1,
        "capture_mode": "teacher_forced",
        "vocab_size": 4,
        "hidden_size": 3,
        "n_hidden_layers": 3,
        "row_count": 3,
        "sample_count": 2,
        "layers": [3],
        "receipt": json.loads((tmp_path / "capture" / "raw-capture.json").read_text())["receipt"],
        "files": {
            "logits": "logits.f32",
            "layer_inputs": ["layer-003.f32"],
            "alignment": "alignment.jsonl",
            "tokenizer": "tokenizer.tsv",
        },
    }
    with pytest.raises(ValueError, match="exceeds model layer count"):
        RawCapture.model_validate(raw)
    raw["layers"] = [0]
    raw["files"]["layer_inputs"] = ["layer-000.f32"]
    raw["row_count"] = 1_000_000
    raw["vocab_size"] = 1_000_000
    with pytest.raises(ValueError, match="per-artifact byte limit"):
        RawCapture.model_validate(raw)


def test_profile_tokenizer_identity_is_pair_bound(tmp_path: Path) -> None:
    candidate_request, _ = _raw_capture(tmp_path / "candidate")
    candidate = finalize_capture(candidate_request)
    reference_request, reference_output = _raw_capture(
        tmp_path / "reference", role=CaptureRole.IDENTITY_CONTROL
    )
    tokenizer = Path(reference_request.profile_tokenizer_path)
    tokenizer.write_bytes(b"different-profile-tokenizer")
    model_manifest = Path(reference_request.model_artifact_manifest_path)
    model_evidence = json.loads(model_manifest.read_text())
    model_evidence["profile_tokenizer_sha256"] = _sha(tokenizer.read_bytes())
    model_manifest.write_text(json.dumps(model_evidence))
    reference_request = CaptureRequest.model_validate(
        {
            **reference_request.model_dump(mode="json"),
            "profile_tokenizer_sha256": _sha(tokenizer.read_bytes()),
            "model_artifact_manifest_sha256": _sha(model_manifest.read_bytes()),
            "request_id": None,
        }
    )
    raw_path = reference_output / "raw-capture.json"
    raw = json.loads(raw_path.read_text())
    raw["receipt"]["profile_tokenizer_sha256"] = reference_request.profile_tokenizer_sha256
    raw["receipt"]["model_artifact_manifest_sha256"] = (
        reference_request.model_artifact_manifest_sha256
    )
    raw["receipt"]["request_id"] = reference_request.request_id
    raw_path.write_text(json.dumps(raw))
    reference = finalize_capture(reference_request)
    with pytest.raises(CaptureValidationError, match="exact candidate model"):
        validate_capture_pair(reference, candidate)


def test_streaming_identity_control_kld_and_cka(tmp_path: Path) -> None:
    candidate_request, candidate_root = _raw_capture(tmp_path / "candidate")
    candidate = finalize_capture(candidate_request)
    reference_request, reference_root = _raw_capture(
        tmp_path / "reference", role=CaptureRole.IDENTITY_CONTROL
    )
    reference = finalize_capture(reference_request)

    report = evaluate_capture_pair(
        reference_root=reference_root,
        reference=reference,
        candidate_root=candidate_root,
        candidate=candidate,
        cka_rows=3,
    )
    assert report.identity_control_passed is True
    assert report.kld.report.overall.token_weighted_mean == pytest.approx(0.0, abs=1e-12)
    assert all(result.score == pytest.approx(1.0) for result in report.layer_cka)


def test_full_vocab_kld_row_matches_analytic_binary_case() -> None:
    measured = full_vocab_kld_row(
        [math.log(0.8), math.log(0.2)],
        [math.log(0.5), math.log(0.5)],
    )
    expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
    assert measured == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_streaming_metrics_reject_tamper_and_false_identity(tmp_path: Path) -> None:
    candidate_request, candidate_root = _raw_capture(tmp_path / "candidate")
    logits = candidate_root / "logits.f32"
    values = list(struct.unpack("<12f", logits.read_bytes()))
    values = [value + 7.0 for value in values]
    logits.write_bytes(struct.pack("<12f", *values))
    for layer in (0, 2):
        path = candidate_root / f"layer-{layer:03d}.f32"
        hidden = list(struct.unpack("<9f", path.read_bytes()))
        path.write_bytes(struct.pack("<9f", *(value * 3.0 for value in hidden)))
    candidate = finalize_capture(candidate_request)
    reference_request, reference_root = _raw_capture(
        tmp_path / "reference", role=CaptureRole.IDENTITY_CONTROL
    )
    reference = finalize_capture(reference_request)
    with pytest.raises(CaptureMetricError, match="identity-control"):
        evaluate_capture_pair(
            reference_root=reference_root,
            reference=reference,
            candidate_root=candidate_root,
            candidate=candidate,
            cka_rows=3,
        )

    clean_request, clean_root = _raw_capture(tmp_path / "clean")
    clean = finalize_capture(clean_request)
    (clean_root / "logits.f32").write_bytes(b"tampered")
    with pytest.raises(CaptureMetricError, match="size/type"):
        evaluate_capture_pair(
            reference_root=reference_root,
            reference=reference,
            candidate_root=clean_root,
            candidate=clean,
            cka_rows=3,
        )
