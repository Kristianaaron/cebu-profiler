"""Pinned llama.cpp teacher-forced capture contracts and artifact finalizer.

The C++ helper writes a private raw directory.  This module independently
validates its dimensions, exact held-out token alignment, tokenizer census,
file sizes, finite FP32 payloads, and hashes before publishing a content-
addressed manifest.  It never launches the helper or a model runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LLAMA_CPP_CAPTURE_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"
CAPTURE_ADAPTER_VERSION = "atlas-llamacpp-capture-v1"
_SHA256 = r"^[0-9a-f]{64}$"
_PLAN_ID = r"^recipe-[0-9a-f]{24}$"
_RUN_ID = r"^run-[0-9a-f]{24}$"
_PROFILE_ID = r"^profile-[0-9a-f]{24}$"
_RECOMMENDATION_ID = r"^rec-[0-9a-f]{24}$"
_REVISION = r"^[0-9a-f]{40}$"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_TOKENS_BYTES = 64 * 1024 * 1024
_MAX_ALIGNMENT_BYTES = 512 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 512 * 1024 * 1024
_MAX_ROWS = 1_000_000
_MAX_DIMENSION = 1_000_000
_MAX_LAYERS = 64
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024 * 1024
_MAX_AGGREGATE_BYTES = 64 * 1024 * 1024 * 1024
_FLOAT_CHUNK_BYTES = 4 * 1024 * 1024
_SPLIT_GGUF = re.compile(r".*-\d{5}-of-\d{5}\.gguf$")


class CaptureValidationError(RuntimeError):
    """A raw capture is not safe to publish as measured evidence."""


class _Frozen(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class CaptureRole(StrEnum):
    CANDIDATE = "candidate"
    IDENTITY_CONTROL = "identity_control"
    NVFP4_SOURCE_REFERENCE = "nvfp4_source_reference"
    BF16_TEACHER = "bf16_teacher"


class CaptureToolIdentity(_Frozen):
    llama_cpp_commit: str = Field(pattern=_REVISION)
    binary_path: str = Field(pattern=r"^/")
    binary_sha256: str = Field(pattern=_SHA256)
    build_contract_path: str = Field(pattern=r"^/")
    build_contract_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _pinned_revision(self) -> CaptureToolIdentity:
        if self.llama_cpp_commit != LLAMA_CPP_CAPTURE_COMMIT:
            raise ValueError("capture tool revision is not the reviewed llama.cpp pin")
        return self


class PrecisionEvidence(_Frozen):
    schema_version: Literal[1]
    model_sha256: str = Field(pattern=_SHA256)
    model_artifact_manifest_sha256: str = Field(pattern=_SHA256)
    precision: Literal["bf16", "nvfp4"]
    producer: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    evidence_kind: Literal["measured"]


class CaptureModelEvidence(_Frozen):
    schema_version: Literal[1]
    model_sha256: str = Field(pattern=_SHA256)
    # For a sharded source this is its canonical recursive manifest digest.
    source_model_sha256: str | None = Field(default=None, pattern=_SHA256)
    profile_tokenizer_sha256: str = Field(pattern=_SHA256)
    profile_id: str = Field(pattern=_PROFILE_ID)
    profile_sha256: str = Field(pattern=_SHA256)
    recommendation_id: str = Field(pattern=_RECOMMENDATION_ID)
    compression_handoff_sha256: str = Field(pattern=_SHA256)
    producer_artifact_sha256: str = Field(pattern=_SHA256)
    recipe_sha256: str = Field(pattern=_SHA256)
    # These are Atlas' exact externally visible IDs, not full content digests.
    # The full recipe digest is carried separately in ``recipe_sha256``.
    plan_id: str = Field(pattern=_PLAN_ID)
    run_id: str = Field(pattern=_RUN_ID)
    evidence_kind: Literal["measured"]


class CaptureRequest(_Frozen):
    schema_version: Literal[1] = 1
    request_id: str | None = Field(default=None, pattern=_SHA256)
    model_id: str = Field(min_length=1)
    model_path: str = Field(pattern=r"^/")
    model_sha256: str = Field(pattern=_SHA256)
    model_artifact_manifest_path: str = Field(pattern=r"^/")
    model_artifact_manifest_sha256: str = Field(pattern=_SHA256)
    source_model_sha256: str | None = Field(default=None, pattern=_SHA256)
    role: CaptureRole
    reference_kind: str = Field(min_length=1)
    precision_evidence_path: str | None = Field(default=None, pattern=r"^/")
    precision_evidence_sha256: str | None = Field(default=None, pattern=_SHA256)
    forced_tokens_path: str = Field(pattern=r"^/")
    forced_tokens_sha256: str = Field(pattern=_SHA256)
    held_out_manifest_path: str = Field(pattern=r"^/")
    held_out_manifest_sha256: str = Field(pattern=_SHA256)
    ordered_sample_ids_sha256: str = Field(pattern=_SHA256)
    profile_tokenizer_path: str = Field(pattern=r"^/")
    profile_tokenizer_sha256: str = Field(pattern=_SHA256)
    output_dir: str = Field(pattern=r"^/")
    layers: tuple[int, ...] = Field(min_length=1)
    context_tokens: int = Field(gt=1)
    batch_tokens: int = Field(gt=0)
    ubatch_tokens: int = Field(gt=0)
    threads: int = Field(gt=0)
    runtime_argv_sha256: str = Field(pattern=_SHA256)
    tool: CaptureToolIdentity

    @field_validator("layers")
    @classmethod
    def _ordered_layers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) > _MAX_LAYERS:
            raise ValueError("capture request exceeds the layer-count limit")
        if any(layer < 0 for layer in value):
            raise ValueError("capture layers must be non-negative")
        if tuple(sorted(set(value))) != value:
            raise ValueError("capture layers must be sorted and unique")
        return value

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"request_id"})

    @model_validator(mode="after")
    def _identity(self) -> CaptureRequest:
        if self.ubatch_tokens > self.batch_tokens:
            raise ValueError("ubatch_tokens cannot exceed batch_tokens")
        if _SPLIT_GGUF.fullmatch(Path(self.model_path).name):
            raise ValueError("capture v1 requires a single-file GGUF model")
        expected_reference = {
            CaptureRole.CANDIDATE: "candidate",
            CaptureRole.IDENTITY_CONTROL: "identity_control",
            CaptureRole.NVFP4_SOURCE_REFERENCE: "nvfp4_source_relative",
            CaptureRole.BF16_TEACHER: "bf16",
        }[self.role]
        if self.reference_kind != expected_reference:
            raise ValueError("capture role and reference_kind disagree")
        requires_precision_evidence = self.role in {
            CaptureRole.NVFP4_SOURCE_REFERENCE,
            CaptureRole.BF16_TEACHER,
        }
        has_precision_evidence = (
            self.precision_evidence_path is not None and self.precision_evidence_sha256 is not None
        )
        if requires_precision_evidence != has_precision_evidence:
            raise ValueError("capture role and precision evidence disagree")
        if (self.precision_evidence_path is None) != (self.precision_evidence_sha256 is None):
            raise ValueError("precision evidence path and digest must be supplied together")
        carries_candidate_lineage = self.role in {
            CaptureRole.CANDIDATE,
            CaptureRole.IDENTITY_CONTROL,
        }
        if carries_candidate_lineage != (self.source_model_sha256 is not None):
            raise ValueError("candidate and identity captures require source model lineage")
        expected = _canonical_sha256(self.identity_payload())
        if self.request_id is not None and self.request_id != expected:
            raise ValueError("capture request_id does not match canonical content")
        object.__setattr__(self, "request_id", expected)
        return self


class RawCaptureFiles(_Frozen):
    logits: Literal["logits.f32"]
    layer_inputs: tuple[str, ...] = Field(min_length=1)
    alignment: Literal["alignment.jsonl"]
    tokenizer: Literal["tokenizer.tsv"]


class RawRuntimeParams(_Frozen):
    model_path: str = Field(pattern=r"^/")
    tokens_jsonl: str = Field(pattern=r"^/")
    output_dir: str = Field(pattern=r"^/")
    layers: tuple[int, ...] = Field(min_length=1)
    context_tokens: int = Field(gt=1)
    batch_tokens: int = Field(gt=0)
    ubatch_tokens: int = Field(gt=0)
    threads: int = Field(gt=0)
    threads_batch: int = Field(gt=0)
    split_mode: str = Field(min_length=1)
    n_gpu_layers: int
    main_gpu: int = Field(ge=0)
    fit_params: bool
    devices: tuple[str, ...] = Field(min_length=1)
    warmup: bool


class RawCaptureReceipt(_Frozen):
    request_id: str = Field(pattern=_SHA256)
    model_sha256: str = Field(pattern=_SHA256)
    measured_model_sha256: str = Field(pattern=_SHA256)
    model_artifact_manifest_sha256: str = Field(pattern=_SHA256)
    tool_binary_sha256: str = Field(pattern=_SHA256)
    measured_tool_binary_sha256: str = Field(pattern=_SHA256)
    tool_build_contract_sha256: str = Field(pattern=_SHA256)
    forced_tokens_sha256: str = Field(pattern=_SHA256)
    measured_forced_tokens_sha256: str = Field(pattern=_SHA256)
    held_out_manifest_sha256: str = Field(pattern=_SHA256)
    ordered_sample_ids_sha256: str = Field(pattern=_SHA256)
    profile_tokenizer_sha256: str = Field(pattern=_SHA256)
    runtime_argv_sha256: str = Field(pattern=_SHA256)
    role: CaptureRole
    reference_kind: str = Field(min_length=1)
    layers: tuple[int, ...] = Field(min_length=1)
    normalized_runtime_argv: tuple[str, ...] = Field(min_length=1)
    runtime_params: RawRuntimeParams


class RawCapture(_Frozen):
    schema_version: Literal[1]
    capture_mode: Literal["teacher_forced"]
    vocab_size: int = Field(gt=1, le=_MAX_DIMENSION)
    hidden_size: int = Field(gt=0, le=_MAX_DIMENSION)
    n_hidden_layers: int = Field(gt=0, le=100_000)
    row_count: int = Field(gt=0, le=_MAX_ROWS)
    sample_count: int = Field(gt=0, le=4096)
    layers: tuple[int, ...] = Field(min_length=1)
    receipt: RawCaptureReceipt
    files: RawCaptureFiles

    @model_validator(mode="after")
    def _consistent_layers(self) -> RawCapture:
        if tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("raw capture layers must be sorted and unique")
        if len(self.layers) > _MAX_LAYERS:
            raise ValueError("raw capture requests too many layers")
        if any(layer >= self.n_hidden_layers for layer in self.layers):
            raise ValueError("raw capture layer exceeds model layer count")
        logits_bytes = self.row_count * self.vocab_size * 4
        layer_bytes = self.row_count * self.hidden_size * 4
        if logits_bytes > _MAX_ARTIFACT_BYTES or layer_bytes > _MAX_ARTIFACT_BYTES:
            raise ValueError("raw capture tensor exceeds the per-artifact byte limit")
        if logits_bytes + layer_bytes * len(self.layers) > _MAX_AGGREGATE_BYTES:
            raise ValueError("raw capture tensors exceed the aggregate byte limit")
        expected = tuple(_layer_name(layer) for layer in self.layers)
        if self.files.layer_inputs != expected:
            raise ValueError("raw layer file names do not match layer identities")
        return self


class AlignmentRow(_Frozen):
    row: int = Field(ge=0)
    sample_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    input_position: int = Field(ge=0)
    target_position: int = Field(gt=0)
    target_token_id: int = Field(ge=0)

    @model_validator(mode="after")
    def _next_token(self) -> AlignmentRow:
        if self.target_position != self.input_position + 1:
            raise ValueError("alignment target must be the next token position")
        return self


class CaptureArtifact(_Frozen):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(ge=0)
    shape: tuple[int, ...] = Field(min_length=1)
    dtype: str = Field(min_length=1)


class CaptureManifest(_Frozen):
    schema_version: Literal[1] = 1
    capture_id: str | None = Field(default=None, pattern=_SHA256)
    request: CaptureRequest
    raw_capture_sha256: str = Field(pattern=_SHA256)
    alignment_sha256: str = Field(pattern=_SHA256)
    tokenizer_table_sha256: str = Field(pattern=_SHA256)
    row_count: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    vocab_size: int = Field(gt=1)
    hidden_size: int = Field(gt=0)
    n_hidden_layers: int = Field(gt=0)
    artifacts: tuple[CaptureArtifact, ...] = Field(min_length=3)

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"capture_id"})

    @model_validator(mode="after")
    def _identity_and_files(self) -> CaptureManifest:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("capture artifact names must be unique")
        expected = _canonical_sha256(self.identity_payload())
        if self.capture_id is not None and self.capture_id != expected:
            raise ValueError("capture_id does not match canonical content")
        object.__setattr__(self, "capture_id", expected)
        return self


class CapturePair(_Frozen):
    reference_capture_id: str = Field(pattern=_SHA256)
    candidate_capture_id: str = Field(pattern=_SHA256)
    alignment_sha256: str = Field(pattern=_SHA256)
    tokenizer_table_sha256: str = Field(pattern=_SHA256)
    row_count: int = Field(gt=0)
    vocab_size: int = Field(gt=1)
    layers: tuple[int, ...] = Field(min_length=1)
    identity_control: bool


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _canonical_value_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _layer_name(layer: int) -> str:
    return f"layer-{layer:03d}.f32"


def canonical_capture_runtime_argv(
    *,
    tool_path: str,
    common_argv: tuple[str, ...],
    forced_tokens_path: str,
    output_dir: str,
    layers: tuple[int, ...],
) -> tuple[str, ...]:
    """Mirror the native tool's canonical argv receipt without executing it."""

    inputs = (Path(tool_path), Path(forced_tokens_path))
    output = Path(output_dir)
    canonical_output = output.parent.resolve(strict=True) / output.name
    if (
        any(not path.is_absolute() or path != path.resolve(strict=True) for path in inputs)
        or not output.is_absolute()
        or output != canonical_output
    ):
        raise CaptureValidationError("capture argv paths must be canonical and existing")
    custom_prefixes = {
        "--tokens-jsonl",
        "--out-dir",
        "--layers",
        "--request-id",
        "--model-sha256",
        "--model-artifact-manifest-sha256",
        "--tool-binary-sha256",
        "--build-contract-sha256",
        "--forced-tokens-sha256",
        "--held-out-manifest-sha256",
        "--ordered-sample-ids-sha256",
        "--profile-tokenizer-sha256",
        "--runtime-argv-sha256",
        "--role",
        "--reference-kind",
    }
    if any(
        argument in custom_prefixes
        or "=" in argument
        or argument in {"-m", "-c", "-b", "-ub", "-t", "-tb"}
        for argument in common_argv
    ):
        raise CaptureValidationError("common capture argv must use canonical long-form options")
    return (
        tool_path,
        *common_argv,
        "--tokens-jsonl",
        forced_tokens_path,
        "--out-dir",
        output_dir,
        "--layers",
        ",".join(str(layer) for layer in layers),
    )


def capture_runtime_argv_sha256(argv: tuple[str, ...]) -> str:
    return _canonical_value_sha256(list(argv))


def _single_option(common_argv: tuple[str, ...], option: str) -> str:
    positions = [index for index, value in enumerate(common_argv) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(common_argv):
        raise CaptureValidationError(f"capture argv requires exactly one {option}")
    return common_argv[positions[0] + 1]


def build_capture_argv(request: CaptureRequest, *, common_argv: tuple[str, ...]) -> tuple[str, ...]:
    """Build the exact non-shell native invocation after contract verification."""

    normalized = canonical_capture_runtime_argv(
        tool_path=request.tool.binary_path,
        common_argv=common_argv,
        forced_tokens_path=request.forced_tokens_path,
        output_dir=request.output_dir,
        layers=request.layers,
    )
    if capture_runtime_argv_sha256(normalized) != request.runtime_argv_sha256:
        raise CaptureValidationError("capture argv differs from the request digest")
    expected_options = {
        "--model": request.model_path,
        "--ctx-size": str(request.context_tokens),
        "--batch-size": str(request.batch_tokens),
        "--ubatch-size": str(request.ubatch_tokens),
        "--threads": str(request.threads),
    }
    if any(_single_option(common_argv, key) != value for key, value in expected_options.items()):
        raise CaptureValidationError("capture argv core parameters differ from the request")
    if _single_option(common_argv, "--model") != str(Path(request.model_path).resolve(strict=True)):
        raise CaptureValidationError("capture argv model path is not canonical")
    if "--no-warmup" not in common_argv:
        raise CaptureValidationError("capture argv must disable warmup")
    receipt_args = (
        "--tokens-jsonl",
        request.forced_tokens_path,
        "--out-dir",
        request.output_dir,
        "--layers",
        ",".join(str(layer) for layer in request.layers),
        "--request-id",
        request.request_id or "",
        "--model-sha256",
        request.model_sha256,
        "--model-artifact-manifest-sha256",
        request.model_artifact_manifest_sha256,
        "--tool-binary-sha256",
        request.tool.binary_sha256,
        "--build-contract-sha256",
        request.tool.build_contract_sha256,
        "--forced-tokens-sha256",
        request.forced_tokens_sha256,
        "--held-out-manifest-sha256",
        request.held_out_manifest_sha256,
        "--ordered-sample-ids-sha256",
        request.ordered_sample_ids_sha256,
        "--profile-tokenizer-sha256",
        request.profile_tokenizer_sha256,
        "--runtime-argv-sha256",
        request.runtime_argv_sha256,
        "--role",
        request.role.value,
        "--reference-kind",
        request.reference_kind,
    )
    return (request.tool.binary_path, *receipt_args, *common_argv)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_bounded_regular(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise CaptureValidationError(f"{path.name} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(encoded) != before.st_size or before_identity != after_identity:
            raise CaptureValidationError(f"{path.name} changed during bounded read")
        return encoded
    finally:
        os.close(descriptor)


def _read_bounded_at(
    root_fd: int, name: str, limit: int
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise CaptureValidationError(f"{name} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(encoded) != before.st_size or _stat_identity(before) != _stat_identity(after):
            raise CaptureValidationError(f"{name} changed during bounded read")
        return encoded, _stat_identity(after)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path, *, expected_size: int | None = None) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CaptureValidationError(f"{path.name} must be a regular file")
        if expected_size is not None and before.st_size != expected_size:
            raise CaptureValidationError(f"{path.name} size does not match declared shape")
        while chunk := os.read(descriptor, _FLOAT_CHUNK_BYTES):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if total != before.st_size or _stat_identity(before) != _stat_identity(after):
            raise CaptureValidationError(f"{path.name} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_finite_f32_at(
    root_fd: int, name: str, expected_size: int
) -> tuple[str, tuple[int, int, int, int, int]]:
    if sys.byteorder != "little":
        raise CaptureValidationError("capture FP32 contract requires a little-endian host")
    if expected_size > _MAX_ARTIFACT_BYTES:
        raise CaptureValidationError(f"{name} exceeds the per-artifact byte limit")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    pending = b""
    read_bytes = 0
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise CaptureValidationError(f"{name} size does not match FP32 shape")
        while chunk := os.read(descriptor, _FLOAT_CHUNK_BYTES):
            read_bytes += len(chunk)
            digest.update(chunk)
            data = pending + chunk
            usable = len(data) - (len(data) % 4)
            for (value,) in struct.iter_unpack("<f", data[:usable]):
                if not math.isfinite(value):
                    raise CaptureValidationError(f"{name} contains non-finite FP32")
            pending = data[usable:]
        if pending or read_bytes != expected_size:
            raise CaptureValidationError(f"{name} is not complete FP32 data")
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise CaptureValidationError(f"{name} changed during FP32 validation")
        return digest.hexdigest(), _stat_identity(after)
    finally:
        os.close(descriptor)


def _forced_alignment(request: CaptureRequest) -> tuple[list[AlignmentRow], int]:
    path = Path(request.forced_tokens_path)
    encoded = _read_bounded_regular(path, _MAX_TOKENS_BYTES)
    if hashlib.sha256(encoded).hexdigest() != request.forced_tokens_sha256:
        raise CaptureValidationError("forced-token JSONL digest drifted")
    rows: list[AlignmentRow] = []
    sample_ids: list[str] = []
    for line_number, line in enumerate(encoded.splitlines(), 1):
        try:
            value = json.loads(line)
        except (TypeError, ValueError) as exc:
            raise CaptureValidationError(
                f"forced-token JSONL line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict) or set(value) != {"sample_id", "domain", "token_ids"}:
            raise CaptureValidationError("forced-token record fields do not match contract")
        sample_id = value["sample_id"]
        domain = value["domain"]
        token_ids = value["token_ids"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(domain, str)
            or not domain
            or not isinstance(token_ids, list)
            or len(token_ids) < 2
            or len(token_ids) > request.context_tokens
            or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids)
        ):
            raise CaptureValidationError("forced-token record is invalid")
        sample_ids.append(sample_id)
        for position, target in enumerate(token_ids[1:]):
            if target < 0:
                raise CaptureValidationError("forced token IDs must be non-negative")
            rows.append(
                AlignmentRow(
                    row=len(rows),
                    sample_id=sample_id,
                    domain=domain,
                    input_position=position,
                    target_position=position + 1,
                    target_token_id=target,
                )
            )
    if not sample_ids:
        raise CaptureValidationError("forced-token corpus is empty")
    ordered_digest = hashlib.sha256(
        json.dumps(sample_ids, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if ordered_digest != request.ordered_sample_ids_sha256:
        raise CaptureValidationError("ordered sample identity digest drifted")
    return rows, len(sample_ids)


def _validate_alignment(
    root_fd: int, name: str, expected: list[AlignmentRow]
) -> tuple[str, int, tuple[int, int, int, int, int]]:
    encoded, identity = _read_bounded_at(root_fd, name, _MAX_ALIGNMENT_BYTES)
    lines = encoded.splitlines()
    if len(lines) != len(expected):
        raise CaptureValidationError("alignment row count differs from forced tokens")
    for index, (line, expected_row) in enumerate(zip(lines, expected, strict=True)):
        try:
            measured = AlignmentRow.model_validate_json(line)
        except ValueError as exc:
            raise CaptureValidationError(f"alignment row {index} is invalid") from exc
        if measured != expected_row:
            raise CaptureValidationError(f"alignment row {index} differs from forced tokens")
    return hashlib.sha256(encoded).hexdigest(), len(encoded), identity


def _validate_tokenizer(
    root_fd: int, name: str, vocab_size: int
) -> tuple[str, int, tuple[int, int, int, int, int]]:
    encoded, identity = _read_bounded_at(root_fd, name, _MAX_TOKENIZER_BYTES)
    lines = encoded.splitlines()
    if len(lines) != vocab_size:
        raise CaptureValidationError("tokenizer table row count differs from vocab_size")
    for expected_id, line in enumerate(lines):
        fields = line.split(b"\t")
        if len(fields) != 2 or fields[0] != str(expected_id).encode("ascii"):
            raise CaptureValidationError("tokenizer table IDs are not canonical and ordered")
        piece = fields[1]
        if len(piece) % 2 or any(byte not in b"0123456789abcdef" for byte in piece):
            raise CaptureValidationError("tokenizer pieces must be lowercase hex")
    return hashlib.sha256(encoded).hexdigest(), len(encoded), identity


def _verify_request_inputs(request: CaptureRequest, root: Path) -> None:
    raw_paths = [
        Path(request.model_path),
        Path(request.model_artifact_manifest_path),
        Path(request.tool.binary_path),
        Path(request.tool.build_contract_path),
        Path(request.forced_tokens_path),
        Path(request.held_out_manifest_path),
        Path(request.profile_tokenizer_path),
    ]
    if request.precision_evidence_path is not None:
        raw_paths.append(Path(request.precision_evidence_path))
    for path in raw_paths:
        if not path.is_absolute() or path.is_symlink():
            raise CaptureValidationError("capture request inputs must be absolute non-symlinks")
    try:
        model = Path(request.model_path).resolve(strict=True)
        model_manifest = Path(request.model_artifact_manifest_path).resolve(strict=True)
        tool = Path(request.tool.binary_path).resolve(strict=True)
        tool_contract = Path(request.tool.build_contract_path).resolve(strict=True)
        tokens = Path(request.forced_tokens_path).resolve(strict=True)
        held_out = Path(request.held_out_manifest_path).resolve(strict=True)
        tokenizer = Path(request.profile_tokenizer_path).resolve(strict=True)
        precision = (
            Path(request.precision_evidence_path).resolve(strict=True)
            if request.precision_evidence_path is not None
            else None
        )
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CaptureValidationError("capture request input path is unavailable") from exc
    if any(path != path.resolve(strict=True) for path in raw_paths):
        raise CaptureValidationError("capture request input path traverses a symlink")
    inputs = [
        ("model", model),
        ("model artifact manifest", model_manifest),
        ("tool", tool),
        ("tool build contract", tool_contract),
        ("forced tokens", tokens),
        ("held-out manifest", held_out),
        ("profile tokenizer", tokenizer),
    ]
    if precision is not None:
        inputs.append(("precision evidence", precision))
    for name, path in inputs:
        if path == resolved_root or resolved_root in path.parents or path in resolved_root.parents:
            raise CaptureValidationError(f"capture output overlaps {name} input")
    if _sha256_file(model) != request.model_sha256:
        raise CaptureValidationError("capture model bytes drifted from the request")
    model_manifest_bytes = _read_bounded_regular(model_manifest, _MAX_JSON_BYTES)
    if hashlib.sha256(model_manifest_bytes).hexdigest() != request.model_artifact_manifest_sha256:
        raise CaptureValidationError("model artifact manifest bytes drifted from the request")
    try:
        model_evidence = CaptureModelEvidence.model_validate_json(model_manifest_bytes)
    except ValueError as exc:
        raise CaptureValidationError("model artifact evidence schema is invalid") from exc
    if (
        model_evidence.model_sha256 != request.model_sha256
        or model_evidence.source_model_sha256 != request.source_model_sha256
        or model_evidence.profile_tokenizer_sha256 != request.profile_tokenizer_sha256
    ):
        raise CaptureValidationError("model artifact evidence does not match the request")
    if _sha256_file(tool) != request.tool.binary_sha256:
        raise CaptureValidationError("capture tool bytes drifted from the request")
    if _sha256_file(tool_contract) != request.tool.build_contract_sha256:
        raise CaptureValidationError("capture tool build contract drifted from the request")
    if _sha256_file(held_out) != request.held_out_manifest_sha256:
        raise CaptureValidationError("held-out manifest bytes drifted from the request")
    if _sha256_file(tokenizer) != request.profile_tokenizer_sha256:
        raise CaptureValidationError("profile tokenizer bytes drifted from the request")
    if precision is not None:
        precision_bytes = _read_bounded_regular(precision, _MAX_JSON_BYTES)
        if (
            request.precision_evidence_sha256 is None
            or hashlib.sha256(precision_bytes).hexdigest() != request.precision_evidence_sha256
        ):
            raise CaptureValidationError("precision evidence bytes drifted from the request")
        try:
            evidence = PrecisionEvidence.model_validate_json(precision_bytes)
        except ValueError as exc:
            raise CaptureValidationError("precision evidence schema is invalid") from exc
        expected_precision = "bf16" if request.role is CaptureRole.BF16_TEACHER else "nvfp4"
        if (
            evidence.model_sha256 != request.model_sha256
            or evidence.model_artifact_manifest_sha256 != request.model_artifact_manifest_sha256
            or evidence.precision != expected_precision
        ):
            raise CaptureValidationError("precision evidence does not describe the model")


def _exclusive_json(root_fd: int, name: str, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    temporary = f".{name}.tmp-{os.getpid()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=root_fd,
    )
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("incomplete capture manifest write")
        os.fsync(descriptor)
    except BaseException:
        os.unlink(temporary, dir_fd=root_fd)
        raise
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
            follow_symlinks=False,
        )
        try:
            os.fsync(root_fd)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(name, dir_fd=root_fd)
                os.fsync(root_fd)
            raise
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=root_fd)


def finalize_capture(request: CaptureRequest) -> CaptureManifest:
    """Validate and publish ``capture-manifest.json`` without executing a model."""

    root = Path(request.output_dir)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise CaptureValidationError("capture output must be an absolute real directory")
    if root != root.resolve(strict=True):
        raise CaptureValidationError("capture output path must be canonical and symlink-free")
    _verify_request_inputs(request, root)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        root_before = os.fstat(root_fd)
        raw_bytes, raw_identity = _read_bounded_at(root_fd, "raw-capture.json", _MAX_JSON_BYTES)
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        try:
            raw = RawCapture.model_validate_json(raw_bytes)
        except ValueError as exc:
            raise CaptureValidationError("raw capture manifest is invalid") from exc
        if raw.layers != request.layers:
            raise CaptureValidationError("captured layers differ from the request")
        if request.request_id is None:
            raise CaptureValidationError("capture request identity is incomplete")
        expected_receipt = (
            request.request_id,
            request.model_sha256,
            request.model_artifact_manifest_sha256,
            request.tool.binary_sha256,
            request.tool.build_contract_sha256,
            request.forced_tokens_sha256,
            request.held_out_manifest_sha256,
            request.ordered_sample_ids_sha256,
            request.profile_tokenizer_sha256,
            request.runtime_argv_sha256,
            request.role,
            request.reference_kind,
        )
        measured_receipt = (
            raw.receipt.request_id,
            raw.receipt.model_sha256,
            raw.receipt.model_artifact_manifest_sha256,
            raw.receipt.tool_binary_sha256,
            raw.receipt.tool_build_contract_sha256,
            raw.receipt.forced_tokens_sha256,
            raw.receipt.held_out_manifest_sha256,
            raw.receipt.ordered_sample_ids_sha256,
            raw.receipt.profile_tokenizer_sha256,
            raw.receipt.runtime_argv_sha256,
            raw.receipt.role,
            raw.receipt.reference_kind,
        )
        if measured_receipt != expected_receipt:
            raise CaptureValidationError("native capture receipt differs from request")
        if (
            raw.receipt.measured_model_sha256 != request.model_sha256
            or raw.receipt.measured_tool_binary_sha256 != request.tool.binary_sha256
            or raw.receipt.measured_forced_tokens_sha256 != request.forced_tokens_sha256
        ):
            raise CaptureValidationError("native capture measurements differ from request")
        if (
            _canonical_value_sha256(list(raw.receipt.normalized_runtime_argv))
            != request.runtime_argv_sha256
        ):
            raise CaptureValidationError("native runtime argv differs from its bound digest")
        runtime = raw.receipt.runtime_params
        if (
            raw.receipt.layers != request.layers
            or runtime.layers != request.layers
            or runtime.context_tokens != request.context_tokens
            or runtime.batch_tokens != request.batch_tokens
            or runtime.ubatch_tokens != request.ubatch_tokens
            or runtime.threads != request.threads
            or Path(runtime.model_path) != Path(request.model_path)
            or Path(runtime.tokens_jsonl) != Path(request.forced_tokens_path)
            or Path(runtime.output_dir) != root
            or runtime.warmup
        ):
            raise CaptureValidationError("native runtime parameters differ from request")

        expected_rows, sample_count = _forced_alignment(request)
        if raw.row_count != len(expected_rows) or raw.sample_count != sample_count:
            raise CaptureValidationError("raw capture counts differ from forced-token inputs")
        if any(row.target_token_id >= raw.vocab_size for row in expected_rows):
            raise CaptureValidationError("forced token ID exceeds captured vocabulary")

        allowed = {"raw-capture.json", "alignment.jsonl", "tokenizer.tsv", "logits.f32"}
        allowed.update(raw.files.layer_inputs)
        names = set(os.listdir(root_fd))
        if names != allowed:
            raise CaptureValidationError("capture directory file set differs from raw manifest")
        for name in names:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise CaptureValidationError("capture directory must contain regular files only")

        alignment_sha, alignment_size, alignment_identity = _validate_alignment(
            root_fd, raw.files.alignment, expected_rows
        )
        tokenizer_sha, tokenizer_size, tokenizer_identity = _validate_tokenizer(
            root_fd, raw.files.tokenizer, raw.vocab_size
        )
        artifact_identities = {
            "raw-capture.json": raw_identity,
            raw.files.alignment: alignment_identity,
            raw.files.tokenizer: tokenizer_identity,
        }
        artifacts: list[CaptureArtifact] = []
        aggregate_bytes = alignment_size + tokenizer_size + len(raw_bytes)
        logits_size = raw.row_count * raw.vocab_size * 4
        logits_sha, logits_identity = _validate_finite_f32_at(
            root_fd, raw.files.logits, logits_size
        )
        artifact_identities[raw.files.logits] = logits_identity
        aggregate_bytes += logits_size
        artifacts.append(
            CaptureArtifact(
                name=raw.files.logits,
                sha256=logits_sha,
                size_bytes=logits_size,
                shape=(raw.row_count, raw.vocab_size),
                dtype="float32-le",
            )
        )
        for name in raw.files.layer_inputs:
            size = raw.row_count * raw.hidden_size * 4
            layer_sha, layer_identity = _validate_finite_f32_at(root_fd, name, size)
            artifact_identities[name] = layer_identity
            aggregate_bytes += size
            if aggregate_bytes > _MAX_AGGREGATE_BYTES:
                raise CaptureValidationError("capture exceeds aggregate byte limit")
            artifacts.append(
                CaptureArtifact(
                    name=name,
                    sha256=layer_sha,
                    size_bytes=size,
                    shape=(raw.row_count, raw.hidden_size),
                    dtype="float32-le",
                )
            )
        artifacts.extend(
            (
                CaptureArtifact(
                    name=raw.files.alignment,
                    sha256=alignment_sha,
                    size_bytes=alignment_size,
                    shape=(raw.row_count,),
                    dtype="jsonl",
                ),
                CaptureArtifact(
                    name=raw.files.tokenizer,
                    sha256=tokenizer_sha,
                    size_bytes=tokenizer_size,
                    shape=(raw.vocab_size,),
                    dtype="token-id-tab-hex-piece",
                ),
            )
        )
        manifest = CaptureManifest(
            request=request,
            raw_capture_sha256=raw_sha,
            alignment_sha256=alignment_sha,
            tokenizer_table_sha256=tokenizer_sha,
            row_count=raw.row_count,
            sample_count=raw.sample_count,
            vocab_size=raw.vocab_size,
            hidden_size=raw.hidden_size,
            n_hidden_layers=raw.n_hidden_layers,
            artifacts=tuple(artifacts),
        )
        if _stat_identity(root_before) != _stat_identity(os.fstat(root_fd)):
            raise CaptureValidationError("capture output directory changed during validation")
        for name, identity in artifact_identities.items():
            if _stat_identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != identity:
                raise CaptureValidationError("capture artifact changed before publication")
        _exclusive_json(root_fd, "capture-manifest.json", manifest.model_dump(mode="json"))
        if set(os.listdir(root_fd)) != allowed | {"capture-manifest.json"}:
            with suppress(FileNotFoundError):
                os.unlink("capture-manifest.json", dir_fd=root_fd)
                os.fsync(root_fd)
            raise CaptureValidationError("capture directory changed during publication")
        for name, identity in artifact_identities.items():
            if _stat_identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != identity:
                with suppress(FileNotFoundError):
                    os.unlink("capture-manifest.json", dir_fd=root_fd)
                    os.fsync(root_fd)
                raise CaptureValidationError("capture artifact changed during publication")
        return manifest
    finally:
        os.close(root_fd)


def validate_capture_pair(reference: CaptureManifest, candidate: CaptureManifest) -> CapturePair:
    """Fail closed unless two captures are exactly aligned for KLD and CKA."""

    if candidate.request.role is not CaptureRole.CANDIDATE:
        raise CaptureValidationError("candidate side must have the candidate role")
    if reference.request.role not in {
        CaptureRole.BF16_TEACHER,
        CaptureRole.NVFP4_SOURCE_REFERENCE,
        CaptureRole.IDENTITY_CONTROL,
    }:
        raise CaptureValidationError("reference side has no teacher/reference role")
    identity_control = reference.request.role is CaptureRole.IDENTITY_CONTROL
    if identity_control and (
        reference.request.model_sha256 != candidate.request.model_sha256
        or reference.request.model_artifact_manifest_sha256
        != candidate.request.model_artifact_manifest_sha256
    ):
        raise CaptureValidationError("identity control must use the exact candidate model")
    if (
        reference.request.role is CaptureRole.NVFP4_SOURCE_REFERENCE
        and candidate.request.source_model_sha256 != reference.request.model_sha256
    ):
        raise CaptureValidationError("candidate lineage does not derive from NVFP4 reference")

    expected = (
        reference.request.forced_tokens_sha256,
        reference.request.held_out_manifest_sha256,
        reference.request.ordered_sample_ids_sha256,
        reference.request.profile_tokenizer_sha256,
        reference.alignment_sha256,
        reference.tokenizer_table_sha256,
        reference.row_count,
        reference.vocab_size,
        reference.hidden_size,
        reference.request.layers,
    )
    measured = (
        candidate.request.forced_tokens_sha256,
        candidate.request.held_out_manifest_sha256,
        candidate.request.ordered_sample_ids_sha256,
        candidate.request.profile_tokenizer_sha256,
        candidate.alignment_sha256,
        candidate.tokenizer_table_sha256,
        candidate.row_count,
        candidate.vocab_size,
        candidate.hidden_size,
        candidate.request.layers,
    )
    if expected != measured:
        raise CaptureValidationError("capture pair is not token/layer aligned")
    if reference.capture_id is None or candidate.capture_id is None:
        raise CaptureValidationError("capture pair identities are incomplete")
    return CapturePair(
        reference_capture_id=reference.capture_id,
        candidate_capture_id=candidate.capture_id,
        alignment_sha256=reference.alignment_sha256,
        tokenizer_table_sha256=reference.tokenizer_table_sha256,
        row_count=reference.row_count,
        vocab_size=reference.vocab_size,
        layers=reference.request.layers,
        identity_control=identity_control,
    )


__all__ = [
    "LLAMA_CPP_CAPTURE_COMMIT",
    "CAPTURE_ADAPTER_VERSION",
    "CaptureValidationError",
    "CaptureRole",
    "CaptureToolIdentity",
    "PrecisionEvidence",
    "CaptureModelEvidence",
    "CaptureRequest",
    "RawCapture",
    "AlignmentRow",
    "CaptureArtifact",
    "CaptureManifest",
    "CapturePair",
    "canonical_capture_runtime_argv",
    "capture_runtime_argv_sha256",
    "build_capture_argv",
    "finalize_capture",
    "validate_capture_pair",
]
