"""Canonical, non-executing GLM-5.2 candidate/identity capture plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_atlas.evaluation.llamacpp_capture import (
    LLAMA_CPP_CAPTURE_COMMIT,
    CaptureModelEvidence,
    CaptureRequest,
    CaptureRole,
    CaptureToolIdentity,
    capture_runtime_argv_sha256,
)

CAPTURE_BINARY = Path(
    "/home/kristianaaron/tmp/atlas-toolchains/llama.cpp/"
    "build-atlas-capture/llama-atlas-capture"
)
CAPTURE_BINARY_SHA256 = "d41b76cfc042b868b86f2e23431a11da1eb736bf570bb8b3dcb99aa23b3f5859"
CAPTURE_BUILD_CONTRACT = Path(
    "/home/kristianaaron/tmp/model-atlas/artifacts/llama-atlas-capture-build.json"
)
CAPTURE_BUILD_CONTRACT_SHA256 = (
    "bca3446b8a98f06d75822d7e4e5e60e17cefe21ee2c428062da69bba1bd983b8"
)
FORCED_TOKENS = Path(
    "/home/kristianaaron/tmp/model-atlas/artifacts/glm52-capture-canary.jsonl"
)
FORCED_TOKENS_SHA256 = "539e8f6682cfe3768195e65e75ffcd2ba83b229282e79d94983856ff70ad07c8"
HELD_OUT_MANIFEST = Path(
    "/home/kristianaaron/tmp/model-atlas/artifacts/glm52-capture-canary-manifest.json"
)
HELD_OUT_MANIFEST_SHA256 = (
    "a141e026b908d745890213371be8fa6f7fd23e727149ade25d44bd3e5b14a885"
)
ORDERED_SAMPLE_IDS_SHA256 = (
    "8c8500753e0450351aed1ec8fd1fe41920163030bba0f95945fcd226edfeb52f"
)
CAPTURE_LAYERS = (0, 26, 52, 77)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Glm52CapturePlan(_Frozen):
    schema_version: Literal[1] = 1
    plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_evidence: CaptureModelEvidence
    model_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_evidence_path: str = Field(pattern=r"^/")
    common_argv: tuple[str, ...]
    candidate: CaptureRequest
    identity_control: CaptureRequest
    quality_claim: Literal[False] = False

    @model_validator(mode="after")
    def _identity(self) -> Glm52CapturePlan:
        evidence_sha = hashlib.sha256(
            json.dumps(
                self.model_evidence.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if evidence_sha != self.model_evidence_sha256:
            raise ValueError("capture model evidence digest disagrees with its payload")
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.plan_sha256 is not None and self.plan_sha256 != expected:
            raise ValueError("capture plan digest disagrees with canonical content")
        object.__setattr__(self, "plan_sha256", expected)
        return self


def _common_argv(model_path: Path) -> tuple[str, ...]:
    return (
        "--model",
        str(model_path),
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
        "4096",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
        "--flash-attn",
        "auto",
        "--load-mode",
        "auto",
        "--fit",
        "off",
        "--no-warmup",
        "--threads",
        "8",
        "--threads-batch",
        "8",
    )


def build_glm52_capture_plan(
    *,
    work_root: Path,
    model_path: Path,
    model_sha256: str,
    source_manifest_sha256: str,
    profile_tokenizer_path: Path,
    profile_tokenizer_sha256: str,
    producer_artifact_sha256: str,
    recipe_sha256: str,
    plan_id: str,
    run_id: str,
    profile_id: str,
    profile_sha256: str,
    recommendation_id: str,
    compression_handoff_sha256: str,
) -> Glm52CapturePlan:
    """Build the exact candidate + same-model identity-control plan."""

    if not work_root.is_absolute():
        raise ValueError("capture work root must be absolute")
    canonical_work_root = work_root.parent.resolve(strict=True) / work_root.name
    if work_root != canonical_work_root or work_root.is_symlink():
        raise ValueError("capture work root must be a canonical non-symlink path")
    canonical_model = model_path.resolve(strict=True)
    canonical_tokenizer = profile_tokenizer_path.resolve(strict=True)
    evidence = CaptureModelEvidence(
        schema_version=1,
        model_sha256=model_sha256,
        source_model_sha256=source_manifest_sha256,
        profile_tokenizer_sha256=profile_tokenizer_sha256,
        profile_id=profile_id,
        profile_sha256=profile_sha256,
        recommendation_id=recommendation_id,
        compression_handoff_sha256=compression_handoff_sha256,
        producer_artifact_sha256=producer_artifact_sha256,
        recipe_sha256=recipe_sha256,
        plan_id=plan_id,
        run_id=run_id,
        evidence_kind="measured",
    )
    evidence_bytes = json.dumps(
        evidence.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    evidence_path = canonical_work_root / "model-artifact-evidence.json"
    common_argv = _common_argv(canonical_model)
    tool = CaptureToolIdentity(
        binary_path=str(CAPTURE_BINARY),
        binary_sha256=CAPTURE_BINARY_SHA256,
        build_contract_path=str(CAPTURE_BUILD_CONTRACT),
        build_contract_sha256=CAPTURE_BUILD_CONTRACT_SHA256,
        llama_cpp_commit=LLAMA_CPP_CAPTURE_COMMIT,
    )

    def request(role: CaptureRole, output_name: str) -> CaptureRequest:
        output = canonical_work_root / output_name
        # This is the exact value returned by canonical_capture_runtime_argv
        # once the execution wrapper has securely created canonical_work_root.
        normalized = (
            str(CAPTURE_BINARY),
            *common_argv,
            "--tokens-jsonl",
            str(FORCED_TOKENS),
            "--out-dir",
            str(output),
            "--layers",
            ",".join(str(layer) for layer in CAPTURE_LAYERS),
        )
        return CaptureRequest(
            model_id="glm52-mixed-gguf",
            model_path=str(canonical_model),
            model_sha256=model_sha256,
            model_artifact_manifest_path=str(evidence_path),
            model_artifact_manifest_sha256=evidence_sha,
            source_model_sha256=source_manifest_sha256,
            role=role,
            reference_kind=("candidate" if role is CaptureRole.CANDIDATE else "identity_control"),
            forced_tokens_path=str(FORCED_TOKENS),
            forced_tokens_sha256=FORCED_TOKENS_SHA256,
            held_out_manifest_path=str(HELD_OUT_MANIFEST),
            held_out_manifest_sha256=HELD_OUT_MANIFEST_SHA256,
            ordered_sample_ids_sha256=ORDERED_SAMPLE_IDS_SHA256,
            profile_tokenizer_path=str(canonical_tokenizer),
            profile_tokenizer_sha256=profile_tokenizer_sha256,
            output_dir=str(output),
            layers=CAPTURE_LAYERS,
            context_tokens=4096,
            batch_tokens=512,
            ubatch_tokens=128,
            threads=8,
            runtime_argv_sha256=capture_runtime_argv_sha256(normalized),
            tool=tool,
        )

    return Glm52CapturePlan(
        model_evidence=evidence,
        model_evidence_sha256=evidence_sha,
        model_evidence_path=str(evidence_path),
        common_argv=common_argv,
        candidate=request(CaptureRole.CANDIDATE, "candidate"),
        identity_control=request(CaptureRole.IDENTITY_CONTROL, "identity-control"),
    )
