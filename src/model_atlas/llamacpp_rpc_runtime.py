"""Pinned two-Spark llama.cpp RPC runtime contract.

This is deliberately separate from the ``llamacpp_gguf_mixed`` artifact
producer.  A filesystem probe can establish that the pinned runtime bytes are
present, but it never executes them and therefore never makes a runtime claim.
Only a matching, measured validation receipt can promote this contract.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

RUNTIME_ID = "llamacpp-rpc-two-spark"
PINNED_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"
EXPECTED_LLAMA_SERVER_SHA256 = (
    "86d791cf2ba2332b75b1589eece04a29488cf37d7fce871584c929fc85f644bb"
)
EXPECTED_RPC_SERVER_SHA256 = (
    "6b448f515e4f674c99c37ce20fd82bde9cbb28c0b2bd1fd9b0e16db3ee81ce76"
)
DEFAULT_TOOLCHAIN_ROOT = Path("/home/kristianaaron/tmp/atlas-toolchains/llama.cpp")
DEFAULT_LLAMA_SERVER = DEFAULT_TOOLCHAIN_ROOT / "build-atlas/bin/llama-server"
DEFAULT_WORKER_RPC_SERVER = DEFAULT_TOOLCHAIN_ROOT / "build-atlas/bin/ggml-rpc-server"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IO_CHUNK = 1 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_IO_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    """Read HEAD from the filesystem; never invoke git or another binary."""
    git_dir = root / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir:"):
            return None
        git_dir = (root / marker.split(":", 1)[1].strip()).resolve()
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head if re.fullmatch(r"[0-9a-f]{40}", head) else None
    ref = head[5:]
    loose = git_dir / ref
    if loose.is_file():
        value = loose.read_text(encoding="utf-8").strip()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref and re.fullmatch(r"[0-9a-f]{40}", commit):
                    return commit
    return None


def _private_worker_host(value: str) -> str:
    try:
        address = ipaddress.IPv4Address(value)
    except ValueError as exc:
        raise ValueError("worker_host must be an IPv4 address") from exc
    if address.is_unspecified or address.is_loopback or not address.is_private:
        raise ValueError("worker_host must be a private, non-loopback address")
    return str(address)


def _loopback_api_host(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError("api_host must be a loopback IP address") from exc
    if not address.is_loopback:
        raise ValueError("api_host must be loopback-only")
    return str(address)


def _port(value: int, label: str) -> int:
    if value < 1 or value > 65535:
        raise ValueError(f"{label} must be in [1, 65535]")
    return value


@dataclass(frozen=True)
class LlamaCppRpcRuntimeConfig:
    """Conservative base run: two devices, 4K, parallel one, no MTP."""

    artifact_path: Path
    artifact_sha256: str
    llama_server_path: Path = DEFAULT_LLAMA_SERVER
    worker_rpc_server_path: Path = DEFAULT_WORKER_RPC_SERVER
    worker_host: str = "169.254.200.197"
    rpc_port: int = 50052
    api_host: str = "127.0.0.1"
    api_port: int = 8892
    context_size: int = 4096
    parallel: int = 1

    def __post_init__(self) -> None:
        if not self.artifact_path.is_absolute():
            raise ValueError("artifact_path must be absolute")
        if not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256")
        if not self.llama_server_path.is_absolute():
            raise ValueError("llama_server_path must be absolute")
        if not self.worker_rpc_server_path.is_absolute():
            raise ValueError("worker_rpc_server_path must be absolute")
        object.__setattr__(self, "worker_host", _private_worker_host(self.worker_host))
        object.__setattr__(self, "api_host", _loopback_api_host(self.api_host))
        _port(self.rpc_port, "rpc_port")
        _port(self.api_port, "api_port")
        if self.context_size != 4096 or self.parallel != 1:
            raise ValueError("base validation is fixed to 4K context and parallel=1")

    def worker_argv(self) -> tuple[str, ...]:
        return (
            str(self.worker_rpc_server_path),
            "--host",
            self.worker_host,
            "--port",
            str(self.rpc_port),
            "--device",
            "CUDA0",
            "--threads",
            "8",
        )

    def head_argv(self) -> tuple[str, ...]:
        # RPC discovery must precede the explicit device list.  Keep this list
        # literal and test its ordering because llama.cpp argument order is part
        # of the validated deployment contract.
        return (
            str(self.llama_server_path),
            "--model",
            str(self.artifact_path),
            "--alias",
            "glm52-mixed-gguf",
            "--rpc",
            f"{self.worker_host}:{self.rpc_port}",
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
            self.api_host,
            "--port",
            str(self.api_port),
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            str(self.parallel),
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

    def canonical_sha256(self) -> str:
        payload = {
            "schema_version": 1,
            "runtime_id": RUNTIME_ID,
            "commit": PINNED_COMMIT,
            "llama_server_sha256": EXPECTED_LLAMA_SERVER_SHA256,
            "worker_rpc_server_sha256": EXPECTED_RPC_SERVER_SHA256,
            "worker_argv": self.worker_argv(),
            "head_argv": self.head_argv(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LlamaCppRpcMtpConfig:
    """Explicit follow-up only; the base contract contains no MTP flags."""

    base: LlamaCppRpcRuntimeConfig
    draft_n_max: int = 3

    def __post_init__(self) -> None:
        if self.draft_n_max != 3:
            raise ValueError("MTP follow-up is fixed to draft_n_max=3")

    def head_argv(self) -> tuple[str, ...]:
        return self.base.head_argv() + (
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            str(self.draft_n_max),
            "--spec-draft-n-min",
            "0",
            "--cache-type-k-draft",
            "q8_0",
            "--cache-type-v-draft",
            "q8_0",
        )


@dataclass(frozen=True)
class LlamaCppRpcToolProbe:
    available: bool
    commit: str | None
    llama_server_path: str
    llama_server_sha256: str
    worker_rpc_server_path: str
    worker_rpc_server_sha256: str
    remote_worker_attested: bool
    artifact_path: str
    artifact_sha256: str
    artifact_verified: bool
    executed_binaries: bool = False


@dataclass(frozen=True)
class LlamaCppRpcWorkerAttestation:
    """Out-of-band hash evidence measured on the remote worker filesystem."""

    host: str
    rpc_server_path: str
    rpc_server_sha256: str
    commit: str
    evidence_kind: str = "measured"


def probe_llamacpp_rpc_runtime(
    config: LlamaCppRpcRuntimeConfig,
    *,
    toolchain_root: Path = DEFAULT_TOOLCHAIN_ROOT,
    expected_llama_server_sha256: str = EXPECTED_LLAMA_SERVER_SHA256,
    expected_worker_rpc_server_sha256: str = EXPECTED_RPC_SERVER_SHA256,
    worker_attestation: LlamaCppRpcWorkerAttestation | None = None,
) -> LlamaCppRpcToolProbe:
    """Hash head files and verify worker-supplied evidence; execute nothing.

    The worker path is remote and is intentionally not read on the head.  Its
    bytes count only through an explicit attestation measured on that worker.
    """
    root = toolchain_root.resolve()
    head = config.llama_server_path
    commit = _git_head(root)
    head_sha = _sha256_file(head) if head.is_file() else ""
    artifact = config.artifact_path
    artifact_sha = _sha256_file(artifact) if artifact.is_file() else ""
    artifact_verified = artifact_sha == config.artifact_sha256
    remote_worker_attested = bool(
        worker_attestation is not None
        and worker_attestation.host == config.worker_host
        and worker_attestation.rpc_server_path == str(config.worker_rpc_server_path)
        and worker_attestation.rpc_server_sha256 == expected_worker_rpc_server_sha256
        and worker_attestation.commit == PINNED_COMMIT
        and worker_attestation.evidence_kind == "measured"
    )
    worker_sha = "" if worker_attestation is None else worker_attestation.rpc_server_sha256
    available = (
        commit == PINNED_COMMIT
        and head.is_file()
        and os.access(head, os.X_OK)
        and head_sha == expected_llama_server_sha256
        and remote_worker_attested
        and artifact_verified
    )
    return LlamaCppRpcToolProbe(
        available=available,
        commit=commit,
        llama_server_path=str(head),
        llama_server_sha256=head_sha,
        worker_rpc_server_path=str(config.worker_rpc_server_path),
        worker_rpc_server_sha256=worker_sha,
        remote_worker_attested=remote_worker_attested,
        artifact_path=str(artifact),
        artifact_sha256=artifact_sha,
        artifact_verified=artifact_verified,
    )


@dataclass(frozen=True)
class LlamaCppRpcValidationReceipt:
    """Measured two-node load/generation evidence produced outside this module."""

    runtime_id: str
    config_sha256: str
    artifact_sha256: str
    commit: str
    llama_server_sha256: str
    worker_rpc_server_sha256: str
    worker_rpc_server_path: str
    worker_hash_attested: bool
    worker_host: str
    observed_devices: tuple[str, ...]
    load_succeeded: bool
    generation_succeeded: bool
    evidence_kind: str


@dataclass(frozen=True)
class LlamaCppRpcRuntimeClaim:
    validated: bool
    runtime_compat: tuple[str, ...]
    reason: str


def validate_runtime_receipt(
    config: LlamaCppRpcRuntimeConfig,
    probe: LlamaCppRpcToolProbe,
    receipt: LlamaCppRpcValidationReceipt | None,
) -> LlamaCppRpcRuntimeClaim:
    """Fail closed: tool presence alone is never a serving compatibility claim."""
    if receipt is None:
        return LlamaCppRpcRuntimeClaim(False, (), "measured runtime receipt is required")
    expected: dict[str, object] = {
        "runtime_id": RUNTIME_ID,
        "config_sha256": config.canonical_sha256(),
        "artifact_sha256": config.artifact_sha256,
        "commit": PINNED_COMMIT,
        "llama_server_sha256": EXPECTED_LLAMA_SERVER_SHA256,
        "worker_rpc_server_sha256": EXPECTED_RPC_SERVER_SHA256,
        "worker_rpc_server_path": str(config.worker_rpc_server_path),
        "worker_hash_attested": True,
        "worker_host": config.worker_host,
        "evidence_kind": "measured",
    }
    actual = asdict(receipt)
    mismatches = [key for key, value in expected.items() if actual[key] != value]
    if not probe.available:
        mismatches.append("fresh_tool_probe")
    if probe.commit != PINNED_COMMIT:
        mismatches.append("probe_commit")
    if probe.llama_server_path != str(config.llama_server_path):
        mismatches.append("probe_llama_server_path")
    if probe.llama_server_sha256 != EXPECTED_LLAMA_SERVER_SHA256:
        mismatches.append("probe_llama_server_sha256")
    if probe.worker_rpc_server_path != str(config.worker_rpc_server_path):
        mismatches.append("probe_worker_rpc_server_path")
    if probe.worker_rpc_server_sha256 != EXPECTED_RPC_SERVER_SHA256:
        mismatches.append("probe_worker_rpc_server_sha256")
    if not probe.remote_worker_attested:
        mismatches.append("probe_remote_worker_attestation")
    if probe.artifact_path != str(config.artifact_path):
        mismatches.append("probe_artifact_path")
    if probe.artifact_sha256 != config.artifact_sha256 or not probe.artifact_verified:
        mismatches.append("probe_artifact_sha256")
    if probe.executed_binaries:
        mismatches.append("probe_executed_binaries")
    if not receipt.load_succeeded:
        mismatches.append("load_succeeded")
    if not receipt.generation_succeeded:
        mismatches.append("generation_succeeded")
    if set(receipt.observed_devices) != {"CUDA0", "RPC0"}:
        mismatches.append("observed_devices")
    if mismatches:
        return LlamaCppRpcRuntimeClaim(
            False, (), "runtime receipt mismatch: " + ", ".join(sorted(set(mismatches)))
        )
    return LlamaCppRpcRuntimeClaim(True, (RUNTIME_ID,), "measured two-Spark receipt matched")


@dataclass(frozen=True)
class LlamaCppRpcRuntimeAdapter:
    """Non-executing adapter for planning, probing, and receipt verification.

    Deliberately does not implement the compression ``BackendAdapter`` API:
    serving a produced artifact is a separate lifecycle from producing it.
    """

    config: LlamaCppRpcRuntimeConfig
    toolchain_root: Path = DEFAULT_TOOLCHAIN_ROOT

    def probe(
        self, worker_attestation: LlamaCppRpcWorkerAttestation | None = None
    ) -> LlamaCppRpcToolProbe:
        return probe_llamacpp_rpc_runtime(
            self.config,
            toolchain_root=self.toolchain_root,
            worker_attestation=worker_attestation,
        )

    def worker_argv(self) -> tuple[str, ...]:
        return self.config.worker_argv()

    def base_head_argv(self) -> tuple[str, ...]:
        return self.config.head_argv()

    def mtp_head_argv(self) -> tuple[str, ...]:
        return LlamaCppRpcMtpConfig(self.config).head_argv()

    def validate_receipt(
        self,
        receipt: LlamaCppRpcValidationReceipt | None,
        *,
        independently_measured_worker: LlamaCppRpcWorkerAttestation | None,
    ) -> LlamaCppRpcRuntimeClaim:
        """Validate only against independently supplied remote measurement.

        The receipt is never allowed to manufacture its own worker attestation.
        The maintenance/runtime runner must measure the remote bytes over its
        authenticated SSH channel and pass that separate evidence object here.
        """
        return validate_runtime_receipt(
            self.config,
            self.probe(independently_measured_worker),
            receipt,
        )
