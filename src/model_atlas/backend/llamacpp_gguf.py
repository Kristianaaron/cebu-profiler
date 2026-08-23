"""Pinned llama.cpp mixed GGUF artifact producer (no runtime claim)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from model_atlas.backend.contract import (
    BackendAdapter,
    BackendRecord,
    BackendUnavailable,
    ParameterSpec,
)
from model_atlas.backend.enforce import cgroup_scope_argv, clamp_threads
from model_atlas.recipe.schema import RecipeStatus

BACKEND_ID = "llamacpp_gguf_mixed"
PINNED_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"
EXPECTED_CONVERTER_SHA256 = "e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f"
EXPECTED_CPU_QUANTIZER_SHA256 = "536a0cd9cafe3172d638ca6eb29402661d05c2d8954631ef2f6f0fac6fb78e48"
EXPECTED_PYTHON_SHA256 = "a7d56a8a764faf7bbf5c164055a48fd072be52287bdeb523a9e07b2042f4e7e1"
DEFAULT_TOOLCHAIN_ROOT = Path("/home/kristianaaron/tmp/atlas-toolchains/llama.cpp")
DEFAULT_PYTHON = Path("/home/kristianaaron/ai-lab/venvs/vllm/bin/python")
CPU_QUANTIZER_RELATIVE_PATH = Path("build-atlas-cpu/bin/llama-quantize")
GENERIC_EXPERT_RULE = r"blk\..*\.ffn_(gate|up|down)_exps\.weight=Q1_0"
_SENSITIVE_RULES = (
    re.compile(r"blk\\\.[0-9]+\\\.ffn_(gate|up|down)_exps\\\.weight=NVFP4"),
    re.compile(r"\^blk\\\.[0-9]+\\\.ffn_(gate|up|down)_exps\\\.weight\$=NVFP4"),
)
_MAX_PLAN_BYTES = 1 << 20
_IO_CHUNK = 1 << 20


def _read_bounded_regular(path: Path, limit: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BackendUnavailable(f"{label} cannot be opened safely") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BackendUnavailable(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(limit + 1)
    finally:
        os.close(descriptor)
    if len(payload) > limit:
        raise BackendUnavailable(f"{label} exceeds size bound")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_IO_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
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
    ref_path = git_dir / ref
    if ref_path.is_file():
        value = ref_path.read_text(encoding="utf-8").strip()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref and re.fullmatch(r"[0-9a-f]{40}", commit):
                    return commit
    return None


@dataclass(frozen=True)
class LlamaCppProbeResult:
    available: bool
    commit: str | None
    converter: str
    converter_sha256: str
    quantizer: str
    quantizer_sha256: str
    python: str
    python_resolved: str
    python_sha256: str
    evidence: str


def probe_llamacpp_gguf(
    toolchain_root: str | Path = DEFAULT_TOOLCHAIN_ROOT,
    python_executable: str | Path = DEFAULT_PYTHON,
    *,
    expected_converter_sha256: str = EXPECTED_CONVERTER_SHA256,
    expected_quantizer_sha256: str = EXPECTED_CPU_QUANTIZER_SHA256,
    expected_python_sha256: str = EXPECTED_PYTHON_SHA256,
) -> LlamaCppProbeResult:
    """Filesystem-only probe; deliberately never executes the CUDA-linked binary."""
    root = Path(toolchain_root).resolve()
    python = Path(python_executable)
    converter = root / "convert_hf_to_gguf.py"
    quantizer = root / CPU_QUANTIZER_RELATIVE_PATH
    resolved_python = python.resolve()
    commit = _git_head(root)
    converter_sha = _sha256_file(converter) if converter.is_file() else ""
    quantizer_sha = _sha256_file(quantizer) if quantizer.is_file() else ""
    python_sha = _sha256_file(resolved_python) if resolved_python.is_file() else ""
    available = (
        commit == PINNED_COMMIT
        and converter.is_file()
        and quantizer.is_file()
        and converter_sha == expected_converter_sha256
        and quantizer_sha == expected_quantizer_sha256
        and os.access(quantizer, os.X_OK)
        and python.is_file()
        and os.access(python, os.X_OK)
        and python_sha == expected_python_sha256
    )
    evidence_obj = {
        "commit": commit,
        "expected_commit": PINNED_COMMIT,
        "converter": str(converter),
        "converter_sha256": converter_sha,
        "expected_converter_sha256": expected_converter_sha256,
        "quantizer": str(quantizer),
        "quantizer_sha256": quantizer_sha,
        "expected_quantizer_sha256": expected_quantizer_sha256,
        "quantizer_build_contract": {
            "GGML_CUDA": False,
            "GGML_RPC": False,
        },
        "python": str(python),
        "python_resolved": str(resolved_python),
        "python_sha256": python_sha,
        "expected_python_sha256": expected_python_sha256,
        "probe_executed_binaries": False,
    }
    return LlamaCppProbeResult(
        available=available,
        commit=commit,
        converter=str(converter),
        converter_sha256=converter_sha,
        quantizer=str(quantizer),
        quantizer_sha256=quantizer_sha,
        python=str(python),
        python_resolved=str(resolved_python),
        python_sha256=python_sha,
        evidence=json.dumps(evidence_obj, sort_keys=True),
    )


class CommandRunner(Protocol):
    def run(self, argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    def __init__(self, timeout_seconds: float = 86_400.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )


def _validate_plan(text: str) -> list[str]:
    if "--prune" in text.lower():
        raise BackendUnavailable("tensor plan must not contain pruning flags")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or lines[-1] != GENERIC_EXPERT_RULE:
        raise BackendUnavailable("tensor plan must end with the generic Q1_0 expert rule")
    if not all(any(pattern.fullmatch(line) for pattern in _SENSITIVE_RULES) for line in lines[:-1]):
        raise BackendUnavailable(
            "tensor plan must place exact per-layer NVFP4 expert rules before generic Q1_0"
        )
    return lines


class LlamaCppGgufMixedAdapter(BackendAdapter):
    backend_id = BACKEND_ID
    produces_derivative = True

    def __init__(
        self,
        *,
        toolchain_root: str | Path = DEFAULT_TOOLCHAIN_ROOT,
        python_executable: str | Path = DEFAULT_PYTHON,
        expected_converter_sha256: str = EXPECTED_CONVERTER_SHA256,
        expected_quantizer_sha256: str = EXPECTED_CPU_QUANTIZER_SHA256,
        expected_python_sha256: str = EXPECTED_PYTHON_SHA256,
        runner: CommandRunner | None = None,
    ) -> None:
        self.toolchain_root = Path(toolchain_root).resolve()
        # Preserve the pinned venv path in provenance/argv even when it is a
        # symlink; resolving it would silently turn the contract into system Python.
        self.python_executable = Path(python_executable)
        self.expected_converter_sha256 = expected_converter_sha256
        self.expected_quantizer_sha256 = expected_quantizer_sha256
        self.expected_python_sha256 = expected_python_sha256
        self.runner = runner or SubprocessRunner()

    def _paths(self, context: dict[str, object]) -> tuple[Path, Path, Path]:
        source_raw = context.get("source")
        staging_raw = context.get("staging_dir")
        if not source_raw or not staging_raw:
            raise BackendUnavailable("llama.cpp GGUF requires canonical source and staging_dir")
        source = Path(str(source_raw)).resolve()
        staging = Path(str(staging_raw)).resolve()
        scratch_path = staging.parent / "llamacpp-work"
        if scratch_path.is_symlink():
            raise BackendUnavailable("GGUF scratch must not be a symlink")
        scratch = scratch_path.resolve(strict=False)
        if source == staging or staging.is_relative_to(source) or source.is_relative_to(staging):
            raise BackendUnavailable("GGUF staging and immutable source must not overlap")
        if source == scratch or scratch.is_relative_to(source) or source.is_relative_to(scratch):
            raise BackendUnavailable("GGUF scratch and immutable source must not overlap")
        return source, staging, scratch

    def _probe(self) -> LlamaCppProbeResult:
        return probe_llamacpp_gguf(
            self.toolchain_root,
            self.python_executable,
            expected_converter_sha256=self.expected_converter_sha256,
            expected_quantizer_sha256=self.expected_quantizer_sha256,
            expected_python_sha256=self.expected_python_sha256,
        )

    @staticmethod
    def _plain(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): LlamaCppGgufMixedAdapter._plain(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [LlamaCppGgufMixedAdapter._plain(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _bind_resume_provenance(
        self,
        context: dict[str, object],
        scratch: Path,
        staging: Path,
        params: dict[str, str],
        plan_sha: str,
        probe: LlamaCppProbeResult,
    ) -> str:
        provenance = {
            "schema_version": 1,
            "source": str(context.get("source", "")),
            "source_identity": self._plain(context.get("source_identity", {})),
            "source_revision": self._plain(context.get("source_revision")),
            "tensor_plan_sha256": plan_sha,
            "toolchain_commit": probe.commit,
            "converter_sha256": probe.converter_sha256,
            "quantizer_sha256": probe.quantizer_sha256,
            "python": probe.python,
            "python_sha256": probe.python_sha256,
            "parameters": self._plain(params),
        }
        encoded = (json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = scratch / "resume-provenance.json"
        existing_outputs = any(
            candidate.exists()
            for candidate in (
                scratch / "source-auto.gguf",
                scratch / "candidate/model.gguf",
                staging / "model.gguf",
            )
        )
        if path.exists():
            if _read_bounded_regular(path, 64 * 1024, "resume provenance") != encoded:
                raise BackendUnavailable(
                    "resume provenance does not match source, plan, or toolchain"
                )
        elif existing_outputs:
            raise BackendUnavailable("unbound GGUF resume artifacts are forbidden")
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            os.replace(temporary, path)
        return digest

    @staticmethod
    def _parameters(context: dict[str, object]) -> dict[str, str]:
        raw = context.get("parameters", {})
        if not isinstance(raw, dict):
            raise BackendUnavailable("GGUF parameters must be a mapping")
        params = {str(key): str(value) for key, value in raw.items()}
        if any("prun" in key.lower() for key in params):
            raise BackendUnavailable("pruning parameters are forbidden for GGUF quantization")
        risk_sha256 = params.get("risk_artifact_sha256", "")
        if risk_sha256 and re.fullmatch(r"[0-9a-f]{64}", risk_sha256) is None:
            raise BackendUnavailable("risk artifact sha256 must be lowercase hexadecimal")
        return params

    def _materialize_plan(self, params: dict[str, str], scratch: Path) -> tuple[Path, str]:
        content = params.get("tensor_plan_content", "")
        source_path = params.get("tensor_plan_path", "")
        declared_hash = params.get("tensor_plan_sha256", "")
        if bool(content) == bool(source_path):
            raise BackendUnavailable(
                "provide exactly one tensor plan mode: content or path with sha256"
            )
        if source_path:
            path = Path(source_path).resolve()
            if not declared_hash or not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
                raise BackendUnavailable("tensor plan path requires a lowercase sha256")
            payload = _read_bounded_regular(path, _MAX_PLAN_BYTES, "tensor plan")
            measured = hashlib.sha256(payload).hexdigest()
            if measured != declared_hash:
                raise BackendUnavailable("tensor plan sha256 mismatch")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise BackendUnavailable("tensor plan is not UTF-8") from exc
        else:
            encoded = content.encode("utf-8")
            if not encoded or len(encoded) > _MAX_PLAN_BYTES:
                raise BackendUnavailable("tensor plan content is empty or exceeds size bound")
            text = content
        lines = _validate_plan(text)
        normalized = "\n".join(lines) + "\n"
        plan = scratch / "tensor-types.txt"
        temporary = plan.with_suffix(".tmp")
        temporary.write_text(normalized, encoding="utf-8")
        os.replace(temporary, plan)
        return plan, hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _run_checked(runner: CommandRunner, argv: list[str], *, cwd: Path, label: str) -> None:
        if any(arg.startswith("--prune") for arg in argv):
            raise BackendUnavailable("pruning flags are forbidden for GGUF quantization")
        try:
            result = runner.run(argv, cwd=cwd)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendUnavailable(f"{label} failed to execute: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-2000:]
            raise BackendUnavailable(f"{label} exited {result.returncode}: {detail}")

    @staticmethod
    def _gguf_ok(directory: Path) -> bool:
        from model_atlas.checkpoint.validators import _gguf_structure

        return _gguf_structure(BACKEND_ID, directory, "gguf").ok

    def prepare(self, context: dict[str, object]) -> str:
        source, staging, scratch = self._paths(context)
        if not source.is_dir():
            raise BackendUnavailable(f"GGUF source is not a directory: {source}")
        probe = self._probe()
        if not probe.available:
            raise BackendUnavailable(f"pinned llama.cpp toolchain unavailable: {probe.evidence}")
        staging.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        if scratch.is_symlink() or not scratch.is_dir():
            raise BackendUnavailable("GGUF scratch must be a private real directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(scratch, flags)
        except OSError as exc:
            raise BackendUnavailable("GGUF scratch cannot be opened without symlinks") from exc
        else:
            os.close(descriptor)
        return f"llamacpp-gguf::{PINNED_COMMIT}"

    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]:
        source, staging, scratch = self._paths(context)
        params = self._parameters(context)
        threads_raw = params.get("threads", "16")
        try:
            threads = int(threads_raw)
        except ValueError as exc:
            raise BackendUnavailable("threads must be an integer") from exc
        if threads < 1 or threads > 256:
            raise BackendUnavailable("threads must be in [1, 256]")
        plan, plan_sha = self._materialize_plan(params, scratch)
        probe = self._probe()
        if not probe.available:
            raise BackendUnavailable(f"pinned llama.cpp toolchain unavailable: {probe.evidence}")
        provenance_sha = self._bind_resume_provenance(
            context, scratch, staging, params, plan_sha, probe
        )
        intermediate = scratch / "source-auto.gguf"
        final = staging / "model.gguf"
        resumed = final.is_file() and self._gguf_ok(staging)
        if not resumed:
            if final.exists():
                final.unlink()
            if not (intermediate.is_file() and self._gguf_ok(scratch)):
                converter_argv, conv_mode = cgroup_scope_argv(
                    [
                        str(self.python_executable),
                        probe.converter,
                        str(source),
                        "--outfile",
                        str(intermediate),
                        "--outtype",
                        "auto",
                    ],
                    label="gguf-converter",
                )
                self._run_checked(
                    self.runner,
                    converter_argv,
                    cwd=self.toolchain_root,
                    label=f"GGUF converter [{conv_mode}]",
                )
                if not intermediate.is_file() or not self._gguf_ok(scratch):
                    raise BackendUnavailable("converter did not produce a structurally valid GGUF")
            candidate_dir = scratch / "candidate"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate = candidate_dir / "model.gguf"
            if candidate.exists():
                candidate.unlink()
            threads = clamp_threads(threads)
            quantizer_argv, quant_mode = cgroup_scope_argv(
                [
                probe.quantizer,
                "--allow-requantize",
                "--tensor-type-file",
                str(plan),
                "--output-tensor-type",
                "Q4_K",
                "--token-embedding-type",
                "Q4_K",
                str(intermediate),
                str(candidate),
                "Q4_K",
                str(threads),
                ],
                label="gguf-quantizer",
            )
            self._run_checked(
                self.runner,
                quantizer_argv,
                cwd=self.toolchain_root,
                label=f"GGUF quantizer [{quant_mode}]",
            )
            if not candidate.is_file() or not self._gguf_ok(candidate_dir):
                raise BackendUnavailable("quantizer did not produce a structurally valid GGUF")
            os.replace(candidate, final)
            candidate_dir.rmdir()
            if not self._gguf_ok(staging):
                raise BackendUnavailable("atomically promoted GGUF failed structural validation")
        if intermediate.exists():
            intermediate.unlink()
        return {
            "derivative": True,
            "format": "gguf",
            "runtime_validated": False,
            "pruning": False,
            "handle": handle,
            "resumed": resumed,
            "tensor_plan_sha256": plan_sha,
            "resume_provenance_sha256": provenance_sha,
            "toolchain": json.loads(probe.evidence),
        }

    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]:
        return self.execute(context, handle)

    def validate(self, context: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
        from model_atlas.checkpoint.validators import _gguf_structure

        del outputs
        _source, staging, _scratch = self._paths(context)
        result = _gguf_structure(self.backend_id, staging, "gguf")
        return {
            "validated": result.ok,
            "status": "passed" if result.ok else "failed",
            **result.to_dict(),
        }


def build_llamacpp_gguf_record(
    *,
    toolchain_root: str | Path = DEFAULT_TOOLCHAIN_ROOT,
    python_executable: str | Path = DEFAULT_PYTHON,
    expected_converter_sha256: str = EXPECTED_CONVERTER_SHA256,
    expected_quantizer_sha256: str = EXPECTED_CPU_QUANTIZER_SHA256,
    expected_python_sha256: str = EXPECTED_PYTHON_SHA256,
    runner: CommandRunner | None = None,
) -> BackendRecord:
    adapter = LlamaCppGgufMixedAdapter(
        toolchain_root=toolchain_root,
        python_executable=python_executable,
        expected_converter_sha256=expected_converter_sha256,
        expected_quantizer_sha256=expected_quantizer_sha256,
        expected_python_sha256=expected_python_sha256,
        runner=runner,
    )

    def availability() -> tuple[bool, str | None, str]:
        result = probe_llamacpp_gguf(
            toolchain_root,
            python_executable,
            expected_converter_sha256=expected_converter_sha256,
            expected_quantizer_sha256=expected_quantizer_sha256,
            expected_python_sha256=expected_python_sha256,
        )
        return result.available, result.commit, result.evidence

    def execution_identity() -> Mapping[str, str]:
        result = probe_llamacpp_gguf(
            toolchain_root,
            python_executable,
            expected_converter_sha256=expected_converter_sha256,
            expected_quantizer_sha256=expected_quantizer_sha256,
            expected_python_sha256=expected_python_sha256,
        )
        if not result.available or result.commit is None:
            raise BackendUnavailable("llama.cpp execution tool identity is unavailable or drifted")
        return {
            "commit": result.commit,
            "converter": result.converter,
            "converter_sha256": result.converter_sha256,
            "quantizer": result.quantizer,
            "quantizer_sha256": result.quantizer_sha256,
            "python": result.python,
            "python_resolved": result.python_resolved,
            "python_sha256": result.python_sha256,
            "quantizer_ggml_cuda": "false",
            "quantizer_ggml_rpc": "false",
        }

    return BackendRecord(
        backend_id=BACKEND_ID,
        display_name="llama.cpp mixed GGUF quantization (artifact only)",
        method_family="llamacpp",
        formats=("gguf",),
        represents_method="mixed GGUF requantization with recipe-bound tensor overrides",
        architectures=("glm-5.2", "any"),
        compute_archs=("gb10-sm121", "any"),
        topologies=("2x-spark", "single", "any"),
        runtime_compat=(),
        conversion_tool_compat=("llama.cpp",),
        status=RecipeStatus.DISCOVERED,
        version=PINNED_COMMIT,
        declared_capabilities=(),
        supported_formats=("gguf",),
        fail_closed=True,
        produces_derivative=True,
        availability_probe=availability,
        execution_identity_probe=execution_identity,
        parameters=(
            ParameterSpec("tensor_plan_content", "string", "recipe-bound tensor override plan"),
            ParameterSpec("tensor_plan_path", "string", "path to tensor override plan"),
            ParameterSpec("tensor_plan_sha256", "string", "required digest for path mode"),
            ParameterSpec(
                "risk_artifact_sha256",
                "string",
                "immutable risk artifact lineage digest",
            ),
            ParameterSpec(
                "threads",
                "int",
                "bounded quantizer threads",
                default="16",
                minimum=1,
                maximum=256,
            ),
        ),
        adapter=adapter,
    )
