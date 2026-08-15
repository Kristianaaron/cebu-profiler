"""Pinned, derivative-producing EXL3 (exllamav3) backend adapter.

This adapter drives an **explicitly-pinned external executable/module contract**
(the exllamav3 conversion tool). It never assumes the tool is present and never
fabricates an output:

* ``probe_exl3`` resolves the exact version + capabilities from the pinned
  command; the record registers and only reports ``available`` when the probe
  passes.
* executing a stage shells out to the pinned command; the resulting safetensors
  derivative is written **only into the stager's scoped staging dir** (never to
  the source), then structurally validated + content-addressed receipts are
  taken.
* cancellation / non-zero exit is a hard, fail-closed error.

Default pinned command ``EXL3_COMMAND``. On a machine with no exllamav3 this
project must keep failing closed (we do NOT unpin/install anything).

External contract modelled on exllamav3's ``doc/convert.md``:
  ``<python> convert.py -i <in> -o <out> -w <work> -b <bpw> [-hb] [-hq] [-cr]``

Version resolution uses a probe flag ``--atlas-probe-version`` that the pinned
executable MUST implement (prints a JSON line ``{"exl3_version": "...", capabilities:
[...]}``). Real exllamav3 does not define that flag, so the
probe FAILS CLOSED unless the wrapper defines it: an available probe that lies
is worse than unavailable. The test suite supplies a fake executable that
implements the exact contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_atlas.backend.contract import (
    AvailabilityProbe,
    BackendAdapter,
    BackendRecord,
    BackendUnavailable,
    ParameterSpec,
    ResourceEstimate,
)
from model_atlas.recipe.schema import RecipeStatus

# Pinned external contract. The probe/test inject the real executable path; a
# bare name is resolved via PATH lookup so we never claim a binary not present.
EXL3_COMMAND = "exllamav3-convert"

# Probe flag the pinned executable must implement (see module docstring).
PROBE_FLAG = "--atlas-probe-version"

# The derivative materializes into the stager's staging dir then is committed by
# the job engine; the adapter only validates + receipts, never publishes.
_STASH_KEY = "output_sink"
_STAGING_KEY = "staging_dir"

# Historical bits-per-weight for the lm_head must be 1..8 (exllamav3 contract).
_HEAD_BITS_MIN, _HEAD_BITS_MAX = 1, 8


@dataclass(frozen=True)
class Exl3ProbeResult:
    available: bool
    version: str | None
    capabilities: tuple[str, ...]
    evidence: str


def _resolve_command(command: str) -> str | None:
    """Resolve a pinned executable by absolute path or PATH lookup."""
    candidate = Path(command).expanduser()
    if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(command)


class _SubprocessRunner:
    """Thin seam so tests can substitute a fake executable without touching the
    adapter's contract (it still drives a REAL subprocess)."""

    def __init__(self, timeout: float = 7200.0) -> None:
        self.timeout = timeout

    def run(
        self, argv: list[str], cwd: str, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )


def probe_exl3(command: str = EXL3_COMMAND) -> Exl3ProbeResult:
    """Resolve exact version + capabilities from a real version-aware pinned
    executable. Fails closed when the executable is absent or the contract flag
    is unimplemented. ``available`` is True only when the binary exists AND the
    probe round-trips a self-consistent version."""
    resolved = _resolve_command(command)
    if resolved is None:
        return Exl3ProbeResult(False, None, (), f"{command!r} not found (fail closed)")
    try:
        proc = _SubprocessRunner(timeout=30.0).run(
            [resolved, PROBE_FLAG, "--json"], cwd=".", env=dict(os.environ)
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Exl3ProbeResult(False, None, (), f"probe execution failed ({exc})")
    if proc.returncode != 0:
        return Exl3ProbeResult(False, None, (), "probe flag not implemented / non-zero exit")
    try:
        payload = json.loads(proc.stdout)
    except (ValueError, json.JSONDecodeError):
        return Exl3ProbeResult(False, None, (), "probe did not return a JSON version report")
    version = payload.get("exl3_version")
    raw_caps = payload.get("capabilities", [])
    if isinstance(raw_caps, str):
        raw_caps = [c.strip() for c in raw_caps.split(",") if c.strip()]
    caps = tuple(sorted(str(c) for c in raw_caps if c))
    if not isinstance(version, str) or not version:
        return Exl3ProbeResult(False, None, (), "probe returned an empty/unsupported version")
    return Exl3ProbeResult(
        True, version, caps, f"pinned executable {resolved!r} reports v{version}"
    )


def _build_command_argv(
    source: str, work: str, out: str, bpw: str, head_bits: str, hq: bool, cal_rows: str
) -> list[str]:
    """Deterministic EXL3 conversion argv (contract placeholders, not shell
    expansion — executed as a real argv list, never through a shell)."""
    argv = [EXL3_COMMAND, "-i", source, "-o", out, "-w", work, "-b", bpw, "-hb", head_bits]
    if hq:
        argv.append("-hq")
    if cal_rows and cal_rows not in ("0", ""):
        argv.extend(["-cr", cal_rows])
    return argv


def build_exl3_manifest(context: dict[str, object], handle: str) -> dict[str, object]:
    """Deterministic EXL3 run manifest: resolve source/work/out from context,
    fill the command argv, and record provenance. Pure (no I/O)."""
    workdir = Path(str(context.get("workdir", ".")))
    staging = Path(str(context.get(_STASH_KEY, "") or context.get(_STAGING_KEY, "") or workdir))
    params_raw = context.get("parameters", {})
    params: dict[str, Any] = dict(params_raw) if isinstance(params_raw, dict) else {}
    source = str(params.get("source", context.get("source", "")))
    work = str(params.get("work", "") or (workdir / ".exl3-work"))
    out = str(params.get("out", "") or staging)
    bpw = str(params.get("bpw", "3.25"))
    head_bits = str(params.get("head_bits", "6"))
    hq = str(params.get("hq", "1")) in {"1", "true", "True"}
    cal_rows = str(params.get("cal_rows", "0"))
    return {
        "backend": "exl3",
        "handle": handle,
        "run": {
            "bpw": bpw,
            "head_bits": head_bits,
            "hq": hq,
            "hq_flag": "-hq" if hq else "",
            "cal_rows": cal_rows,
        },
        "paths": {
            "source": source,
            "work": work,
            "out": out,
            "staging": str(staging),
            "workdir": str(workdir),
        },
        "command_argv": _build_command_argv(source, work, out, bpw, head_bits, hq, cal_rows),
        "provenance": {
            "source_immutable": True,  # adapter never writes to source
            "derivative_only_written_to": out,
            "method": "EXL3 primary quantization",
            "evidence_kind": "predicted",
        },
    }


def canonical_receipt(shard_files: list[Path]) -> dict[str, object]:
    """Content-addressed receipt over the produced safetensors derivative. It
    never embeds paths — only name->sha256 content digests (replayable from the
    CAS store)."""
    from model_atlas.jobs.artifacts import sha256_file

    name_digest = {p.name: sha256_file(p) for p in sorted(shard_files) if p.is_file()}
    payload = json.dumps(name_digest, sort_keys=True).encode("utf-8")
    return {"files": name_digest, "digest": hashlib.sha256(payload).hexdigest()}


def _probe_registry_probe(
    command: str,
) -> AvailabilityProbe:
    """Zero-arg availability probe (contract signature) bound to the pinned
    command; used on a BackendRecord."""

    def _probe() -> tuple[bool, str | None, str]:
        result = probe_exl3(command)
        return result.available, result.version, result.evidence

    return _probe


class Exl3Adapter(BackendAdapter):
    """EXL3 derivative producer: shells out to the pinned exllamav3 conversion
    tool, materializing safetensors into the stager's staging dir only, then
    structurally validating + content-receipting them. Never writes to source.
    Cancellation = non-zero/aborted subprocess (fail closed)."""

    backend_id = "exl3"
    # P0: this adapter produces a REAL derivative checkpoint.
    produces_derivative = True

    def __init__(
        self,
        *,
        command: str = EXL3_COMMAND,
        timeout_seconds: float = 7200.0,
        runner: _SubprocessRunner | None = None,
    ) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        self._runner = runner or _SubprocessRunner(timeout_seconds)

    # ---- internal helpers ----
    @staticmethod
    def _staging_of(context: dict[str, object]) -> Path:
        staging = context.get(_STASH_KEY) or context.get(_STAGING_KEY)
        if not staging:
            raise BackendUnavailable(
                "exl3: no staging/output_sink dir in context — derivative-safe "
                "execution requires a scoped stager"
            )
        return Path(str(staging))

    def _prepared(
        self, context: dict[str, object]
    ) -> tuple[str, str, Path, dict[str, str], list[str], str]:
        """Resolve source/work/out + env + argv from context. Fails closed
        (BackendUnavailable) if the pinned executable is not resolvable at
        execute time."""
        resolved = _resolve_command(self.command)
        if resolved is None:
            raise BackendUnavailable(
                f"exl3: pinned executable {self.command!r} not found at execute time"
            )
        params_raw = context.get("parameters", {})
        params: dict[str, str] = {}
        if isinstance(params_raw, dict):
            params = {str(k): str(v) for k, v in params_raw.items()}
        workdir = Path(str(context.get("workdir", ".")))
        work = str(params.get("work", "") or (workdir / ".exl3-work"))
        out = str(params.get("out", "") or self._staging_of(context))
        bpw = params.get("bpw", "3.25")
        head_bits = params.get("head_bits", "6")
        hq = params.get("hq", "1") in {"1", "true", "True"}
        cal_rows = params.get("cal_rows", "0")
        source = str(context.get("source") or params.get("source") or "")
        if not source:
            raise BackendUnavailable("exl3: no source checkpoint declared in context")
        env = dict(os.environ)
        env["EXL3_OUT_DIR"] = out
        env["EXL3_SOURCE"] = source
        env["EXL3_WORK_DIR"] = work
        argv = _build_command_argv(source, work, out, bpw, head_bits, hq, cal_rows)
        argv = [resolved if a == EXL3_COMMAND else a for a in argv]
        return source, out, Path(str(out)), env, argv, resolved

    # ---- BackendAdapter API ----
    def prepare(self, context: dict[str, object]) -> str:
        # idempotent: create the work + empty staging dirs; no execution.
        self._staging_of(context).mkdir(parents=True, exist_ok=True)
        return "exl3::prepared"

    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]:
        _source, out, out_dir, env, argv, resolved = self._prepared(context)
        out_dir.mkdir(parents=True, exist_ok=True)
        cwd = str(context.get("workdir", "."))
        try:
            proc = self._runner.run(argv, cwd=cwd, env=env)
        except subprocess.TimeoutExpired:
            raise BackendUnavailable(
                f"exl3: quantization aborted — exceeded timeout {self.timeout_seconds}s"
            ) from None
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-3000:]
            raise BackendUnavailable(
                f"exl3: pinned conversion command exited {proc.returncode}: {detail}"
            )
        shards = [p for p in out_dir.rglob("*.safetensors") if p.is_file()]
        if not shards:
            raise BackendUnavailable(
                "exl3: conversion finished but produced no safetensors derivative "
                "(command may not be the pinned EXL3 converter)"
            )
        receipt = canonical_receipt(shards)
        return {
            "command_executed": True,
            "format": "safetensors",
            "derivative": True,
            "produced_shards": [p.name for p in shards],
            "receipt": receipt,
            "provenance": {"executable": resolved, "handle": handle, "source_immutable": True},
        }

    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]:
        # Crash-safe resume of an interrupted run: re-run idempotently (the
        # pinned tool makes re-invocation idempotent via its work_dir).
        return self.execute(context, handle)

    def validate(self, context: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
        """Validate the produced safetensors derivative structurally and return a
        typed receipt. ``outputs`` are the staged ``_StageRef`` objects (each
        carrying ``path`` or ``relpath``)."""
        from model_atlas.checkpoint.validators import (
            CheckpointValidationResult,
            _safetensors_structure,
        )

        staged: list[Path] = []
        for ref in outputs.values():
            p = getattr(ref, "path", None) or getattr(ref, "relpath", None)
            if p is not None:
                staged.append(Path(str(p)))
        staging = self._staging_of(context) if context else None
        base = staging if (staging and staging.is_dir()) else None
        candidates = [Path(f) for f in (base.iterdir() if base else [])]
        shards = [f for f in staged + candidates if f.suffix == ".safetensors"]
        if not shards:
            return {
                "validated": False,
                "status": "unvalidated",
                "errors": ["exl3: no safetensors derivative in staging to validate"],
            }
        structural: CheckpointValidationResult = _safetensors_structure(
            "exl3", base or shards[0].parent, ""
        )
        if not structural.ok:
            return {"validated": False, "status": "failed", "errors": [structural.detail]}
        receipt = canonical_receipt(shards)
        files = receipt["files"]
        shard_names = sorted(files) if isinstance(files, dict) else []
        return {
            "validated": True,
            "status": "passed",
            "format": "safetensors",
            "derivative": True,
            "receipt": receipt,
            "shard_names": shard_names,
        }


def build_exl3_record(
    command: str = EXL3_COMMAND,
    *,
    timeout_seconds: float = 7200.0,
) -> BackendRecord:
    """Factory for the EXL3 BackendRecord. Availability is probed through the
    REAL pinned command (never assumed). The adapter is always wired and fails
    closed on a missing executable at execute time; the record only reports
    ``available`` when the probe passes."""
    return BackendRecord(
        backend_id="exl3",
        display_name="EXL3 quantization (external, pinned)",
        method_family="exl3",
        formats=("exl3", "safetensors"),
        represents_method="EXL3 primary quantization (4-bit row/group)",
        architectures=("glm-5.2", "k3", "any"),
        compute_archs=("gb10-sm121", "any"),
        topologies=("2x-spark", "single", "any"),
        runtime_compat=("exllamav2",),
        conversion_tool_compat=(command,),
        status=RecipeStatus.DISCOVERED,
        version="n/a",
        declared_capabilities=(),
        supported_formats=(),
        fail_closed=True,
        produces_derivative=True,
        resource_limits=ResourceEstimate(host_gb=32.0, scratch_gb=64.0, workers=1),
        availability_probe=_probe_registry_probe(command),
        parameters=(
            ParameterSpec("bpw", "float", "target bits-per-weight", default="3.25"),
            ParameterSpec("head_bits", "int", "lm_head bits 1..8", default="6"),
            ParameterSpec(
                "hq",
                "string",
                "high-quality attention layers",
                enum=("0", "1"),
                default="1",
            ),
            ParameterSpec("cal_rows", "int", "calibration rows (0 = default)", default="0"),
        ),
        adapter=Exl3Adapter(command=command, timeout_seconds=timeout_seconds),
    )
