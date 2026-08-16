"""Typed backend plugin contract and capability registry.

The Atlas compression backend contract is **interface-agnostic**: any method
family (EXL3/EXO quant, ModelOpt-NVFP4, LLM-Compressor, Eval-Lab, a custom
research backend, or an existing in-repo Atlas operation) registers a concrete
adapter and declares:

* what formats it consumes/produces,
* which architecture/runtime it is compatible with,
* a parameter schema (json-schema subset we validate against),
* resource estimates + availability probe,
* the lifecycle status it honestly holds today,
* and the prepare/execute/resume/validate methods.

A backend that is not installed, not importable, or otherwise unavailable is
``UNAVAILABLE`` and **fails closed** — the compiler and the job engine refuse to
run a recipe stage on it. Nothing ever simulates success.

Status lifecycle: ``UNAVAILABLE -> DISCOVERED -> EXPERIMENTAL -> VALIDATED ->
RECOMMENDED``. A method may only reach ``EXPERIMENTAL``/``VALIDATED`` on
captured evidence, never by fiat.
"""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from model_atlas.recipe.schema import RecipeStatus

if TYPE_CHECKING:
    from model_atlas.backend.registry import BackendRegistry


class AvailabilityProbe(Protocol):
    """Probe signature: returns (available, version, evidence_note)."""

    def __call__(self) -> tuple[bool, str | None, str]: ...


class ExecutionIdentityProbe(Protocol):
    """Return the fresh, canonical identity of execution-affecting tools.

    This probe must be read-only.  It exists separately from the cached
    availability probe because a compiled artifact must bind the exact bytes
    that will execute and start must re-check those bytes under its lock.
    """

    def __call__(self) -> Mapping[str, str]: ...


def module_present(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def command_exists(cmd: str) -> bool:
    import shutil

    return shutil.which(cmd) is not None


@dataclass(frozen=True)
class ParameterSpec:
    """One declared stage parameter (subset of JSON Schema)."""

    name: str
    type: str = "string"
    description: str = ""
    default: str | None = None
    required: bool = False
    enum: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, value: str) -> list[str]:
        errors: list[str] = []
        if self.type == "int":
            try:
                int(value)
            except ValueError:
                errors.append(f"{self.name}: expected int, got {value!r}")
        elif self.type == "float":
            try:
                float(value)
            except ValueError:
                errors.append(f"{self.name}: expected float, got {value!r}")
        if self.enum and value not in self.enum:
            errors.append(f"{self.name}: {value!r} not in {sorted(self.enum)}")
        if self.minimum is not None:
            try:
                if float(value) < self.minimum:
                    errors.append(f"{self.name}: < minimum {self.minimum}")
            except ValueError:
                pass
        if self.maximum is not None:
            try:
                if float(value) > self.maximum:
                    errors.append(f"{self.name}: > maximum {self.maximum}")
            except ValueError:
                pass
        return errors


@dataclass(frozen=True)
class ResourceEstimate:
    """Predicted resource footprint of one stage (never claimed measured)."""

    host_gb: float = 0.0
    scratch_gb: float = 0.0
    workers: int = 1
    wall_seconds: float | None = None  # None = not estimated
    evidence: str = "predicted"


@dataclass
class BackendRecord:
    """One registered backend + its truthful capability/status state."""

    backend_id: str
    display_name: str
    method_family: str  # exl3 | modelopt | llm_compressor | eval_lab | atlas | custom
    formats: tuple[str, ...]
    represents_method: str = ""  # e.g. "EXL3 quantization"
    architectures: tuple[str, ...] = ("glm-5.2", "k3", "k3-mini", "any")
    compute_archs: tuple[str, ...] = ("gb10-sm121", "any")
    topologies: tuple[str, ...] = ("2x-spark", "single", "any")
    # SERVING runtime implementations that can RUN the backend's output (truthful
    # only; never telemetry/compute-arch strings; no wildcard).
    runtime_compat: tuple[str, ...] = ()
    # CONVERSION-tool compatibility — a SEPARATE dimension from serving runtime:
    # which conversion tool produces the backend's format (e.g. exl3.exe,
    # llm-compressor). Never conflated with runtime_compat.
    conversion_tool_compat: tuple[str, ...] = ()

    # lifecycle status: UNAVAILABLE / DISCOVERED / EXPERIMENTAL / VALIDATED / RECOMMENDED
    status: RecipeStatus = RecipeStatus.UNAVAILABLE
    version: str = "n/a"
    # declared support
    declared_capabilities: tuple[str, ...] = ()  # e.g. "pruning", "hybrid:nvfp4+fp8"
    supported_formats: tuple[str, ...] = ()
    # failure-closed guarantee
    fail_closed: bool = True
    # P0: only true for adapters that produce a REAL derivative checkpoint.
    produces_derivative: bool = False
    # P1: bounded resource envelope this backend can serve (host/workers); None =
    # unbounded-declared (compiler skips the resource-bounds check).
    resource_limits: ResourceEstimate | None = None
    # probe
    availability_probe: AvailabilityProbe | None = None
    execution_identity_probe: ExecutionIdentityProbe | None = None
    _availability_cached: tuple[bool, str | None, str] | None = field(default=None, repr=False)

    # typed parameter schema (validated before execute)
    parameters: tuple[ParameterSpec, ...] = ()

    # adapter (optional until contract requires execution)
    adapter: BackendAdapter | None = None

    def is_available(self, registry: BackendRegistry) -> bool:
        if self.availability_probe is None:
            return False
        ok, _, _ = self.probe(registry)
        return ok

    def probe(self, registry: BackendRegistry) -> tuple[bool, str | None, str]:
        """Probe availability, with per-registry caching (stale within a compile
        pass is acceptable and deterministic — probes are cached per registry
        instance to keep compilation pure)."""
        if self._availability_cached is None:
            if self.availability_probe is None:
                self._availability_cached = (
                    False,
                    None,
                    f"{self.backend_id}: no availability probe registered (unavailable)",
                )
            else:
                self._availability_cached = self.availability_probe()
        return self._availability_cached

    def probe_fresh(self) -> tuple[bool, str | None, str]:
        """Perform an uncached availability check for dispatch-time safety."""
        if self.availability_probe is None:
            return False, None, f"{self.backend_id}: no availability probe registered"
        result = self.availability_probe()
        self._availability_cached = result
        return result

    def fresh_execution_identity(self) -> Mapping[str, str]:
        """Return fresh tool identity, or a stable in-process backend identity."""
        if self.execution_identity_probe is not None:
            return self.execution_identity_probe()
        adapter = self.adapter
        return {
            "backend_id": self.backend_id,
            "backend_version": self.version,
            "adapter": (
                "none-adapter"
                if adapter is None
                else f"{adapter.backend_id}::{adapter.__class__.__name__}"
            ),
        }

    # ------------------------------------------------------------ lifetime
    def note_discovered(self, evidence: str) -> None:
        """DISCOVERED: a working reference/implementation was found (its binary
        or source); execution may be REHEARSED but results are not validated."""
        self.status = RecipeStatus.DISCOVERED
        self._evidence = evidence

    def note_experimental(self, evidence: str) -> None:
        """EXPERIMENTAL: results reproduced on controlled small runs."""
        self.status = RecipeStatus.EXPERIMENTAL
        self._evidence = evidence

    def note_validated(self, evidence: str) -> None:
        """VALIDATED: the gate (eq/held-out/runtime) passed for a real output."""
        self.status = RecipeStatus.VALIDATED
        self._evidence = evidence

    def note_recommended(self, evidence: str) -> None:
        """RECOMMENDED: validated on the canonical product recipe."""
        self.status = RecipeStatus.RECOMMENDED
        self._evidence = evidence

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "display_name": self.display_name,
            "method_family": self.method_family,
            "formats": list(self.formats),
            "represents_method": self.represents_method,
            "architectures": list(self.architectures),
            "runtime_compat": list(self.runtime_compat),
            "status": self.status.value,
            "version": self.version,
            "declared_capabilities": list(self.declared_capabilities),
            "supported_formats": list(self.supported_formats),
            "fail_closed": self.fail_closed,
            "produces_derivative": self.produces_derivative,
            "parameters": [p.__dict__ for p in self.parameters],
        }


class BackendAdapter(ABC):
    """Concrete executable backend. All methods may raise BackendUnavailable."""

    backend_id: str = ""
    # P0: whether execute() produces a real derivative checkpoint. Probe-only
    # math / analysis adapters must keep this False; the job engine refuses to
    # mark a compression stage DONE from a non-derivative producer.
    produces_derivative: bool = False

    @abstractmethod
    def prepare(self, context: dict[str, object]) -> str:
        """Idempotent preparation; returns an opaque handle id."""

    @abstractmethod
    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]:
        """Run the stage; return typed evidence + output references."""

    @abstractmethod
    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]:
        """Crash-safe resume of a staged/incomplete run."""

    @abstractmethod
    def validate(self, context: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
        """Validate stage outputs against declared gates; returns typed result."""

    def describe(self) -> str:
        return f"{self.__class__.__name__}({self.backend_id})"


class CommandBackedAdapter(BackendAdapter):
    """Adapter that shells out to a pinned external command/venv python.

    Execute is command-backed: it runs ``prepare_cmd``(prepare) or
    ``run_cmd``(execute) with ``{params}``/``{handle}``/``{workdir}`` placeholders.
    A missing command is a hard error at prepare time (fail closed). The
    implementation is intentionally thin — it only *drives* a real dependency;
    it cannot make an absent dependency appear present.
    """

    execute_kind: str = "command"  # command | module | importable

    def __init__(
        self,
        *,
        backend_id: str,
        prepare_cmd: str | None = None,
        run_cmd: str | None = None,
        resume_cmd: str | None = None,
        validate_cmd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.backend_id = backend_id
        self._prepare_cmd = prepare_cmd
        self._run_cmd = run_cmd
        self._resume_cmd = resume_cmd
        self._validate_cmd = validate_cmd
        self._env = env or {}

    def _expand(self, cmd: str, context: dict[str, object], handle: str) -> str:
        workdir = str(context.get("workdir", "."))
        return (
            cmd.replace("{workdir}", workdir)
            .replace("{handle}", handle)
            .replace("{params}", _json_arg(context.get("parameters", {})))
        )

    def _run_command(self, cmd: str, context: dict[str, object]) -> None:
        expanded = self._expand(cmd, context, self._handle_of(context))
        try:
            timeout = context.get("timeout_seconds", 3600)
            subprocess.run(
                expanded,
                shell=True,
                cwd=str(context.get("workdir", ".")),
                env={**__import__("os").environ, **self._env},
                check=True,
                capture_output=True,
                text=True,
                timeout=int(str(timeout)),
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"{self.backend_id} command failed (exit {e.returncode}): {e.stderr[-2000:]}"
            ) from e
        except FileNotFoundError as e:
            raise BackendUnavailable(f"{self.backend_id}: command not found ({e})") from e

    def _handle_of(self, context: dict[str, object]) -> str:
        h = context.get("handle")
        return h if isinstance(h, str) else "auto"

    def prepare(self, context: dict[str, object]) -> str:
        if not self._prepare_cmd:
            return "auto"
        self._run_command(self._prepare_cmd, context)
        return "auto"

    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]:
        if not self._run_cmd:
            raise BackendUnavailable(
                f"{self.backend_id}: no execute command wired — dependency integration "
                "is not implemented; refusing to fabricate an output"
            )
        self._run_command(self._run_cmd, context)
        return {"command_executed": True}

    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]:
        if not self._resume_cmd:
            # crash-safe resume defaults to re-running execute idempotently.
            return self.execute(context, handle)
        self._run_command(self._resume_cmd, context)
        return {"command_resumed": True}

    def validate(self, context: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
        if not self._validate_cmd:
            # No validator wired: declare the results UNVALIDATED, not pass/fail.
            return {"validated": False, "status": "unvalidated", "errors": ["no validator"]}
        self._run_command(self._validate_cmd, context)
        return {"validated": True, "status": "passed"}


class BackendUnavailable(RuntimeError):
    """Raised when a dependency is missing at execution time (fail closed)."""


def _json_arg(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@runtime_checkable
class BackendAdapterLike(Protocol):
    backend_id: str

    def prepare(self, context: dict[str, object]) -> str: ...
    def execute(self, context: dict[str, object], handle: str) -> dict[str, object]: ...
    def resume(self, context: dict[str, object], handle: str) -> dict[str, object]: ...

    def validate(
        self, context: dict[str, object], outputs: dict[str, object]
    ) -> dict[str, object]: ...
