"""Fail-closed, local-only implementation of the two-node telemetry probe."""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from model_atlas.fit_telemetry import (
    NodeRole,
    ProcessMemory,
    ProcessRole,
    TelemetrySample,
)
from model_atlas.schemas.evidence import EvidenceKind

_MIB = 1024**2
_KIB = 1024


class TelemetryProbeError(RuntimeError):
    """Fail-closed probe error that intentionally contains no payload content."""


class TextRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str: ...


class TextReader(Protocol):
    def __call__(self, path: Path) -> str: ...


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class SubprocessTextRunner:
    def __call__(self, argv: Sequence[str]) -> str:
        result = subprocess.run(list(argv), check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise TelemetryProbeError("GPU query failed")
        return result.stdout


_DEFAULT_RUNNER = SubprocessTextRunner()


@dataclass(frozen=True)
class TelemetryProbeConfig:
    node: NodeRole
    process_role: ProcessRole
    pid: int
    sample_set_id: str
    phase_id: str
    context_tokens: int
    rdma_interface: str
    disk_device: str
    proc_root: Path = Path("/proc")
    sys_class_net: Path = Path("/sys/class/net")
    diskstats_path: Path = Path("/proc/diskstats")
    gpu_index: int = 0

    def __post_init__(self) -> None:
        expected = ProcessRole.SERVER if self.node is NodeRole.HEAD else ProcessRole.RPC
        if self.process_role is not expected:
            raise ValueError("process role does not match node")
        if self.pid <= 0 or self.context_tokens < 0:
            raise ValueError("PID must be positive and context non-negative")
        if not self.sample_set_id or not self.phase_id:
            raise ValueError("sample and phase identities are required")
        if not self.rdma_interface or "/" in self.rdma_interface:
            raise ValueError("RDMA interface must be a single interface name")
        if not self.disk_device or "/" in self.disk_device:
            raise ValueError("disk device must be a single device name")
        if self.gpu_index < 0:
            raise ValueError("GPU index must be non-negative")
        for path in (self.proc_root, self.sys_class_net, self.diskstats_path):
            if not path.is_absolute():
                raise ValueError("telemetry paths must be absolute")


def _records(payload: str, *, label: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for raw in payload.splitlines():
        fields = raw.replace(":", " ", 1).split()
        if len(fields) < 2:
            continue
        try:
            value = int(fields[1])
        except ValueError as exc:
            raise TelemetryProbeError(f"malformed {label}") from exc
        if value < 0:
            raise TelemetryProbeError(f"malformed {label}")
        values[fields[0]] = value
    return values


def _required(values: dict[str, int], *names: str, label: str) -> tuple[int, ...]:
    try:
        return tuple(values[name] for name in names)
    except KeyError as exc:
        raise TelemetryProbeError(f"missing {label} field") from exc


def _gpu(payload: str) -> tuple[int, int, float, float, float]:
    rows = [line.strip() for line in payload.splitlines() if line.strip()]
    if len(rows) != 1:
        raise TelemetryProbeError("GPU query returned an unexpected row count")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 5:
        raise TelemetryProbeError("GPU query returned malformed fields")
    try:
        used, free = int(fields[0]), int(fields[1])
        util, temperature, power = (float(value) for value in fields[2:])
    except ValueError as exc:
        raise TelemetryProbeError("GPU query returned malformed values") from exc
    if used < 0 or free < 0 or not 0 <= util <= 100 or temperature < 0 or power < 0:
        raise TelemetryProbeError("GPU query returned malformed values")
    return used * _MIB, free * _MIB, util, temperature, power


def _disk_bytes(payload: str, device: str) -> tuple[int, int]:
    found: list[list[str]] = [
        line.split() for line in payload.splitlines() if line.split()[2:3] == [device]
    ]
    if len(found) != 1 or len(found[0]) < 10:
        raise TelemetryProbeError("disk device is unavailable")
    try:
        sectors_read, sectors_written = int(found[0][5]), int(found[0][9])
    except ValueError as exc:
        raise TelemetryProbeError("disk counters are malformed") from exc
    if sectors_read < 0 or sectors_written < 0:
        raise TelemetryProbeError("disk counters are malformed")
    return sectors_read * 512, sectors_written * 512


def _single_int(reader: TextReader, path: Path, *, label: str) -> int:
    raw = reader(path).strip()
    if not raw.isdecimal():
        raise TelemetryProbeError(f"malformed {label}")
    return int(raw)


def collect_sample(
    config: TelemetryProbeConfig,
    *,
    reader: TextReader = read_text,
    runner: TextRunner = _DEFAULT_RUNNER,
    hostname: Callable[[], str] = socket.gethostname,
) -> TelemetrySample:
    """Read every required field once.  Missing state is a hard failure."""
    try:
        gpu = _gpu(
            runner(
                (
                    "nvidia-smi",
                    f"--id={config.gpu_index}",
                    "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                )
            )
        )
        memory = _records(reader(config.proc_root / "meminfo"), label="meminfo")
        mem_available_kib, swap_free_kib = _required(
            memory, "MemAvailable", "SwapFree", label="meminfo"
        )
        vmstat = _records(reader(config.proc_root / "vmstat"), label="vmstat")
        pswpin, pswpout = _required(vmstat, "pswpin", "pswpout", label="vmstat")
        process_root = config.proc_root / str(config.pid)
        smaps = _records(reader(process_root / "smaps_rollup"), label="smaps_rollup")
        pss, private_clean, private_dirty, private_hugetlb = _required(
            smaps,
            "Pss",
            "Private_Clean",
            "Private_Dirty",
            "Private_Hugetlb",
            label="smaps_rollup",
        )
        status = _records(reader(process_root / "status"), label="status")
        rss, swap = _required(status, "VmRSS", "VmSwap", label="status")
        process_io = _records(reader(process_root / "io"), label="process io")
        read_bytes, write_bytes = _required(
            process_io, "read_bytes", "write_bytes", label="process io"
        )
        net_root = config.sys_class_net / config.rdma_interface / "statistics"
        rdma_rx = _single_int(reader, net_root / "rx_bytes", label="RDMA rx counter")
        rdma_tx = _single_int(reader, net_root / "tx_bytes", label="RDMA tx counter")
        disk_read, disk_write = _disk_bytes(reader(config.diskstats_path), config.disk_device)
        host = hostname()
        if not host:
            raise TelemetryProbeError("hostname unavailable")
    except TelemetryProbeError:
        raise
    except (OSError, UnicodeError) as exc:
        raise TelemetryProbeError("required telemetry state unavailable") from exc
    return TelemetrySample(
        sample_set_id=config.sample_set_id,
        timestamp=datetime.now(UTC),
        node=config.node,
        hostname=host,
        phase_id=config.phase_id,
        context_tokens=config.context_tokens,
        gpu_used_bytes=gpu[0],
        gpu_free_bytes=gpu[1],
        gpu_util_percent=gpu[2],
        gpu_temperature_c=gpu[3],
        gpu_power_watts=gpu[4],
        mem_available_bytes=mem_available_kib * _KIB,
        swap_free_bytes=swap_free_kib * _KIB,
        pswpin_pages=pswpin,
        pswpout_pages=pswpout,
        process=ProcessMemory(
            role=config.process_role,
            pid=config.pid,
            rss_bytes=rss * _KIB,
            pss_bytes=pss * _KIB,
            private_bytes=(private_clean + private_dirty + private_hugetlb) * _KIB,
            swap_bytes=swap * _KIB,
        ),
        rdma_rx_bytes=rdma_rx,
        rdma_tx_bytes=rdma_tx,
        disk_read_bytes=disk_read,
        disk_write_bytes=disk_write,
        evidence_kind=EvidenceKind.MEASURED,
    )
