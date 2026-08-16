from __future__ import annotations

from pathlib import Path

import pytest

from model_atlas.fit_telemetry import NodeRole, ProcessRole
from model_atlas.telemetry_probe import (
    TelemetryProbeConfig,
    TelemetryProbeError,
    collect_sample,
)


def _config() -> TelemetryProbeConfig:
    return TelemetryProbeConfig(
        node=NodeRole.HEAD,
        process_role=ProcessRole.SERVER,
        pid=101,
        sample_set_id="set-1",
        phase_id="load",
        context_tokens=4096,
        rdma_interface="rdma0",
        disk_device="nvme0n1",
        proc_root=Path("/proc"),
        sys_class_net=Path("/sys/class/net"),
        diskstats_path=Path("/proc/diskstats"),
    )


def _files() -> dict[Path, str]:
    return {
        Path("/proc/meminfo"): "MemAvailable: 20 kB\nSwapFree: 30 kB\n",
        Path("/proc/vmstat"): "pswpin 4\npswpout 5\n",
        Path("/proc/101/smaps_rollup"): (
            "Pss: 1 kB\nPrivate_Clean: 2 kB\nPrivate_Dirty: 3 kB\nPrivate_Hugetlb: 4 kB\n"
        ),
        Path("/proc/101/status"): "VmRSS: 6 kB\nVmSwap: 7 kB\n",
        Path("/proc/101/io"): "read_bytes: 8\nwrite_bytes: 9\n",
        Path("/sys/class/net/rdma0/statistics/rx_bytes"): "10\n",
        Path("/sys/class/net/rdma0/statistics/tx_bytes"): "11\n",
        Path("/proc/diskstats"): "259 0 nvme0n1 0 0 12 0 0 0 13 0 0 0 0\n",
    }


def test_collects_one_strict_sample_without_real_gpu_or_proc_access() -> None:
    files = _files()
    commands: list[tuple[str, ...]] = []

    def reader(path: Path) -> str:
        return files[path]

    def runner(argv: object) -> str:
        commands.append(tuple(argv))  # type: ignore[arg-type]
        return "100, 200, 50, 60, 70\n"

    sample = collect_sample(_config(), reader=reader, runner=runner, hostname=lambda: "spark-head")
    assert commands == [
        (
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        )
    ]
    assert sample.hostname == "spark-head"
    assert sample.gpu_used_bytes == 100 * 1024**2
    assert sample.mem_available_bytes == 20 * 1024
    assert sample.process.private_bytes == 9 * 1024
    assert sample.disk_read_bytes == 12 * 512
    assert sample.disk_write_bytes == 13 * 512


def test_fails_closed_on_missing_required_process_field() -> None:
    files = _files()
    files[Path("/proc/101/status")] = "VmRSS: 6 kB\n"
    with pytest.raises(TelemetryProbeError, match="missing status"):
        collect_sample(
            _config(),
            reader=lambda path: files[path],
            runner=lambda _argv: "100, 200, 50, 60, 70\n",
            hostname=lambda: "spark-head",
        )


def test_fails_closed_on_missing_pid_state_and_bad_gpu_row() -> None:
    files = _files()
    files.pop(Path("/proc/101/io"))

    def missing_reader(path: Path) -> str:
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    with pytest.raises(TelemetryProbeError, match="required telemetry"):
        collect_sample(
            _config(),
            reader=missing_reader,
            runner=lambda _argv: "100, 200, 50, 60, 70\n",
            hostname=lambda: "spark-head",
        )
    with pytest.raises(TelemetryProbeError, match="GPU"):
        collect_sample(
            _config(),
            reader=lambda path: _files()[path],
            runner=lambda _argv: "bad\n",
            hostname=lambda: "spark-head",
        )


def test_config_requires_role_and_explicit_local_interface_and_disk() -> None:
    with pytest.raises(ValueError, match="process role"):
        TelemetryProbeConfig(**{**_config().__dict__, "process_role": ProcessRole.RPC})
    with pytest.raises(ValueError, match="RDMA"):
        TelemetryProbeConfig(**{**_config().__dict__, "rdma_interface": "bad/name"})
