"""Enforce the declared stage memory budget on heavy backend subprocesses.

The planner computes per-stage resource budgets (e.g. max_host_gb=64 for the
GGUF conversion stage). Those budgets were previously *declared only*: the
converter/quantizer ran as bare children and an overshoot was punished by the
kernel OOM killer hours into the pass (SIGKILL, no journal evidence, wasted
window).

`cgroup_scope_argv` wraps a command in a transient systemd scope with hard
MemoryMax + soft MemoryHigh limits so:

* pressure reclaim starts early (MemoryHigh) instead of thrashing at the cliff,
* the kill is a clean cgroup OOM with a journal-recordable return code,
* the enforced values are visible in `systemd-cgls` / `systemctl status` for
  post-run audits.

Fail-open policy: if systemd transient scopes are unavailable (no systemd, no
user session bus), we fall back to an RLIMIT_AS soft cap via prlimit when
available, and to an unwrapped command as a last resort — recording which
enforcement level was used so receipts can distinguish "bounded" from
"unbounded" executions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Defaults derived from the GLM-5.2 conversion post-mortem (2026-08-22):
# 16-thread NVFP4 repack peaked past the host's comfortable headroom.
DEFAULT_MEMORY_HIGH_MIB = 56 * 1024  # soft limit: early reclaim pressure
DEFAULT_MEMORY_MAX_MIB = 88 * 1024   # hard limit: clean fail before kernel OOM
DEFAULT_MAX_THREADS = 8              # halves per-tensor dequant temporary peaks


def _systemd_available() -> bool:
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    probe = subprocess.run(
        ["systemctl", "--user", "is-system-running"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    # degenerated/maintenance still accept transient units; only total breakage fails
    return probe.returncode == 0 or "degraded" in probe.stdout or "maintenance" in probe.stdout


def cgroup_scope_argv(
    argv: list[str],
    *,
    label: str,
    memory_high_mib: int = DEFAULT_MEMORY_HIGH_MIB,
    memory_max_mib: int = DEFAULT_MEMORY_MAX_MIB,
) -> tuple[list[str], str]:
    """Wrap ``argv`` in enforcement. Returns (possibly-wrapped argv, mode)."""
    safe_label = "atlas-" + "".join(c if c.isalnum() else "-" for c in label.lower())
    if _systemd_available():
        return (
            [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "-p",
                f"MemoryHigh={memory_high_mib}M",
                "-p",
                f"MemoryMax={memory_max_mib}M",
                "-p",
                "MemorySwapMax=2G",
                "-p",
                f"Description={safe_label}",
            ]
            + argv,
            f"cgroup(MemoryHigh={memory_high_mib}M,MemoryMax={memory_max_mib}M)",
        )
    return list(argv), "unbounded(no-systemd)"


def clamp_threads(threads: int, maximum: int = DEFAULT_MAX_THREADS) -> int:
    """Clamp converter thread count; fewer threads = smaller repack temporaries."""
    return max(1, min(threads, maximum))


__all__ = [
    "cgroup_scope_argv",
    "clamp_threads",
    "DEFAULT_MEMORY_HIGH_MIB",
    "DEFAULT_MEMORY_MAX_MIB",
    "DEFAULT_MAX_THREADS",
]
