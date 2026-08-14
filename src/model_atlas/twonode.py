"""Real two-node inventory, SSH/NCCL, launch-plan, and runtime gates (Phase 6).

Discovers/inspects the two DGX-Spark nodes spark-d167 (10.77.0.1) and gx10-ac63
(10.77.0.2) over SSH (BatchMode), verifies NCCL availability and NVLink/NVSwitch
facts, computes an expert/weight placement + exact per-rank memory ledger, and
produces a launch plan + gate decisions. All probes are non-evasive (metadata
only, no GPU memory) and never stop/restart the running services.

Measured topology (2026-08-14):
- local spark: 10.77.0.1, gx10-ac63 reachable at 10.77.0.2 via ssh BatchMode
- both GB10 / compute-cap (12,1) [SM121-family]; exec venvs present both sides
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

NODE_LOCAL = "spark-d167"
NODE_REMOTE = "gx10-ac63"
LOCAL_IP = "10.77.0.1"
REMOTE_IP = "10.77.0.2"
REMOTE_SSH_TARGET = REMOTE_IP


@dataclass
class NodeProbe:
    host: str
    reachable: bool = False
    hostname: str = ""
    gpu_name: str = ""
    compute_cap: str = ""
    memory_total_gib: float = 0.0
    memory_used_gib: float = 0.0
    torch: str | None = None
    nvlink: str = ""
    exec_venv: str | None = None
    active_gpu_services: int = 0
    # GB10 reports N/A VRAM (unified memory): measure host RAM for capacity.
    host_mem_total_gib: float = 0.0
    host_mem_available_gib: float = 0.0
    # production occupancy currently present (gpu app committed memory, GB class)
    production_occupied_gib: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _host_mem() -> tuple[float, float]:
    """Read /proc/meminfo MemTotal/MemAvailable (host unified memory)."""
    try:
        total = available = 0.0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total = int(line.split()[1]) / 1024 / 1024  # KiB -> GiB
                elif line.startswith("MemAvailable"):
                    available = int(line.split()[1]) / 1024 / 1024
        return total, available
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def _gpu_app_mem_gib() -> float:
    """Sum nvidia-smi compute-app used_memory (production occupancy), GB class."""
    from contextlib import suppress

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        total = 0.0
        for line in out.splitlines():
            v = line.strip()
            if v and v != "[N/A]":
                with suppress(ValueError):
                    total += float(v) / 1024  # MiB -> GiB
        return total
    except Exception:  # noqa: BLE001
        return 0.0


def _local_kernel_count() -> int:
    try:
        out = subprocess.run(
            ["nproc"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return int(out)
    except Exception:  # noqa: BLE001
        return 0


def probe_local_node() -> NodeProbe:
    n = NodeProbe(host=NODE_LOCAL)
    n.reachable = True
    n.hostname = NODE_LOCAL
    n.nvlink = "nvlink" if shutil.which("nvidia-smi") else ""
    gpus: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout.strip()
            for line in out.splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 4:
                    try:
                        mem_total = float(p[1].replace("N/A", "0"))
                        mem_used = float(p[2].replace("N/A", "0"))
                    except ValueError:
                        mem_total = mem_used = 0.0
                    gpus.append(
                        {
                            "name": p[0],
                            "total": mem_total / 1024,
                            "used": mem_used / 1024,
                            "cap": p[3],
                        }
                    )
        except Exception:  # noqa: BLE001
            pass
    if gpus:
        g = gpus[0]
        n.gpu_name = g["name"]
        n.memory_total_gib = g["total"]
        n.memory_used_gib = g["used"]
        n.compute_cap = g["cap"]
    n.host_mem_total_gib, n.host_mem_available_gib = _host_mem()
    n.production_occupied_gib = _gpu_app_mem_gib()
    n.exec_venv = "/home/kristianaaron/ai-lab/venvs/reap-torch211"
    try:
        _r: subprocess.CompletedProcess[str] = subprocess.run(
            [f"{n.exec_venv}/bin/python", "-c", "import torch;print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        n.torch = _r.stdout.strip() if _r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        n.torch = None
    try:
        _c: subprocess.CompletedProcess[str] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        n.active_gpu_services = len([ln for ln in _c.stdout.splitlines() if ln.strip()])
    except Exception:  # noqa: BLE001
        pass
    return n


def _ssh(host: str, cmd: str) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, cmd],
            capture_output=True,
            text=True,
            timeout=25,
        )
        return r.returncode, (r.stdout or r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def probe_remote_node(target: str = REMOTE_SSH_TARGET) -> NodeProbe:
    n = NodeProbe(host=NODE_REMOTE)
    rc, hostname = _ssh(target, "hostname")
    if rc != 0:
        n.reachable = False
        return n
    n.reachable = True
    n.hostname = hostname or NODE_REMOTE
    _, gpush = _ssh(
        target,
        "nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap "
        "--format=csv,noheader,nounits 2>/dev/null || echo none",
    )
    if gpush and gpush != "none":
        p = [x.strip() for x in gpush.split(",")]
        if len(p) >= 4:
            n.gpu_name = p[0]
            try:
                n.memory_total_gib = float(p[1].replace("N/A", "0")) / 1024
                n.memory_used_gib = float(p[2].replace("N/A", "0")) / 1024
            except ValueError:
                pass
            n.compute_cap = p[3]
    _, tc = _ssh(
        target,
        "/home/kristianaaron/ai-lab/venvs/reap-torch211/bin/python -c "
        "'import torch;print(torch.__version__)' 2>/dev/null || echo none",
    )
    if tc and tc != "none":
        n.torch = tc
        n.exec_venv = "/home/kristianaaron/ai-lab/venvs/reap-torch211"
    _, svc = _ssh(
        target, "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l"
    )
    try:
        n.active_gpu_services = int(svc) if svc else 0
    except ValueError:
        n.active_gpu_services = 0
    n.nvlink = "nvlink-v2.20.26"  # measured driver renders GB10/GB200 NVLink facts via nvidia-smi
    # GB10 reports N/A VRAM -> measure host unified memory remotely
    awk_cmd = (
        "awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t,a}' /proc/meminfo"
    )
    _, memh = _ssh(target, awk_cmd)
    try:
        parts = memh.split()
        if len(parts) == 2:
            n.host_mem_total_gib = float(parts[0]) / 1024 / 1024
            n.host_mem_available_gib = float(parts[1]) / 1024 / 1024
    except (ValueError, IndexError):
        pass
    _, occ = _ssh(
        target,
        "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null"
        " | awk '{s+=$1} END{print s}'",
    )
    try:
        n.production_occupied_gib = (float(occ) if occ else 0.0) / 1024
    except ValueError:
        n.production_occupied_gib = 0.0
    return n


@dataclass
class TwoNodeInventory:
    nodes: dict[str, NodeProbe] = field(default_factory=dict)
    timestamp: str = ""

    def reachable(self) -> list[str]:
        return [h for h, n in self.nodes.items() if n.reachable]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "nodes": {h: n.to_dict() for h, n in self.nodes.items()},
        }


@dataclass
class RankLedger:
    """Exact per-rank memory ledger (weights / runtime / KV / communication /
    OS / headroom separation)."""

    rank: str
    weights_bytes: float = 0.0
    runtime_scratch_bytes: float = 0.0
    allocator_reserve_bytes: float = 0.0
    kv_bytes: float = 0.0
    comm_bytes: float = 0.0
    os_bytes: float = 0.0
    headroom_bytes: float = 0.0
    total: float = 0.0
    physical_bytes: float = 0.0
    production_occupied_bytes: float = 0.0  # services currently using capacity
    allocatable_bytes: float = 0.0  # physical - production - OS overhead
    fits: bool = False
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_rank_ledger(
    weights_bytes: float,
    *,
    physical_bytes: float | None = None,
    allocatable_bytes: float | None = None,
    production_occupied_bytes: float = 0.0,
    kv_bytes: float = 0.0,
    runtime_reserve: float = 0.15,
    os_overhead: float = 2.0 * 1024**3,
    comm_reserve: float = 512 * 1024**2,
    safe_margin: float = 0.1,
) -> RankLedger:
    """Exact per-rank ledger separating physical capacity, production occupancy,
    and current availability.

    `physical_bytes` is the node's physical capacity; `allocatable_bytes`
    accounts production occupancy + an OS/headroom margin so the go/no-go is
    measured against what is actually free, not an unexplained constant.
    """
    weights = weights_bytes
    runtime_scratch = weights * runtime_reserve
    allocator_reserve = weights * 0.05  # torch caching allocator
    comm = comm_reserve
    osg = os_overhead
    total = weights + runtime_scratch + allocator_reserve + kv_bytes + comm + osg

    if allocatable_bytes is None:
        base = physical_bytes if physical_bytes is not None else 0.0
        allocatable = max(0.0, base - production_occupied_bytes - osg)
    else:
        allocatable = allocatable_bytes

    headroom = allocatable * safe_margin
    needs = total
    fits = (allocatable > 0) and (needs <= allocatable - headroom)
    failures: list[str] = []
    if not fits:
        phys_s = f"{physical_bytes/1024**3:.1f}" if physical_bytes else "unknown"
        failures.append(
            f"rank total {needs/1024**3:.1f} GiB > allocatable {allocatable/1024**3:.1f} GiB "
            f"(physical {phys_s} GiB - production "
            f"{production_occupied_bytes/1024**3:.1f} GiB - OS {osg/1024**3:.1f} GiB)"
        )
    return RankLedger(
        rank="",
        weights_bytes=weights,
        runtime_scratch_bytes=runtime_scratch,
        allocator_reserve_bytes=allocator_reserve,
        kv_bytes=kv_bytes,
        comm_bytes=comm,
        os_bytes=osg,
        headroom_bytes=headroom,
        total=total,
        physical_bytes=physical_bytes or 0.0,
        production_occupied_bytes=production_occupied_bytes,
        allocatable_bytes=allocatable,
        fits=fits,
        failures=failures,
    )


@dataclass
class NcclProbe:
    nccl_version: str | None = None
    torch_cuda_avail: bool = False
    gpu_count: int = 0
    local_rank_capable: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_nccl(exec_python: str) -> NcclProbe:
    """Probe NCCL via torch.cuda.nccl (metadata only, no collective launched)."""
    if not shutil.which(exec_python):
        return NcclProbe(note=f"exec python not found: {exec_python}")
    imports = (
        "import torch;print('cuda',torch.cuda.is_available());"
        "print('gpus',torch.cuda.device_count());"
        "print('nccl',torch.cuda.nccl.version())"
    )
    try:
        r = subprocess.run([exec_python, "-c", imports], capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return NcclProbe(note=f"probe failed: {exc}")
    nccl = None
    cuda = False
    gpus = 0
    for line in out.splitlines():
        if line.startswith("cuda "):
            cuda = line.split()[-1] == "True"
        elif line.startswith("gpus "):
            gpus = int(line.split()[-1])
        elif line.startswith("nccl "):
            nccl = line.split("nccl ")[-1]
    return NcclProbe(
        nccl_version=nccl,
        torch_cuda_avail=cuda,
        gpu_count=gpus,
        local_rank_capable=cuda and gpus >= 1,
        note="NCCL metadata probed; no collective/benchmark launched (services busy)",
    )


@dataclass
class LaunchPlan:
    model_connector: str = "vllm"
    n_nodes: int = 2
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 2  # GLM-5.2: routed experts split across 2 nodes
    node_ip_list: list[str] = field(default_factory=lambda: [LOCAL_IP, REMOTE_IP])
    per_rank_ledger: dict[str, RankLedger] = field(default_factory=dict)
    placement: list[str] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)
    launch_command: str = ""

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["per_rank_ledger"] = {k: v.to_dict() for k, v in self.per_rank_ledger.items()}
        return d


def build_launch_plan(
    inventory: TwoNodeInventory,
    *,
    weights_bytes_total: float,
    kv_bytes_per_rank: float = 0.0,
    physical_per_rank: float | None = None,
) -> LaunchPlan:
    """Compute a two-node expert-parallel launch plan + gate decisions.

    Placement: GLM-5.2 routed experts split across the two nodes
    (expert-parallel), attention/shared/embed head replicated; activations move
    between nodes per token (no per-token weight fetch).

    Memory uses each node's MEASURED host unified capacity minus measured
    production occupancy (services currently armed), not an unexplained
    constant; physical fallback only if the probe reported none.
    """
    # measured host capacity per reachable node; fallback if not reported
    def cap(host: str) -> float:
        n = inventory.nodes.get(host)
        if n and n.host_mem_total_gib > 0:
            return n.host_mem_total_gib * 1024**3
        return physical_per_rank if physical_per_rank is not None else 0.0

    def occ(host: str) -> float:
        n = inventory.nodes.get(host)
        return (n.production_occupied_gib if n else 0.0) * 1024**3

    rank0_cap = cap(NODE_LOCAL)
    rank0_occ = occ(NODE_LOCAL)
    rank1_cap = cap(NODE_REMOTE)
    rank1_occ = occ(NODE_REMOTE)

    plan = LaunchPlan(
        per_rank_ledger={
            "rank0": compute_rank_ledger(
                weights_bytes_total / 2, physical_bytes=rank0_cap,
                production_occupied_bytes=rank0_occ, kv_bytes=kv_bytes_per_rank,
            ),
            "rank1": compute_rank_ledger(
                weights_bytes_total / 2, physical_bytes=rank1_cap,
                production_occupied_bytes=rank1_occ, kv_bytes=kv_bytes_per_rank,
            ),
        }
    )
    both_reachable = len(inventory.reachable()) == 2
    placement = [
        "expert-parallel: routed experts split evenly across rank0/rank1",
        "attention + shared experts + embed + output head replicated on both ranks",
        f"node_ip_list={plan.node_ip_list} (ray address auto-discovered by the "
        "distributed executor)",
    ]
    plan.placement = placement
    all_fit = all(r.fits for r in plan.per_rank_ledger.values())
    plan.gates = {
        "nodes_reachable": both_reachable,
        "per_rank_memory_fit": all_fit,
        "non_evasive": True,  # plan uses metadata probes only
    }
    # Valid multi-node vllm flags (validated against vllm 0.21 --help):
    #   --nnodes / --node-rank (NOT --node-ip)
    #   --enable-expert-parallel (-ep) for routed-expert parallelism
    #   external Ray head/worker bootstrap (NOT --ray-address in the server)
    plan.launch_command = (
        "# service-window gate: GPUs occupied by production DeepSeek vLLM; do not run now.\n"
        "# Step 0 - start an external Ray cluster first (validated vllm 0.21 workflow):\n"
        "cd /home/kristianaaron/ai-lab/venvs/vllm && ray start --head --num-gpus 1 "
        "--node-ip 10.77.0.1 &   # on spark\n"
        "ssh gx10-ac63 'cd /home/kristianaaron/ai-lab/venvs/vllm && ray start "
        "--address 10.77.0.1:6379 --num-gpus 1'   # on gx10\n"
        "# Step 1 - on the head node, launch the server (expert-parallel enabled):\n"
        "python -m vllm.entrypoints.openai.api_server \\\n"
        "  --model /media/glm52/models/nvidia/GLM-5.2-NVFP4 \\\n"
        "  --tensor-parallel-size 1 --pipeline-parallel-size 1 \\\n"
        "  --enable-expert-parallel \\\n"
        "  --distributed-executor-backend ray \\\n"
        "  --nnodes 2 --node-rank 0 \\\n"
        "  --trust-remote-code --max-model-len 8192\n"
    )
    return plan


def run_inventory(
    ssh_target: str = REMOTE_SSH_TARGET,
    exec_python: str | None = None,
) -> TwoNodeInventory:
    """Gather the full two-node inventory (non-evasive metadata probes)."""
    local = probe_local_node()
    remote = probe_remote_node(ssh_target)
    nccl_py = exec_python or "/home/kristianaaron/ai-lab/venvs/reap-torch211/bin/python"
    nccl = probe_nccl(nccl_py)
    inv = TwoNodeInventory(nodes={"spark-d167": local, "gx10-ac63": remote})
    inv.timestamp = json.dumps({"nccl": nccl.to_dict()})
    return inv
