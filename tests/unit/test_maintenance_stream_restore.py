import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_maintenance_runner import ALL_PRODUCTION, FakeRunner, config  # noqa: E402

from model_atlas.ops.maintenance import CommandResult, MaintenanceCoordinator


class StreamingRunner(FakeRunner):
    """FakeRunner that also implements run_streaming, emitting shard lines."""

    def run_streaming(self, argv, on_line):
        # simulate a start-script that prints vLLM shard load lines
        for n in range(1, 49):
            if n % 8 == 0:
                on_line(f"Loading safetensors checkpoint shards: {n}/48")
        self.mutations.append(tuple(argv))
        self.active.add("deepseek-v4-flash-vllm-dspark-1")
        return CommandResult(returncode=0, stdout="")


def _events(cfg):
    path: Path = cfg.journal_dir / "maintenance-events.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_restore_streams_dsv4_shard_progress(tmp_path: Path) -> None:
    runner = StreamingRunner(active=set(ALL_PRODUCTION))
    cfg = config(tmp_path).model_copy(update={"dsv4_model_shards": 48})
    MaintenanceCoordinator(cfg, runner, execute=True).run(["./capture"])

    loaded = [e for e in _events(cfg)
              if e["phase"] == "restore" and e["status"] == "shard_loaded"]
    assert loaded, "expected live shard_loaded events during restore"
    # last one should report full progress
    assert loaded[-1]["shard_current"] == 48
    assert loaded[-1]["shard_total"] == 48
    # dsv4 also marked loaded for the resume summary
    assert any(e.get("service") == "dsv4" and e["status"] == "load"
               for e in _events(cfg))
