#!/usr/bin/env python3
"""Durable operator entry point for the real GLM-5.2 mixed-GGUF canary.

This runs the pinned backend adapter directly so conversion can proceed with the
CPU-only quantizer while production GPU services remain online. The resulting
artifact is still runtime-unvalidated and must be adopted/evaluated by Atlas
before it can be promoted as a platform candidate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
from pathlib import Path

from model_atlas.backend.llamacpp_gguf import (
    LlamaCppGgufMixedAdapter,
    SubprocessRunner,
)
from model_atlas.recipes.builtin import (
    GLM52_GGUF_TENSOR_PLAN_PATH,
    GLM52_GGUF_TENSOR_PLAN_SHA256,
    GLM52_SOURCE_PATH,
)

_DEFAULT_ROOT = Path("/home/kristianaaron/tmp/model-atlas/experiments/glm52-mixed-gguf")
_HF_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"


class StreamingRunner(SubprocessRunner):
    """Stream long-running tool output to one append-only operator log."""

    def __init__(self, log_path: Path, timeout_seconds: float = 86_400.0) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.log_path = log_path

    def run(self, argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab", buffering=0) as log:
            log.write(("\n$ " + shlex.join(argv) + "\n").encode())
            result = subprocess.run(
                argv,
                cwd=str(cwd),
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
            )
        detail = ""
        if result.returncode:
            with self.log_path.open("rb") as log:
                log.seek(max(0, self.log_path.stat().st_size - 4000))
                detail = log.read(4000).decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(argv, result.returncode, "", detail)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="required safety acknowledgement")
    parser.add_argument("--source", default=GLM52_SOURCE_PATH)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--plan", default=GLM52_GGUF_TENSOR_PLAN_PATH)
    parser.add_argument("--plan-sha256", default=GLM52_GGUF_TENSOR_PLAN_SHA256)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required")

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "run.lock"
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another GLM mixed-GGUF run holds the lock") from exc

    staging = root / "stage/staging"
    context: dict[str, object] = {
        "source": str(Path(args.source).resolve()),
        "source_revision": _HF_REVISION,
        "source_identity": {
            "source_id": "nvidia/GLM-5.2-NVFP4",
            "checkpoint_revision": _HF_REVISION,
            "checkpoint_path": str(Path(args.source).resolve()),
            "identity_evidence": "local read-only HF snapshot tree metadata",
        },
        "staging_dir": str(staging),
        "parameters": {
            "tensor_plan_path": str(Path(args.plan).resolve()),
            "tensor_plan_sha256": args.plan_sha256,
            "threads": str(args.threads),
        },
    }
    _atomic_json(root / "run-context.json", context)
    adapter = LlamaCppGgufMixedAdapter(runner=StreamingRunner(root / "commands.log"))
    handle = adapter.prepare(context)
    result = adapter.execute(context, handle)
    _atomic_json(root / "run-result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
