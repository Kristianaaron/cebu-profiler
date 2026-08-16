#!/usr/bin/env python3
"""Build the GLM-5.2 mounted-source identity and mixed-GGUF profile.

The command is intentionally dry-run unless --execute is supplied: hashing the
mounted model is an explicit expensive operation, never an import side effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_atlas.glm52_source_profile import (
    build_glm52_mixed_gguf_profile,
    build_resumable_source_manifest,
)

_SOURCE = "/media/glm52/models/nvidia/GLM-5.2-NVFP4"
_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="perform mounted-source hashing")
    parser.add_argument("--source", default=_SOURCE)
    parser.add_argument("--revision", default=_REVISION)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/glm52-source-manifest.state.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/glm52-source-manifest.json")
    )
    parser.add_argument(
        "--profile", type=Path, default=Path("profiles/glm52-nvfp4-mixed-gguf.json")
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--risk", type=Path, default=Path("artifacts/glm52-nvfp4-quant-risk.json"))
    parser.add_argument(
        "--tensor-plan", type=Path, default=Path("artifacts/glm52-gguf-tensor-types.txt")
    )
    args = parser.parse_args()
    tokenizer = args.tokenizer or Path(args.source) / "tokenizer.json"
    if not args.execute:
        print(
            json.dumps(
                {
                    "would_build_manifest": str(args.manifest),
                    "would_checkpoint": str(args.checkpoint),
                    "would_build_profile": str(args.profile),
                    "source": args.source,
                    "requires": "--execute",
                },
                sort_keys=True,
            )
        )
        return 0
    result = build_resumable_source_manifest(
        args.source, checkpoint_path=args.checkpoint, output_path=args.manifest
    )
    profile = build_glm52_mixed_gguf_profile(
        manifest_path=args.manifest,
        source_path=args.source,
        source_revision=args.revision,
        tokenizer_path=tokenizer,
        risk_path=args.risk,
        tensor_plan_path=args.tensor_plan,
        output_path=args.profile,
    )
    print(
        json.dumps(
            {
                "manifest_digest": result.digest,
                "hashed_files": result.hashed_files,
                "reused_files": result.reused_files,
                "profile": str(args.profile),
                "profile_kind": profile["profile_kind"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
