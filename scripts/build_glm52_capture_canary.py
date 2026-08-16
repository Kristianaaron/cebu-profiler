#!/usr/bin/env python3
"""Build the frozen forced-token corpus for the GLM pipeline identity control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

_TOKENIZER_ROOT = Path("/media/glm52/models/nvidia/GLM-5.2-NVFP4")
_TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
_TOKENIZER_CONFIG_SHA256 = "77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a"
_CHAT_TEMPLATE_SHA256 = "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679"
_SAMPLES = (
    (
        "canary-math-001",
        "math",
        "Compute 17 multiplied by 19, then subtract 23. Show the arithmetic.",
    ),
    (
        "canary-code-001",
        "code",
        "Explain why a stable sort preserves the order of records with equal keys.",
    ),
    (
        "canary-reason-001",
        "reasoning",
        "Which is larger, 7/11 or 5/8? Compare them using exact integer arithmetic.",
    ),
    (
        "canary-science-001",
        "science",
        "At standard pressure, what phase change occurs when liquid water reaches zero "
        "degrees Celsius?",
    ),
    (
        "canary-json-001",
        "instruction",
        "Return a JSON object with keys name, count, and enabled for a widget count of three.",
    ),
    (
        "canary-summary-001",
        "summarization",
        "Summarize in one sentence: careful measurements must record inputs, versions, "
        "and uncertainty.",
    ),
    (
        "canary-logic-001",
        "logic",
        "All ravens in the garden are black. This bird is a raven in the garden. What follows?",
    ),
    (
        "canary-safety-001",
        "safety",
        "State a safe first response when an unknown electrical device begins smoking indoors.",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError("incomplete canary artifact write")
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--tokens-output", type=Path, default=Path("artifacts/glm52-capture-canary.jsonl")
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/glm52-capture-canary-manifest.json"),
    )
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute_required": True,
                    "sample_count": len(_SAMPLES),
                    "tokens_output": str(args.tokens_output),
                    "manifest_output": str(args.manifest_output),
                    "quality_claim": False,
                },
                sort_keys=True,
            )
        )
        return 0

    tokenizer_path = _TOKENIZER_ROOT / "tokenizer.json"
    config_path = _TOKENIZER_ROOT / "tokenizer_config.json"
    template_path = _TOKENIZER_ROOT / "chat_template.jinja"
    expected = (
        (tokenizer_path, _TOKENIZER_SHA256),
        (config_path, _TOKENIZER_CONFIG_SHA256),
        (template_path, _CHAT_TEMPLATE_SHA256),
    )
    if any(_sha256_file(path) != digest for path, digest in expected):
        raise SystemExit("tokenizer inputs differ from the reviewed GLM-5.2 identities")

    import tokenizers  # type: ignore[import-not-found]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    records: list[dict[str, object]] = []
    prompt_rows: list[dict[str, str]] = []
    for sample_id, domain, prompt in _SAMPLES:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(token_ids) < 2 or len(token_ids) > 256 or any(token < 0 for token in token_ids):
            raise SystemExit(f"tokenized canary sample {sample_id} violates bounds")
        records.append({"sample_id": sample_id, "domain": domain, "token_ids": token_ids})
        prompt_rows.append(
            {
                "sample_id": sample_id,
                "domain": domain,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    tokens_bytes = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in records
    )
    ordered_ids = [sample_id for sample_id, _, _ in _SAMPLES]
    manifest = {
        "schema_version": 1,
        "purpose": "pipeline_identity_control",
        "partition": "canary_not_quality_evidence",
        "quality_claim": False,
        "sample_count": len(records),
        "samples": prompt_rows,
        "ordered_sample_ids_sha256": hashlib.sha256(
            json.dumps(ordered_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "forced_tokens_sha256": hashlib.sha256(tokens_bytes).hexdigest(),
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": _TOKENIZER_SHA256,
            "config_sha256": _TOKENIZER_CONFIG_SHA256,
            "chat_template_sha256": _CHAT_TEMPLATE_SHA256,
            "implementation": "tokenizers.Tokenizer.from_file",
            "implementation_version": tokenizers.__version__,
            "add_special_tokens": False,
            "local_files_only": True,
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _exclusive_write(args.tokens_output, tokens_bytes)
    try:
        _exclusive_write(args.manifest_output, manifest_bytes)
    except BaseException:
        args.tokens_output.unlink(missing_ok=True)
        raise
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
