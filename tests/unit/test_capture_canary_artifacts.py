from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
TOKENS = ROOT / "artifacts/glm52-capture-canary.jsonl"
MANIFEST = ROOT / "artifacts/glm52-capture-canary-manifest.json"


def test_capture_canary_is_frozen_bounded_and_not_quality_evidence() -> None:
    token_bytes = TOKENS.read_bytes()
    manifest_bytes = MANIFEST.read_bytes()
    assert hashlib.sha256(token_bytes).hexdigest() == (
        "539e8f6682cfe3768195e65e75ffcd2ba83b229282e79d94983856ff70ad07c8"
    )
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "a141e026b908d745890213371be8fa6f7fd23e727149ade25d44bd3e5b14a885"
    )
    rows = [json.loads(line) for line in token_bytes.splitlines()]
    manifest = json.loads(manifest_bytes)
    assert len(rows) == manifest["sample_count"] == 8
    assert manifest["purpose"] == "pipeline_identity_control"
    assert manifest["partition"] == "canary_not_quality_evidence"
    assert manifest["quality_claim"] is False
    assert manifest["forced_tokens_sha256"] == hashlib.sha256(token_bytes).hexdigest()
    assert all(set(row) == {"sample_id", "domain", "token_ids"} for row in rows)
    assert len({row["sample_id"] for row in rows}) == len(rows)
    assert all(2 <= len(row["token_ids"]) <= 256 for row in rows)
    ordered_ids = [row["sample_id"] for row in rows]
    ordered_digest = hashlib.sha256(
        json.dumps(ordered_ids, separators=(",", ":")).encode()
    ).hexdigest()
    assert manifest["ordered_sample_ids_sha256"] == ordered_digest


def test_capture_canary_tokenizer_identity_is_pinned() -> None:
    manifest = json.loads(MANIFEST.read_bytes())
    tokenizer = manifest["tokenizer"]
    assert tokenizer == {
        "path": "/media/glm52/models/nvidia/GLM-5.2-NVFP4/tokenizer.json",
        "sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "config_sha256": "77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a",
        "chat_template_sha256": "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
        "implementation": "tokenizers.Tokenizer.from_file",
        "implementation_version": "0.22.2",
        "add_special_tokens": False,
        "local_files_only": True,
    }
