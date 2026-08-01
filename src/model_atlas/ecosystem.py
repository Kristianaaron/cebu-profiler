"""Native eval-lab ⇄ model-atlas pipeline (ecosystem bridge).

Lets the two platforms talk data, not just links:

- ingress: read eval-lab's real task corpus (`tasks/**/prompt.md` + domain
  paths) into an Atlas calibration corpus tagged with capability labels and a
  data partition, so REAP saliency runs on the harness's actual tasks.
- egress: emit measured atlas findings / derivatives as JSON manifests that
  eval-lab (the `eval-lab atlas` plugin + ModelAssetService) can consume.

Everything downstream is the same measured F3–F13 runtime; only the corpus
source is real eval-lab tasks instead of the synthetic generator. Deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from model_atlas.atlas.reap import CalibrationSample, run_calibration
from model_atlas.atlas.runtime import MiniMoE
from model_atlas.registry.architectures import get_registry
from model_atlas.schemas.ontology import CapabilityLabel, DataPartition, TrajectoryStage

_DOMAIN_HINTS: dict[str, CapabilityLabel] = {
    "coding": CapabilityLabel.CODE_GENERATION,
    "frontend": CapabilityLabel.FRONTEND_FROM_SPEC,
    "voxel": CapabilityLabel.VOXEL_SPATIAL,
    "reasoning": CapabilityLabel.GENERAL_REASONING,
    "mathematics": CapabilityLabel.MATHEMATICAL_REASONING,
    "long_context": CapabilityLabel.LONG_CONTEXT_RETRIEVAL,
    "agentic": CapabilityLabel.PLANNING,
    "tool_calling": CapabilityLabel.TOOL_SELECTION,
    "general": CapabilityLabel.GENERAL_REASONING,
    "hardware": CapabilityLabel.GENERAL_REASONING,
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def label_for_path(path: Path) -> CapabilityLabel:
    """Best-effort capability label from an eval-lab task's relative path."""
    for part in path.parts:
        key = part.lower().replace("-", "_")
        if key in _DOMAIN_HINTS:
            return _DOMAIN_HINTS[key]
    return CapabilityLabel.GENERAL_REASONING


def tokens_from_text(text: str, vocab: int, seed: int = 0, max_len: int = 256) -> list[int]:
    """Deterministic token ids from words (no external tokenizer needed)."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return []
    out: list[int] = []
    for i in range(min(len(words), max_len)):
        w = words[i]
        h = int.from_bytes(hashlib.sha256(f"{seed}:{w}".encode()).digest()[:8], "big")
        out.append(h % max(1, vocab))
    return out


def prompt_corpus(
    eval_lab_root: str,
    *,
    vocab: int,
    seed: int = 0,
    partition: DataPartition = DataPartition.ATLAS_CALIBRATION,
    max_samples: int | None = None,
    skip: set[str] | None = None,
) -> list[CalibrationSample]:
    """Ingest eval-lab's task prompts into an Atlas calibration corpus."""
    root = Path(eval_lab_root)
    prompts = [
        p for p in root.rglob("prompt.md") if not p.is_dir() and not p.name.startswith("exists_")
    ]
    if skip:
        prompts = [p for p in prompts if not any(s in p.as_posix() for s in skip)]
    if max_samples is not None:
        prompts = prompts[:max_samples]
    stages = list(TrajectoryStage)
    samples: list[CalibrationSample] = []
    for i, p in enumerate(prompts):
        text = p.read_text(encoding="utf-8", errors="ignore")
        tokens = tokens_from_text(text, vocab, seed=seed)
        if not tokens:
            continue
        rel = p.relative_to(root)
        samples.append(
            CalibrationSample(
                tokens=tokens,
                labels=[label_for_path(rel)],
                stage=stages[i % len(stages)],
            )
        )
    return samples


def pipeline_summary(
    eval_lab_root: str,
    *,
    seed: int = 0,
    arch_name: str = "k3-mini",
    vocab: int | None = None,
    partition: DataPartition = DataPartition.ATLAS_CALIBRATION,
) -> dict[str, Any]:
    """Run Atlas REAP saliency over eval-lab's real tasks and summarize per label."""
    model: MiniMoE = build_mini_moe_for(arch_name, seed)
    vocab = vocab or model.arch.vocabulary_size or 1000
    corpus = prompt_corpus(eval_lab_root, vocab=vocab, seed=seed, partition=partition)
    if not corpus:
        raise ValueError(f"no eval-lab task prompts under {eval_lab_root}")
    saliency = run_calibration(model, corpus, top_k=2)
    labels_seen = sorted({lab.value for s in corpus for lab in s.labels if lab})
    per_label: dict[str, list[dict[str, Any]]] = {}
    for lab in labels_seen:
        from model_atlas.schemas.ontology import CapabilityLabel

        try:
            enum_lab = CapabilityLabel(lab)
        except ValueError:
            continue
        per_label[lab] = [
            {"layer": lay, "expert": e, "score": round(s, 5)}
            for lay, e, s in saliency.rank(enum_lab, topk=5)
        ]
    return {
        "source": "eval-lab task corpus",
        "eval_lab_root": str(Path(eval_lab_root).resolve()),
        "n_tasks": len(corpus),
        "arch": arch_name,
        "seed": seed,
        "partition": partition.value,
        "saliency_per_label": per_label,
    }


def build_mini_moe_for(arch_name: str, seed: int) -> MiniMoE:
    arch = get_registry().get(arch_name)
    return build_mini_moe_impl(arch, seed)


def build_mini_moe_impl(arch: Any, seed: int) -> MiniMoE:
    from model_atlas.atlas.runtime import build_mini_moe

    return build_mini_moe(arch, seed=seed)


def write_manifest(payload: dict[str, Any], out_path: str) -> str:
    """Write a JSON manifest (atlas findings) for eval-lab to consume."""
    Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path
