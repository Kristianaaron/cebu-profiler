"""Semantic expert/channel map (blueprint §8.1, §10).

Associates experts with capability labels from measured REAP saliency, as a
protection + explanation signal (never "Expert 417 = Python" claims — just
"this expert contributes disproportionately to label X".) The resulting
associations can protect important trajectories and enrich the compression
scores (``ChannelScore.semantic``).
"""

from __future__ import annotations

from dataclasses import dataclass

from cebu_profiler.profiler.reap import CalibrationSample, run_calibration
from cebu_profiler.profiler.runtime import MiniMoE
from cebu_profiler.schemas.ontology import CapabilityLabel
from cebu_profiler.scoring.base import (
    ChannelScore,
    ProfilerScorer,
    ScoreNeed,
    ScorerRequirements,
    ScoreTable,
)


@dataclass
class SemanticAssociation:
    label: str
    layer: int
    expert: int
    strength: float


class SemanticMap:
    """Measured label -> expert associations, from REAP saliency."""

    def __init__(self, associations: list[SemanticAssociation]) -> None:
        self.associations = associations

    def experts_for(self, label: str) -> list[tuple[int, int]]:
        return [(a.layer, a.expert) for a in self.associations if a.label == label]

    def experts_all(self) -> set[tuple[int, int]]:
        return {(a.layer, a.expert) for a in self.associations}

    def strongest(self, threshold: float) -> set[tuple[int, int]]:
        """(layer, expert) pairs whose best label association exceeds `threshold`."""
        best: dict[tuple[int, int], float] = {}
        for a in self.associations:
            if a.strength > best.get((a.layer, a.expert), 0.0):
                best[(a.layer, a.expert)] = a.strength
        return {k for k, v in best.items() if v >= threshold}


def semantic_map(
    model: MiniMoE,
    samples: list[CalibrationSample],
    top_k: int | None = None,
    topk_per_label: int = 8,
) -> SemanticMap:
    """Top-labelled experts per capability, from measured calibration saliency."""
    acc = run_calibration(model, samples, top_k=top_k)
    associations: list[SemanticAssociation] = []
    for label in CapabilityLabel:
        for layer, expert, score in acc.rank(label, topk=topk_per_label):
            associations.append(SemanticAssociation(label.value, layer, expert, score))
    return SemanticMap(associations)


def expert_semantic_score(
    model: MiniMoE,
    samples: list[CalibrationSample],
    top_k: int | None = None,
    topk_per_label: int = 8,
) -> dict[tuple[int, int], float]:
    """Per-expert semantic strength (its strongest measured label association)."""
    m = semantic_map(model, samples, top_k=top_k, topk_per_label=topk_per_label)
    best: dict[tuple[int, int], float] = {}
    for a in m.associations:
        if a.strength > best.get((a.layer, a.expert), 0.0):
            best[(a.layer, a.expert)] = a.strength
    return best


class SemanticScorer(ProfilerScorer):
    name = "semantic"
    version = "1.0"

    def __init__(
        self, model: MiniMoE, samples: list[CalibrationSample], top_k: int | None = None
    ) -> None:
        self._model = model
        self._per_expert = expert_semantic_score(model, samples, top_k=top_k)

    def requirements(self) -> ScorerRequirements:
        return ScorerRequirements(
            frozenset({ScoreNeed.FORWARD_ACTIVATIONS, ScoreNeed.RAW_EXPERT_TENSORS})
        )

    def finalize(self) -> ScoreTable:
        rows: list[ChannelScore] = []
        for layer, layer_w in enumerate(self._model.layers):
            for e in range(self._model.n_exp):
                sem = self._per_expert.get((layer, e), 0.0)
                for c in range(len(layer_w.experts[e]["gate"])):
                    rows.append(
                        ChannelScore(
                            layer=layer,
                            expert=e,
                            channel=c,
                            semantic=sem,
                        )
                    )
        return ScoreTable(
            model=self._model.arch.name,
            scorer_versions={self.name: self.version},
            rows=rows,
        )
