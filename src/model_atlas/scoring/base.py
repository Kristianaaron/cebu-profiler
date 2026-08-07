"""First-class scorer contracts (blueprint §9).

Defines `ScoreNeed` / `ScorerRequirements` (what each scorer needs to run) and
the versioned `ScoreTable` score-table format that every scorer emits. Scorers
are pure functions/aggregators over measured telemetry — they never touch model
weights (`Atlas` observes, scores, plans; it does not mutate).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ScoreNeed(StrEnum):
    """A dependency a scorer declares to run."""

    FORWARD_ACTIVATIONS = "forward_activations"
    GRADIENTS = "gradients"
    CAUSAL_RERUNS = "causal_reruns"
    HIGH_PRECISION_WEIGHTS = "high_precision_weights"
    RAW_EXPERT_TENSORS = "raw_expert_tensors"
    ROUTER_LOGITS = "router_logits"


@dataclass(frozen=True)
class ScorerRequirements:
    """The inputs a scorer needs; drives what can run now vs later.

    A scorer requiring only `FORWARD_ACTIVATIONS` + `RAW_EXPERT_TENSORS` can run
    immediately on the NVFP4 checkpoint; one requiring `GRADIENTS` /
    `HIGH_PRECISION_WEIGHTS` is marked authoritative-pending (blueprint §9.1).
    """

    needs: frozenset[ScoreNeed]
    note: str = ""

    @property
    def forward_only(self) -> bool:
        """True when only forward activations / tensor reads are needed."""
        return not (self.needs & {ScoreNeed.GRADIENTS, ScoreNeed.HIGH_PRECISION_WEIGHTS})


class ChannelScore(BaseModel):
    """One (layer, expert, channel) scoring row across all views."""

    model_config = ConfigDict(extra="forbid")

    layer: int
    expert: int
    channel: int
    tenp: float | None = Field(default=None, ge=0.0)
    taylor: float | None = Field(default=None, ge=0.0)
    causal: float | None = Field(default=None, ge=0.0)
    stability: float | None = Field(default=None, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rank_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: dict[str, int] = Field(default_factory=dict)


class ScoreTable(BaseModel):
    """Versioned, deterministic score-table (blueprint §9 / §10)."""

    model_config = ConfigDict(extra="forbid")

    score_table_version: int = 1
    model: str
    scorer_versions: dict[str, str] = Field(default_factory=dict)
    rows: list[ChannelScore] = Field(default_factory=list)


class AtlasScorer(ABC):
    """Base scorer interface (blueprint §9.1)."""

    name: str
    version: str

    @abstractmethod
    def requirements(self) -> ScorerRequirements:
        ...

    @abstractmethod
    def finalize(self) -> ScoreTable:
        ...
