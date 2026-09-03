"""Sparse features and vocabulary projections (v2 §15–§16).

Vocabulary projection (§16): project a direction (expert output, residual/MoE
contribution) through the unembedding/LM head and report the tokens it
promotes/suppresses — a real interpretation probe, not a label.

Sparse features (§15): learn a small dictionary over real per-token hidden
states (k-means atoms) and produce k-sparse codes by matching pursuit, then
link each feature to the labels of the tokens that activate it, the experts
that contributed on those tokens, and its redundancy (polysemanticity proxy).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from cebu_profiler.profiler.reap import CalibrationSample
from cebu_profiler.profiler.runtime import MiniMoE, forward
from cebu_profiler.schemas.ontology import CapabilityLabel


def _dot(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def _norm(x: list[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def _normalize(x: list[float]) -> list[float]:
    n = _norm(x)
    return [v / n for v in x] if n else x


# --------------------------------------------------------------------------- #
# Vocabulary projection (v2 §16)
# --------------------------------------------------------------------------- #


def project_vocabulary(model: MiniMoE, direction: list[float]) -> list[tuple[int, float]]:
    """Score each vocab token by `lm_head[token] · direction`, desc."""
    scores = [_dot(row, direction) for row in model.lm_head]
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


def promoted_suppressed(
    model: MiniMoE, direction: list[float], topk: int = 10
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """(promoted, suppressed) token projections for a direction."""
    ranked = project_vocabulary(model, direction)
    return ranked[:topk], list(reversed(ranked))[:topk]


@dataclass
class VocabProjection:
    promoted: list[int]
    suppressed: list[int]
    scores: dict[int, float]
    confidence: float  # normalized separation between promoted and suppressed means


def directional_projection(
    model: MiniMoE, direction: list[float], topk: int = 10
) -> VocabProjection:
    """Structured projection with promoted/suppressed ids, scores, and confidence."""
    ranked = project_vocabulary(model, direction)
    promoted, suppressed = ranked[:topk], list(reversed(ranked))[:topk]
    scores = {t: s for t, s in ranked}
    mean_p = sum(s for _, s in promoted) / max(1, len(promoted))
    mean_s = sum(s for _, s in suppressed) / max(1, len(suppressed))
    spread = max(abs(s) for _, s in ranked) or 1.0
    confidence = max(0.0, min(1.0, (mean_p - mean_s) / spread))
    return VocabProjection(
        promoted=[t for t, _ in promoted],
        suppressed=[t for t, _ in suppressed],
        scores=scores,
        confidence=confidence,
    )


def expert_direction(
    model: MiniMoE, tokens: list[int], layer: int, expert: int, token_index: int = 0
) -> list[float]:
    """MoE-output direction of one expert, isolated by forcing it as the whole route.

    Forcing [expert]*top_k yields the combined vector == that expert's output
    (probs are uniform 1/k over repeats, summing to that expert's output).
    """
    k = model.arch.moe.top_k
    result = forward(model, tokens, route_override={(layer, token_index): [expert] * k})
    return list(result.traces[layer].combined[token_index])


def residual_direction(model: MiniMoE, tokens: list[int], token_index: int = 0) -> list[float]:
    """MoE-contribution direction under the original (real) route at the last layer."""
    result = forward(model, tokens)
    layer = len(result.traces) - 1
    return list(result.traces[layer].combined[token_index])


# --------------------------------------------------------------------------- #
# Sparse features (v2 §15)
# --------------------------------------------------------------------------- #


def _kmeans(X: list[list[float]], k: int, seed: int, iters: int = 10) -> list[list[float]]:
    rng = random.Random(seed)
    dim = len(X[0])
    centroids = [list(X[i]) for i in rng.sample(range(len(X)), k)]
    for _ in range(iters):
        clusters: list[list[list[float]]] = [[] for _ in range(k)]
        for x in X:
            bi = min(range(k), key=lambda i: sum((centroids[i][d] - x[d]) ** 2 for d in range(dim)))
            clusters[bi].append(x)
        for i, cl in enumerate(clusters):
            if cl:
                centroids[i] = [sum(c[d] for c in cl) / len(cl) for d in range(dim)]
    return [_normalize(c) for c in centroids]


def _sparse_codes_topk(x: list[float], atoms: list[list[float]], k: int) -> list[int]:
    """k-sparse activation: indices of the best-matching atoms (matching pursuit)."""
    dots = [_dot(a, x) for a in atoms]
    order = sorted(range(len(atoms)), key=lambda i: dots[i], reverse=True)
    return sorted(order[:k])


@dataclass
class Feature:
    id: int
    activation_frequency: float
    top_labels: list[str]
    top_experts: list[tuple[int, int]]
    max_redundancy: float
    atom: list[float] = field(repr=False)


@dataclass
class FeatureDictionary:
    n_features: int
    hidden: int
    k_sparse: int
    features: list[Feature]

    def top_labels(self, feature_id: int) -> list[str]:
        return self.features[feature_id].top_labels


def _states_and_experts(
    model: MiniMoE, samples: list[CalibrationSample]
) -> tuple[list[list[float]], list[list[CapabilityLabel]], list[list[tuple[int, int]]]]:
    states: list[list[float]] = []
    labs: list[list[CapabilityLabel]] = []
    experts: list[list[tuple[int, int]]] = []
    for s in samples:
        r = forward(model, s.tokens)
        last = r.traces[-1]
        for t_idx, h in enumerate(r.final_hidden_states):
            states.append(h)
            labs.append(s.labels)
            contrib = sorted(
                enumerate(last.router_weighted[t_idx]),
                key=lambda x: -x[1],
            )[:3]
            experts.append([(last.layer, e) for e, _ in contrib])
    return states, labs, experts


def learn_features(
    model: MiniMoE,
    samples: list[CalibrationSample],
    *,
    n_features: int = 16,
    k_sparse: int = 2,
    seed: int = 0,
) -> FeatureDictionary:
    """Learn a k-means dictionary + k-sparse codes over real per-token hidden states."""
    states, labs, experts = _states_and_experts(model, samples)
    if not states:
        return FeatureDictionary(n_features, model.hidden, k_sparse, [])

    atoms = _kmeans(states, n_features, seed)
    codes = [_sparse_codes_topk(x, atoms, k_sparse) for x in states]
    N = len(states)

    freq = [0] * n_features
    feat_labels: list[defaultdict[str, int]] = [defaultdict(int) for _ in range(n_features)]
    feat_experts: list[defaultdict[tuple[int, int], int]] = [
        defaultdict(int) for _ in range(n_features)
    ]
    for c, labs_i, exps_i in zip(codes, labs, experts, strict=True):
        for a in c:
            freq[a] += 1
            for lab in labs_i:
                feat_labels[a][lab.value] += 1
            for ex in exps_i:
                feat_experts[a][ex] += 1

    redundancy: list[float] = []
    for i in range(n_features):
        m = 0.0
        for j in range(n_features):
            if i != j:
                m = max(m, _dot(atoms[i], atoms[j]))
        redundancy.append(m)

    features: list[Feature] = []
    for i in range(n_features):
        top_labels = sorted(feat_labels[i], key=lambda v: feat_labels[i][v], reverse=True)[:4]
        top_experts = sorted(feat_experts[i], key=lambda v: feat_experts[i][v], reverse=True)[:4]
        features.append(
            Feature(
                id=i,
                activation_frequency=freq[i] / N,
                top_labels=top_labels,
                top_experts=top_experts,
                max_redundancy=redundancy[i],
                atom=atoms[i],
            )
        )
    return FeatureDictionary(
        n_features=n_features, hidden=model.hidden, k_sparse=k_sparse, features=features
    )
