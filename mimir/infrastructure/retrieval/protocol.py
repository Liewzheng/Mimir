"""Protocols for pluggable memory retrieval strategies."""

from __future__ import annotations

from typing import Protocol

from mimir.domain.model import Memory


class MemoryScorer(Protocol):
    """Score memories against a query.

    Implementations must be stateless with respect to the scoring session: they
    may pre-compute indices from the memory list, but must not mutate the input
    memories.
    """

    def score(self, query: str, memories: list[Memory]) -> dict[int, float]:
        """Return a mapping from memory index to relevance score.

        Higher scores mean more relevant. Memories that cannot be scored should
        be omitted from the result.
        """
        ...


class FusionStrategy(Protocol):
    """Combine multiple per-memory rankings into a single ranking."""

    def fuse(self, rankings: list[dict[int, float]]) -> dict[int, float]:
        """Return a fused mapping from memory index to score.

        The caller is responsible for normalizing or sorting the returned scores.
        """
        ...


class RankFusion:
    """Reciprocal Rank Fusion (RRF).

    RRF combines rankings without requiring score calibration. It is robust to
    different score scales and naturally down-ranks memories that only appear in
    a single scorer.

    Args:
        k: Ranking constant. Larger values reduce the penalty for lower ranks.
            The canonical RRF paper uses k=60.
    """

    def __init__(self, k: float = 60.0) -> None:
        self.k = k

    def fuse(self, rankings: list[dict[int, float]]) -> dict[int, float]:
        """Fuse rankings into a single score per memory index."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            # Sort by score descending and assign 1-based ranks.
            sorted_items = sorted(
                ranking.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for rank, (idx, _score) in enumerate(sorted_items, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.k + rank)
        return fused


class WeightedFusion:
    """Weighted linear fusion of normalized scores.

    Each ranking is min-max normalized to [0, 1] before weights are applied.
    Single-entry rankings receive a score of 1.0 for that entry and 0.0 for all
    others (if other rankings cover them).
    """

    def __init__(self, weights: list[float]) -> None:
        if len(weights) < 1:
            raise ValueError("At least one weight is required")
        self.weights = weights

    def _normalize(self, ranking: dict[int, float]) -> dict[int, float]:
        if not ranking:
            return {}
        values = list(ranking.values())
        min_val = min(values)
        max_val = max(values)
        span = max_val - min_val
        if span == 0:
            # All scores are identical: assign 1.0 to every item.
            return dict.fromkeys(ranking, 1.0)
        return {idx: (score - min_val) / span for idx, score in ranking.items()}

    def fuse(self, rankings: list[dict[int, float]]) -> dict[int, float]:
        """Fuse normalized rankings with per-ranking weights."""
        if len(rankings) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} rankings, got {len(rankings)}"
            )

        fused: dict[int, float] = {}
        for ranking, weight in zip(rankings, self.weights, strict=True):
            normalized = self._normalize(ranking)
            for idx, score in normalized.items():
                fused[idx] = fused.get(idx, 0.0) + score * weight
        return fused
