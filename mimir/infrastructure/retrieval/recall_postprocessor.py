"""Post-processing layer for Mimir hybrid recall.

This module implements a three-stage pipeline on top of the existing retrieval
scorers:

1. Candidate generation (kept in ``InMemoryAgentAdapter.recall``): vector +
   BM25 + RRF fusion.
2. Semantic clustering deduplication: near-duplicate memories are grouped by
   embedding cosine similarity and only one representative per cluster is kept.
3. Multiplicative reranking: the fused retrieval score is amplified by a
   normalized lifecycle signal (recency, importance, access patterns).

The goal is to avoid returning multiple nearly-identical memories and to keep
ordering stable even when lifecycle metadata varies widely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mimir.domain.model import Memory


@dataclass
class PostprocessorConfig:
    """Configuration for the recall post-processing pipeline."""

    dedup_threshold: float = 0.9
    """Cosine similarity threshold for semantic clustering. A value of 1.0
    effectively disables clustering."""

    ranking_mode: str = "multiplicative"
    """Either ``multiplicative`` or ``additive``. Multiplicative keeps the
    retrieval score as a base and amplifies it with lifecycle metadata."""

    lifecycle_weight: float = 0.3
    """Weight of the lifecycle signal in the final blend. For the multiplicative
    mode the score becomes ``retrieval * (1 + lifecycle_weight * norm_lifecycle)``.
    """

    recency_half_life_days: float = 30.0
    """Half-life (in days) used for any internal recency computations. The
    caller is expected to pass lifecycle scores already containing recency; this
    is reserved for future extensions."""

    max_candidates_for_clustering: int = 50
    """Only cluster the top-N retrieval candidates to keep pairwise similarity
    computation bounded. Larger buffers are truncated before clustering."""


class RecallPostprocessor:
    """Post-process retrieval candidates: cluster and rerank."""

    def __init__(self, config: PostprocessorConfig | None = None) -> None:
        """Initialize the postprocessor.

        Args:
            config: Configuration. Defaults to a reasonable multiplicative
                pipeline with semantic deduplication at 0.9.
        """
        self.config = config or PostprocessorConfig()
        if self.config.ranking_mode not in {"multiplicative", "additive"}:
            raise ValueError(
                f"Unknown ranking_mode: {self.config.ranking_mode!r}. "
                "Use 'multiplicative' or 'additive'."
            )
        if not 0.0 <= self.config.dedup_threshold <= 1.0:
            raise ValueError("dedup_threshold must be in [0.0, 1.0]")
        if self.config.max_candidates_for_clustering < 1:
            raise ValueError("max_candidates_for_clustering must be positive")
        if not 0.0 <= self.config.lifecycle_weight <= 1.0:
            raise ValueError("lifecycle_weight must be in [0.0, 1.0]")

    def normalize_scores(
        self, scores: dict[int, float], fallback: float = 1.0
    ) -> dict[int, float]:
        """Min-max normalize scores to [0, 1].

        When all scores are identical the span is zero; in that case the
        fallback value is used so that equal candidates are treated consistently
        (e.g., all receive the full lifecycle boost rather than none). The
        fallback is suppressed when all scores are zero, so an absent lifecycle
        signal does not artificially amplify every candidate.
        """
        if not scores:
            return {}
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        span = max_val - min_val
        if span == 0:
            effective_fallback = fallback if max_val > 0 else 0.0
            return dict.fromkeys(scores, effective_fallback)
        return {idx: (score - min_val) / span for idx, score in scores.items()}

    def similarity_matrix(self, memories: list[Memory]) -> torch.Tensor:
        """Return the pairwise cosine similarity matrix for the given memories.

        Returns an ``(n, n)`` tensor where the diagonal is 1.0. An empty input
        returns a ``(0, 0)`` tensor.
        """
        if not memories:
            return torch.empty(0, 0)

        embeddings = torch.tensor(
            [memory.embedding for memory in memories], dtype=torch.float32
        )
        norms = torch.linalg.norm(embeddings, dim=1, keepdim=True)
        # Avoid division by zero; zero vectors will have zero similarity.
        normalized = embeddings / torch.where(norms == 0, torch.ones_like(norms), norms)
        return torch.matmul(normalized, normalized.T)

    def _cluster(
        self,
        memories: list[Memory],
        retrieval_scores: dict[int, float],
        lifecycle_scores: dict[int, float] | None,
    ) -> list[Memory]:
        """Group near-duplicate memories and return one representative per cluster.

        The representative is the memory with the highest combined score within
        its cluster. The combined score is computed with the configured ranking
        mode so that the representative content and its ordering score are
        aligned.

        Args:
            memories: Candidate memories in arbitrary order.
            retrieval_scores: Mapping from candidate index to fused retrieval score.
            lifecycle_scores: Optional mapping from candidate index to lifecycle
                score.

        Returns:
            A list of representative memories, one per cluster, with their score
            set to the representative's combined score.
        """
        if not memories or self.config.dedup_threshold >= 1.0:
            return memories

        normalized_lifecycle = self.normalize_scores(lifecycle_scores or {})
        combined = self._combined_scores(retrieval_scores, normalized_lifecycle)

        # Sort candidates by combined score so the highest-scoring member of a
        # cluster becomes the representative naturally.
        order = sorted(
            range(len(memories)),
            key=lambda idx: combined.get(idx, 0.0),
            reverse=True,
        )

        sim = self.similarity_matrix(memories)
        assigned = [False] * len(memories)
        clusters: list[int] = []

        for idx in order:
            if assigned[idx]:
                continue
            # Start a new cluster with ``idx`` as representative.
            clusters.append(idx)
            assigned[idx] = True
            for j in order:
                if assigned[j]:
                    continue
                if sim[idx, j].item() >= self.config.dedup_threshold:
                    assigned[j] = True

        representatives: list[Memory] = []
        for rep_idx in clusters:
            memory = memories[rep_idx]
            memory.score = combined.get(rep_idx, memory.score)
            representatives.append(memory)
        return representatives

    def _combined_scores(
        self,
        retrieval_scores: dict[int, float],
        normalized_lifecycle: dict[int, float],
    ) -> dict[int, float]:
        """Compute final scores for each candidate index."""
        combined: dict[int, float] = {}
        for idx, retrieval in retrieval_scores.items():
            lifecycle = normalized_lifecycle.get(idx, 0.0)
            if self.config.ranking_mode == "multiplicative":
                # Retrieval score is the base; lifecycle can amplify it by up to
                # (1 + lifecycle_weight). When lifecycle is 0 the score is
                # unchanged, so low-lifecycle candidates are not zeroed out.
                combined[idx] = retrieval * (
                    1.0 + self.config.lifecycle_weight * lifecycle
                )
            else:
                # Additive blend kept for backward compatibility.
                combined[idx] = (
                    1.0 - self.config.lifecycle_weight
                ) * retrieval + self.config.lifecycle_weight * lifecycle
        return combined

    def process(
        self,
        memories: list[Memory],
        retrieval_scores: dict[int, float],
        lifecycle_scores: dict[int, float] | None = None,
    ) -> list[Memory]:
        """Run the post-processing pipeline on retrieval candidates.

        Steps:
          1. Truncate to ``max_candidates_for_clustering`` by retrieval score.
          2. Cluster near-duplicate memories and keep one representative per
             cluster.
          3. Re-rank representatives by the configured ranking mode.

        Args:
            memories: Candidate memories returned by the retrieval scorers.
                Expected to already satisfy any ``min_score`` filter.
            retrieval_scores: Mapping from candidate index to fused retrieval score.
            lifecycle_scores: Optional mapping from candidate index to lifecycle
                composite score. If omitted, lifecycle amplification is skipped.

        Returns:
            Memories sorted by final score descending. Each returned memory is a
            representative (possibly the original candidate if no clustering
            occurred) with ``memory.score`` set to the final score.
        """
        if not memories:
            return []

        # Work on indices that have retrieval scores; drop others because they
        # cannot be ranked.
        candidate_indices = [idx for idx in range(len(memories)) if idx in retrieval_scores]
        if not candidate_indices:
            return []

        # Truncate to the top retrieval candidates before clustering.
        if len(candidate_indices) > self.config.max_candidates_for_clustering:
            candidate_indices = sorted(
                candidate_indices,
                key=lambda idx: retrieval_scores[idx],
                reverse=True,
            )[: self.config.max_candidates_for_clustering]

        candidates = [memories[idx] for idx in candidate_indices]
        candidate_retrieval = {i: retrieval_scores[candidate_indices[i]] for i in range(len(candidate_indices))}
        candidate_lifecycle: dict[int, float] | None = None
        if lifecycle_scores:
            candidate_lifecycle = {
                i: lifecycle_scores[candidate_indices[i]]
                for i in range(len(candidate_indices))
                if candidate_indices[i] in lifecycle_scores
            }

        clustered = self._cluster(candidates, candidate_retrieval, candidate_lifecycle)
        normalized_lifecycle = self.normalize_scores(candidate_lifecycle or {})
        final_scores = self._combined_scores(candidate_retrieval, normalized_lifecycle)

        # Map final scores back to the clustered representatives using object
        # identity, because dataclass value equality can map a representative to
        # the wrong candidate index when multiple memories are value-equal.
        candidate_index_by_id = {id(candidate): i for i, candidate in enumerate(candidates)}
        scored: list[tuple[Memory, float]] = []
        for memory in clustered:
            idx = candidate_index_by_id[id(memory)]
            score = final_scores[idx]
            memory.score = score
            scored.append((memory, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [memory for memory, _ in scored]
