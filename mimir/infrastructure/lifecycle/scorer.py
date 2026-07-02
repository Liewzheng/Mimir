"""Lifecycle scoring: recency, importance, access patterns, staleness."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from mimir.domain.model import Memory
from mimir.infrastructure.lifecycle.metadata import MemoryMetadata


def _default_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _content_hash(text: str) -> str:
    """Return a stable SHA-256 hash of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class LifecycleWeights:
    """Weights for the lifecycle composite score.

    All weights are non-negative and need not sum to 1.0; the final score is a
    weighted sum that callers can re-sort or threshold as needed.
    """

    recency: float = 1.0
    importance: float = 1.0
    access: float = 0.5
    staleness_penalty: float = 0.8


class LifecycleScorer:
    """Score memories by their lifecycle metadata.

    The composite score combines:
      - recency: exponential decay since creation
      - importance: user/system assigned weight
      - access: log-scaled access count
      - staleness penalty: multiplied by (1 - staleness_penalty) if stale

    This scorer is intentionally simple and dependency-free. The half-life and
    weights can be tuned per workspace or use-case.
    """

    def __init__(
        self,
        weights: LifecycleWeights | None = None,
        recency_half_life_days: float = 30.0,
        now: datetime | None = None,
    ) -> None:
        """Initialize the scorer.

        Args:
            weights: Composite score weights. Defaults to equal emphasis on
                recency and importance, with moderate access bonus.
            recency_half_life_days: Number of days for the recency score to
                decay by half. Must be positive.
            now: Optional reference time for scoring. Defaults to UTC now.
        """
        if recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        self.weights = weights or LifecycleWeights()
        self.half_life_days = recency_half_life_days
        self.now = now or _default_now()

    def _recency_score(self, created_at: datetime) -> float:
        """Exponential decay from 1.0 at creation to 0.0 as age → ∞."""
        age = self.now - created_at
        age_days = age.total_seconds() / 86_400.0
        return math.exp2(-age_days / self.half_life_days)

    def _access_score(self, access_count: int) -> float:
        """Log-scaled access score: 0 accesses → 0, grows sub-linearly."""
        if access_count <= 0:
            return 0.0
        return math.log1p(access_count)

    def _metadata(self, memory: Memory) -> MemoryMetadata:
        """Return lifecycle metadata from memory, normalizing it in place."""
        meta = memory.metadata.get("lifecycle")
        if isinstance(meta, MemoryMetadata):
            return meta
        if isinstance(meta, dict):
            # Preserve the memory's own creation time if the legacy dict lacks it.
            if not meta.get("created_at"):
                meta = {**meta, "created_at": memory.created_at.isoformat()}
            normalized = MemoryMetadata.from_dict(meta)
            memory.metadata["lifecycle"] = normalized
            return normalized
        normalized = MemoryMetadata(created_at=memory.created_at)
        memory.metadata["lifecycle"] = normalized
        return normalized

    def score(self, memories: list[Memory]) -> dict[int, float]:
        """Return a lifecycle score for each memory.

        Args:
            memories: List of memories to score.

        Returns:
            Mapping from memory index to composite lifecycle score.
        """
        scores: dict[int, float] = {}
        for idx, memory in enumerate(memories):
            meta = self._metadata(memory)
            recency = self._recency_score(meta.created_at)
            access = self._access_score(meta.access_count)
            composite = (
                self.weights.recency * recency
                + self.weights.importance * meta.importance
                + self.weights.access * access
            )
            if meta.stale:
                composite *= max(0.0, 1.0 - self.weights.staleness_penalty)
            scores[idx] = composite
        return scores

    def mark_stale(
        self,
        memories: list[Memory],
        max_age_days: float = 90.0,
        min_access_count: int = 1,
        min_importance: float = 1.0,
    ) -> list[int]:
        """Mark stale any memories that are old AND rarely accessed AND unimportant.

        A memory is marked stale if all of the following hold:
          - age > max_age_days
          - access_count < min_access_count
          - importance <= min_importance

        Args:
            memories: List of memories to evaluate.
            max_age_days: Age threshold in days.
            min_access_count: Access count ceiling.
            min_importance: Importance ceiling.

        Returns:
            List of indices that were newly marked stale.
        """
        newly_stale: list[int] = []
        for idx, memory in enumerate(memories):
            meta = self._metadata(memory)
            age = (self.now - meta.created_at).total_seconds() / 86_400.0
            if (
                age > max_age_days
                and meta.access_count < min_access_count
                and meta.importance <= min_importance
                and not meta.stale
            ):
                meta.stale = True
                memory.metadata["lifecycle"] = meta
                newly_stale.append(idx)
        return newly_stale


def ensure_lifecycle_metadata(memory: Memory) -> MemoryMetadata:
    """Ensure memory.metadata['lifecycle'] exists and has a content_hash.

    If the memory already has lifecycle metadata, its content_hash is updated
    to match memory.text and its created_at is preserved. If no metadata
    exists, a default one is created using ``memory.created_at`` as the
    creation time.

    Returns:
        The MemoryMetadata instance attached to the memory.
    """
    meta = memory.metadata.get("lifecycle")
    if isinstance(meta, dict):
        meta = MemoryMetadata.from_dict(meta)
    elif not isinstance(meta, MemoryMetadata):
        meta = MemoryMetadata(created_at=memory.created_at)
    meta.content_hash = _content_hash(memory.text)
    memory.metadata["lifecycle"] = meta
    return meta


def deduplicate_memories(memories: list[Memory]) -> list[Memory]:
    """Remove memories with duplicate content, keeping the most recent.

    When duplicates are found, the memory with the latest created_at is kept.
    If no lifecycle metadata exists, a content hash is computed on the fly.

    Args:
        memories: List of memories to deduplicate.

    Returns:
        A new list with duplicates removed.
    """
    seen: dict[str, Memory] = {}
    for memory in memories:
        meta = ensure_lifecycle_metadata(memory)
        if meta.content_hash is None:
            meta.content_hash = _content_hash(memory.text)
        existing = seen.get(meta.content_hash)
        if existing is None or memory.created_at > existing.created_at:
            seen[meta.content_hash] = memory
    return list(seen.values())
