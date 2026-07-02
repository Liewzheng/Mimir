"""Agent-facing memory interface for Mimir.

This module exposes a high-level abstraction that agent CLIs (opencode,
kimi code, claude code, codex, etc.) can use to add plastic memory to
their sessions without depending on Mimir internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

from mimir.application.factories import create_mimir
from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.domain.model import Memory, Message
from mimir.domain.model.engine import EmbeddingEngine
from mimir.infrastructure.lifecycle import (
    LifecycleScorer,
    ensure_lifecycle_metadata,
)
from mimir.infrastructure.lifecycle.metadata import MemoryMetadata
from mimir.infrastructure.retrieval import (
    BM25Scorer,
    RankFusion,
    VectorScorer,
)


def _iso(dt: datetime) -> str:
    """Return ISO 8601 representation with timezone."""
    return dt.isoformat()


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 datetime string."""
    return datetime.fromisoformat(value)


def _serialize_embedding(embedding: Any) -> list[float]:
    """Convert an embedding tensor/array to a JSON-serializable list."""
    if isinstance(embedding, torch.Tensor):
        return cast(list[float], embedding.tolist())
    if hasattr(embedding, "tolist"):
        return cast(list[float], embedding.tolist())
    return cast(list[float], embedding)


def _deserialize_embedding(embedding: Any) -> Any:
    """Normalize an embedding loaded from JSON.

    Embeddings serialized via :func:`_serialize_embedding` are plain lists,
    which is exactly what JSON deserialization produces.  This helper exists
    so callers can add future formats without scattering type checks.
    """
    return embedding


class AgentMemoryInterface(ABC):
    """Generic interface for agent-session memory backed by Mimir.

    Implementations may store memories in-memory, on disk, or in a remote
    service.  The interface intentionally avoids Mimir-specific types so
    that agent CLIs can swap backends without changing call sites.
    """

    @abstractmethod
    def observe(self, messages: list[Message]) -> None:
        """Observe a batch of messages and update memory."""

    @abstractmethod
    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[Memory]:
        """Retrieve memories most relevant to the query."""

    @abstractmethod
    def consolidate(self) -> None:
        """Explicitly reinforce recent or important memories."""

    @abstractmethod
    def checkpoint(self, path: str | Path) -> None:
        """Persist the memory state to disk."""

    @abstractmethod
    def restore(self, path: str | Path) -> None:
        """Restore the memory state from disk."""

    @abstractmethod
    def reset(self) -> None:
        """Clear session memory without deleting persisted state."""


class InMemoryAgentAdapter(AgentMemoryInterface):
    """Default in-memory adapter backed by an Mimir instance.

    Keeps a rolling buffer of observed messages and uses Mimir embeddings
    for both storage and similarity-based retrieval.
    """

    def __init__(
        self,
        config: MimirConfig | None = None,
        engine: EmbeddingEngine | None = None,
        mimir: Mimir | None = None,
        max_memories: int = 10_000,
        learn_on_observe: bool = False,
        checkpoint_dir: str | Path | None = None,
        max_text_length: int = 10_000,
    ) -> None:
        """Initialize the adapter.

        Args:
            config: Mimir configuration. Ignored if ``mimir`` is provided.
            engine: Embedding engine. Ignored if ``mimir`` is provided.
            mimir: Pre-built Mimir instance.
            max_memories: Maximum number of memories to retain in memory.
            learn_on_observe: Whether to call ``Mimir.learn`` for each
                observed message. Defaults to False so that learning is
                explicit via ``consolidate()`` and messages are not learned
                twice.
            checkpoint_dir: Base directory that restricts all checkpoint/restore
                paths. Defaults to ``~/.mimir/checkpoints``.
            max_text_length: Maximum characters allowed for any message content
                or recall query. Longer inputs are rejected.
        """
        if mimir is not None and config is not None:
            raise ValueError("Cannot provide both 'mimir' and 'config'. Pass only one of them.")
        if mimir is not None:
            self._mimir = mimir
        elif config is not None:
            self._mimir = create_mimir(config, engine=engine)
        else:
            raise ValueError("Must provide either config or mimir")

        self._max_memories = max_memories
        self._learn_on_observe = learn_on_observe
        self._checkpoint_dir = (
            Path(checkpoint_dir).expanduser().resolve()
            if checkpoint_dir is not None
            else Path.home() / ".mimir" / "checkpoints"
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._max_text_length = max_text_length
        self._memories: list[Memory] = []

    @property
    def memory_count(self) -> int:
        """Return the number of memories in the working buffer."""
        return len(self._memories)

    def _resolve_checkpoint_path(self, path: str | Path) -> Path:
        """Resolve a checkpoint path relative to the sandboxed checkpoint_dir.

        Absolute paths are rejected to prevent accidental writes outside the
        sandbox.  Pass a path relative to ``checkpoint_dir`` instead.
        """
        path = Path(path).expanduser()
        if path.is_absolute():
            raise ValueError(
                f"Checkpoint path must be relative to checkpoint_dir "
                f"'{self._checkpoint_dir}', got absolute path '{path}'"
            )
        resolved = (self._checkpoint_dir / path).resolve()
        try:
            resolved.relative_to(self._checkpoint_dir)
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint path '{path}' escapes checkpoint_dir '{self._checkpoint_dir}'"
            ) from exc
        return resolved

    def _validate_text(self, text: str, label: str) -> None:
        """Reject oversized input to prevent resource exhaustion."""
        if len(text) > self._max_text_length:
            raise ValueError(
                f"{label} exceeds max_text_length ({self._max_text_length}): {len(text)} characters"
            )

    def observe(self, messages: list[Message]) -> None:
        """Encode messages and add them to working memory.

        If ``learn_on_observe`` is enabled, each message text is also passed
        to ``Mimir.learn`` so the prototype matrix adapts to the session.

        Each memory is annotated with lifecycle metadata (importance, access
        count, content hash) to support deduplication and lifecycle scoring
        during recall.
        """
        if not messages:
            return

        texts = []
        for msg in messages:
            self._validate_text(msg.content, "Message content")
            texts.append(msg.content)

        embeddings = self._mimir.encode(texts)

        now = datetime.now(timezone.utc)
        for msg, emb in zip(messages, embeddings, strict=True):
            memory = Memory(
                text=msg.content,
                embedding=emb.tolist(),
                score=0.0,
                created_at=now,
                source=msg,
                metadata={"role": msg.role, **msg.metadata},
            )
            ensure_lifecycle_metadata(memory)
            self._memories.append(memory)

        if self._learn_on_observe:
            self._mimir.learn(texts)

        # Enforce capacity limit by dropping oldest memories.
        if len(self._memories) > self._max_memories:
            self._memories = self._memories[-self._max_memories :]

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        use_bm25: bool = True,
        use_lifecycle: bool = True,
        lifecycle_weight: float = 0.3,
    ) -> list[Memory]:
        """Return the top-k memories most relevant to the query.

        By default this performs a hybrid retrieval: vector cosine similarity
        is fused with BM25 keyword matching via Reciprocal Rank Fusion (RRF),
        and the result is reranked by lifecycle metadata (recency, importance,
        access count). Pass ``use_bm25=False`` or ``use_lifecycle=False`` to
        fall back to the previous pure-vector behavior.

        Args:
            query: The recall query text.
            top_k: Maximum number of memories to return.
            min_score: Minimum fused retrieval score. Applied before lifecycle
                reranking; memories below this threshold are dropped.
            use_bm25: Whether to include BM25 keyword scores in the fusion.
            use_lifecycle: Whether to apply lifecycle reranking.
            lifecycle_weight: Weight of the lifecycle score in the final blend.
                0.0 means no lifecycle reranking; 1.0 means lifecycle only.

        Returns:
            A list of Memory objects sorted by final relevance score.
        """
        self._validate_text(query, "Recall query")
        if not self._memories:
            return []

        query_emb = self._mimir.encode([query])[0]

        rankings: list[dict[int, float]] = []
        vector_scorer = VectorScorer(query_embedding=query_emb)
        vector_scores = vector_scorer.score(query, self._memories)
        rankings.append(vector_scores)

        if use_bm25:
            bm25_scorer = BM25Scorer()
            bm25_scores = bm25_scorer.score(query, self._memories)
            if bm25_scores:
                rankings.append(bm25_scores)

        fusion = RankFusion()
        fused_scores = fusion.fuse(rankings)

        # RRF scores are ranking-based and not calibrated to [0, 1]. Normalize
        # so that downstream thresholds (e.g. the hook recall cutoff) and the
        # lifecycle blend have a consistent, interpretable scale.
        if fused_scores:
            max_fused = max(fused_scores.values())
            min_fused = min(fused_scores.values())
            span = max_fused - min_fused
            if span > 0:
                fused_scores = {
                    idx: (score - min_fused) / span
                    for idx, score in fused_scores.items()
                }
            else:
                fused_scores = dict.fromkeys(fused_scores, 1.0)

        lifecycle_scores: dict[int, float] = {}
        if use_lifecycle:
            lifecycle_scorer = LifecycleScorer()
            lifecycle_scores = lifecycle_scorer.score(self._memories)

        # Build candidates, applying min_score to the fused retrieval score.
        scored: list[tuple[Memory, float]] = []
        for idx, memory in enumerate(self._memories):
            retrieval_score = fused_scores.get(idx, 0.0)
            if retrieval_score < min_score:
                continue

            final_score = retrieval_score
            if use_lifecycle and lifecycle_scores:
                lifecycle_score = lifecycle_scores.get(idx, 0.0)
                # Normalize lifecycle scores to [0, 1] using the max observed.
                max_lifecycle = max(lifecycle_scores.values())
                normalized_lifecycle = (
                    lifecycle_score / max_lifecycle if max_lifecycle > 0 else 0.0
                )
                final_score = (
                    1.0 - lifecycle_weight
                ) * retrieval_score + lifecycle_weight * normalized_lifecycle

            meta = memory.metadata.get("lifecycle")
            if isinstance(meta, dict):
                meta = MemoryMetadata.from_dict(meta)
                memory.metadata["lifecycle"] = meta
            if isinstance(meta, MemoryMetadata):
                meta.touch()

            scored.append(
                (
                    Memory(
                        text=memory.text,
                        embedding=memory.embedding,
                        score=final_score,
                        created_at=memory.created_at,
                        source=memory.source,
                        metadata=memory.metadata,
                    ),
                    final_score,
                )
            )

        scored.sort(key=lambda item: item[1], reverse=True)
        return [memory for memory, _ in scored[:top_k]]

    def consolidate(self) -> None:
        """Reinforce all memories currently held in the buffer.

        If ``learn_on_observe`` is enabled, memories have already been
        learned once during ``observe()``.  Calling ``consolidate()`` will
        reinforce them an additional time.
        """
        if not self._memories:
            return
        texts = [m.text for m in self._memories]
        self.learn(texts)

    def learn(self, texts: list[str], importance: float = 1.0) -> dict[str, Any]:
        """Run a learning step on the given texts."""
        return self._mimir.learn(texts, importance=importance)

    def checkpoint(self, path: str | Path) -> None:
        """Persist Mimir state and memory buffer."""
        resolved = self._resolve_checkpoint_path(path)
        self._mimir.save(resolved)

    def restore(self, path: str | Path) -> None:
        """Restore Mimir state. Working memory is intentionally cleared."""
        resolved = self._resolve_checkpoint_path(path)
        self._mimir.load(resolved)
        self._memories.clear()

    def reset(self) -> None:
        """Clear working memory and reset the underlying Mimir state."""
        self._memories.clear()
        self._mimir.reset()

    def clear_memories(self) -> None:
        """Clear the working memory buffer without changing Mimir state."""
        self._memories.clear()

    @property
    def prototype_capacity(self) -> int:
        """Return the configured number of Mimir prototypes."""
        return self._mimir.config.num_prototypes

    @property
    def capacity_usage(self) -> float:
        """Return the current Mimir prototype capacity usage ratio."""
        return self._mimir.store.capacity_usage()

    @property
    def step(self) -> int:
        """Return the current Mimir learning step."""
        return self._mimir.step

    def memories_state(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of the working memory buffer."""
        result = []
        for memory in self._memories:
            source: dict[str, Any] | None = None
            if memory.source is not None:
                source = {
                    "role": memory.source.role,
                    "content": memory.source.content,
                    "metadata": memory.source.metadata,
                    "timestamp": _iso(memory.source.timestamp),
                }
            embedding = _serialize_embedding(memory.embedding)
            metadata = dict(memory.metadata)
            lifecycle = metadata.get("lifecycle")
            if isinstance(lifecycle, MemoryMetadata):
                metadata["lifecycle"] = lifecycle.to_dict()
            result.append(
                {
                    "text": memory.text,
                    "embedding": embedding,
                    "score": memory.score,
                    "created_at": _iso(memory.created_at),
                    "source": source,
                    "metadata": metadata,
                }
            )
        return result

    def load_memories_state(self, state: list[dict[str, Any]]) -> None:
        """Restore the working memory buffer from a JSON-serializable snapshot."""
        self._memories.clear()
        for item in state:
            source = item.get("source")
            message = None
            if source is not None:
                message = Message(
                    role=source["role"],
                    content=source["content"],
                    metadata=source.get("metadata", {}),
                    timestamp=_parse_iso(source["timestamp"]),
                )
            embedding = _deserialize_embedding(item["embedding"])
            self._memories.append(
                Memory(
                    text=item["text"],
                    embedding=embedding,
                    score=item["score"],
                    created_at=_parse_iso(item["created_at"]),
                    source=message,
                    metadata=item.get("metadata", {}),
                )
            )
