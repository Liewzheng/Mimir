"""Agent-facing memory interface for Mimir.

This module exposes a high-level abstraction that agent CLIs (opencode,
kimi code, claude code, codex, etc.) can use to add plastic memory to
their sessions without depending on Mimir internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

from mimir.application.factories import create_mimir
from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.domain.model.engine import EmbeddingEngine


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


@dataclass
class Message:
    """A single message in an agent conversation."""

    role: str  # e.g. "user", "assistant", "system", "tool"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Memory:
    """A retrieved memory with relevance score and optional source messages."""

    text: str
    embedding: list[float]
    score: float
    created_at: datetime
    source: Message | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    ) -> list[Memory]:
        """Return the top-k memories most similar to the query."""
        self._validate_text(query, "Recall query")
        if not self._memories:
            return []

        query_emb = self._mimir.encode([query])[0]
        memory_embs = torch.tensor(
            [m.embedding for m in self._memories],
            dtype=query_emb.dtype,
            device=query_emb.device,
        )

        # Cosine similarity.
        query_norm = query_emb / torch.linalg.norm(query_emb)
        memory_norms = memory_embs / torch.linalg.norm(memory_embs, dim=1, keepdim=True)
        sims = torch.matmul(memory_norms, query_norm).tolist()

        scored = [
            Memory(
                text=m.text,
                embedding=m.embedding,
                score=score,
                created_at=m.created_at,
                source=m.source,
                metadata=m.metadata,
            )
            for m, score in zip(self._memories, sims, strict=True)
            if score >= min_score
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

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
            result.append(
                {
                    "text": memory.text,
                    "embedding": embedding,
                    "score": memory.score,
                    "created_at": _iso(memory.created_at),
                    "source": source,
                    "metadata": memory.metadata,
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
