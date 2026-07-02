"""Core memory domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
