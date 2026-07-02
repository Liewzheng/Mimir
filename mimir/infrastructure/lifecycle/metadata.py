"""Lifecycle metadata and scoring for Mimir memories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryMetadata:
    """Mutable lifecycle metadata attached to a memory.

    These fields are stored alongside the memory text and embedding and drive
    lifecycle scoring (recency, importance, access patterns) and deduplication.

    All datetime fields are UTC. Callers should normalize timezone-aware
    datetimes before construction.
    """

    importance: float = 1.0
    """User- or system-assigned importance. Higher values resist decay."""

    access_count: int = 0
    """Number of times this memory has been recalled or observed."""

    last_accessed_at: datetime | None = None
    """Last time the memory was recalled. None if never accessed."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Creation timestamp."""

    content_hash: str | None = None
    """Stable hash of memory.text, used for deduplication."""

    stale: bool = False
    """True if the memory has been marked stale due to age or low utility."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
            "created_at": self.created_at.isoformat(),
            "content_hash": self.content_hash,
            "stale": self.stale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryMetadata:
        """Deserialize from a dictionary."""
        last_accessed = data.get("last_accessed_at")
        created_at_raw = data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw)
            if created_at_raw
            else datetime.now(timezone.utc)
        )
        return cls(
            importance=float(data.get("importance", 1.0)),
            access_count=int(data.get("access_count", 0)),
            last_accessed_at=(
                datetime.fromisoformat(last_accessed) if last_accessed else None
            ),
            created_at=created_at,
            content_hash=data.get("content_hash"),
            stale=bool(data.get("stale", False)),
        )

    def touch(self) -> None:
        """Record an access: bump count and update last_accessed_at."""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)
