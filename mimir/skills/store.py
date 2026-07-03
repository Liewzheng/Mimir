"""Persistent store for extracted skills."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass
class Skill:
    """A reusable skill extracted from repeated agent commands."""

    id: str
    type: Literal["alias", "workflow"]
    name: str
    trigger_pattern: str
    expansion: str | None = None
    template: str | None = None
    required_context: list[str] = field(default_factory=list)
    confidence: float = 0.0
    usage_count: int = 0
    failure_count: int = 0
    version: int = 1
    deprecated: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        return cls(**data)


class SkillStore:
    """JSONL-based store for skills."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_lines(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def load(self) -> list[Skill]:
        """Return all non-deprecated skills."""
        return [
            Skill.from_dict(record)
            for record in self._load_lines()
            if not record.get("deprecated", False)
        ]

    def load_all(self) -> list[Skill]:
        """Return all skills including deprecated ones."""
        return [Skill.from_dict(record) for record in self._load_lines()]

    def add(self, skill: Skill) -> None:
        """Append a skill to the store."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(skill.to_dict(), ensure_ascii=False) + "\n")

    def replace(self, skills: list[Skill]) -> None:
        """Overwrite the store with the given skills."""
        self.path.write_text(
            "\n".join(json.dumps(s.to_dict(), ensure_ascii=False) for s in skills) + "\n",
            encoding="utf-8",
        )

    def get_by_id(self, skill_id: str) -> Skill | None:
        for skill in self.load_all():
            if skill.id == skill_id:
                return skill
        return None
