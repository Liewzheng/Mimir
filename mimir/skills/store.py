"""Persistent store for extracted skills."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mimir.infrastructure.persistence.atomic_write import atomic_write

logger = logging.getLogger(__name__)


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

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupted skill record at line %d", line_no)
        return records

    def _load_skills(self) -> list[Skill]:
        return [Skill.from_dict(record) for record in self._load_records()]

    def load(self) -> list[Skill]:
        """Return all non-deprecated skills."""
        return [s for s in self._load_skills() if not s.deprecated]

    def load_all(self) -> list[Skill]:
        """Return all skills including deprecated ones."""
        return self._load_skills()

    def add(self, skill: Skill) -> None:
        """Append a skill to the store, replacing an existing skill with the same ID."""
        skills = self._load_skills()
        skills = [s for s in skills if s.id != skill.id]
        skills.append(skill)
        self.replace(skills)

    def update(self, skill: Skill) -> None:
        """Replace an existing skill by ID, keeping its created_at timestamp."""
        skills = self._load_skills()
        existing = next((s for s in skills if s.id == skill.id), None)
        if existing is None:
            raise ValueError(f"Skill {skill.id!r} not found in store")
        skill.created_at = existing.created_at
        skills = [s for s in skills if s.id != skill.id]
        skills.append(skill)
        self.replace(skills)

    def deprecate(self, skill_id: str) -> None:
        """Mark the skill with the given ID as deprecated."""
        skill = self.get_by_id(skill_id)
        if skill is None:
            return
        skill.deprecated = True
        self.update(skill)

    def replace(self, skills: list[Skill]) -> None:
        """Overwrite the store with the given skills atomically."""
        text = "\n".join(json.dumps(s.to_dict(), ensure_ascii=False) for s in skills)
        if text:
            text += "\n"
        atomic_write(self.path, text)

    def get_by_id(self, skill_id: str) -> Skill | None:
        for skill in self._load_skills():
            if skill.id == skill_id:
                return skill
        return None
