"""Inject top skills into agent context."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mimir.skills.store import Skill


class InjectorConfig:
    """Configuration for skill context injection."""

    def __init__(self, max_active: int = 10, min_confidence: float = 0.85) -> None:
        self.max_active = max_active
        self.min_confidence = min_confidence


class SkillInjector:
    """Select and format skills for injection into the agent context."""

    def __init__(self, config: InjectorConfig | None = None) -> None:
        self.config = config or InjectorConfig()

    def select(self, skills: list[Skill]) -> list[Skill]:
        """Return the top-N active skills sorted by confidence and usage."""
        active = [s for s in skills if not s.deprecated and s.confidence >= self.config.min_confidence]

        def _score(skill: Skill) -> tuple[float, float]:
            try:
                ts = datetime.fromisoformat(skill.last_used).timestamp()
            except ValueError:
                ts = 0.0
            return (skill.confidence * (1 + skill.usage_count), ts)

        scored = sorted(active, key=_score, reverse=True)
        return scored[: self.config.max_active]

    def _sanitize(self, text: str) -> str:
        """Escape backticks to prevent markdown injection in context."""
        return text.replace("`", "'")

    def format(self, skills: list[Skill]) -> str:
        """Format selected skills as a markdown block for context injection."""
        if not skills:
            return ""
        lines = ["## Your active shortcuts", ""]
        for skill in skills:
            if skill.type == "alias" and skill.expansion:
                lines.append(f"- `{self._sanitize(skill.name)}` = `{self._sanitize(skill.expansion)}`")
            elif skill.template:
                lines.append(f"- `{self._sanitize(skill.name)}` = `{self._sanitize(skill.template)}`")
        lines.append("")
        return "\n".join(lines)

    def inject(self, skills: list[Skill]) -> dict[str, Any]:
        """Return a JSON-ready injection payload."""
        selected = self.select(skills)
        return {
            "active_skills": [s.to_dict() for s in selected],
            "formatted": self.format(selected),
        }
