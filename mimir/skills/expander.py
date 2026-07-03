"""Expand skill shortcuts into full tool inputs.

The expander matches an incoming tool input against active skills and fills in
any `{var}` slots. It is intentionally conservative: if the match is ambiguous or
the expansion would be dangerous, it returns ``None`` so the original input is
preserved.
"""

from __future__ import annotations

from typing import Any

from mimir.skills.store import Skill
from mimir.skills.validator import SafeCommandClassifier, _match_template


class SkillExpander:
    """Expand a tool input when it matches a stored skill."""

    def __init__(
        self,
        skills: list[Skill],
        min_confidence: float = 0.85,
        classifier: SafeCommandClassifier | None = None,
    ) -> None:
        self.skills = [s for s in skills if not s.deprecated and s.confidence >= min_confidence]
        self.classifier = classifier or SafeCommandClassifier()

    def expand(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
        """Return an expanded tool input if a skill matches, otherwise ``None``."""
        if tool_name != "Shell":
            return None
        command = tool_input.get("command", "")
        if not isinstance(command, str) or not command:
            return None

        for skill in self._sorted_skills():
            expanded = self._try_expand(skill, command)
            if expanded is not None:
                return {"command": expanded}

        return None

    def _sorted_skills(self) -> list[Skill]:
        """Prefer longer, higher-confidence patterns to avoid greedy matches."""
        return sorted(
            self.skills,
            key=lambda s: (len(s.trigger_pattern or ""), s.confidence),
            reverse=True,
        )

    def _try_expand(self, skill: Skill, command: str) -> str | None:
        """Expand a single skill against the command.

        Alias: exact match on ``trigger_pattern`` → ``expansion``.
        Workflow: ``template`` with ``{var}`` slots → filled command.
        """
        trigger = skill.trigger_pattern
        if not trigger:
            return None

        if skill.type == "alias" and skill.expansion:
            if trigger.strip() == command.strip():
                return skill.expansion
            return None

        template = skill.template or skill.expansion
        if not template or not _match_template(template, command):
            return None

        return _fill_template(template, command)


def _fill_template(template: str, command: str) -> str:
    """Fill ``{var}`` slots in ``template`` with values from ``command``."""
    template_tokens = template.split()
    command_tokens = command.split()

    filled: list[str] = []
    var_index = 0
    cmd_i = 0
    for tmpl_tok in template_tokens:
        if tmpl_tok.startswith("{") and tmpl_tok.endswith("}"):
            var_index += 1
            if cmd_i < len(command_tokens):
                filled.append(command_tokens[cmd_i])
                cmd_i += 1
            else:
                filled.append(f"{{{tmpl_tok[1:-1]}{var_index}}}")
        else:
            filled.append(tmpl_tok)
            # Advance cmd_i past the matching literal token(s).
            while cmd_i < len(command_tokens) and command_tokens[cmd_i] != tmpl_tok:
                cmd_i += 1
            if cmd_i < len(command_tokens) and command_tokens[cmd_i] == tmpl_tok:
                cmd_i += 1

    return " ".join(filled)
