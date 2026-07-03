"""Validate extracted skills against executed commands and update confidence.

Validation is post-hoc: we never re-run a command. We only compare the command
that was already executed with the templates of active skills and adjust
confidence / failure counters accordingly.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from mimir.skills.store import Skill, SkillStore

#: Shell verbs/prefixes that are generally read-only and safe to validate.
_SAFE_VERBS: set[str] = {
    "cat",
    "ls",
    "find",
    "grep",
    "rg",
    "head",
    "tail",
    "wc",
    "sort",
    "uniq",
    "diff",
    "stat",
    "echo",
    "pwd",
    "which",
    "git",
}


class SafeCommandClassifier:
    """Decide whether a command is safe enough for automatic validation."""

    def is_safe(self, tool_name: str, command: str) -> bool:
        """Return True if the command is read-only and safe to validate."""
        if tool_name != "Shell":
            # Non-shell tool calls are not validated automatically in this phase.
            return False
        tokens = [t for t in command.split() if t]
        if not tokens:
            return False
        first = tokens[0].lower()

        # Some git subcommands are destructive; whitelist only read-only ones.
        if first == "git":
            if len(tokens) < 2:
                return False
            return tokens[1].lower() in {
                "status",
                "log",
                "show",
                "branch",
                "remote",
                "diff",
                "ls-files",
                "ls-tree",
            }

        return first in _SAFE_VERBS


class SkillValidator:
    """Compare executed commands with active skills and update confidence."""

    def __init__(
        self,
        store: SkillStore,
        classifier: SafeCommandClassifier | None = None,
        min_confidence: float = 0.5,
        max_failures: int = 5,
    ) -> None:
        self.store = store
        self.classifier = classifier or SafeCommandClassifier()
        self.min_confidence = min_confidence
        self.max_failures = max_failures

    def validate(
        self,
        tool_name: str,
        command: str,
        success: bool,
    ) -> Skill | None:
        """Validate a single command against active skills.

        Returns a revised skill if the command did not match an existing skill
        but a revision was generated from the tracker's cluster data. Otherwise
        returns None.
        """
        if not self.classifier.is_safe(tool_name, command):
            return None

        skills = self.store.load()
        matching_skill = self._find_match(skills, command)

        if matching_skill is not None:
            self._record_result(matching_skill, success)
            self.store.update(matching_skill)
            return None

        # No match. If the command succeeded, a new pattern may be emerging.
        # We do not create a brand-new skill here; extraction is the tracker/observer's
        # responsibility. The caller may trigger revision if cluster data is available.
        if success:
            return None

        # On failure without a match, there is nothing to update yet.
        return None

    def _find_match(self, skills: list[Skill], command: str) -> Skill | None:
        """Return the first active skill whose template matches the command."""
        for skill in skills:
            template = skill.template or skill.expansion
            if template and _match_template(template, command):
                return skill
        return None

    def _record_result(self, skill: Skill, success: bool) -> None:
        """Adjust confidence and counters based on validation result."""
        if success:
            skill.confidence = min(1.0, skill.confidence * 0.9 + 0.1)
            skill.usage_count += 1
            skill.last_used = datetime.now(timezone.utc).isoformat()
        else:
            skill.confidence = max(0.0, skill.confidence * 0.95 - 0.05)
            skill.failure_count += 1

        if (
            skill.failure_count >= self.max_failures
            or skill.usage_count >= self.max_failures
            and skill.confidence < self.min_confidence
        ):
            skill.deprecated = True


def _match_template(template: str, command: str) -> bool:
    """Check whether ``command`` matches ``template`` with ``{var}`` slots.

    The template is converted to a regex where each ``{...}`` token becomes a
    non-greedy capture of non-whitespace characters. This is intentionally
    simple and conservative: tokens are separated by whitespace.
    """
    regex_parts: list[str] = []
    for token in template.split():
        if token.startswith("{") and token.endswith("}"):
            regex_parts.append(r"\S+")
        else:
            regex_parts.append(re.escape(token))
    pattern = re.compile(r"^\s*" + r"\s+".join(regex_parts) + r"\s*$")
    return bool(pattern.match(command))


def _extract_result(payload: dict[str, Any]) -> bool | None:
    """Infer command success from the tool result payload.

    Returns True if the result clearly indicates success, False if it clearly
    indicates failure, and None if the result is absent or ambiguous.
    """
    tool_result = payload.get("tool_result")
    if tool_result is None:
        return None

    if isinstance(tool_result, bool):
        return tool_result

    if isinstance(tool_result, dict):
        if "error" in tool_result or "exit_code" in tool_result and tool_result["exit_code"] != 0:
            return False
        if "success" in tool_result:
            return bool(tool_result["success"])
        return True

    if isinstance(tool_result, str):
        lower = tool_result.lower()
        return not any(lower.startswith(p) for p in ("error:", "exception:", "failed"))

    return None
