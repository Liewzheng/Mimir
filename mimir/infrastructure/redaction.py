"""Secret and sensitive-data redaction for memory text.

The redactor scans text before it is stored as memory and replaces common
secret patterns with a placeholder. This prevents API keys, tokens, passwords,
and other credentials from being persisted in the long-term memory file.

Pattern coverage is intentionally conservative: it is better to redact a few
false positives than to leak a real secret. Users can disable the default
patterns or add their own via configuration.
"""

from __future__ import annotations

import collections.abc
import re
from dataclasses import dataclass
from typing import Protocol

_REDACTION_PLACEHOLDER = "[REDACTED]"


class RedactionPattern(Protocol):
    """A single redaction rule."""

    def redact(self, text: str) -> str:
        """Return *text* with sensitive content replaced."""
        pass


@dataclass(frozen=True)
class RegexRedactionPattern:
    """Redaction rule based on a regular expression."""

    name: str
    pattern: re.Pattern[str]
    placeholder: str = _REDACTION_PLACEHOLDER

    def redact(self, text: str) -> str:
        return self.pattern.sub(self.placeholder, text)


def _as_pattern(pattern: str | RedactionPattern) -> RedactionPattern:
    """Convert a plain regex string into a compiled redaction pattern."""
    if isinstance(pattern, str):
        return RegexRedactionPattern(name=pattern, pattern=re.compile(pattern))
    return pattern


# Default patterns. Keep them simple and over-inclusive: the goal is to avoid
# leaking secrets, not to preserve every valid-looking string.
DEFAULT_PATTERNS: list[RegexRedactionPattern] = [
    # Standalone OpenAI / Anthropic style API keys.
    RegexRedactionPattern(
        name="openai_style_key",
        pattern=re.compile(
            r"\b(sk-[a-z0-9]{16,}|sk-proj-[a-z0-9_-]{16,}|sk-ant-[a-z0-9]{16,})\b",
        ),
    ),
    # OpenAI / Anthropic / generic API keys assigned in shell or code.
    RegexRedactionPattern(
        name="api_key_assignment",
        pattern=re.compile(
            r"(?i)(?:api[_-]?key|apikey|secret[_-]?key|openai[_-]?key|anthropic[_-]?key)\s*[:=]\s*['\"]?([a-z0-9_\-]{16,})['\"]?",
        ),
    ),
    # GitHub personal access tokens (classic and fine-grained).
    RegexRedactionPattern(
        name="github_token",
        pattern=re.compile(
            r"\b(gh[pousr]_[A-Za-z0-9_]{36,}|github[_-]?pat[_-]?[A-Za-z0-9]{20,})\b",
        ),
    ),
    # AWS access key id + secret access key pair.
    RegexRedactionPattern(
        name="aws_credentials",
        pattern=re.compile(
            r"(?i)\b(AKI[A-Z0-9]{16})\b.*?(?:secret[_-]?access[_-]?key\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?|[A-Za-z0-9/+=]{40})",
            re.DOTALL,
        ),
    ),
    # JSON Web Tokens (three base64url segments separated by dots).
    RegexRedactionPattern(
        name="jwt",
        pattern=re.compile(
            r"\b(eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*)\b",
        ),
    ),
    # Bearer / Basic authorization headers.
    RegexRedactionPattern(
        name="authorization_header",
        pattern=re.compile(
            r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)[A-Za-z0-9_\-./+=]+",
        ),
    ),
    # Password assignments and URL credentials.
    RegexRedactionPattern(
        name="password",
        pattern=re.compile(
            r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{4,})['\"]?",
        ),
    ),
    # URL credentials: https://user:pass@host
    RegexRedactionPattern(
        name="url_credentials",
        pattern=re.compile(
            r"([a-z+]+://[^:/\s]+:)([^@\s]+)(@[^\s]+)",
            re.IGNORECASE,
        ),
    ),
    # Generic high-entropy secret assignments like MY_SECRET=xxxxxx.
    RegexRedactionPattern(
        name="env_secret_assignment",
        pattern=re.compile(
            r"(?i)([A-Z_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|PWD|APIKEY|API_KEY)[A-Z_]*\s*=\s*['\"]?)([A-Za-z0-9_\-./+=]{8,})['\"]?",
        ),
    ),
]


class Redactor:
    """Redact sensitive patterns from text before storage."""

    def __init__(
        self,
        patterns: collections.abc.Sequence[str | RedactionPattern] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.patterns = (
            [_as_pattern(p) for p in patterns]
            if patterns is not None
            else DEFAULT_PATTERNS
        )
        self.enabled = enabled

    def redact(self, text: str) -> str:
        """Return *text* with all configured patterns redacted."""
        if not self.enabled or not text:
            return text
        result = text
        for pattern in self.patterns:
            result = pattern.redact(result)
        return result

    def redact_messages(
        self,
        messages: list[dict[str, str]],
        content_key: str = "content",
    ) -> list[dict[str, str]]:
        """Return a copy of *messages* with sensitive content redacted.

        Only string values under *content_key* are processed; other fields are
        left untouched.
        """
        if not self.enabled:
            return messages
        redacted: list[dict[str, str]] = []
        for msg in messages:
            copy = dict(msg)
            content = copy.get(content_key)
            if isinstance(content, str):
                copy[content_key] = self.redact(content)
            redacted.append(copy)
        return redacted
