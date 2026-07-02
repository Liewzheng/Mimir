"""Filter provider abstraction and default implementation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class FilterProvider(Protocol):
    """Provides script-specific small-talk detection and importance scoring."""

    scripts: frozenset[str]

    def is_small_talk(self, text: str) -> bool:
        """Return True if *text* is considered small-talk in this provider's scripts."""
        ...

    def score_importance(self, text: str) -> float:
        """Return an importance score in [0, 1] for *text*."""
        ...


@dataclass(frozen=True)
class JsonRulePack:
    """A rule pack loaded from a JSON resource file."""

    scripts: frozenset[str]
    small_talk_exact: frozenset[str]
    small_talk_prefixes: tuple[str, ...]
    small_talk_suffixes: tuple[str, ...]
    high_signals: frozenset[str]
    low_signals: frozenset[str]

    _prefix_re: re.Pattern[str] | None = None
    _suffix_re: re.Pattern[str] | None = None

    def __post_init__(self) -> None:
        # Compile prefix/suffix regexes once. Because the dataclass is frozen,
        # we use object.__setattr__ to attach cached patterns.
        if self.small_talk_prefixes:
            pattern = "^(?:" + "|".join(re.escape(p) for p in self.small_talk_prefixes) + ")"
            object.__setattr__(self, "_prefix_re", re.compile(pattern, re.IGNORECASE))
        if self.small_talk_suffixes:
            pattern = "(?:" + "|".join(re.escape(s) for s in self.small_talk_suffixes) + ")$"
            object.__setattr__(self, "_suffix_re", re.compile(pattern, re.IGNORECASE))

    def is_small_talk(self, text: str) -> bool:
        # Strip surrounding punctuation so "OK。" matches "ok".
        normalized = text.strip().lower().rstrip("。.!?！？")
        return normalized in self.small_talk_exact

    def score_importance(self, text: str) -> float:
        lower = text.lower()
        score = 0.5
        score += sum(0.08 for w in self.high_signals if w in lower)
        score -= sum(0.1 for w in self.low_signals if lower == w or lower.startswith(w + " "))
        return max(0.0, min(1.0, score))

    @classmethod
    def from_path(cls, path: Path) -> JsonRulePack:
        data = json.loads(path.read_text(encoding="utf-8"))
        small = data.get("small_talk", {})
        return cls(
            scripts=frozenset(data.get("scripts", [])),
            small_talk_exact=frozenset(w.lower() for w in small.get("exact", [])),
            small_talk_prefixes=(),
            small_talk_suffixes=(),
            high_signals=frozenset(data.get("high_signals", [])),
            low_signals=frozenset(data.get("low_signals", [])),
        )


class RulePackProvider:
    """A concrete FilterProvider backed by one or more JsonRulePacks."""

    def __init__(self, packs: list[JsonRulePack]) -> None:
        self._packs = packs
        self.scripts = frozenset().union(*(p.scripts for p in packs))

    def is_small_talk(self, text: str) -> bool:
        return any(pack.is_small_talk(text) for pack in self._packs)

    def score_importance(self, text: str) -> float:
        if not self._packs:
            return 0.5
        return max(pack.score_importance(text) for pack in self._packs)


def load_rule_packs(resource_dir: Path) -> dict[str, JsonRulePack]:
    """Load all JSON rule packs in *resource_dir*."""
    packs: dict[str, JsonRulePack] = {}
    if not resource_dir.exists():
        return packs
    for path in sorted(resource_dir.glob("*.json")):
        if path.name == "metadata.json":
            continue
        try:
            pack = JsonRulePack.from_path(path)
            for script in pack.scripts:
                packs[script] = pack
        except (json.JSONDecodeError, KeyError):
            continue
    return packs
