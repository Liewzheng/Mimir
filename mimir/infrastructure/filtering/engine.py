"""Central filtering engine for deciding what should be stored as memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .provider import RulePackProvider
from .registry import ProviderRegistry, default_registry
from .rules import is_gibberish, split_sentences


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for the memory filtering engine."""

    enabled: bool = True
    min_store_length: int = 1
    min_hook_length: int = 5
    min_hook_importance: float = 0.35
    small_talk_ratio_threshold: float = 0.85
    gibberish_min_density: float = 0.3
    gibberish_max_repetition: float = 0.5
    gibberish_max_punctuation: float = 0.7
    resource_dir: Path | None = None
    user_resource_dir: Path | None = None


@dataclass(frozen=True)
class FilterResult:
    """Outcome of a filtering decision."""

    store: bool
    score: float
    reason: str


class FilterEngine:
    """Decides whether a piece of text is worth remembering.

    The engine is intentionally conservative: it is better to store a slightly
    low-value sentence than to drop a valuable one. Filtering is applied most
    strictly to hook-captured text, where the agent did not explicitly ask to
    remember something.
    """

    def __init__(
        self,
        config: FilterConfig | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self.config = config or FilterConfig()
        self.registry = registry or default_registry(
            resource_dir=self.config.resource_dir,
            user_resource_dir=self.config.user_resource_dir,
        )

    def should_store(
        self,
        text: str,
        *,
        source: Literal["mcp", "hook"] = "mcp",
        force: bool = False,
    ) -> FilterResult:
        """Return a filtering decision for *text*.

        Args:
            text: The candidate text.
            source: "mcp" for explicit tool calls (lenient), "hook" for
                automatic hook capture (strict).
            force: If True, bypass all filters.
        """
        if force:
            return FilterResult(True, 1.0, "forced")

        if not self.config.enabled:
            return FilterResult(True, 0.5, "filtering_disabled")

        normalized = (text or "").strip()
        if not normalized:
            return FilterResult(False, 0.0, "empty")

        min_len = (
            self.config.min_hook_length
            if source == "hook"
            else self.config.min_store_length
        )
        if len(normalized) < min_len:
            return FilterResult(False, 0.0, "too_short")

        gibberish, reason = is_gibberish(
            normalized,
            min_density=self.config.gibberish_min_density,
            max_repetition=self.config.gibberish_max_repetition,
            max_punctuation=self.config.gibberish_max_punctuation,
        )
        if gibberish:
            return FilterResult(False, 0.0, reason)

        # Hook captures are automatic and need strict small-talk / importance
        # filtering. Explicit MCP store calls are intentional, so we only guard
        # against empty/gibberish input to preserve backward compatibility.
        if source == "mcp":
            return FilterResult(True, 0.5, "passed")

        providers = self.registry.providers_for(normalized)
        ratio = self._small_talk_ratio(normalized, providers)
        if ratio > self.config.small_talk_ratio_threshold:
            return FilterResult(False, 0.0, "mostly_small_talk")

        non_small = self._non_small_sentences(normalized, providers)
        if non_small:
            score = max(
                (p.score_importance(" ".join(non_small)) for p in providers),
                default=0.5,
            )
        else:
            score = 0.0

        if score < self.config.min_hook_importance:
            return FilterResult(False, score, "low_importance")

        return FilterResult(True, score, "passed")

    def _small_talk_ratio(
        self,
        text: str,
        providers: list[RulePackProvider],
    ) -> float:
        sentences = split_sentences(text)
        if not sentences:
            return 1.0
        if not providers:
            return 0.0
        small = sum(
            1
            for sentence in sentences
            if any(p.is_small_talk(sentence) for p in providers)
        )
        return small / len(sentences)

    def _non_small_sentences(
        self,
        text: str,
        providers: list[RulePackProvider],
    ) -> list[str]:
        sentences = split_sentences(text)
        if not providers:
            return sentences
        return [
            sentence
            for sentence in sentences
            if not any(p.is_small_talk(sentence) for p in providers)
        ]

    def clean_small_talk(self, text: str) -> str:
        """Return *text* with small-talk sentences removed.

        This is useful for storing only the informative part of a message.
        """
        providers = self.registry.providers_for(text)
        non_small = self._non_small_sentences(text, providers)
        return " ".join(non_small)
