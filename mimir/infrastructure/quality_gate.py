"""Quality gate for memory ingestion.

The gate runs before a new memory is learned. It currently provides:

1. Duplicate detection: reject or merge memories that are too similar to an
   existing memory (cosine similarity over embeddings).
2. Simple contradiction hints: surface pairs of memories that contain opposite
   polarity on the same subject. This is a lightweight heuristic, not a full
   natural-language inference model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

DEFAULT_DUPLICATE_THRESHOLD = 0.95
DEFAULT_CONTRADICTION_THRESHOLD = 0.85


@dataclass(frozen=True)
class QualityResult:
    """Outcome of the quality gate."""

    ok: bool
    reason: str = ""
    similar_memory: str | None = None


class QualityGate:
    """Check a candidate memory against existing memories for quality issues."""

    def __init__(
        self,
        *,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    ) -> None:
        self.duplicate_threshold = duplicate_threshold
        self.contradiction_threshold = contradiction_threshold

    def check_duplicate(
        self,
        text: str,
        embedding: list[float],
        existing_texts: list[str],
        existing_embeddings: list[list[float]],
    ) -> QualityResult:
        """Return a duplicate warning if *embedding* is too close to an existing one."""
        if not existing_embeddings:
            return QualityResult(ok=True)

        query = torch.tensor(embedding, dtype=torch.float32)
        candidates = torch.tensor(existing_embeddings, dtype=torch.float32)
        similarities = torch.nn.functional.cosine_similarity(
            query.unsqueeze(0), candidates, dim=1
        )
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= self.duplicate_threshold:
            return QualityResult(
                ok=False,
                reason="duplicate",
                similar_memory=existing_texts[best_idx],
            )
        return QualityResult(ok=True)

    def find_contradictions(self, texts: list[str]) -> list[tuple[int, int, str]]:
        """Return pairs of indices that may contradict each other.

        This is a simple heuristic: it looks for sentences that share keywords
        but contain negation asymmetry (e.g. "use Python" vs "don't use Python").
        It is not a substitute for an LLM-based contradiction detector.
        """
        contradictions: list[tuple[int, int, str]] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                hint = self._contradiction_hint(texts[i], texts[j])
                if hint:
                    contradictions.append((i, j, hint))
        return contradictions

    def _contradiction_hint(self, a: str, b: str) -> str:
        """Return a short hint if *a* and *b* look contradictory, else empty."""
        a_lower = a.lower()
        b_lower = b.lower()

        # Strip common leading words that don't affect polarity.
        a_core = self._strip_prefix(a_lower)
        b_core = self._strip_prefix(b_lower)

        # Direct negation of the same sentence.
        negators = ("don't ", "do not ", "won't ", "will not ", "can't ", "cannot ", "isn't ", "is not ", "aren't ", "are not ", "no ")
        for neg in negators:
            if a_core.startswith(neg) and b_core == a_core[len(neg):]:
                return f"negation: '{a}' vs '{b}'"
            if b_core.startswith(neg) and a_core == b_core[len(neg):]:
                return f"negation: '{a}' vs '{b}'"

        # Keyword overlap plus polarity words.
        a_words = set(re.findall(r"[a-z0-9]+", a_core))
        b_words = set(re.findall(r"[a-z0-9]+", b_core))
        shared = a_words & b_words
        if len(shared) >= 3:
            polarity_a = self._has_negation(a_lower)
            polarity_b = self._has_negation(b_lower)
            if polarity_a != polarity_b:
                return f"opposite polarity on shared terms: {sorted(shared)}"

        return ""

    def _strip_prefix(self, text: str) -> str:
        prefixes = ("i ", "we ", "they ", "he ", "she ", "it ", "you ", "the ", "a ", "an ")
        result = text
        while any(result.startswith(p) for p in prefixes):
            for p in prefixes:
                if result.startswith(p):
                    result = result[len(p):]
                    break
        return result

    def _has_negation(self, text: str) -> bool:
        negation_words = {
            "not", "no", "never", "none", "nothing", "don't", "do not",
            "won't", "will not", "can't", "cannot", "isn't", "is not",
            "aren't", "are not", "wasn't", "was not", "weren't", "were not",
        }
        words = set(re.findall(r"[a-z0-9']+", text))
        return bool(words & negation_words)
