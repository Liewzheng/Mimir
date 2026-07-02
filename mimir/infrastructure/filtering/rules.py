"""Language-agnostic content quality heuristics.

These rules intentionally avoid language-specific knowledge. They rely on
statistical properties of text (entropy, repetition, punctuation density) that
are valid across human languages.
"""

import math
from collections import Counter


def shannon_entropy(text: str) -> float:
    """Compute the Shannon entropy of *text* in bits."""
    counts = Counter(text)
    length = len(text)
    if length == 0:
        return 0.0
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def information_density(text: str) -> float:
    """Return normalized entropy in [0, 1].

    Low for repetitive strings like '哈哈哈哈', 'okokok', or '继续继续'.
    """
    text_set = set(text)
    if len(text_set) <= 1:
        return 0.0
    entropy = shannon_entropy(text)
    max_entropy = math.log2(len(text_set))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def repetition_ratio(text: str) -> float:
    """Return the fraction of characters that repeat the previous character."""
    if len(text) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])
    return repeats / (len(text) - 1)


def punctuation_ratio(text: str) -> float:
    """Return the fraction of characters that are punctuation or whitespace."""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isspace() or _is_punctuation(c)) / len(text)


def _is_punctuation(ch: str) -> bool:
    """Return True for common sentence/phrase punctuation across scripts."""
    return ch in ".,;:!?。，；：！？·•・、""''""«»„““”‘’"


def is_gibberish(
    text: str,
    min_density: float = 0.3,
    max_repetition: float = 0.5,
    max_punctuation: float = 0.7,
) -> tuple[bool, str]:
    """Return (is_gibberish, reason) for *text*.

    A string is considered gibberish if it is too repetitive, too punctuation-heavy,
    or lacks character diversity. These checks are script-independent.
    """
    if len(text) < 2:
        return True, "too_short"

    density = information_density(text)
    if density < min_density:
        return True, "low_density"

    repetition = repetition_ratio(text)
    if repetition > max_repetition:
        return True, "high_repetition"

    punct = punctuation_ratio(text)
    if punct > max_punctuation:
        return True, "high_punctuation"

    if all(c.isdigit() for c in text if not c.isspace()):
        return True, "pure_digits"

    return False, ""


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using cross-language delimiters.

    Falls back to the whole string if no delimiter is found, so a single
    long clause is never lost.
    """
    import re

    if not text.strip():
        return []

    # CJK sentence endings do not require whitespace; Latin endings should be
    # followed by whitespace or end-of-string to avoid splitting abbreviations.
    parts = re.split(r"(?<=[。！？])\s*|(?<=[.!?])(?:\s+|$)", text.strip())
    return [part.strip() for part in parts if part.strip()]


def sentence_has_content(sentence: str) -> bool:
    """Return True if *sentence* contains at least one alphabetic/CJK/digit token."""
    return any(c.isalnum() for c in sentence)
