"""Unicode script detection utilities for language-aware filtering."""

from collections import Counter

# Common Unicode script blocks used for filtering. The list is intentionally
# conservative: new scripts can be added without changing the architecture.
SCRIPT_BLOCKS: dict[str, tuple[int, int]] = {
    "Latin": (0x0041, 0x024F),
    "CJK": (0x4E00, 0x9FFF),
    "Hiragana": (0x3040, 0x309F),
    "Katakana": (0x30A0, 0x30FF),
    "Hangul": (0xAC00, 0xD7AF),
    "Cyrillic": (0x0400, 0x04FF),
    "Arabic": (0x0600, 0x06FF),
    "Devanagari": (0x0900, 0x097F),
    "Greek": (0x0370, 0x03FF),
    "Hebrew": (0x0590, 0x05FF),
    "Thai": (0x0E00, 0x0E7F),
}

# CJK-related scripts are grouped together for resource lookup.
CJK_SCRIPTS: set[str] = {"CJK", "Hiragana", "Katakana", "Hangul"}


def detect_scripts(text: str, threshold: float = 0.15) -> set[str]:
    """Return the set of scripts present in *text* above *threshold*.

    The threshold is the minimum fraction of identifiable characters that must
    belong to a script before it is reported. This avoids classifying a mixed
    string by one or two stray characters (e.g. an English word in a Chinese
    sentence).

    Characters that do not belong to any known block are ignored.
    """
    counts: Counter[str] = Counter()
    for ch in text.strip():
        code = ord(ch)
        for script, (start, end) in SCRIPT_BLOCKS.items():
            if start <= code <= end:
                counts[script] += 1
                break

    if not counts:
        return {"Unknown"}

    total = counts.total()
    return {script for script, count in counts.items() if count / total >= threshold}


def is_cjk_script(scripts: set[str]) -> bool:
    """Return True if any of *scripts* is a CJK-related script."""
    return bool(scripts & CJK_SCRIPTS)


def script_groups(scripts: set[str]) -> set[str]:
    """Collapse CJK-related scripts into a single 'CJK' resource group."""
    groups = set(scripts)
    if groups & CJK_SCRIPTS:
        groups -= CJK_SCRIPTS
        groups.add("CJK")
    return groups
