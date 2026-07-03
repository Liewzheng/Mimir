"""Extract reusable command skeletons from a cluster of similar strings."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import NamedTuple


class Skeleton(NamedTuple):
    """A command template with fixed parts and variable slots."""

    template: str
    fixed_part: str
    fixed_ratio: float
    variable_count: int


def _looks_like_variable(token: str) -> bool:
    """Heuristically decide whether a differing token is likely a variable."""
    if not token:
        return False
    # Hex, UUID, hash-like, or plain numeric.
    if re.fullmatch(r"[0-9a-fA-F\-]{4,}", token):
        return True
    if re.fullmatch(r"\d+", token):
        return True
    # Absolute or relative paths.
    return bool(re.search(r"[/\\]", token))


def _tokenize(command: str) -> list[str]:
    """Split a command into whitespace-separated tokens while preserving structure."""
    return command.split()


def extract_skeleton(commands: list[str]) -> Skeleton:
    """Return the longest common literal skeleton of ``commands``.

    The algorithm works in token space for shell-like commands:

    1. Take the first command as the reference.
    2. For every other command, find the matching token blocks.
    3. Mark positions in the reference that appear in all other commands.
    4. Consecutive unmarked positions are collapsed into ``{var}`` slots.

    This is a simple prototype; it intentionally avoids heavy semantic parsing.
    """
    if not commands:
        return Skeleton("", "", 0.0, 0)
    if len(commands) == 1:
        return Skeleton(commands[0], commands[0], 1.0, 0)

    reference_tokens = _tokenize(commands[0])
    if not reference_tokens:
        return Skeleton("", "", 0.0, 0)

    # fixed_mask[i] == True if reference_tokens[i] is present in every other command.
    fixed_mask = [True] * len(reference_tokens)

    for other in commands[1:]:
        other_tokens = _tokenize(other)
        # Build a set of matching token positions in the reference for this pair.
        pair_matches = set()
        matcher = SequenceMatcher(None, reference_tokens, other_tokens)
        for block in matcher.get_matching_blocks():
            for i in range(block.a, block.a + block.size):
                pair_matches.add(i)
        for i in range(len(reference_tokens)):
            if i not in pair_matches:
                fixed_mask[i] = False

    # Build the template from fixed/unfixed runs.
    template_parts: list[str] = []
    fixed_parts: list[str] = []
    variable_count = 0
    i = 0
    while i < len(reference_tokens):
        if fixed_mask[i]:
            fixed_parts.append(reference_tokens[i])
            template_parts.append(reference_tokens[i])
            i += 1
        else:
            # Collect a run of unfixed tokens.
            run: list[str] = []
            while i < len(reference_tokens) and not fixed_mask[i]:
                run.append(reference_tokens[i])
                i += 1
            if run:
                variable_count += 1
                # If all tokens in the run look like variables, keep a generic slot;
                # otherwise include a hint from the first token for readability.
                hint = run[0] if not _looks_like_variable(run[0]) else "var"
                template_parts.append(f"{{{hint}}}" if variable_count == 1 else f"{{{hint}{variable_count}}}")

    template = " ".join(template_parts)
    fixed_literal = " ".join(fixed_parts)
    avg_length = sum(len(c) for c in commands) / len(commands)
    fixed_ratio = len(fixed_literal) / avg_length if avg_length else 0.0

    return Skeleton(template, fixed_literal, fixed_ratio, variable_count)
