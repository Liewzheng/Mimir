"""Pure-Python BM25 scorer for keyword retrieval over memory text."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from mimir.domain.model import Memory
from mimir.infrastructure.retrieval.protocol import MemoryScorer


def _default_tokenizer(text: str) -> list[str]:
    return _tokenize(text)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25.

    This tokenizer is intentionally simple and dependency-free. It:
      - lowercases the text
      - extracts contiguous ASCII alphanumeric sequences as tokens
      - extracts individual CJK characters as tokens

    This gives reasonable behavior for English (word-level) and Chinese
    (character-level) without requiring ICU, jieba, or other heavy dependencies.
    """
    text = text.lower()
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9]+|[^\x00-\x7f]", text):
        token = match.group(0)
        # Split CJK characters individually.
        if token.isascii():
            tokens.append(token)
        else:
            tokens.extend(list(token))
    return tokens


class BM25Scorer(MemoryScorer):
    """BM25Okapi scorer for keyword matching.

    Implementation follows the standard Robertson et al. formula:

        score(q, d) = sum(IDF(q_i) * (f(q_i, d) * (k1 + 1)) /
                          (f(q_i, d) + k1 * (1 - b + b * |d| / avgdl)))

    The scorer is stateless across calls but builds an inverted index from the
    provided memory list for efficiency.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        """Initialize BM25 hyperparameters.

        Args:
            k1: Term frequency saturation parameter. Typical range 1.2-2.0.
            b: Length normalization parameter. 0.0 disables length norm;
                1.0 full length norm. Typical 0.75.
            epsilon: Minimum IDF floor to avoid negative scores for very common
                terms. Standard BM25+ uses 0.25-0.5.
            tokenizer: Optional custom tokenizer. Defaults to a lightweight
                English + CJK tokenizer.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.tokenizer = tokenizer or _default_tokenizer

    def score(self, query: str, memories: list[Memory]) -> dict[int, float]:
        """Return BM25 scores for each memory.

        Args:
            query: Query text.
            memories: List of memories. Only `memory.text` is used.

        Returns:
            Mapping from memory index to BM25 score. Memories with no matching
            tokens are omitted.
        """
        if not memories:
            return {}

        tokenized_docs = [self.tokenizer(memory.text) for memory in memories]
        doc_freqs: Counter[str] = Counter()
        doc_lengths: list[int] = []
        for tokens in tokenized_docs:
            doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1

        avgdl = sum(doc_lengths) / len(doc_lengths)
        n_docs = len(memories)

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return {}

        # Pre-compute IDF for each query token.
        idf: dict[str, float] = {}
        for token in set(query_tokens):
            df = doc_freqs.get(token, 0)
            # BM25 IDF with a floor to avoid negative scores for common terms.
            idf[token] = max(
                math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0),
                self.epsilon,
            )

        scores: dict[int, float] = {}
        for idx, (tokens, doc_len) in enumerate(zip(tokenized_docs, doc_lengths, strict=True)):
            if not tokens:
                continue
            token_counts = Counter(tokens)
            denom_left = 1.0 - self.b
            denom_right = self.b * (doc_len / avgdl) if avgdl > 0 else 0.0
            length_factor = denom_left + denom_right

            score = 0.0
            for token in query_tokens:
                freq = token_counts.get(token, 0)
                if freq == 0:
                    continue
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * length_factor
                score += idf[token] * (numerator / denominator)

            if score > 0:
                scores[idx] = score

        return scores
