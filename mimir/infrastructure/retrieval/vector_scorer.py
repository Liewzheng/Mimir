"""Cosine-similarity scorer backed by Mimir embeddings."""

from __future__ import annotations

import torch

from mimir.domain.model import Memory
from mimir.infrastructure.retrieval.protocol import MemoryScorer


class VectorScorer(MemoryScorer):
    """Score memories by cosine similarity between query embedding and memory
    embeddings.

    This scorer relies on an external encoder (usually the Mimir instance) to
    embed the query. The memory objects are expected to already contain their
    embeddings.
    """

    def __init__(self, query_embedding: torch.Tensor) -> None:
        """Initialize with the pre-computed query embedding.

        Args:
            query_embedding: 1-D tensor of shape [dim].
        """
        if query_embedding.dim() != 1:
            raise ValueError("query_embedding must be a 1-D tensor")
        self.query_embedding = query_embedding

    def score(self, query: str, memories: list[Memory]) -> dict[int, float]:
        """Return cosine similarity scores for each memory.

        Args:
            query: Ignored; the query embedding was supplied at construction.
            memories: List of memories with embeddings.

        Returns:
            Mapping from memory index to cosine similarity score in [-1, 1].
        """
        if not memories:
            return {}

        query_norm = self.query_embedding / torch.linalg.norm(self.query_embedding)
        embeddings = torch.tensor(
            [memory.embedding for memory in memories],
            dtype=query_norm.dtype,
            device=query_norm.device,
        )
        memory_norms = embeddings / torch.linalg.norm(embeddings, dim=1, keepdim=True)
        similarities = torch.matmul(memory_norms, query_norm).tolist()
        return dict(enumerate(similarities))
