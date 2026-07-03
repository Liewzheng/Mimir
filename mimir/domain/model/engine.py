"""Embedding engine protocol."""

from typing import Protocol

import torch


class EmbeddingEngine(Protocol):
    """Protocol for slow-weight embedding engines."""

    output_dim: int

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Embed a list of texts into a tensor of shape (n, output_dim)."""
        ...
