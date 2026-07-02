"""Inference pipeline: read-only path from base embeddings to plastic embeddings."""

import torch

from mimir.core.config import MimirConfig
from mimir.core.prototype_store import PrototypeStore


class InferencePipeline:
    """Compute plastic embeddings from base embeddings.

    This pipeline is intentionally stateless: it only reads from the
    PrototypeStore and never writes back.
    """

    def __init__(self, store: PrototypeStore, config: MimirConfig) -> None:
        self.store = store
        self.config = config

    def encode(self, base: torch.Tensor) -> torch.Tensor:
        """Apply prototype-based residual modulation.

        Args:
            base: Tensor of shape [batch_size, dim].

        Returns:
            Tensor of shape [batch_size, dim].
        """
        residual = self.store.lookup(base)
        return base + residual
