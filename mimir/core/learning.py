"""Learning pipeline: update prototypes and metadata from new inputs."""

import torch

from mimir.core.config import MimirConfig
from mimir.core.prototype_store import PrototypeStore


class LearningPipeline:
    """Handle explicit and implicit learning updates."""

    def __init__(self, store: PrototypeStore, config: MimirConfig) -> None:
        self.store = store
        self.config = config

    def learn(
        self,
        base: torch.Tensor,
        step: int,
        importance: float = 1.0,
    ) -> dict[str, object]:
        """Update the prototype store based on base embeddings.

        Args:
            base: Tensor of shape [batch_size, dim].
            step: Global time step.
            importance: Multiplier for update strength.

        Returns:
            A report dict with update statistics.
        """
        updated_ids = self.store.update_nearest(base, step, importance)
        self.store.decay(step)

        return {
            "updated": len(updated_ids),
            "unique_updated": len(set(updated_ids)),
            "capacity_usage": self.store.capacity_usage(),
            "updated_ids": updated_ids,
        }
