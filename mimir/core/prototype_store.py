"""Prototype store: the core fast-weight memory matrix."""

import torch
import torch.nn.functional as functional

from mimir.core.config import MimirConfig
from mimir.domain.policy.learning_policy import LearningPolicy


class PrototypeStore:
    """Maintain a fixed-capacity matrix of semantic prototypes.

    Metadata tensor columns (float32):
      0: strength
      1: access_count
      2: created_step
      3: last_access_step
    """

    _STRENGTH_IDX = 0
    _ACCESS_COUNT_IDX = 1
    _CREATED_STEP_IDX = 2
    _LAST_ACCESS_STEP_IDX = 3

    def __init__(
        self,
        dim: int,
        config: MimirConfig,
        policy: LearningPolicy,
    ) -> None:
        self.dim = dim
        self.config = config
        self.policy = policy
        self.prototypes: torch.Tensor = torch.zeros(config.num_prototypes, dim, dtype=torch.float32)
        self.metadata: torch.Tensor = torch.zeros(config.num_prototypes, 4, dtype=torch.float32)
        self._init_prototypes()

    def _init_prototypes(self) -> None:
        """Initialize prototypes as small random unit vectors.

        The created_step column is initialized to -1 so that a prototype created
        at global step 0 is not mistaken for uninitialized on later updates.
        """
        self.prototypes.normal_(mean=0.0, std=self.config.prototype_init_scale)
        self.prototypes = functional.normalize(self.prototypes, dim=1)
        self.metadata[:, self._STRENGTH_IDX] = 1.0
        self.metadata[:, self._CREATED_STEP_IDX] = -1.0

    def lookup(self, base: torch.Tensor) -> torch.Tensor:
        """Return residual modulation for the given base embeddings.

        Args:
            base: Tensor of shape [batch_size, dim].

        Returns:
            Residual tensor of shape [batch_size, dim].
        """
        weights = self.activation_weights(base)
        residual = torch.matmul(weights, self.prototypes)  # [batch, dim]
        return self.config.residual_scale * residual

    def activation_weights(self, base: torch.Tensor) -> torch.Tensor:
        """Compute prototype activation weights for base embeddings.

        If ``config.top_k`` is set, only the top-k prototypes by cosine
        similarity receive non-zero weight.  The weights are normalized
        via softmax over the selected subset.

        Args:
            base: Tensor of shape [batch_size, dim].

        Returns:
            Weight tensor of shape [batch_size, num_prototypes].
        """
        base_norm = functional.normalize(base, dim=1)
        proto_norm = functional.normalize(self.prototypes, dim=1)
        # Cosine similarity is the dot product of unit vectors.
        sim = torch.matmul(base_norm, proto_norm.t())  # [batch, num_protos]

        if self.config.top_k is None:
            # Dense attention: every prototype contributes, weighted by softmax.
            return functional.softmax(sim / self.config.temperature, dim=1)

        top_k = min(self.config.top_k, self.config.num_prototypes)
        # Build a mask that keeps only the top-k similarities per sample.
        threshold = torch.topk(sim, top_k, dim=1).values[:, -1:]  # [batch, 1]
        mask = sim >= threshold  # [batch, num_protos]
        # Masked-out entries become -inf so softmax assigns them zero weight.
        masked_sim = sim.masked_fill(~mask, float("-inf"))
        return functional.softmax(masked_sim / self.config.temperature, dim=1)

    def update_nearest(
        self,
        base: torch.Tensor,
        step: int,
        importance: float = 1.0,
    ) -> list[int]:
        """Update the nearest prototype for each base embedding.

        Args:
            base: Tensor of shape [batch_size, dim].
            step: Global time step.
            importance: Multiplier for the update magnitude.

        Returns:
            List of updated prototype ids.
        """
        base_norm = functional.normalize(base, dim=1)
        proto_norm = functional.normalize(self.prototypes, dim=1)
        # Winner-take-all: each input updates exactly one prototype.
        sim = torch.matmul(base_norm, proto_norm.t())  # [batch, num_protos]
        nearest_ids = sim.argmax(dim=1).tolist()

        for idx, proto_id in enumerate(nearest_ids):
            input_vector = base_norm[idx]
            prototype = self.prototypes[proto_id]
            access_count = int(self.metadata[proto_id, self._ACCESS_COUNT_IDX].item())

            # Compute the policy-specific weight update (e.g. Oja's rule).
            delta = self.policy.compute_delta(
                prototype=prototype,
                input_vector=input_vector,
                access_count=access_count,
                learning_rate_base=self.config.learning_rate_base,
                learning_rate_decay=self.config.learning_rate_decay,
            )

            # Re-normalize so prototypes stay on the unit hypersphere.
            self.prototypes[proto_id] = functional.normalize(prototype + importance * delta, dim=0)

            # Update metadata.
            self.metadata[proto_id, self._ACCESS_COUNT_IDX] = access_count + 1
            self.metadata[proto_id, self._LAST_ACCESS_STEP_IDX] = float(step)
            if self.metadata[proto_id, self._CREATED_STEP_IDX].item() < 0:
                self.metadata[proto_id, self._CREATED_STEP_IDX] = float(step)

            # Strength grows logarithmically and caps at 5.0.
            strength = self.metadata[proto_id, self._STRENGTH_IDX].item()
            delta_strength = 1.0 / (1.0 + access_count * 0.2)
            self.metadata[proto_id, self._STRENGTH_IDX] = min(strength + delta_strength, 5.0)

        return nearest_ids

    def decay(self, step: int) -> None:
        """Apply global forgetting to all prototypes."""
        self.prototypes *= self.config.forgetting_decay
        self.metadata[:, self._STRENGTH_IDX] *= self.config.forgetting_decay

    def capacity_usage(self) -> float:
        """Return a rough indicator of how much the store is utilized.

        Based on the mean L2 norm of prototypes relative to init scale.
        """
        norms = torch.linalg.norm(self.prototypes, dim=1)
        threshold = self.config.prototype_init_scale * 3.0
        active = (norms > threshold).float().mean().item()
        return float(active)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a serializable snapshot of the store state."""
        return {
            "prototypes": self.prototypes.clone(),
            "metadata": self.metadata.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the store from a snapshot."""
        self.prototypes = state["prototypes"].clone()
        self.metadata = state["metadata"].clone()
