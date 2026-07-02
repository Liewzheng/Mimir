"""Oja learning policy implementation."""

import torch

from mimir.domain.policy.learning_policy import LearningPolicy


class OjaLearningPolicy(LearningPolicy):
    """Normalized Hebbian learning rule.

    Oja's rule prevents unbounded weight growth by adding a weight-decay term
    proportional to the post-synaptic activity:

        delta_p = lr * y * (x - y * p)

    where y = p · x is the post-synaptic activation.
    """

    def compute_delta(
        self,
        prototype: torch.Tensor,
        input_vector: torch.Tensor,
        access_count: int,
        learning_rate_base: float,
        learning_rate_decay: float,
    ) -> torch.Tensor:
        """Compute the Oja update delta for a single prototype.

        Args:
            prototype: Current prototype vector, shape [dim].
            input_vector: Input base embedding, shape [dim].
            access_count: How many times this prototype has been updated.
            learning_rate_base: Base learning rate.
            learning_rate_decay: Decay factor for familiar prototypes.

        Returns:
            Update delta, shape [dim].
        """
        # Familiar prototypes get a smaller learning rate.
        lr = learning_rate_base / (1.0 + access_count * learning_rate_decay)
        # Post-synaptic activation is the dot product of unit vectors.
        y = torch.dot(prototype, input_vector)
        # Oja's rule: Hebbian growth (x) minus decay (y * p) to prevent explosion.
        return lr * y * (input_vector - y * prototype)
