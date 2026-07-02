"""Mimir configuration."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class MimirConfig:
    """Configuration for the Mimir system.

    All hyperparameters are exposed here to keep the core modules
    dependency-injection friendly and easy to test.
    """

    # Slow weights: base embedding model.
    base_model: str
    base_device: Literal["cpu", "cuda", "mps", "auto"] = "auto"

    # Prototype matrix.
    num_prototypes: int = 1024
    prototype_init_scale: float = 0.01

    # Modulation.
    residual_scale: float = 0.3
    temperature: float = 1.0
    top_k: int | None = None  # Sparse activation: keep only top-k prototypes.

    # Learning.
    learning_rate_base: float = 0.01
    learning_rate_decay: float = 0.1

    # Forgetting.
    forgetting_decay: float = 0.995
    max_prototypes: int = 1024

    # Persistence.
    checkpoint_dir: Path | None = None

    # Memory filtering (language-aware small-talk and quality gating).
    filter_enabled: bool = True
    filter_min_store_length: int = 1
    filter_min_hook_length: int = 5
    filter_min_hook_importance: float = 0.35
    filter_small_talk_ratio_threshold: float = 0.85
    filter_user_resource_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if self.max_prototypes <= 0:
            raise ValueError("max_prototypes must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if self.learning_rate_base < 0:
            raise ValueError("learning_rate_base must be non-negative")
        if self.learning_rate_base == 0:
            warnings.warn(
                "learning_rate_base is 0; Mimir.learn() will be a no-op.",
                stacklevel=2,
            )
