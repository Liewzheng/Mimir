"""Mimir configuration."""

import re
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

    # Secret redaction.
    redaction_enabled: bool = True
    redaction_patterns: list[str] | None = None  # None = all defaults; [] = none

    # Quality gating (duplicate blocking and contradiction hints).
    quality_gate_enabled: bool = True
    quality_gate_duplicate_threshold: float = 0.95
    quality_gate_contradiction_threshold: float = 0.85

    # Project context ingestion.
    project_context_enabled: bool = True
    project_context_importance: float = 1.5

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
        if not 0 <= self.quality_gate_duplicate_threshold <= 1:
            raise ValueError("quality_gate_duplicate_threshold must be in [0, 1]")
        if not 0 <= self.quality_gate_contradiction_threshold <= 1:
            raise ValueError("quality_gate_contradiction_threshold must be in [0, 1]")
        if self.project_context_importance < 0:
            raise ValueError("project_context_importance must be non-negative")
        if self.redaction_patterns is not None:
            for pattern in self.redaction_patterns:
                if not pattern:
                    raise ValueError("Redaction patterns must be non-empty strings")
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"Invalid redaction pattern {pattern!r}: {exc}") from exc
