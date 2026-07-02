"""Tests for MimirConfig validation."""

import pytest

from mimir.core.config import MimirConfig


def test_config_defaults() -> None:
    config = MimirConfig(base_model="test-model")
    assert config.num_prototypes == 1024
    assert config.residual_scale == 0.3


def test_config_rejects_invalid_num_prototypes() -> None:
    with pytest.raises(ValueError, match="num_prototypes must be positive"):
        MimirConfig(base_model="test-model", num_prototypes=0)


def test_config_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate_base must be non-negative"):
        MimirConfig(base_model="test-model", learning_rate_base=-0.1)


def test_config_allows_zero_learning_rate() -> None:
    with pytest.warns(UserWarning, match="learning_rate_base is 0"):
        config = MimirConfig(base_model="test-model", learning_rate_base=0.0)
    assert config.learning_rate_base == 0.0
