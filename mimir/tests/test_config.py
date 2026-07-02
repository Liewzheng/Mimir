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


def test_config_rejects_invalid_quality_gate_thresholds() -> None:
    with pytest.raises(ValueError, match="quality_gate_duplicate_threshold must be in"):
        MimirConfig(base_model="test-model", quality_gate_duplicate_threshold=1.5)
    with pytest.raises(ValueError, match="quality_gate_contradiction_threshold must be in"):
        MimirConfig(base_model="test-model", quality_gate_contradiction_threshold=-0.1)
    with pytest.raises(ValueError, match="quality_gate_duplicate_threshold must be in"):
        MimirConfig(base_model="test-model", quality_gate_duplicate_threshold=-0.01)


def test_config_rejects_negative_project_context_importance() -> None:
    with pytest.raises(ValueError, match="project_context_importance must be non-negative"):
        MimirConfig(base_model="test-model", project_context_importance=-1.0)


def test_config_allows_zero_project_context_importance() -> None:
    config = MimirConfig(base_model="test-model", project_context_importance=0.0)
    assert config.project_context_importance == 0.0


def test_config_rejects_invalid_redaction_patterns() -> None:
    with pytest.raises(ValueError, match="Invalid redaction pattern"):
        MimirConfig(base_model="test-model", redaction_patterns=["("])


def test_config_rejects_empty_redaction_pattern() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MimirConfig(base_model="test-model", redaction_patterns=[""])


def test_config_allows_empty_redaction_pattern_list() -> None:
    config = MimirConfig(base_model="test-model", redaction_patterns=[])
    assert config.redaction_patterns == []


def test_config_allows_disabled_project_context() -> None:
    config = MimirConfig(base_model="test-model", project_context_enabled=False)
    assert config.project_context_enabled is False


def test_config_allows_disabled_quality_gate() -> None:
    config = MimirConfig(base_model="test-model", quality_gate_enabled=False)
    assert config.quality_gate_enabled is False


def test_config_allows_custom_redaction_patterns() -> None:
    config = MimirConfig(base_model="test-model", redaction_patterns=[r"secret-\w+"])
    assert config.redaction_patterns == [r"secret-\w+"]
