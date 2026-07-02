"""Tests for the evaluation harness."""

from unittest import mock

import pytest

from mimir.eval import (
    EvalConfig,
    _check_pass,
    _default_config,
    _mean_pairwise_similarity,
    _select_engine_factory,
    eval_convergence,
    eval_forgetting,
    eval_latency,
    eval_prediction,
    main,
    run_all,
)


def test_mean_pairwise_similarity() -> None:
    """Identical vectors have cosine similarity 1."""
    import torch

    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert _mean_pairwise_similarity(embeddings) == pytest.approx(1.0)


def test_select_engine_factory_fake() -> None:
    """The fake backend returns a usable engine."""
    factory = _select_engine_factory("fake")
    engine = factory()
    assert engine.output_dim > 0


def test_select_engine_factory_unknown() -> None:
    """Unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        _select_engine_factory("unknown")


def test_default_config_fake() -> None:
    """Default config for fake backend preserves defaults."""
    config = _default_config("fake")
    assert config.num_prototypes == 8


def test_default_config_unknown_backend() -> None:
    """Default config rejects unknown backend."""
    with pytest.raises(ValueError, match="Unknown backend"):
        _default_config("unknown")


def test_check_pass_fake_backend() -> None:
    """Check pass with all thresholds satisfied."""
    metrics = {
        "fruit_sim_delta": 0.03,
        "residual_norm_delta": 0.01,
        "cross_sim_delta": 0.01,
        "strength_decay_ratio": 0.9,
        "state_finite": 1.0,
        "overhead_ratio": 1.2,
        "predicted_after_c": 2.0,
        "surprise_unexpected": 0.8,
        "surprise_expected": 0.2,
    }
    passes = _check_pass(metrics, "fake")
    assert all(passes.values())


def test_eval_convergence() -> None:
    """Convergence evaluation returns expected keys."""
    config = EvalConfig(learn_iterations=2, decay_steps=2)
    result = eval_convergence(config)
    assert "fruit_sim_delta" in result
    assert "embedding_shift" in result


def test_eval_forgetting() -> None:
    """Forgetting evaluation returns expected keys."""
    config = EvalConfig(decay_steps=2)
    result = eval_forgetting(config)
    assert "strength_decay_ratio" in result
    assert "state_finite" in result


def test_eval_latency() -> None:
    """Latency evaluation returns overhead metrics."""
    config = EvalConfig()
    result = eval_latency(config)
    assert "base_time_ms" in result
    assert "overhead_ratio" in result


def test_eval_prediction() -> None:
    """Prediction evaluation returns expected keys."""
    config = EvalConfig(learn_iterations=2)
    result = eval_prediction(config)
    assert "predicted_after_c" in result
    assert "surprise_expected" in result


def test_run_all() -> None:
    """run_all aggregates results from all evaluators."""
    config = EvalConfig(learn_iterations=2, decay_steps=2)
    report = run_all(config, backend="fake")
    assert "convergence" in report
    assert "forgetting" in report
    assert "latency" in report
    assert "prediction" in report
    assert "pass" in report


def test_main_runs() -> None:
    """eval main prints a report and returns 0 on fake backend."""
    with mock.patch("sys.argv", ["mimir.eval", "--backend", "fake"]):
        assert main() == 0
