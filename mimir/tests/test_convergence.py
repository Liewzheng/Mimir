"""Regression tests for learning convergence."""

import torch

from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.eval_data import CODE_THEME, FRUIT_THEME
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy
from mimir.infrastructure.persistence.torch_checkpoint_repository import (
    TorchCheckpointRepository,
)
from mimir.tests.fake_engine import make_fake_engine


def _mean_pairwise_similarity(embeddings: torch.Tensor) -> float:
    embeddings = embeddings / torch.linalg.norm(embeddings, dim=1, keepdims=True)
    sim_matrix = torch.matmul(embeddings, embeddings.t())
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim_matrix[mask].mean().item()


def test_same_semantic_converges() -> None:
    config = MimirConfig(
        base_model="eval",
        num_prototypes=8,  # Small relative to themes to force clustering.
        learning_rate_base=0.1,
        temperature=0.5,
    )
    mimir = Mimir(
        config,
        engine=make_fake_engine(dim=16),
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )

    before = mimir.encode(FRUIT_THEME)
    sim_before = _mean_pairwise_similarity(before)

    for _ in range(20):
        mimir.learn(FRUIT_THEME)

    after = mimir.encode(FRUIT_THEME)
    sim_after = _mean_pairwise_similarity(after)

    assert sim_after > sim_before, (
        f"Same-theme similarity should increase: {sim_before:.4f} -> {sim_after:.4f}"
    )


def test_cross_semantic_stable() -> None:
    config = MimirConfig(
        base_model="eval",
        num_prototypes=8,
        learning_rate_base=0.1,
        temperature=0.5,
    )
    mimir = Mimir(
        config,
        engine=make_fake_engine(dim=16),
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )

    before = torch.cat([mimir.encode(FRUIT_THEME), mimir.encode(CODE_THEME)], dim=0)
    sim_before = _mean_pairwise_similarity(before)

    for _ in range(20):
        mimir.learn(FRUIT_THEME)
        mimir.learn(CODE_THEME)

    after = torch.cat([mimir.encode(FRUIT_THEME), mimir.encode(CODE_THEME)], dim=0)
    sim_after = _mean_pairwise_similarity(after)

    delta = abs(sim_after - sim_before)
    assert delta < 0.1, (
        f"Cross-theme similarity should stay stable: {sim_before:.4f} -> {sim_after:.4f}"
    )


def test_residual_grows_after_learning() -> None:
    torch.manual_seed(42)
    config = MimirConfig(
        base_model="eval",
        num_prototypes=8,
        learning_rate_base=0.1,
        temperature=0.5,
    )
    mimir = Mimir(
        config,
        engine=make_fake_engine(dim=16),
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )

    base = mimir.engine.encode(FRUIT_THEME)
    residual_before = mimir.store.lookup(base)
    norm_before = torch.linalg.norm(residual_before, dim=1).mean().item()

    for _ in range(20):
        mimir.learn(FRUIT_THEME)

    residual_after = mimir.store.lookup(base)
    norm_after = torch.linalg.norm(residual_after, dim=1).mean().item()

    assert norm_after > norm_before, (
        f"Residual norm should grow: {norm_before:.4f} -> {norm_after:.4f}"
    )
