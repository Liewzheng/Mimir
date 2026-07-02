"""Tests for prediction and surprise evaluation."""

from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.eval import eval_prediction
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy
from mimir.infrastructure.persistence.torch_checkpoint_repository import (
    TorchCheckpointRepository,
)
from mimir.infrastructure.prediction.first_order_markov_policy import (
    FirstOrderMarkovPredictionPolicy,
)
from mimir.tests.fake_engine import make_fake_engine


def test_eval_prediction_passes() -> None:
    result = eval_prediction()

    assert "predicted_after_c" in result
    assert "surprise_expected" in result
    assert "surprise_unexpected" in result
    assert "mean_random_surprise" in result

    # Repeated sequence should be predictable and surprises discriminable.
    assert result["surprise_expected"] <= result["surprise_unexpected"]
    assert result["predicted_after_c"] >= 0.0


def test_repeated_sequence_prediction() -> None:
    config = MimirConfig(base_model="eval", num_prototypes=8)
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=8)
    mimir = Mimir(
        config,
        engine=make_fake_engine(dim=16),
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
        prediction_policy=policy,
    )

    repeated = ["A", "B", "C"]
    for _ in range(10):
        for text in repeated:
            mimir.learn(text)

    # Find the prototype ids for A, B, C.
    ids: list[int] = []
    for text in repeated:
        report = mimir.learn(text)
        updated_ids = report.get("updated_ids", [])
        assert isinstance(updated_ids, list)
        ids.append(int(updated_ids[-1]))

    proto_a, proto_b, proto_c = ids[-3], ids[-2], ids[-1]
    predicted = mimir.predict_next(proto_id=proto_c)
    assert predicted == proto_a

    # Expected transition A->B should be no more surprising than A->C.
    policy.smoothing = 0.1
    surprise_expected = policy.surprise_score(proto_b, last_proto_id=proto_a)
    surprise_unexpected = policy.surprise_score(proto_c, last_proto_id=proto_a)
    assert surprise_expected <= surprise_unexpected


def test_random_sequence_higher_surprise() -> None:
    config = MimirConfig(base_model="eval", num_prototypes=8)
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=8)
    mimir = Mimir(
        config,
        engine=make_fake_engine(dim=16),
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
        prediction_policy=policy,
    )

    # Learn a repeated sequence first to establish expectations.
    for _ in range(5):
        for text in ["A", "B", "C"]:
            mimir.learn(text)

    policy.reset()
    random_texts = ["X", "Y", "Z", "W", "Q"]
    surprises: list[float] = []
    for text in random_texts:
        report = mimir.learn(text)
        score = report.get("surprise_score", 1.0)
        assert isinstance(score, (int, float))
        surprises.append(float(score))

    mean_random = sum(surprises) / len(surprises)
    # Random inputs should produce relatively high surprise.
    assert mean_random > 0.5
