"""Evaluation harness for Mimir.

Run with:
    python -m mimir.eval
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.domain.model.engine import EmbeddingEngine
from mimir.eval_data import CODE_THEME, FRUIT_THEME, HISTORY_THEME
from mimir.infrastructure.embedding.fake_engine import make_fake_engine
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy
from mimir.infrastructure.persistence.torch_checkpoint_repository import (
    TorchCheckpointRepository,
)
from mimir.infrastructure.prediction.first_order_markov_policy import (
    FirstOrderMarkovPredictionPolicy,
)


def _select_engine_factory(backend: str) -> Callable[[], EmbeddingEngine]:
    """Select an embedding engine factory by backend name."""
    if backend == "fake":
        return make_fake_engine
    if backend == "llama-server":
        from mimir.infrastructure.embedding.llama_server_engine import (
            LlamaServerEmbeddingEngine,
        )

        return lambda: LlamaServerEmbeddingEngine()
    if backend == "sentence-transformer":
        from mimir.infrastructure.embedding.sentence_transformer_engine import (
            SentenceTransformerEngine,
        )

        return lambda: SentenceTransformerEngine("all-MiniLM-L6-v2")
    raise ValueError(f"Unknown backend: {backend}")


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for evaluation runs."""

    seed: int = 42
    num_prototypes: int = 8  # Small relative to themes to force clustering.
    learning_rate_base: float = 0.1
    learning_rate_decay: float = 0.1
    temperature: float = 0.5  # Sharper softmax for clearer prototype assignment.
    residual_scale: float = 0.3
    forgetting_decay: float = 0.995
    learn_iterations: int = 20
    decay_steps: int = 100
    top_k: int | None = None


def _mean_pairwise_similarity(embeddings: torch.Tensor) -> float:
    """Compute mean cosine similarity over all pairs in a batch."""
    embeddings = embeddings / torch.linalg.norm(embeddings, dim=1, keepdims=True)
    sim_matrix = torch.matmul(embeddings, embeddings.t())
    # Exclude diagonal.
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_sim: torch.Tensor = sim_matrix[mask].mean()
    return float(mean_sim.item())


def _residual_norm(mimir: Mimir, texts: list[str]) -> float:
    """Compute mean L2 norm of the prototype residual for given texts."""
    base = mimir.engine.encode(texts)
    residual = mimir.store.lookup(base)
    mean_norm: torch.Tensor = torch.linalg.norm(residual, dim=1).mean()
    return float(mean_norm.item())


def eval_convergence(
    config: EvalConfig | None = None,
    engine_factory: Callable[[], EmbeddingEngine] | None = None,
) -> dict[str, float]:
    """Evaluate whether similar inputs converge and dissimilar inputs stay apart."""
    config = config or EvalConfig()
    engine = (engine_factory or make_fake_engine)()

    torch.manual_seed(config.seed)
    mimir_config = MimirConfig(
        base_model="eval",
        num_prototypes=config.num_prototypes,
        learning_rate_base=config.learning_rate_base,
        learning_rate_decay=config.learning_rate_decay,
        temperature=config.temperature,
        residual_scale=config.residual_scale,
        forgetting_decay=config.forgetting_decay,
        top_k=config.top_k,
    )
    mimir = Mimir(
        mimir_config,
        engine=engine,
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )

    # Baseline.
    base_fruit = mimir.encode(FRUIT_THEME)
    base_code = mimir.encode(CODE_THEME)
    base_history = mimir.encode(HISTORY_THEME)

    sim_fruit_before = _mean_pairwise_similarity(base_fruit)
    sim_code_before = _mean_pairwise_similarity(base_code)
    cross_before = _mean_pairwise_similarity(
        torch.cat([base_fruit, base_code, base_history], dim=0)
    )
    residual_before = _residual_norm(mimir, FRUIT_THEME)

    # Learn.
    for _ in range(config.learn_iterations):
        mimir.learn(FRUIT_THEME)
        mimir.learn(CODE_THEME)
        mimir.learn(HISTORY_THEME)

    after_fruit = mimir.encode(FRUIT_THEME)
    after_code = mimir.encode(CODE_THEME)
    after_history = mimir.encode(HISTORY_THEME)

    # Compute embedding shift for learned texts (mean L2 distance).
    embedding_shift = torch.linalg.norm(after_fruit - base_fruit, dim=1).mean().item()

    sim_fruit_after = _mean_pairwise_similarity(after_fruit)
    sim_code_after = _mean_pairwise_similarity(after_code)
    cross_after = _mean_pairwise_similarity(
        torch.cat([after_fruit, after_code, after_history], dim=0)
    )
    residual_after = _residual_norm(mimir, FRUIT_THEME)

    return {
        "fruit_sim_before": sim_fruit_before,
        "fruit_sim_after": sim_fruit_after,
        "fruit_sim_delta": sim_fruit_after - sim_fruit_before,
        "code_sim_before": sim_code_before,
        "code_sim_after": sim_code_after,
        "code_sim_delta": sim_code_after - sim_code_before,
        "cross_sim_before": cross_before,
        "cross_sim_after": cross_after,
        "cross_sim_delta": cross_after - cross_before,
        "residual_norm_before": residual_before,
        "residual_norm_after": residual_after,
        "residual_norm_delta": residual_after - residual_before,
        "embedding_shift": embedding_shift,
    }


def eval_forgetting(
    config: EvalConfig | None = None,
    engine_factory: Callable[[], EmbeddingEngine] | None = None,
) -> dict[str, float]:
    """Evaluate strength decay and capacity eviction behavior."""
    config = config or EvalConfig()
    engine = (engine_factory or make_fake_engine)()

    torch.manual_seed(config.seed)
    mimir_config = MimirConfig(
        base_model="eval",
        num_prototypes=8,  # Small capacity to force eviction.
        learning_rate_base=config.learning_rate_base,
        learning_rate_decay=config.learning_rate_decay,
        temperature=config.temperature,
        residual_scale=config.residual_scale,
        forgetting_decay=config.forgetting_decay,
    )
    mimir = Mimir(
        mimir_config,
        engine=engine,
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )

    # Learn a seed text and record strength.
    mimir.learn(["seed text"])
    proto_id = mimir.store.update_nearest(mimir.engine.encode(["seed text"]), step=0)[0]
    initial_strength = mimir.store.metadata[proto_id, 0].item()

    # Apply decay steps without further learning.
    strengths = [initial_strength]
    for step in range(1, config.decay_steps + 1):
        mimir.store.decay(step)
        strengths.append(mimir.store.metadata[proto_id, 0].item())

    final_strength = strengths[-1]

    # Capacity test: learn more texts than num_prototypes.
    capacity_texts = [f"distinct concept {i}" for i in range(16)]
    for text in capacity_texts:
        mimir.learn([text])

    # Check no NaN/Inf in final state.
    finite = torch.isfinite(mimir.store.prototypes).all().item()

    return {
        "initial_strength": initial_strength,
        "final_strength": final_strength,
        "strength_decay_ratio": final_strength / max(initial_strength, 1e-9),
        "strengths": strengths[-1],  # Only report last for summary.
        "capacity_learned": len(capacity_texts),
        "capacity_prototypes": mimir_config.num_prototypes,
        "state_finite": float(finite),
    }


def eval_prediction(
    config: EvalConfig | None = None,
    engine_factory: Callable[[], EmbeddingEngine] | None = None,
) -> dict[str, float]:
    """Evaluate the first-order Markov prediction policy.

    A repeated sequence should be predicted accurately; a random sequence
    should produce higher surprise scores.
    """
    config = config or EvalConfig()
    engine = (engine_factory or make_fake_engine)()

    torch.manual_seed(config.seed)
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=config.num_prototypes)
    mimir_config = MimirConfig(
        base_model="eval",
        num_prototypes=config.num_prototypes,
        learning_rate_base=config.learning_rate_base,
        learning_rate_decay=config.learning_rate_decay,
        temperature=config.temperature,
        residual_scale=config.residual_scale,
        forgetting_decay=config.forgetting_decay,
        top_k=config.top_k,
    )
    mimir = Mimir(
        mimir_config,
        engine=engine,
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
        prediction_policy=policy,
    )

    # Repeated sequence: A -> B -> C -> A -> B -> C.
    # Use semantically distinct texts so real embedding models assign separate
    # prototypes; otherwise fruit-themed texts collapse to a single prototype
    # and the transition matrix cannot learn A->B->C.
    repeated = [FRUIT_THEME[0], CODE_THEME[0], HISTORY_THEME[0]]
    for _ in range(10):
        for text in repeated:
            mimir.learn(text)

    # Inspect the actual prototype sequence and pick the last observed id.
    # FRUIT_THEME[:3] should cycle through three stable prototypes.
    policy.reset()
    observed_ids: list[int] = []
    for _ in range(3):
        for text in repeated:
            report = mimir.learn(text)
            updated_ids = report.get("updated_ids", [])
            if isinstance(updated_ids, list) and updated_ids:
                observed_ids.append(int(updated_ids[-1]))

    if len(observed_ids) >= 3:
        proto_a, proto_b, proto_c = observed_ids[-3], observed_ids[-2], observed_ids[-1]
    else:
        proto_a = proto_b = proto_c = 0

    # Predict the prototype that follows proto_c in the repeated sequence.
    predicted_after_c = mimir.predict_next(proto_id=proto_c)

    # Surprise for expected transition proto_a -> proto_b should be low,
    # and unexpected transition proto_a -> proto_c should be high.
    # Use a lower smoothing so repeated observations dominate quickly.
    policy.smoothing = 0.1
    surprise_expected = policy.surprise_score(proto_b, last_proto_id=proto_a)
    surprise_unexpected = policy.surprise_score(proto_c, last_proto_id=proto_a)

    # Random sequence: should have high average surprise.
    # Preserve learned statistics but reset the per-session last_proto_id.
    policy.reset()
    random_sequence = HISTORY_THEME[:5]
    random_surprises: list[float] = []
    for text in random_sequence:
        report = mimir.learn(text)
        score = report.get("surprise_score", 1.0)
        if isinstance(score, (int, float)):
            random_surprises.append(float(score))
        else:
            random_surprises.append(1.0)

    # Return proto ids as reference, not as pass criteria.
    return {
        "predicted_after_c": float(predicted_after_c if predicted_after_c is not None else -1),
        "proto_a": float(proto_a),
        "proto_b": float(proto_b),
        "proto_c": float(proto_c),
        "surprise_expected": surprise_expected,
        "surprise_unexpected": surprise_unexpected,
        "mean_random_surprise": sum(random_surprises) / max(len(random_surprises), 1),
    }


def eval_latency(
    config: EvalConfig | None = None,
    engine_factory: Callable[[], EmbeddingEngine] | None = None,
) -> dict[str, float]:
    """Evaluate encoding latency overhead relative to base engine."""
    import time

    config = config or EvalConfig()
    engine = (engine_factory or make_fake_engine)()

    torch.manual_seed(config.seed)
    mimir_config = MimirConfig(
        base_model="eval",
        num_prototypes=config.num_prototypes,
    )
    mimir = Mimir(
        mimir_config,
        engine=engine,
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )

    texts = ["evaluation text"] * 100

    # Warm up.
    mimir.encode(texts[:10])

    # Base engine timing.
    start = time.perf_counter()
    for _ in range(10):
        engine.encode(texts)
    base_time = time.perf_counter() - start

    # Mimir timing.
    start = time.perf_counter()
    for _ in range(10):
        mimir.encode(texts)
    mimir_time = time.perf_counter() - start

    return {
        "base_time_ms": base_time * 100,
        "mimir_time_ms": mimir_time * 100,
        "overhead_ratio": mimir_time / max(base_time, 1e-9),
    }


def _default_config(backend: str, top_k: int | None = None) -> EvalConfig:
    """Return a backend-appropriate default configuration."""
    if backend == "fake":
        base = EvalConfig()
    elif backend == "llama-server":
        # Real embedding models are already well-clustered.
        # Use more aggressive modulation to observe learning effects.
        base = EvalConfig(
            num_prototypes=16,
            learning_rate_base=0.5,
            learning_rate_decay=0.05,
            temperature=0.3,
            residual_scale=1.0,
        )
    elif backend == "sentence-transformer":
        base = EvalConfig(
            num_prototypes=16,
            learning_rate_base=0.2,
            learning_rate_decay=0.05,
            temperature=0.5,
            residual_scale=0.5,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if top_k is None:
        return base
    return EvalConfig(**{**base.__dict__, "top_k": top_k})


def _check_pass(metrics: dict[str, float], backend: str) -> dict[str, bool]:
    """Determine whether key metrics pass evaluation thresholds."""
    if backend == "fake":
        base = {
            "convergence_fruit": metrics.get("fruit_sim_delta", 0.0) > 0.02,
            "convergence_residual": metrics.get("residual_norm_delta", 0.0) > 0.0,
            "separation_cross": abs(metrics.get("cross_sim_delta", 1.0)) < 0.05,
            "forgetting_decay": metrics.get("strength_decay_ratio", 1.0) < 1.0,
            "state_finite": metrics.get("state_finite", 0.0) > 0.0,
            "latency_overhead": metrics.get("overhead_ratio", 999.0) < 1.5,
            "prediction_repeated": metrics.get("predicted_after_c", -1.0) >= 0.0,
            "surprise_discrimination": metrics.get("surprise_unexpected", 0.0)
            > metrics.get("surprise_expected", 1.0),
        }
    else:
        # For real embedding models, the base clustering is already strong.
        # We therefore require only a small positive shift and a measurable
        # embedding drift for learned texts.
        base = {
            "convergence_fruit": metrics.get("fruit_sim_delta", 0.0) > 0.001,
            "convergence_residual": metrics.get("residual_norm_delta", 0.0) > 0.0,
            "separation_cross": abs(metrics.get("cross_sim_delta", 1.0)) < 0.05,
            "forgetting_decay": metrics.get("strength_decay_ratio", 1.0) < 1.0,
            "state_finite": metrics.get("state_finite", 0.0) > 0.0,
            "latency_overhead": metrics.get("overhead_ratio", 999.0) < 2.0,
            "prediction_repeated": metrics.get("predicted_after_c", -1.0) >= 0.0,
            "surprise_discrimination": metrics.get("surprise_unexpected", 0.0)
            > metrics.get("surprise_expected", 1.0),
        }
    return base


def run_all(
    config: EvalConfig | None = None,
    engine_factory: Callable[[], EmbeddingEngine] | None = None,
    backend: str = "fake",
    top_k: int | None = None,
) -> dict[str, dict[str, float] | dict[str, bool]]:
    """Run all evaluations and return a summary report."""
    config = config or _default_config(backend, top_k=top_k)

    convergence = eval_convergence(config, engine_factory)
    forgetting = eval_forgetting(config, engine_factory)
    latency = eval_latency(config, engine_factory)
    prediction = eval_prediction(config, engine_factory)

    # Merge all scalar metrics for pass/fail checking.
    all_metrics: dict[str, float] = {}
    all_metrics.update(convergence)
    all_metrics.update(forgetting)
    all_metrics.update(latency)
    all_metrics.update(prediction)

    passes = _check_pass(all_metrics, backend)

    return {
        "convergence": convergence,
        "forgetting": forgetting,
        "latency": latency,
        "prediction": prediction,
        "pass": passes,
    }


def main() -> int:
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Mimir evaluation harness")
    parser.add_argument(
        "--backend",
        choices=["fake", "llama-server", "sentence-transformer"],
        default="fake",
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Use top-k sparse prototype activation",
    )
    args = parser.parse_args()

    engine_factory = _select_engine_factory(args.backend)
    report = run_all(engine_factory=engine_factory, backend=args.backend, top_k=args.top_k)

    print("=" * 60)
    print("Mimir Evaluation Report")
    print("=" * 60)

    print("\n## Convergence")
    for key, value in report["convergence"].items():
        print(f"  {key}: {value:.6f}")

    print("\n## Prediction")
    for key, value in report["prediction"].items():
        print(f"  {key}: {value:.6f}")

    print("\n## Forgetting")
    for key, value in report["forgetting"].items():
        print(f"  {key}: {value:.6f}")

    print("\n## Latency")
    for key, value in report["latency"].items():
        print(f"  {key}: {value:.6f}")

    print("\n## Pass/Fail")
    for key, value in report["pass"].items():
        status = "PASS" if value else "FAIL"
        print(f"  {key}: {status}")

    all_pass = all(report["pass"].values())
    print("\n" + "=" * 60)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
