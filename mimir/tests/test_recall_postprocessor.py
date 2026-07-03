"""Tests for the recall post-processing pipeline."""

from datetime import datetime, timedelta, timezone

import pytest

from mimir.domain.model import Memory
from mimir.infrastructure.retrieval.recall_postprocessor import (
    PostprocessorConfig,
    RecallPostprocessor,
)


def _memory(
    text: str,
    embedding: list[float] | None = None,
    created_at: datetime | None = None,
) -> Memory:
    return Memory(
        text=text,
        embedding=embedding or [0.0] * 8,
        score=0.0,
        created_at=created_at or datetime.now(timezone.utc),
    )


class TestRecallPostprocessorBasics:
    def test_empty_input_returns_empty(self) -> None:
        pp = RecallPostprocessor()
        assert pp.process([], {}) == []

    def test_unknown_ranking_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ranking_mode"):
            RecallPostprocessor(config=PostprocessorConfig(ranking_mode="magic"))

    def test_dedup_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="dedup_threshold"):
            RecallPostprocessor(config=PostprocessorConfig(dedup_threshold=1.5))

    def test_lifecycle_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="lifecycle_weight"):
            RecallPostprocessor(config=PostprocessorConfig(lifecycle_weight=-0.1))

    def test_max_candidates_for_clustering_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="max_candidates_for_clustering"):
            RecallPostprocessor(config=PostprocessorConfig(max_candidates_for_clustering=0))

    def test_lifecycle_weight_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="lifecycle_weight"):
            RecallPostprocessor(config=PostprocessorConfig(lifecycle_weight=1.5))

    def test_single_memory_returns_unchanged(self) -> None:
        memory = _memory("hello", embedding=[1.0, 0.0, 0.0])
        pp = RecallPostprocessor()
        result = pp.process([memory], {0: 0.8})
        assert len(result) == 1
        assert result[0].text == "hello"
        assert result[0].score == pytest.approx(0.8)


class TestSemanticClustering:
    def test_clusters_near_duplicates(self) -> None:
        """Memories with nearly identical embeddings should collapse to one."""
        base = [1.0, 0.0, 0.0, 0.0]
        memories = [
            _memory("remember to use Python", embedding=base),
            _memory("use Python for this task", embedding=[0.99, 0.14, 0.0, 0.0]),
            _memory("unrelated gardening tip", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        retrieval = {0: 0.7, 1: 0.65, 2: 0.5}
        pp = RecallPostprocessor(config=PostprocessorConfig(dedup_threshold=0.9))
        result = pp.process(memories, retrieval)

        texts = {m.text for m in result}
        assert len(result) == 2
        assert "unrelated gardening tip" in texts
        assert len([m for m in result if "Python" in m.text]) == 1

    def test_representative_is_highest_scoring_member(self) -> None:
        """The representative of a cluster should be the highest-scoring member."""
        memories = [
            _memory("low", embedding=[1.0, 0.0]),
            _memory("high", embedding=[0.99, 0.14]),
        ]
        retrieval = {0: 0.5, 1: 0.9}
        pp = RecallPostprocessor(config=PostprocessorConfig(dedup_threshold=0.9))
        result = pp.process(memories, retrieval)

        assert len(result) == 1
        assert result[0].text == "high"

    def test_disabled_clustering_with_threshold_one(self) -> None:
        memories = [
            _memory("a", embedding=[1.0, 0.0]),
            _memory("b", embedding=[0.99, 0.14]),
        ]
        retrieval = {0: 0.8, 1: 0.7}
        pp = RecallPostprocessor(config=PostprocessorConfig(dedup_threshold=1.0))
        result = pp.process(memories, retrieval)

        assert len(result) == 2

    def test_max_candidates_truncates_before_clustering(self) -> None:
        memories = [_memory(f"m{i}") for i in range(10)]
        retrieval = {i: i / 10.0 for i in range(10)}
        pp = RecallPostprocessor(
            config=PostprocessorConfig(max_candidates_for_clustering=3)
        )
        result = pp.process(memories, retrieval)
        # Zero embeddings have zero cosine similarity, so no clustering happens
        # and the top-3 retrieval candidates are returned.
        assert len(result) == 3

    def test_dedup_threshold_zero_clusters_all(self) -> None:
        memories = [
            _memory("a", embedding=[1.0, 0.0]),
            _memory("b", embedding=[0.0, 1.0]),
        ]
        retrieval = {0: 0.8, 1: 0.7}
        pp = RecallPostprocessor(config=PostprocessorConfig(dedup_threshold=0.0))
        result = pp.process(memories, retrieval)

        assert len(result) == 1


class TestMultiplicativeRanking:
    def test_lifecycle_amplifies_retrieval_score(self) -> None:
        memories = [
            _memory("high lifecycle", embedding=[1.0, 0.0]),
            _memory("low lifecycle", embedding=[0.0, 1.0]),
        ]
        retrieval = {0: 0.8, 1: 0.8}
        lifecycle = {0: 10.0, 1: 1.0}
        pp = RecallPostprocessor(
            config=PostprocessorConfig(
                ranking_mode="multiplicative", lifecycle_weight=0.3
            )
        )
        result = pp.process(memories, retrieval, lifecycle)

        assert result[0].text == "high lifecycle"
        assert result[0].score > result[1].score
        # High lifecycle: 0.8 * (1 + 0.3 * 1.0) = 1.04
        # Low lifecycle: 0.8 * (1 + 0.3 * 0.0) = 0.8
        assert result[0].score == pytest.approx(1.04)
        assert result[1].score == pytest.approx(0.8)

    def test_zero_lifecycle_does_not_zero_score(self) -> None:
        memory = _memory("only", embedding=[1.0, 0.0])
        pp = RecallPostprocessor(
            config=PostprocessorConfig(
                ranking_mode="multiplicative", lifecycle_weight=0.3
            )
        )
        result = pp.process([memory], {0: 0.5}, {0: 0.0})
        assert result[0].score == pytest.approx(0.5)

    def test_additive_mode_is_linear_blend(self) -> None:
        memories = [
            _memory("high lifecycle", embedding=[1.0, 0.0]),
            _memory("low lifecycle", embedding=[0.0, 1.0]),
        ]
        retrieval = {0: 0.8, 1: 0.8}
        lifecycle = {0: 10.0, 1: 1.0}
        pp = RecallPostprocessor(
            config=PostprocessorConfig(
                ranking_mode="additive", lifecycle_weight=0.3
            )
        )
        result = pp.process(memories, retrieval, lifecycle)

        # normalized lifecycle: high=1.0, low=0.0
        # additive: 0.7*retrieval + 0.3*lifecycle
        assert result[0].score == pytest.approx(0.7 * 0.8 + 0.3 * 1.0)
        assert result[1].score == pytest.approx(0.7 * 0.8 + 0.3 * 0.0)

    def test_retrieval_order_with_lifecycle_boost(self) -> None:
        """A lower-retrieval but very fresh/important memory can overtake."""
        now = datetime.now(timezone.utc)
        memories = [
            _memory(
                "old but relevant",
                embedding=[1.0, 0.0],
                created_at=now - timedelta(days=60),
            ),
            _memory(
                "recent context",
                embedding=[0.0, 1.0],
                created_at=now,
            ),
        ]
        # Recent context is slightly less relevant.
        retrieval = {0: 0.9, 1: 0.7}
        # Lifecycle scorer weights: recency=1, importance=1, access=0.5.
        # Old memory: recency 0.5 + importance 1.0 = 1.5
        # New memory: recency 1.0 + importance 1.0 = 2.0
        lifecycle = {0: 1.5, 1: 2.0}
        pp = RecallPostprocessor(
            config=PostprocessorConfig(
                ranking_mode="multiplicative", lifecycle_weight=0.5
            )
        )
        result = pp.process(memories, retrieval, lifecycle)

        # With strong lifecycle weight, the recent memory should win.
        assert result[0].text == "recent context"


    def test_process_drops_memories_without_retrieval_scores(self) -> None:
        memories = [
            _memory("scored", embedding=[1.0, 0.0]),
            _memory("unscored", embedding=[0.0, 1.0]),
        ]
        pp = RecallPostprocessor()
        result = pp.process(memories, {0: 0.8})

        assert len(result) == 1
        assert result[0].text == "scored"

    def test_missing_lifecycle_scores_default_to_zero(self) -> None:
        memories = [
            _memory("with lifecycle", embedding=[1.0, 0.0]),
            _memory("without lifecycle", embedding=[0.0, 1.0]),
        ]
        retrieval = {0: 0.8, 1: 0.8}
        lifecycle = {0: 10.0}
        pp = RecallPostprocessor(
            config=PostprocessorConfig(
                ranking_mode="multiplicative", lifecycle_weight=0.3
            )
        )
        result = pp.process(memories, retrieval, lifecycle)

        # Candidate 1 lacks a lifecycle score and should receive no boost.
        assert result[0].score == pytest.approx(0.8 * (1 + 0.3 * 1.0))
        assert result[1].score == pytest.approx(0.8)


class TestNormalize:
    def test_equal_positive_scores_use_full_fallback(self) -> None:
        pp = RecallPostprocessor()
        normalized = pp.normalize_scores({0: 2.0, 1: 2.0})
        assert normalized[0] == 1.0
        assert normalized[1] == 1.0

    def test_all_zero_scores_use_zero_fallback(self) -> None:
        pp = RecallPostprocessor()
        normalized = pp.normalize_scores({0: 0.0, 1: 0.0})
        assert normalized[0] == 0.0
        assert normalized[1] == 0.0

    def test_normalize_scales_to_zero_one(self) -> None:
        pp = RecallPostprocessor()
        normalized = pp.normalize_scores({0: 1.0, 1: 3.0})
        assert normalized[0] == 0.0
        assert normalized[1] == 1.0


class TestEmbeddingSimilarity:
    def test_similarity_matrix_range(self) -> None:
        memories = [
            _memory("same", embedding=[1.0, 0.0, 0.0]),
            _memory("opposite", embedding=[-1.0, 0.0, 0.0]),
            _memory("orthogonal", embedding=[0.0, 1.0, 0.0]),
        ]
        pp = RecallPostprocessor()
        sim = pp.similarity_matrix(memories)
        assert sim[0, 0].item() == pytest.approx(1.0)
        assert sim[0, 1].item() == pytest.approx(-1.0)
        assert sim[0, 2].item() == pytest.approx(0.0)
