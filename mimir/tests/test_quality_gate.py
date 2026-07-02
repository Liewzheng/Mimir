"""Tests for the memory quality gate."""

import pytest

from mimir.infrastructure.quality_gate import QualityGate


@pytest.fixture
def gate() -> QualityGate:
    return QualityGate(duplicate_threshold=0.95, contradiction_threshold=0.85)


class TestDuplicateDetection:
    def test_no_existing_memories(self, gate: QualityGate) -> None:
        result = gate.check_duplicate(
            "I like Python", [1.0, 0.0], [], []
        )
        assert result.ok

    def test_exact_duplicate_blocked(self, gate: QualityGate) -> None:
        text = "I like Python"
        embedding = [1.0, 0.0]
        result = gate.check_duplicate(
            text,
            embedding,
            [text],
            [embedding],
        )
        assert not result.ok
        assert result.reason == "duplicate"
        assert result.similar_memory == text

    def test_different_memories_allowed(self, gate: QualityGate) -> None:
        result = gate.check_duplicate(
            "I like Python",
            [1.0, 0.0],
            ["I prefer TypeScript"],
            [[0.0, 1.0]],
        )
        assert result.ok

    def test_similar_but_not_duplicate(self, gate: QualityGate) -> None:
        result = gate.check_duplicate(
            "I like Python",
            [0.707, 0.707],
            ["I like Python"],
            [[1.0, 0.0]],
        )
        assert result.ok


class TestContradictionHints:
    def test_direct_negation(self, gate: QualityGate) -> None:
        hints = gate.find_contradictions(
            ["I use Python", "I don't use Python"]
        )
        assert len(hints) == 1

    def test_no_contradiction_similar_subject(self, gate: QualityGate) -> None:
        hints = gate.find_contradictions(
            ["I use Python", "I use TypeScript"]
        )
        assert len(hints) == 0

    def test_opposite_polarity_shared_terms(self, gate: QualityGate) -> None:
        hints = gate.find_contradictions(
            [
                "The backend is running on port 8080",
                "The backend is not running on port 8080",
            ]
        )
        assert len(hints) == 1

    def test_empty_list(self, gate: QualityGate) -> None:
        assert gate.find_contradictions([]) == []

    def test_single_memory(self, gate: QualityGate) -> None:
        assert gate.find_contradictions(["Only one memory"]) == []
