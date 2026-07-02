"""Re-export FakeEngine for backward compatibility with tests."""

from mimir.infrastructure.embedding.fake_engine import (
    FakeEngine,
    make_fake_engine,
)

__all__ = ["FakeEngine", "make_fake_engine"]
