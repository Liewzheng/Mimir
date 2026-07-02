"""Tests for the shared embedding engine factory."""

import pytest

from mimir.infrastructure.embedding import create_engine
from mimir.infrastructure.embedding.fake_engine import FakeEngine
from mimir.infrastructure.embedding.llama_server_engine import (
    LlamaServerEmbeddingEngine,
)


def test_create_engine_llama_server() -> None:
    engine = create_engine("llama-server", base_url="http://127.0.0.1:9999")
    assert isinstance(engine, LlamaServerEmbeddingEngine)


def test_create_engine_fake() -> None:
    engine = create_engine("fake", fake_dim=8)
    assert isinstance(engine, FakeEngine)
    assert engine.output_dim == 8


def test_create_engine_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Supported backends"):
        create_engine("unknown")


def test_create_engine_sentence_transformer_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting sentence-transformer without the package raises ImportError."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if "sentence_transformer_engine" in name:
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match="sentence-transformers"):
        create_engine("sentence-transformer", model="all-MiniLM-L6-v2")
