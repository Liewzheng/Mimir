"""Tests for optional embedding backends using mocks."""

from unittest import mock

import pytest
import torch


def test_sentence_transformer_engine_encode() -> None:
    """SentenceTransformerEngine delegates to the underlying model."""
    pytest.importorskip("sentence_transformers")
    from mimir.infrastructure.embedding.sentence_transformer_engine import (
        SentenceTransformerEngine,
    )

    engine = SentenceTransformerEngine("all-MiniLM-L6-v2", device="cpu")
    fake_embeddings = torch.randn(2, 384)
    model_mock = mock.MagicMock()
    model_mock.get_sentence_embedding_dimension.return_value = 384
    model_mock.encode.return_value = fake_embeddings
    engine._model = model_mock
    engine.output_dim = 384

    result = engine.encode(["hello", "world"])
    assert result.shape == (2, 384)


def test_deepseek_engine_encode() -> None:
    """DeepSeekEmbeddingEngine calls the OpenAI-compatible embedding endpoint."""
    pytest.importorskip("openai")
    from mimir.infrastructure.embedding.deepseek_engine import (
        DeepSeekEmbeddingEngine,
    )

    engine = DeepSeekEmbeddingEngine(api_key="test-key", base_url="https://test.example")
    response_mock = mock.MagicMock()
    response_mock.data = [
        mock.MagicMock(embedding=[0.1, 0.2, 0.3]),
        mock.MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    client_mock = mock.MagicMock()
    client_mock.embeddings.create.return_value = response_mock
    engine._client = client_mock

    result = engine.encode(["hello", "world"])
    assert result.shape == (2, 3)
    assert engine.output_dim == 3


def test_deepseek_engine_requires_api_key() -> None:
    """DeepSeekEmbeddingEngine raises without an API key."""
    pytest.importorskip("openai")
    from mimir.infrastructure.embedding.deepseek_engine import (
        DeepSeekEmbeddingEngine,
    )

    with (
        mock.patch("os.environ.get", return_value=None),
        pytest.raises(ValueError, match="API key is required"),
    ):
        DeepSeekEmbeddingEngine(api_key=None)


def test_deepseek_engine_wraps_api_errors() -> None:
    """DeepSeekEmbeddingEngine wraps raw API errors in RuntimeError."""
    pytest.importorskip("openai")
    from mimir.infrastructure.embedding.deepseek_engine import (
        DeepSeekEmbeddingEngine,
    )

    engine = DeepSeekEmbeddingEngine(api_key="test-key")
    client_mock = mock.MagicMock()
    client_mock.embeddings.create.side_effect = ConnectionError("timeout")
    engine._client = client_mock

    with pytest.raises(RuntimeError, match="DeepSeek embedding request failed"):
        engine.encode(["hello"])


def test_ollama_engine_encode() -> None:
    """OllamaEmbeddingEngine calls the Ollama embeddings endpoint."""
    pytest.importorskip("ollama")
    from mimir.infrastructure.embedding.ollama_engine import OllamaEmbeddingEngine

    engine = OllamaEmbeddingEngine("mxbai-embed-large")
    client_mock = mock.MagicMock()
    client_mock.embeddings.side_effect = [
        {"embedding": [0.1, 0.2]},
        {"embedding": [0.3, 0.4]},
    ]
    engine._client = client_mock

    result = engine.encode(["hello", "world"])
    assert result.shape == (2, 2)
    assert engine.output_dim == 2


def test_ollama_engine_requires_package() -> None:
    """OllamaEmbeddingEngine raises a helpful error if ollama is missing."""
    from mimir.infrastructure.embedding.ollama_engine import OllamaEmbeddingEngine

    engine = OllamaEmbeddingEngine("mxbai-embed-large")
    with (
        mock.patch.dict("sys.modules", {"ollama": None}),
        pytest.raises(ImportError, match="ollama is required"),
    ):
        engine.encode(["hello"])


def test_ollama_engine_wraps_api_errors() -> None:
    """OllamaEmbeddingEngine wraps raw client errors in RuntimeError."""
    pytest.importorskip("ollama")
    from mimir.infrastructure.embedding.ollama_engine import OllamaEmbeddingEngine

    engine = OllamaEmbeddingEngine("mxbai-embed-large")
    client_mock = mock.MagicMock()
    client_mock.embeddings.side_effect = RuntimeError("service down")
    engine._client = client_mock

    with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
        engine.encode(["hello"])
