"""Shared factory for creating embedding engines by name."""

from __future__ import annotations

from mimir.domain.model.engine import EmbeddingEngine
from mimir.infrastructure.embedding.fake_engine import FakeEngine
from mimir.infrastructure.embedding.llama_server_engine import (
    LlamaServerEmbeddingEngine,
)
from mimir.infrastructure.embedding.ollama_engine import OllamaEmbeddingEngine


def create_engine(
    backend: str,
    base_url: str = "http://127.0.0.1:11435",
    model: str = "all-MiniLM-L6-v2",
    fake_dim: int = 16,
) -> EmbeddingEngine:
    """Create and return an embedding engine by backend name.

    Args:
        backend: One of ``llama-server``, ``sentence-transformer``, ``ollama``,
            or ``fake``.
        base_url: URL for the llama-server backend. For ``ollama`` this is the
            Ollama host URL (default ``http://localhost:11434`` is used if not
            provided via argument or ``OLLAMA_HOST`` environment variable).
        model: Model name for the sentence-transformer or ollama backend.
        fake_dim: Dimension for the fake backend (for testing only).

    Returns:
        A configured embedding engine.

    Raises:
        ValueError: If the backend name is unknown.
        ImportError: If ``sentence-transformer`` is selected but the package
            is not installed.
    """
    if backend == "llama-server":
        return LlamaServerEmbeddingEngine(base_url=base_url)
    if backend == "sentence-transformer":
        try:
            from mimir.infrastructure.embedding.sentence_transformer_engine import (
                SentenceTransformerEngine,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'sentence-transformer' backend requires the 'sentence-transformers' package. "
                'Install it with: pip install "mimir[server]"'
            ) from exc
        return SentenceTransformerEngine(model)
    if backend == "ollama":
        return OllamaEmbeddingEngine(model=model, base_url=base_url)
    if backend == "fake":
        return FakeEngine(dim=fake_dim)
    raise ValueError(
        f"Unknown backend: {backend!r}. "
        "Supported backends: llama-server, sentence-transformer, ollama, fake"
    )
