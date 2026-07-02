"""Ollama embedding engine."""

import os
from typing import Any

import torch


class OllamaEmbeddingEngine:
    """Use a local Ollama model as the slow weights.

    Expects Ollama to be running locally (default http://localhost:11434).
    Set OLLAMA_HOST to override the base URL.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client: Any | None = None
        self.output_dim = 0

    def _load(self) -> None:
        """Lazy-load the Ollama client."""
        if self._client is not None:
            return
        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "ollama is required for OllamaEmbeddingEngine. Install with: pip install ollama"
            ) from exc

        self._client = ollama.Client(host=self._base_url)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return base embeddings via Ollama embeddings API.

        Args:
            texts: List of input strings.
            batch_size: Ignored (Ollama processes one at a time here).

        Returns:
            Tensor of shape [len(texts), output_dim].
        """
        self._load()
        if self._client is None:
            raise RuntimeError("Ollama client failed to initialize")

        embeddings: list[list[float]] = []
        try:
            for text in texts:
                response = self._client.embeddings(model=self.model, prompt=text)
                embeddings.append(response["embedding"])
        except (ConnectionError, TimeoutError, RuntimeError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

        tensor = torch.tensor(embeddings, dtype=torch.float32)
        if self.output_dim == 0:
            self.output_dim = tensor.shape[1]
        return tensor
