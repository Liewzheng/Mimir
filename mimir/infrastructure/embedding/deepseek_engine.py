"""DeepSeek API embedding engine."""

import os
from typing import Any, Protocol

import torch


class _OpenAIClient(Protocol):
    """Minimal protocol for the OpenAI client we use."""

    class _Embeddings:
        def create(
            self,
            *,
            model: str,
            input: list[str],
            encoding_format: str,
        ) -> Any: ...

    embeddings: _Embeddings


class DeepSeekEmbeddingEngine:
    """Use DeepSeek's embedding API as the slow weights.

    Expects the following environment variables:
      - DEEPSEEK_API_KEY
      - DEEPSEEK_BASE_URL (optional, defaults to https://api.deepseek.com)

    The model name follows the OpenAI-compatible embedding endpoint.
    As of DeepSeek docs, use "deepseek-embedding" or the model name provided
    by the platform.
    """

    def __init__(
        self,
        model: str = "deepseek-embedding",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the DeepSeek embedding engine.

        Args:
            model: DeepSeek embedding model name.
            api_key: Optional API key; falls back to DEEPSEEK_API_KEY.
            base_url: Optional API base URL; falls back to DEEPSEEK_BASE_URL.
        """
        self.model = model
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._client: _OpenAIClient | None = None
        self.output_dim = 0

        if not self._api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY or pass api_key.")

    def _load(self) -> None:
        """Lazy-load the OpenAI client."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for DeepSeekEmbeddingEngine. "
                "Install with: pip install mimir[api]"
            ) from exc

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)  # type: ignore[assignment]

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return base embeddings via DeepSeek embedding API.

        Args:
            texts: List of input strings.
            batch_size: Ignored for API calls (single request).

        Returns:
            Tensor of shape [len(texts), output_dim].
        """
        self._load()
        if self._client is None:
            raise RuntimeError("DeepSeek client failed to initialize")

        try:
            from openai import APIConnectionError, APIError, APITimeoutError
        except ImportError:
            api_error_types: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
        else:
            api_error_types = (APIError, APIConnectionError, APITimeoutError, ConnectionError, TimeoutError)

        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
            )
        except api_error_types as exc:
            raise RuntimeError(f"DeepSeek embedding request failed: {exc}") from exc

        embeddings = [item.embedding for item in response.data]
        tensor = torch.tensor(embeddings, dtype=torch.float32)

        if self.output_dim == 0:
            self.output_dim = tensor.shape[1]

        return tensor
