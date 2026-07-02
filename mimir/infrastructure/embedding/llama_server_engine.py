"""Direct llama.cpp server embedding engine."""

from typing import Any

import httpx
import torch


class LlamaServerEmbeddingEngine:
    """Connect to a running llama-server with --embeddings enabled.

    This bypasses Ollama's orchestration and talks directly to the underlying
    llama-server HTTP API. Useful when Ollama app itself does not expose
    embeddings.

    Expected endpoint:
        POST http://{host}:{port}/embedding
        Body: {"content": "text to embed"}
        Response formats handled:
          - {"embedding": [v1, v2, ...]}
          - [v1, v2, ...]
          - [{"index": 0, "embedding": [[tok1...], [tok2...]]}]  (per-token)
    """

    def __init__(
        self,
        model_path: str | None = None,
        base_url: str = "http://127.0.0.1:11435",
        timeout: float = 120.0,
    ) -> None:
        self.model_path = model_path
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.output_dim = 0

    def _post(self, text: str) -> Any:
        """Send a single embedding request."""
        url = f"{self.base_url}/embedding"
        try:
            response = httpx.post(
                url,
                json={"content": text},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"llama-server request failed for {url}: {exc}") from exc
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"llama-server returned invalid JSON: {exc}") from exc

    @staticmethod
    def _extract_embedding(data: Any) -> list[float]:
        """Normalize various llama-server response shapes to a 1-D vector."""
        if isinstance(data, dict):
            data = data.get("embedding", [])

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # [{"index": 0, "embedding": [[tok1...], [tok2...]]}]
            token_embeddings = data[0].get("embedding", [])
            if token_embeddings and isinstance(token_embeddings[0], list):
                # Mean pool per-token embeddings.
                tensor = torch.tensor(token_embeddings, dtype=torch.float32)
                pooled = tensor.mean(dim=0)
                return pooled.tolist()
            result: list[float] = token_embeddings
            return result

        if isinstance(data, list) and data and isinstance(data[0], list):
            # [[tok1...], [tok2...]]
            tensor = torch.tensor(data, dtype=torch.float32)
            return tensor.mean(dim=0).tolist()

        if isinstance(data, list):
            return data

        raise ValueError(f"Unexpected embedding response: {type(data)}")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return embeddings for the given texts.

        llama-server /embedding endpoint processes one text per request.
        We send requests sequentially here for simplicity.
        """
        embeddings: list[list[float]] = []
        for text in texts:
            data = self._post(text)
            embedding = self._extract_embedding(data)
            embeddings.append(embedding)

        tensor = torch.tensor(embeddings, dtype=torch.float32)
        if self.output_dim == 0:
            self.output_dim = tensor.shape[1]
        return tensor
