"""Direct llama.cpp server embedding engine."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import torch


class InputTooLongError(RuntimeError):
    """Raised when a single llama-server embedding request is too large."""


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
        max_workers: int = 32,
        max_input_tokens: int = 1024,
        chars_per_token: int = 4,
    ) -> None:
        self.model_path = model_path
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_workers = max_workers
        self._max_input_tokens = max_input_tokens
        self._chars_per_token = chars_per_token
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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 500:
                raise InputTooLongError(
                    f"llama-server request too large for {url}: {exc}"
                ) from exc
            raise RuntimeError(f"llama-server request failed for {url}: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"llama-server returned invalid JSON: {exc}") from exc

    def _post_and_extract(
        self,
        text: str,
        min_chunk_chars: int = 64,
    ) -> list[float]:
        """Send a single embedding request and normalize the response.

        If the server rejects the request as too large, the text is recursively
        split in half and the resulting chunk embeddings are averaged.
        """
        try:
            data = self._post(text)
        except InputTooLongError:
            if len(text) <= min_chunk_chars:
                raise
            mid = len(text) // 2
            left = self._post_and_extract(text[:mid], min_chunk_chars)
            right = self._post_and_extract(text[mid:], min_chunk_chars)
            return [(a + b) / 2.0 for a, b in zip(left, right, strict=True)]
        return self._extract_embedding(data)

    def _split_text(self, text: str) -> list[str]:
        """Split a long text into chunks that fit within the server's batch limit.

        llama-server's /embedding endpoint treats the input as a single batch of
        tokens; if the text exceeds the server's physical batch size the request
        fails.  We approximate tokens by characters and send shorter chunks, then
        average the resulting chunk embeddings for the original text.
        """
        max_chunk_chars = self._max_input_tokens * self._chars_per_token
        if len(text) <= max_chunk_chars:
            return [text]
        return [text[i : i + max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]

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

        llama-server /embedding endpoint processes one text per request.  We send
        requests concurrently so that embedding a large session does not become
        bottlenecked by network latency.  Long inputs are transparently chunked
        and their chunk embeddings are mean-pooled to respect the server's batch
        limits.
        """
        chunk_texts: list[str] = []
        chunk_map: list[int] = []
        for idx, text in enumerate(texts):
            chunks = self._split_text(text)
            chunk_texts.extend(chunks)
            chunk_map.extend([idx] * len(chunks))

        chunk_embeddings: list[list[float] | None] = [None] * len(chunk_texts)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self._post_and_extract, text): idx
                for idx, text in enumerate(chunk_texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk_embeddings[idx] = future.result()

        # Aggregate chunk embeddings for each original text by mean pooling.
        grouped: defaultdict[int, list[torch.Tensor]] = defaultdict(list)
        for idx, emb in zip(chunk_map, chunk_embeddings, strict=False):
            grouped[idx].append(torch.tensor(emb, dtype=torch.float32))

        embeddings: list[torch.Tensor] = []
        for idx in range(len(texts)):
            chunk_tensors = grouped[idx]
            if not chunk_tensors:
                raise RuntimeError(f"No embeddings produced for text {idx}")
            mean = torch.stack(chunk_tensors).mean(dim=0)
            embeddings.append(mean)

        tensor = torch.stack(embeddings)
        if self.output_dim == 0:
            self.output_dim = tensor.shape[1]
        return tensor
