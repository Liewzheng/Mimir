"""SentenceTransformer-based slow weights engine."""

from typing import Any, Protocol

import torch


class _SentenceTransformerModel(Protocol):
    """Minimal protocol for the sentence-transformer model we use."""

    def get_sentence_embedding_dimension(self) -> int: ...

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_tensor: bool,
        show_progress_bar: bool,
    ) -> Any: ...


class SentenceTransformerEngine:
    """Load and run a sentence-transformer model as the slow weights."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        """Initialize the sentence-transformer engine.

        Args:
            model_name: Hugging Face model name or local path.
            device: Target device ("auto", "cuda", "mps", or "cpu").
        """
        self.model_name = model_name
        self._device = self._resolve_device(device)
        self._model: _SentenceTransformerModel | None = None
        self.output_dim = 0

    def _resolve_device(self, device: str) -> str:
        """Resolve "auto" to the best available torch device."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        """Lazy-load the underlying model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEngine"
            ) from exc

        from typing import cast

        model = SentenceTransformer(self.model_name, device=self._device)
        self._model = cast(_SentenceTransformerModel, model)
        self.output_dim = model.get_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return base embeddings for the given texts.

        Args:
            texts: List of input strings.
            batch_size: Encoding batch size.

        Returns:
            Tensor of shape [len(texts), output_dim].
        """
        self._load()
        if self._model is None:
            raise RuntimeError("Model failed to load")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return torch.as_tensor(embeddings).cpu().float()
