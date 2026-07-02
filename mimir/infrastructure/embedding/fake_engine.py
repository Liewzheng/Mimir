"""Shared fake embedding engine for tests and evaluation."""

import hashlib

import torch

from mimir.domain.model.engine import EmbeddingEngine


class FakeEngine(EmbeddingEngine):
    """Deterministic embedding engine with theme-aware correlations.

    Each text is embedded as a weighted combination of:
      - a shared theme vector (based on which theme the text belongs to)
      - a per-text noise vector

    This ensures that texts within the same theme have high cosine similarity,
    while texts from different themes are less similar.
    """

    def __init__(self, dim: int = 16) -> None:
        self.output_dim = dim
        self._device = "cpu"
        self._theme_vectors: dict[str, torch.Tensor] = {}

    def _seed_for(self, text: str) -> int:
        """Return a stable integer seed for ``text``.

        ``hash()`` is randomized per Python process, which makes tests flaky
        across runs.  Use SHA-256 instead so embeddings are reproducible.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def set_theme(self, theme: str, texts: list[str]) -> None:
        """Assign a deterministic shared direction to a theme."""
        torch.manual_seed(self._seed_for(theme))
        vec = torch.randn(self.output_dim)
        vec = vec / torch.linalg.norm(vec)
        self._theme_vectors[theme] = vec
        # Also register each text so encode can resolve theme.
        for text in texts:
            self._theme_vectors[text] = vec

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Return theme-aware deterministic embeddings."""
        vectors = []
        for text in texts:
            theme_vec = self._theme_vectors.get(text)
            if theme_vec is None:
                # Fallback for unseen texts.
                torch.manual_seed(self._seed_for(text))
                theme_vec = torch.randn(self.output_dim)
                theme_vec = theme_vec / torch.linalg.norm(theme_vec)

            torch.manual_seed((self._seed_for(text) + 1) % (2**31))
            noise = torch.randn(self.output_dim) * 0.3
            vec = theme_vec + noise
            vec = vec / torch.linalg.norm(vec)
            vectors.append(vec)
        return torch.stack(vectors)


def make_fake_engine(dim: int = 16) -> FakeEngine:
    """Create a FakeEngine preloaded with evaluation themes."""
    from mimir.eval_data import CODE_THEME, FRUIT_THEME, HISTORY_THEME

    engine = FakeEngine(dim=dim)
    engine.set_theme("fruit", FRUIT_THEME)
    engine.set_theme("code", CODE_THEME)
    engine.set_theme("history", HISTORY_THEME)
    return engine
