"""Ollama LLM reader and judge for evaluation harnesses."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


class OllamaReader:
    """Use a local Ollama chat model as a reader and judge.

    Expects Ollama to be running at ``http://localhost:11434`` by default.
    Set ``OLLAMA_HOST`` or pass ``base_url`` to override.
    """

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = httpx.Client(timeout=300.0)

    def _chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Call Ollama /api/chat and return the assistant content."""
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "options": {"temperature": temperature},
                "stream": False,
            },
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        message = data.get("message", {})
        return str(message.get("content", ""))

    def answer(self, context: str, question: str, temperature: float = 0.0) -> str:
        """Generate an answer using the retrieved context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise assistant. Answer the user's question using only "
                    "the provided context. Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        return self._chat(messages, temperature=temperature)

    def judge(self, question: str, reference: str, prediction: str, temperature: float = 0.0) -> bool:
        """Return True if the prediction is semantically correct."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. Given a question, a reference answer, "
                    "and a predicted answer, decide if the prediction is correct. "
                    "Return only JSON with a single boolean field 'correct'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Reference answer: {reference}\n"
                    f"Predicted answer: {prediction}\n\n"
                    "Return JSON: {{\"correct\": true/false}}"
                ),
            },
        ]
        raw = self._chat(messages, temperature=temperature)
        try:
            # Ollama sometimes wraps JSON in markdown fences.
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1].strip("json").strip()
            result = json.loads(cleaned)
            return bool(result.get("correct", False))
        except (json.JSONDecodeError, AttributeError):
            # Fall back to a permissive heuristic.
            return prediction.lower() in reference.lower() or reference.lower() in prediction.lower()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
