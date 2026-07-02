"""Tests for secret redaction."""

import pytest

from mimir.infrastructure.redaction import Redactor


@pytest.fixture
def redactor() -> Redactor:
    return Redactor()


class TestRedactor:
    def test_openai_api_key(self, redactor: Redactor) -> None:
        text = "My OpenAI key is sk-abc123def456ghi789jkl012mnop345qrst678uvw"
        result = redactor.redact(text)
        assert "sk-abc123" not in result
        assert "[REDACTED]" in result

    def test_github_token(self, redactor: Redactor) -> None:
        text = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = redactor.redact(text)
        assert "ghp_xxx" not in result
        assert "[REDACTED]" in result

    def test_jwt(self, redactor: Redactor) -> None:
        text = "token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = redactor.redact(text)
        assert "eyJhbGci" not in result
        assert "[REDACTED]" in result

    def test_password(self, redactor: Redactor) -> None:
        text = "password=super_secret_123"
        result = redactor.redact(text)
        assert "super_secret_123" not in result
        assert "[REDACTED]" in result

    def test_url_credentials(self, redactor: Redactor) -> None:
        text = "https://user:pass123@github.com/repo.git"
        result = redactor.redact(text)
        assert "pass123" not in result
        assert "[REDACTED]" in result

    def test_env_secret_assignment(self, redactor: Redactor) -> None:
        text = "MY_API_TOKEN=abcd1234efgh5678"
        result = redactor.redact(text)
        assert "abcd1234efgh5678" not in result
        assert "[REDACTED]" in result

    def test_bearer_header(self, redactor: Redactor) -> None:
        text = "Authorization: Bearer abc.def.ghi"
        result = redactor.redact(text)
        assert "abc.def.ghi" not in result
        assert "[REDACTED]" in result

    def test_innocuous_text_unchanged(self, redactor: Redactor) -> None:
        text = "I decided to use Python for the backend."
        assert redactor.redact(text) == text

    def test_disabled_redactor(self) -> None:
        redactor = Redactor(enabled=False)
        text = "password=secret123"
        assert redactor.redact(text) == text

    def test_redact_messages(self, redactor: Redactor) -> None:
        messages = [
            {"role": "user", "content": "my key is sk-1234567890abcdef"},
            {"role": "assistant", "content": "ok"},
        ]
        result = redactor.redact_messages(messages)
        assert "sk-1234567890abcdef" not in result[0]["content"]
        assert result[0]["content"] != messages[0]["content"]
        assert result[1]["content"] == "ok"
