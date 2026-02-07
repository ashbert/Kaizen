"""
Tests for OpenAI-compatible LLM provider.

This module tests:
- OpenAICompatProvider configuration
- Successful response parsing (chat/completions and completions formats)
- Error handling (connection, timeout, HTTP errors)
- Protocol and base class compliance

Integration tests that require a running API server are marked
with @pytest.mark.integration and skipped by default.
"""

import os

import pytest

from kaizen.llm import (
    LLMProvider,
    LLMProviderProtocol,
    LLMError,
    OpenAICompatProvider,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestOpenAICompatProviderConfig:
    """Tests for OpenAICompatProvider configuration."""

    def test_default_config(self) -> None:
        """Verify default configuration."""
        provider = OpenAICompatProvider(base_url="http://localhost:8000")

        assert provider.model_name == "default"
        assert provider.base_url == "http://localhost:8000"

    def test_custom_config(self) -> None:
        """Verify custom configuration."""
        provider = OpenAICompatProvider(
            base_url="https://my-server.modal.run",
            model="Qwen/Qwen2.5-72B-Instruct",
            api_key="sk-test-123",
            timeout=300.0,
            endpoint="/v1/completions",
        )

        assert provider.model_name == "Qwen/Qwen2.5-72B-Instruct"
        assert provider.base_url == "https://my-server.modal.run"

    def test_base_url_strips_trailing_slash(self) -> None:
        """Verify trailing slash is stripped from base URL."""
        provider = OpenAICompatProvider(base_url="http://localhost:8000/")

        assert provider.base_url == "http://localhost:8000"

    def test_repr(self) -> None:
        """Verify __repr__ returns useful string."""
        provider = OpenAICompatProvider(
            base_url="http://localhost:8000",
            model="test-model",
        )

        repr_str = repr(provider)
        assert "OpenAICompatProvider" in repr_str
        assert "test-model" in repr_str

    def test_implements_protocol(self) -> None:
        """Verify OpenAICompatProvider implements LLMProviderProtocol."""
        provider = OpenAICompatProvider(base_url="http://localhost:8000")
        assert isinstance(provider, LLMProviderProtocol)

    def test_is_llm_provider(self) -> None:
        """Verify OpenAICompatProvider is an LLMProvider."""
        provider = OpenAICompatProvider(base_url="http://localhost:8000")
        assert isinstance(provider, LLMProvider)


# =============================================================================
# MOCKED RESPONSE TESTS
# =============================================================================


class TestOpenAICompatProviderMocked:
    """Tests for OpenAICompatProvider with mocked HTTP."""

    def test_complete_chat_completions(self, monkeypatch) -> None:
        """Verify complete() parses chat/completions response format."""
        captured = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hello, world!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "model": "test-model",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                    },
                }

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                captured["url"] = url
                captured["payload"] = json
                captured["headers"] = headers
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(
            base_url="http://localhost:8000",
            model="test-model",
        )
        response = provider.complete("Test prompt", system="Be helpful")

        assert response.text == "Hello, world!"
        assert response.model == "test-model"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

        # Verify messages were built correctly
        messages = captured["payload"]["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Test prompt"}

    def test_complete_without_system(self, monkeypatch) -> None:
        """Verify complete() works without system message."""
        captured = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {
                    "choices": [{"message": {"content": "OK"}}],
                    "model": "test",
                }

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                captured["payload"] = json
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(base_url="http://localhost:8000")
        provider.complete("Test prompt")

        messages = captured["payload"]["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Test prompt"}

    def test_complete_completions_fallback(self, monkeypatch) -> None:
        """Verify complete() falls back to completions response format."""
        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {
                    "choices": [{"text": "Fallback response"}],
                    "model": "test-model",
                }

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(base_url="http://localhost:8000")
        response = provider.complete("Test")

        assert response.text == "Fallback response"

    def test_complete_with_api_key(self, monkeypatch) -> None:
        """Verify complete() includes API key in Authorization header."""
        captured = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"choices": [{"message": {"content": "OK"}}], "model": "test"}

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                captured["headers"] = headers
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(
            base_url="http://localhost:8000",
            api_key="sk-test-key",
        )
        provider.complete("Test")

        assert captured["headers"]["Authorization"] == "Bearer sk-test-key"

    def test_complete_connection_error(self, monkeypatch) -> None:
        """Verify complete() handles connection errors."""
        import httpx

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(base_url="http://localhost:8000")

        with pytest.raises(LLMError) as exc_info:
            provider.complete("Test")

        assert "Cannot connect" in exc_info.value.message
        assert exc_info.value.provider == "openai_compat"

    def test_complete_timeout(self, monkeypatch) -> None:
        """Verify complete() handles timeouts."""
        import httpx

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                raise httpx.TimeoutException("Timeout")

        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(base_url="http://localhost:8000")

        with pytest.raises(LLMError) as exc_info:
            provider.complete("Test")

        assert "timed out" in exc_info.value.message

    def test_complete_http_error(self, monkeypatch) -> None:
        """Verify complete() handles HTTP errors."""
        import httpx

        class MockResponse:
            status_code = 500
            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "Server error",
                    request=None,
                    response=self,
                )

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(base_url="http://localhost:8000")

        with pytest.raises(LLMError) as exc_info:
            provider.complete("Test")

        assert "500" in exc_info.value.message

    def test_complete_custom_endpoint(self, monkeypatch) -> None:
        """Verify complete() uses custom endpoint path."""
        captured = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"choices": [{"message": {"content": "OK"}}], "model": "test"}

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json, headers=None):
                captured["url"] = url
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OpenAICompatProvider(
            base_url="http://localhost:8000",
            endpoint="/v1/completions",
        )
        provider.complete("Test")

        assert captured["url"] == "http://localhost:8000/v1/completions"


# =============================================================================
# INTEGRATION TESTS (REQUIRE RUNNING API SERVER)
# =============================================================================


@pytest.mark.integration
class TestOpenAICompatProviderIntegration:
    """Integration tests requiring a running OpenAI-compatible API server."""

    @pytest.fixture(autouse=True)
    def _require_model_url(self) -> None:
        """Skip if KAIZEN_MODEL_URL is not set."""
        if not os.environ.get("KAIZEN_MODEL_URL"):
            pytest.skip("KAIZEN_MODEL_URL not set")

    def test_complete_real(self) -> None:
        """Test real completion against a running endpoint."""
        provider = OpenAICompatProvider(
            base_url=os.environ["KAIZEN_MODEL_URL"],
            model=os.environ.get("KAIZEN_MODEL_NAME", "default"),
            api_key=os.environ.get("KAIZEN_API_KEY"),
        )

        response = provider.complete("What is 2+2? Reply with just the number.")

        assert isinstance(response.text, str)
        assert len(response.text) > 0
