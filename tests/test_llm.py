"""
Tests for LLM providers.

This module tests:
- LLMProvider base class
- LLMResponse dataclass
- LLMError exception
- OllamaProvider (with mocked HTTP calls)

Integration tests that require a running Ollama server are marked
with @pytest.mark.integration and skipped by default.
"""

import pytest

from kaizen.llm import (
    LLMProvider,
    LLMProviderProtocol,
    LLMResponse,
    LLMError,
    OllamaProvider,
)


# =============================================================================
# LLM RESPONSE TESTS
# =============================================================================


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self) -> None:
        """Verify LLMResponse can be created."""
        response = LLMResponse(
            text="Hello, world!",
            model="test-model",
        )

        assert response.text == "Hello, world!"
        assert response.model == "test-model"
        assert response.usage is None

    def test_response_with_usage(self) -> None:
        """Verify LLMResponse handles usage info."""
        response = LLMResponse(
            text="Hello",
            model="test",
            usage={"input_tokens": 10, "output_tokens": 5},
        )

        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_response_with_alt_usage_keys(self) -> None:
        """Verify LLMResponse handles alternate usage keys."""
        # OpenAI-style keys
        response = LLMResponse(
            text="Hello",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_response_no_usage(self) -> None:
        """Verify LLMResponse handles missing usage."""
        response = LLMResponse(text="Hello", model="test")

        assert response.input_tokens is None
        assert response.output_tokens is None


# =============================================================================
# LLM ERROR TESTS
# =============================================================================


class TestLLMError:
    """Tests for LLMError exception."""

    def test_create_error(self) -> None:
        """Verify LLMError can be created."""
        error = LLMError(
            message="Connection failed",
            provider="test",
        )

        assert error.message == "Connection failed"
        assert error.provider == "test"
        assert error.details == {}

    def test_error_with_details(self) -> None:
        """Verify LLMError handles details."""
        error = LLMError(
            message="Failed",
            provider="test",
            details={"status": 500, "url": "http://example.com"},
        )

        assert error.details["status"] == 500
        assert error.details["url"] == "http://example.com"

    def test_error_str(self) -> None:
        """Verify LLMError string representation."""
        error = LLMError(message="Test error", provider="ollama")

        assert "[ollama]" in str(error)
        assert "Test error" in str(error)

    def test_error_is_exception(self) -> None:
        """Verify LLMError can be raised and caught."""
        with pytest.raises(LLMError) as exc_info:
            raise LLMError("Test", "provider")

        assert exc_info.value.message == "Test"


# =============================================================================
# OLLAMA PROVIDER TESTS (MOCKED)
# =============================================================================


class TestOllamaProviderConfig:
    """Tests for OllamaProvider configuration."""

    def test_default_config(self) -> None:
        """Verify default configuration."""
        provider = OllamaProvider()

        assert provider.model_name == "llama3.1:8b"
        assert provider.base_url == "http://localhost:11434"

    def test_custom_model(self) -> None:
        """Verify custom model configuration."""
        provider = OllamaProvider(model="mistral:7b")

        assert provider.model_name == "mistral:7b"

    def test_custom_base_url(self) -> None:
        """Verify custom base URL configuration."""
        provider = OllamaProvider(base_url="http://gpu-server:11434")

        assert provider.base_url == "http://gpu-server:11434"

    def test_base_url_strips_trailing_slash(self) -> None:
        """Verify trailing slash is stripped from base URL."""
        provider = OllamaProvider(base_url="http://localhost:11434/")

        assert provider.base_url == "http://localhost:11434"

    def test_repr(self) -> None:
        """Verify __repr__ returns useful string."""
        provider = OllamaProvider(model="test:7b")

        repr_str = repr(provider)
        assert "OllamaProvider" in repr_str
        assert "test:7b" in repr_str

    def test_implements_protocol(self) -> None:
        """Verify OllamaProvider implements LLMProviderProtocol."""
        provider = OllamaProvider()
        assert isinstance(provider, LLMProviderProtocol)

    def test_is_llm_provider(self) -> None:
        """Verify OllamaProvider is an LLMProvider."""
        provider = OllamaProvider()
        assert isinstance(provider, LLMProvider)


class TestOllamaProviderMocked:
    """Tests for OllamaProvider with mocked HTTP."""

    def test_complete_success(self, monkeypatch) -> None:
        """Verify complete() handles successful response."""
        # Mock httpx.Client
        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {
                    "response": "Hello, world!",
                    "model": "llama3.1:8b",
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                }

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json):
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()
        response = provider.complete("Test prompt")

        assert response.text == "Hello, world!"
        assert response.model == "llama3.1:8b"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_complete_with_system(self, monkeypatch) -> None:
        """Verify complete() includes system message."""
        captured_payload = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"response": "OK", "model": "test"}

        class MockClient:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(self, url, json):
                captured_payload.update(json)
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()
        provider.complete("Prompt", system="System message")

        assert captured_payload["system"] == "System message"
        assert captured_payload["prompt"] == "Prompt"

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
            def post(self, url, json):
                raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()

        with pytest.raises(LLMError) as exc_info:
            provider.complete("Test")

        assert "Cannot connect" in exc_info.value.message
        assert exc_info.value.provider == "ollama"

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
            def post(self, url, json):
                raise httpx.TimeoutException("Timeout")

        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()

        with pytest.raises(LLMError) as exc_info:
            provider.complete("Test")

        assert "timed out" in exc_info.value.message

    def test_complete_model_not_found(self, monkeypatch) -> None:
        """Verify complete() handles 404 for missing model."""
        import httpx

        class MockResponse:
            status_code = 404
            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "Not found",
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
            def post(self, url, json):
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider(model="nonexistent:7b")

        with pytest.raises(LLMError) as exc_info:
            provider.complete("Test")

        assert "not found" in exc_info.value.message
        assert "ollama pull" in exc_info.value.message


# =============================================================================
# KWARGS PASS-THROUGH TESTS
# =============================================================================


class TestOllamaProviderKwargs:
    """Tests for per-call kwargs in OllamaProvider."""

    def test_complete_with_kwargs(self, monkeypatch) -> None:
        """Verify complete() passes recognized kwargs to Ollama options."""
        captured_payload = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"response": "OK", "model": "test"}

        class MockClient:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def post(self, url, json):
                captured_payload.update(json)
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()
        provider.complete("Prompt", temperature=0.5, max_tokens=100)

        assert captured_payload["options"]["temperature"] == 0.5
        assert captured_payload["options"]["num_predict"] == 100

    def test_complete_maps_max_tokens_to_num_predict(self, monkeypatch) -> None:
        """Verify max_tokens is mapped to Ollama's num_predict."""
        captured_payload = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"response": "OK", "model": "test"}

        class MockClient:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def post(self, url, json):
                captured_payload.update(json)
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()
        provider.complete("Prompt", max_tokens=256)

        assert "max_tokens" not in captured_payload.get("options", {})
        assert captured_payload["options"]["num_predict"] == 256

    def test_complete_ignores_unknown_kwargs(self, monkeypatch) -> None:
        """Verify complete() silently ignores unrecognized kwargs."""
        captured_payload = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"response": "OK", "model": "test"}

        class MockClient:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def post(self, url, json):
                captured_payload.update(json)
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()
        provider.complete("Prompt", unknown_param="should be ignored")

        assert "options" not in captured_payload

    def test_complete_no_kwargs_no_options(self, monkeypatch) -> None:
        """Verify no options key when no kwargs passed."""
        captured_payload = {}

        class MockResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"response": "OK", "model": "test"}

        class MockClient:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def post(self, url, json):
                captured_payload.update(json)
                return MockResponse()

        import httpx
        monkeypatch.setattr(httpx, "Client", MockClient)

        provider = OllamaProvider()
        provider.complete("Prompt")

        assert "options" not in captured_payload


# =============================================================================
# INTEGRATION TESTS (REQUIRE RUNNING OLLAMA)
# =============================================================================


@pytest.mark.integration
class TestOllamaProviderIntegration:
    """Integration tests requiring a running Ollama server."""

    def test_is_available(self) -> None:
        """Test checking if Ollama is available."""
        provider = OllamaProvider()
        # This might be True or False depending on whether Ollama is running
        result = provider.is_available()
        assert isinstance(result, bool)

    def test_list_models(self) -> None:
        """Test listing available models."""
        provider = OllamaProvider()
        if not provider.is_available():
            pytest.skip("Ollama not available")

        models = provider.list_models()
        assert isinstance(models, list)

    def test_complete_real(self) -> None:
        """Test real completion (requires Ollama running)."""
        provider = OllamaProvider()
        if not provider.is_available():
            pytest.skip("Ollama not available")

        response = provider.complete("What is 2+2? Reply with just the number.")

        assert isinstance(response.text, str)
        assert len(response.text) > 0
