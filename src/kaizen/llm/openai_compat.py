"""
OpenAI-Compatible LLM Provider for Kaizen.

This module provides an LLM provider that works with any OpenAI-compatible
API endpoint (vLLM, Modal, Together, Groq, OpenAI, etc.).

Configuration:
- base_url: API server URL (e.g., https://your-modal-endpoint.modal.run)
- model: Model name/identifier
- api_key: Optional API key for authenticated endpoints
- timeout: Request timeout in seconds (default: 120)

Example usage:
    provider = OpenAICompatProvider(
        base_url="https://my-endpoint.modal.run",
        model="Qwen/Qwen2.5-72B-Instruct",
    )
    response = provider.complete("What is 2+2?")
    print(response.text)
"""

from typing import Any

import httpx

from kaizen.llm.base import LLMProvider, LLMResponse, LLMError


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "default"
DEFAULT_TIMEOUT = 120.0
DEFAULT_ENDPOINT = "/v1/chat/completions"


# =============================================================================
# OPENAI-COMPATIBLE PROVIDER
# =============================================================================


class OpenAICompatProvider(LLMProvider):
    """
    LLM provider for any OpenAI-compatible API endpoint.

    Works with vLLM, Modal, Together AI, Groq, OpenAI, and any server
    that implements the OpenAI chat completions API.

    Attributes:
        base_url: URL of the API server.
        model: Name of the model to use.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        endpoint: str = DEFAULT_ENDPOINT,
        max_tokens: int | None = None,
    ) -> None:
        """
        Initialize the OpenAI-compatible provider.

        Args:
            base_url: URL of the API server.
            model: Model name/identifier.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            endpoint: API endpoint path (default: /v1/chat/completions).
            max_tokens: Max tokens for completion (None = server default).
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._endpoint = endpoint
        self._max_tokens = max_tokens

        self._client_timeout = httpx.Timeout(timeout, connect=10.0)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def base_url(self) -> str:
        """Get the API server URL."""
        return self._base_url

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion using an OpenAI-compatible API.

        Uses the chat completions endpoint by default, with fallback
        parsing for the completions response format.

        Args:
            prompt: The prompt to complete.
            system: Optional system message.
            **kwargs: Per-call overrides. Recognized: max_tokens (overrides
                      constructor default), temperature, top_p,
                      frequency_penalty, presence_penalty, stop, seed.

        Returns:
            LLMResponse: The generated response.

        Raises:
            LLMError: If the request fails.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self._model,
            "messages": messages,
        }

        # max_tokens: per-call overrides constructor default
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Pass through recognized OpenAI-compatible parameters
        _OPENAI_PARAMS = {"temperature", "top_p", "frequency_penalty",
                          "presence_penalty", "stop", "seed"}
        for key in _OPENAI_PARAMS:
            if key in kwargs:
                payload[key] = kwargs[key]

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}{self._endpoint}"

        try:
            with httpx.Client(timeout=self._client_timeout, follow_redirects=True) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError as e:
            raise LLMError(
                message=f"Cannot connect to API server at {self._base_url}. "
                        f"Is the server running?",
                provider="openai_compat",
                details={"base_url": self._base_url, "error": str(e)},
            ) from e

        except httpx.TimeoutException as e:
            raise LLMError(
                message=f"Request to API server timed out after {self._timeout}s",
                provider="openai_compat",
                details={"timeout": self._timeout, "error": str(e)},
            ) from e

        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            raise LLMError(
                message=f"API request failed with status {e.response.status_code}",
                provider="openai_compat",
                details={"status": e.response.status_code, "error": str(e), "body": body},
            ) from e

        # Parse response - try chat/completions format first, then completions
        text = ""
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            if "message" in choice:
                text = choice["message"].get("content", "")
            elif "text" in choice:
                text = choice["text"]

        # Build usage info if available
        usage = None
        usage_data = data.get("usage")
        if usage_data:
            usage = {
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
            }

        return LLMResponse(
            text=text,
            model=data.get("model", self._model),
            usage=usage,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"OpenAICompatProvider(model={self._model!r}, base_url={self._base_url!r})"
