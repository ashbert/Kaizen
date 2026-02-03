"""
Ollama LLM Provider for Trace.

This module provides an LLM provider that uses a local Ollama server
for text generation. Ollama allows running open-source LLMs locally.

Default model: llama3.1:8b

Requirements:
- Ollama must be installed and running locally
- The desired model must be pulled (e.g., `ollama pull llama3.1:8b`)

Configuration:
- base_url: Ollama server URL (default: http://localhost:11434)
- model: Model name (default: llama3.1:8b)
- timeout: Request timeout in seconds (default: 120)

Example usage:
    provider = OllamaProvider()  # Uses defaults
    response = provider.complete("What is 2+2?")
    print(response.text)

    # With custom configuration
    provider = OllamaProvider(
        model="codellama:7b",
        base_url="http://gpu-server:11434",
        timeout=300,
    )
"""

import httpx

from trace.llm.base import LLMProvider, LLMResponse, LLMError


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Default Ollama server URL (local installation)
DEFAULT_BASE_URL = "http://localhost:11434"

# Default model - Llama 3.1 8B is a good balance of quality and speed
DEFAULT_MODEL = "llama3.1:8b"

# Default timeout for requests (2 minutes for slower machines)
DEFAULT_TIMEOUT = 120.0


# =============================================================================
# OLLAMA PROVIDER
# =============================================================================


class OllamaProvider(LLMProvider):
    """
    LLM provider using local Ollama server.

    Ollama is a tool for running open-source LLMs locally. This provider
    connects to a running Ollama instance and uses its API for completions.

    The provider uses the /api/generate endpoint for completions. It does
    NOT use the /api/chat endpoint as we want raw completions for the
    planner's structured output needs.

    Attributes:
        base_url: URL of the Ollama server.
        model: Name of the model to use.
        timeout: Request timeout in seconds.

    Example:
        # Using defaults (localhost, llama3.1:8b)
        provider = OllamaProvider()

        # Custom model
        provider = OllamaProvider(model="mistral:7b")

        # Remote server
        provider = OllamaProvider(
            base_url="http://gpu-server:11434",
            model="llama3.1:70b",
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model: Name of the Ollama model to use.
            base_url: URL of the Ollama server.
            timeout: Request timeout in seconds.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")  # Remove trailing slash
        self._timeout = timeout

        # Create HTTP client
        # We create a new client for each request to avoid connection issues
        # (Ollama can have long-running requests that might timeout keepalive)
        self._client_timeout = httpx.Timeout(timeout, connect=10.0)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def base_url(self) -> str:
        """Get the Ollama server URL."""
        return self._base_url

    def complete(
        self,
        prompt: str,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Generate a completion using Ollama.

        Uses the /api/generate endpoint for raw completions.

        Args:
            prompt: The prompt to complete.
            system: Optional system message.

        Returns:
            LLMResponse: The generated response.

        Raises:
            LLMError: If the request fails.
        """
        # Build the request payload
        # See: https://github.com/ollama/ollama/blob/main/docs/api.md
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,  # We want the full response, not streaming
        }

        # Add system message if provided
        if system:
            payload["system"] = system

        # Make the request
        try:
            with httpx.Client(timeout=self._client_timeout) as client:
                response = client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError as e:
            raise LLMError(
                message=f"Cannot connect to Ollama server at {self._base_url}. "
                       f"Is Ollama running?",
                provider="ollama",
                details={"base_url": self._base_url, "error": str(e)},
            ) from e

        except httpx.TimeoutException as e:
            raise LLMError(
                message=f"Request to Ollama timed out after {self._timeout}s",
                provider="ollama",
                details={"timeout": self._timeout, "error": str(e)},
            ) from e

        except httpx.HTTPStatusError as e:
            # Handle specific HTTP errors
            status = e.response.status_code
            if status == 404:
                raise LLMError(
                    message=f"Model '{self._model}' not found. "
                           f"Try: ollama pull {self._model}",
                    provider="ollama",
                    details={"model": self._model, "status": status},
                ) from e
            else:
                raise LLMError(
                    message=f"Ollama request failed with status {status}",
                    provider="ollama",
                    details={"status": status, "error": str(e)},
                ) from e

        except Exception as e:
            raise LLMError(
                message=f"Unexpected error calling Ollama: {type(e).__name__}: {e}",
                provider="ollama",
                details={"error_type": type(e).__name__, "error": str(e)},
            ) from e

        # Extract the response
        text = data.get("response", "")

        # Build usage info if available
        usage = None
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            }

        return LLMResponse(
            text=text,
            model=data.get("model", self._model),
            usage=usage,
        )

    def is_available(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            bool: True if server responds, False otherwise.
        """
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                response = client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """
        List available models on the Ollama server.

        Returns:
            list[str]: Names of available models.

        Raises:
            LLMError: If the request fails.
        """
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                response = client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]

        except Exception as e:
            raise LLMError(
                message=f"Failed to list models: {e}",
                provider="ollama",
                details={"error": str(e)},
            ) from e

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"OllamaProvider(model={self._model!r}, base_url={self._base_url!r})"
