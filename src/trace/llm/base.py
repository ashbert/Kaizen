"""
LLM Provider base classes for Trace.

This module defines the abstract interface for LLM providers that can
be used by the Planner to generate capability call sequences.

The provider abstraction allows the Planner to work with different
LLM backends (Ollama, Claude, OpenAI, etc.) without changing the
planning logic.

Key design decisions:
1. Simple interface: just complete(prompt, system) -> str
2. Providers handle their own configuration (API keys, URLs, etc.)
3. Providers are responsible for error handling and retries
4. No streaming support in V1 (simplicity over features)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# =============================================================================
# LLM RESPONSE
# =============================================================================


@dataclass
class LLMResponse:
    """
    Response from an LLM completion request.

    Attributes:
        text: The generated text response.
        model: The model that generated the response.
        usage: Optional token usage information.
    """

    text: str
    model: str
    usage: dict[str, int] | None = None

    @property
    def input_tokens(self) -> int | None:
        """Get input token count if available."""
        if self.usage:
            return self.usage.get("input_tokens") or self.usage.get("prompt_tokens")
        return None

    @property
    def output_tokens(self) -> int | None:
        """Get output token count if available."""
        if self.usage:
            return self.usage.get("output_tokens") or self.usage.get("completion_tokens")
        return None


# =============================================================================
# LLM PROVIDER PROTOCOL
# =============================================================================


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """
    Protocol for LLM providers (structural typing).

    Any class with a complete() method matching this signature can be
    used as an LLM provider, even without inheriting from LLMProvider.
    """

    def complete(
        self,
        prompt: str,
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt."""
        ...


# =============================================================================
# LLM PROVIDER BASE CLASS
# =============================================================================


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Subclasses must implement the complete() method to generate
    text completions from the underlying LLM.

    The provider is responsible for:
    - Managing connections to the LLM service
    - Handling authentication (API keys, etc.)
    - Converting errors to LLMError exceptions
    - Optionally implementing retries

    Example subclass:

        class MyProvider(LLMProvider):
            def __init__(self, api_key: str):
                self.api_key = api_key

            def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
                # Call the LLM API...
                return LLMResponse(text="...", model="my-model")

            @property
            def model_name(self) -> str:
                return "my-model"
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt/message to complete.
            system: Optional system message to guide the model's behavior.

        Returns:
            LLMResponse: The generated response.

        Raises:
            LLMError: If the completion fails.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name/identifier of the model being used.

        Returns:
            str: Model name (e.g., "llama3.1:8b", "claude-3-opus").
        """
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(model={self.model_name})"


# =============================================================================
# LLM ERROR
# =============================================================================


class LLMError(Exception):
    """
    Exception raised when an LLM operation fails.

    This provides a consistent error type across all providers,
    making error handling easier for callers.

    Attributes:
        message: Human-readable error message.
        provider: Name of the provider that raised the error.
        details: Optional additional details about the error.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        details: dict | None = None,
    ) -> None:
        """
        Initialize an LLM error.

        Args:
            message: Human-readable error message.
            provider: Name of the provider (e.g., "ollama", "claude").
            details: Optional additional error details.
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation."""
        return f"[{self.provider}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"LLMError(provider={self.provider!r}, message={self.message!r})"
