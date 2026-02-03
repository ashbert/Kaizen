"""
LLM Provider abstractions for Kaizen.

This module contains the LLM provider protocol and implementations
for various backends (Ollama, Claude, etc.).

The provider abstraction allows the Planner to work with different
LLM backends without changing the planning logic.

Available providers:
- OllamaProvider: Local Ollama server (default: llama3.1:8b)
- (Future) ClaudeProvider: Anthropic Claude API
"""

from kaizen.llm.base import LLMProvider, LLMProviderProtocol, LLMResponse, LLMError
from kaizen.llm.ollama import OllamaProvider

__all__ = [
    "LLMProvider",
    "LLMProviderProtocol",
    "LLMResponse",
    "LLMError",
    "OllamaProvider",
]
