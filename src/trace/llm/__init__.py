"""
LLM Provider abstractions for Trace.

This module contains the LLM provider protocol and implementations
for various backends (Ollama, Claude, etc.).

The provider abstraction allows the Planner to work with different
LLM backends without changing the planning logic.

Available providers:
- OllamaProvider: Local Ollama server (default: llama3.1:8b)
- (Future) ClaudeProvider: Anthropic Claude API
"""

# Exports will be added as modules are implemented:
# - LLMProvider (base.py)
# - OllamaProvider (ollama.py)
