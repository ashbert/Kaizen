#!/usr/bin/env python3
"""
Sample 04: Creating Custom Agents

This example demonstrates how to create your own custom agent:
- Implementing the Agent protocol
- Defining capabilities
- Reading and writing session state
- Recording actions in the trajectory
- Proper error handling

No external dependencies required (no Ollama needed).
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kaizen import Session, Dispatcher, Agent, AgentInfo, InvokeResult
from kaizen.types import EntryType


# =============================================================================
# CUSTOM AGENT: Word Counter
# =============================================================================


class WordCountAgent(Agent):
    """
    A custom agent that counts words in text.

    Capabilities:
    - count_words: Counts words in text at a given key, stores result
    """

    def info(self) -> AgentInfo:
        """Return metadata about this agent."""
        return AgentInfo(
            agent_id="word_count_agent_v1",
            name="Word Count Agent",
            version="1.0.0",
            capabilities=["count_words"],
            description="Counts words in text and stores the count",
        )

    def invoke(
        self,
        capability: str,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        """Execute a capability."""
        if capability == "count_words":
            return self._count_words(session, params)
        else:
            return self._unknown_capability(capability)

    def _count_words(
        self,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        """Count words in text at the given key."""
        # Get parameters
        text_key = params.get("text_key", "text")
        count_key = params.get("count_key", "word_count")

        # Validate text exists
        text = session.get(text_key)
        if text is None:
            return self._invalid_params(
                "count_words",
                f"No text found at key '{text_key}'",
            )

        if not isinstance(text, str):
            return self._invalid_params(
                "count_words",
                f"Value at '{text_key}' must be a string, got {type(text).__name__}",
            )

        # Count words
        words = text.split()
        word_count = len(words)

        # Record action in trajectory
        agent_info = self.info()
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_INVOKED,
            content={
                "capability": "count_words",
                "text_key": text_key,
                "count_key": count_key,
            },
        )

        # Store result
        session.set(count_key, word_count)

        # Record completion
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": "count_words",
                "text": text,
                "word_count": word_count,
            },
        )

        return InvokeResult.ok(
            result={"word_count": word_count, "text": text},
            agent_id=agent_info.agent_id,
            capability="count_words",
        )


# =============================================================================
# CUSTOM AGENT: Text Summarizer (Simple)
# =============================================================================


class SimpleSummarizerAgent(Agent):
    """
    A custom agent that creates a simple summary by truncating text.

    Capabilities:
    - summarize: Truncates text to first N words
    """

    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="simple_summarizer_v1",
            name="Simple Summarizer",
            version="1.0.0",
            capabilities=["summarize"],
            description="Creates a summary by keeping first N words",
        )

    def invoke(
        self,
        capability: str,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        if capability == "summarize":
            return self._summarize(session, params)
        return self._unknown_capability(capability)

    def _summarize(
        self,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        key = params.get("key", "text")
        max_words = params.get("max_words", 10)

        text = session.get(key)
        if text is None:
            return self._invalid_params("summarize", f"No text at '{key}'")

        if not isinstance(text, str):
            return self._invalid_params(
                "summarize",
                f"Value must be string, got {type(text).__name__}",
            )

        # Summarize
        words = text.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + "..."
        else:
            summary = text

        # Record and update
        agent_info = self.info()
        session.append(
            agent_info.agent_id,
            EntryType.AGENT_INVOKED,
            {"capability": "summarize", "key": key, "max_words": max_words},
        )

        session.set(key, summary)

        session.append(
            agent_info.agent_id,
            EntryType.AGENT_COMPLETED,
            {"original_length": len(words), "summary_length": len(summary.split())},
        )

        return InvokeResult.ok(
            result={"summary": summary, "original_words": len(words)},
            agent_id=agent_info.agent_id,
            capability="summarize",
        )


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("Sample 04: Creating Custom Agents")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Register Custom Agents
    # -------------------------------------------------------------------------
    print("\n1. Registering Custom Agents")
    print("-" * 40)

    dispatcher = Dispatcher()
    dispatcher.register(WordCountAgent())
    dispatcher.register(SimpleSummarizerAgent())

    print(f"Available capabilities: {dispatcher.get_capabilities()}")
    print("\nRegistered agents:")
    for agent_info in dispatcher.get_registered_agents():
        print(f"  - {agent_info.name} ({agent_info.agent_id})")
        print(f"    Capabilities: {agent_info.capabilities}")

    # -------------------------------------------------------------------------
    # Use Word Count Agent
    # -------------------------------------------------------------------------
    print("\n2. Using Word Count Agent")
    print("-" * 40)

    session = Session()
    session.set("text", "The quick brown fox jumps over the lazy dog")
    print(f"Text: '{session.get('text')}'")

    result = dispatcher.dispatch_single(
        "count_words",
        session,
        {"text_key": "text", "count_key": "word_count"},
    )

    print(f"Word count: {session.get('word_count')}")
    print(f"Result: {result.result}")

    # -------------------------------------------------------------------------
    # Use Summarizer Agent
    # -------------------------------------------------------------------------
    print("\n3. Using Summarizer Agent")
    print("-" * 40)

    long_text = (
        "Kaizen is a minimal deterministic substrate for agent workflows. "
        "It provides session management with state trajectory and artifacts. "
        "Sessions can be saved to disk and resumed later. "
        "All actions are recorded for auditability."
    )
    session.set("article", long_text)
    print(f"Original ({len(long_text.split())} words):")
    print(f"  '{long_text[:60]}...'")

    result = dispatcher.dispatch_single(
        "summarize",
        session,
        {"key": "article", "max_words": 8},
    )

    print(f"Summary: '{session.get('article')}'")
    print(f"Result: {result.result}")

    # -------------------------------------------------------------------------
    # Chain Custom Agents
    # -------------------------------------------------------------------------
    print("\n4. Chaining Custom Agents")
    print("-" * 40)

    session.set("document", "This is a test document with several words in it")
    print(f"Original: '{session.get('document')}'")

    # First count words
    dispatcher.dispatch_single("count_words", session, {"text_key": "document"})
    print(f"Word count: {session.get('word_count')}")

    # Then summarize
    dispatcher.dispatch_single("summarize", session, {"key": "document", "max_words": 5})
    print(f"After summarize: '{session.get('document')}'")

    # -------------------------------------------------------------------------
    # Inspect Trajectory
    # -------------------------------------------------------------------------
    print("\n5. Inspecting Custom Agent Actions")
    print("-" * 40)

    print("Custom agent entries in trajectory:")
    for entry in session.get_trajectory():
        if entry.agent_id in ("word_count_agent_v1", "simple_summarizer_v1"):
            print(f"  [{entry.seq_num}] {entry.agent_id}")
            print(f"       Type: {entry.entry_type.value}")
            print(f"       Content: {entry.content}")

    print("\n" + "=" * 60)
    print("Sample complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
