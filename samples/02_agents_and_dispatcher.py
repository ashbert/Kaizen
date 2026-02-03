#!/usr/bin/env python3
"""
Sample 02: Agents and Dispatcher

This example demonstrates:
- Using the built-in agents (ReverseAgent, UppercaseAgent)
- Registering agents with the Dispatcher
- Executing single and multiple capability calls
- Error handling with fail-fast behavior

No external dependencies required (no Ollama needed).
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kaizen import Session, Dispatcher
from kaizen.agents import ReverseAgent, UppercaseAgent
from kaizen.types import CapabilityCall


def main():
    print("=" * 60)
    print("Sample 02: Agents and Dispatcher")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Setting Up
    # -------------------------------------------------------------------------
    print("\n1. Setting Up Session and Dispatcher")
    print("-" * 40)

    session = Session()
    session.set("text", "hello world")
    print(f"Initial text: '{session.get('text')}'")

    dispatcher = Dispatcher()
    dispatcher.register(ReverseAgent())
    dispatcher.register(UppercaseAgent())

    print(f"Registered capabilities: {dispatcher.get_capabilities()}")

    # -------------------------------------------------------------------------
    # Single Capability Execution
    # -------------------------------------------------------------------------
    print("\n2. Single Capability Execution")
    print("-" * 40)

    # Reset text
    session.set("text", "hello world")

    # Execute reverse
    result = dispatcher.dispatch_single("reverse", session, {"key": "text"})
    print(f"After reverse: '{session.get('text')}'")
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")

    # Execute uppercase
    result = dispatcher.dispatch_single("uppercase", session, {"key": "text"})
    print(f"After uppercase: '{session.get('text')}'")
    print(f"  Success: {result.success}")

    # -------------------------------------------------------------------------
    # Sequence Execution
    # -------------------------------------------------------------------------
    print("\n3. Sequence Execution")
    print("-" * 40)

    # Reset text
    session.set("text", "kaizen is awesome")
    print(f"Starting text: '{session.get('text')}'")

    # Execute multiple calls
    calls = [
        CapabilityCall("reverse", {"key": "text"}),
        CapabilityCall("uppercase", {"key": "text"}),
    ]

    result = dispatcher.dispatch_sequence(calls, session)
    print(f"Final text: '{session.get('text')}'")
    print(f"Dispatch result:")
    print(f"  Success: {result.success}")
    print(f"  Executed: {result.executed_count} calls")

    # -------------------------------------------------------------------------
    # Error Handling
    # -------------------------------------------------------------------------
    print("\n4. Error Handling (Fail-Fast)")
    print("-" * 40)

    session.set("text", "test")
    print(f"Starting text: '{session.get('text')}'")

    # This sequence will fail at step 1 (nonexistent key)
    calls = [
        CapabilityCall("reverse", {"key": "text"}),          # Will succeed
        CapabilityCall("reverse", {"key": "nonexistent"}),   # Will fail
        CapabilityCall("uppercase", {"key": "text"}),        # Never reached
    ]

    result = dispatcher.dispatch_sequence(calls, session)
    print(f"Dispatch result:")
    print(f"  Success: {result.success}")
    print(f"  Failed at step: {result.failed_at}")
    print(f"  Error: {result.error['message']}")
    print(f"  Executed: {result.executed_count} of {len(calls)} calls")
    print(f"Final text: '{session.get('text')}'")  # Only first reverse applied

    # -------------------------------------------------------------------------
    # Unknown Capability
    # -------------------------------------------------------------------------
    print("\n5. Unknown Capability")
    print("-" * 40)

    result = dispatcher.dispatch_single("unknown_capability", session, {})
    print(f"Success: {result.success}")
    print(f"Error code: {result.error['error_code']}")
    print(f"Message: {result.error['message']}")

    # -------------------------------------------------------------------------
    # Inspecting Trajectory
    # -------------------------------------------------------------------------
    print("\n6. Inspecting Trajectory")
    print("-" * 40)

    print("Agent actions recorded in trajectory:")
    for entry in session.get_trajectory():
        if entry.agent_id not in ("system", "dispatcher"):
            print(f"  [{entry.seq_num}] {entry.agent_id}: {entry.entry_type.value}")
            if "original" in entry.content:
                print(f"       {entry.content.get('original')} -> {entry.content.get('reversed') or entry.content.get('uppercased')}")

    print("\n" + "=" * 60)
    print("Sample complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
