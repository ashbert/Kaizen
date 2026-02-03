#!/usr/bin/env python3
"""
Sample 03: Complete Workflow with LLM Planning

This example demonstrates the full Trace workflow:
1. Create a session with initial state
2. Set up agents and dispatcher
3. Use LLM planner to convert natural language to capability calls
4. Execute the plan
5. Save and resume the session
6. Inspect the trajectory

REQUIREMENTS: This sample requires Ollama to be running.
    ollama serve
    ollama pull llama3.1:8b
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trace import Session, Dispatcher, Planner
from trace.agents import ReverseAgent, UppercaseAgent
from trace.llm import OllamaProvider


def main():
    print("=" * 60)
    print("Sample 03: Complete Workflow with LLM Planning")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Check Ollama Availability
    # -------------------------------------------------------------------------
    print("\n1. Checking Ollama Availability")
    print("-" * 40)

    llm = OllamaProvider()
    if not llm.is_available():
        print("ERROR: Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        print("And pull the model with: ollama pull llama3.1:8b")
        sys.exit(1)

    print(f"Ollama is available at {llm.base_url}")
    print(f"Using model: {llm.model_name}")

    # -------------------------------------------------------------------------
    # Create Session
    # -------------------------------------------------------------------------
    print("\n2. Creating Session")
    print("-" * 40)

    session = Session()
    session.set("text", "hello world")
    print(f"Session ID: {session.session_id[:8]}...")
    print(f"Initial text: '{session.get('text')}'")

    # -------------------------------------------------------------------------
    # Set Up Dispatcher
    # -------------------------------------------------------------------------
    print("\n3. Setting Up Dispatcher")
    print("-" * 40)

    dispatcher = Dispatcher()
    dispatcher.register(ReverseAgent())
    dispatcher.register(UppercaseAgent())
    print(f"Available capabilities: {dispatcher.get_capabilities()}")

    # -------------------------------------------------------------------------
    # Plan with LLM
    # -------------------------------------------------------------------------
    print("\n4. Planning with LLM")
    print("-" * 40)

    planner = Planner(llm, capabilities=dispatcher.get_capabilities())

    user_input = "Reverse the text and then make it uppercase"
    print(f"User request: \"{user_input}\"")
    print("Calling LLM to generate plan...")

    plan_result = planner.plan(user_input, session=session)

    if not plan_result.success:
        print(f"Planning failed: {plan_result.error}")
        sys.exit(1)

    print(f"Plan generated with {len(plan_result.calls)} steps:")
    for i, call in enumerate(plan_result.calls):
        print(f"  {i+1}. {call.capability}({call.params})")

    # -------------------------------------------------------------------------
    # Execute Plan
    # -------------------------------------------------------------------------
    print("\n5. Executing Plan")
    print("-" * 40)

    dispatch_result = dispatcher.dispatch_sequence(plan_result.calls, session)

    if dispatch_result.success:
        print(f"All {dispatch_result.executed_count} steps executed successfully!")
    else:
        print(f"Execution failed at step {dispatch_result.failed_at}")
        print(f"Error: {dispatch_result.error}")
        sys.exit(1)

    print(f"Final text: '{session.get('text')}'")

    # -------------------------------------------------------------------------
    # Save Session
    # -------------------------------------------------------------------------
    print("\n6. Saving Session")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as f:
        session_path = f.name

    session.save(session_path)
    print(f"Session saved to: {session_path}")

    # -------------------------------------------------------------------------
    # Resume Session
    # -------------------------------------------------------------------------
    print("\n7. Resuming Session")
    print("-" * 40)

    restored = Session.load(session_path)
    print(f"Restored session ID: {restored.session_id[:8]}...")
    print(f"Restored text: '{restored.get('text')}'")
    print(f"State version: {restored.get_state_version()}")

    # -------------------------------------------------------------------------
    # Inspect Trajectory
    # -------------------------------------------------------------------------
    print("\n8. Inspecting Trajectory")
    print("-" * 40)

    print(f"Total trajectory entries: {restored.get_trajectory_length()}")
    print("\nKey events:")
    for entry in restored.get_trajectory():
        # Format timestamp
        ts = entry.timestamp.strftime("%H:%M:%S")

        # Skip verbose state_set entries
        if entry.entry_type.value == "state_set":
            continue

        print(f"  [{entry.seq_num:2}] {ts} {entry.agent_id:20} {entry.entry_type.value}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    Path(session_path).unlink()

    print("\n" + "=" * 60)
    print("Sample complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
