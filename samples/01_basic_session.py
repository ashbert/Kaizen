#!/usr/bin/env python3
"""
Sample 01: Basic Session Usage

This example demonstrates the fundamental Session operations:
- Creating a session
- Getting and setting state
- State versioning
- Saving and loading sessions

No external dependencies required (no Ollama needed).
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kaizen import Session


def main():
    print("=" * 60)
    print("Sample 01: Basic Session Usage")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Creating a Session
    # -------------------------------------------------------------------------
    print("\n1. Creating a Session")
    print("-" * 40)

    session = Session()
    print(f"Session ID: {session.session_id}")
    print(f"Initial state version: {session.get_state_version()}")

    # -------------------------------------------------------------------------
    # Working with State
    # -------------------------------------------------------------------------
    print("\n2. Working with State")
    print("-" * 40)

    # Set some values
    session.set("name", "Alice")
    session.set("count", 42)
    session.set("tags", ["python", "kaizen", "agents"])
    session.set("config", {"debug": True, "timeout": 30})

    print(f"name = {session.get('name')}")
    print(f"count = {session.get('count')}")
    print(f"tags = {session.get('tags')}")
    print(f"config = {session.get('config')}")
    print(f"State version after 4 sets: {session.get_state_version()}")

    # Get with default
    print(f"missing key with default: {session.get('missing', default='N/A')}")

    # -------------------------------------------------------------------------
    # State Versioning
    # -------------------------------------------------------------------------
    print("\n3. State Versioning")
    print("-" * 40)

    v1 = session.get_state_version()
    session.set("count", 43)
    v2 = session.get_state_version()
    session.set("count", 44)
    v3 = session.get_state_version()

    print(f"Version progression: {v1} -> {v2} -> {v3}")
    print("Each set() increments the version by 1")

    # -------------------------------------------------------------------------
    # Viewing the Trajectory
    # -------------------------------------------------------------------------
    print("\n4. Viewing the Trajectory")
    print("-" * 40)

    print(f"Total trajectory entries: {session.get_trajectory_length()}")
    print("\nRecent entries:")
    for entry in session.get_trajectory(limit=5):
        print(f"  [{entry.seq_num}] {entry.entry_type.value}: {list(entry.content.keys())}")

    # -------------------------------------------------------------------------
    # Saving and Loading
    # -------------------------------------------------------------------------
    print("\n5. Saving and Loading")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix=".kaizen", delete=False) as f:
        session_path = f.name

    session.save(session_path)
    print(f"Saved to: {session_path}")

    # Load it back
    restored = Session.load(session_path)
    print(f"Loaded session ID: {restored.session_id}")
    print(f"Restored name: {restored.get('name')}")
    print(f"Restored count: {restored.get('count')}")
    print(f"Restored state version: {restored.get_state_version()}")

    # Verify it matches
    assert restored.session_id == session.session_id
    assert restored.get("name") == session.get("name")
    assert restored.get_state_version() == session.get_state_version()
    print("\n✓ All values preserved correctly!")

    # Cleanup
    Path(session_path).unlink()

    print("\n" + "=" * 60)
    print("Sample complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
