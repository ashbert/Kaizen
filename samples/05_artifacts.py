#!/usr/bin/env python3
"""
Sample 05: Working with Artifacts

This example demonstrates artifact management:
- Writing binary artifacts to a session
- Reading artifacts back
- Listing artifacts
- Artifacts surviving save/load cycles

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
    print("Sample 05: Working with Artifacts")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Creating Artifacts
    # -------------------------------------------------------------------------
    print("\n1. Creating Artifacts")
    print("-" * 40)

    session = Session()

    # Write a text file artifact
    session.write_artifact("input.txt", b"Hello, this is the input text!")
    print("Created: input.txt")

    # Write a JSON artifact
    import json
    config = {"version": 1, "settings": {"debug": True}}
    session.write_artifact("config.json", json.dumps(config).encode())
    print("Created: config.json")

    # Write a binary artifact (simulated image header)
    png_header = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    session.write_artifact("image.png", png_header + b"\x00" * 100)
    print("Created: image.png (simulated)")

    # -------------------------------------------------------------------------
    # Listing Artifacts
    # -------------------------------------------------------------------------
    print("\n2. Listing Artifacts")
    print("-" * 40)

    artifacts = session.list_artifacts()
    print(f"Artifacts ({len(artifacts)}):")
    for name in artifacts:
        size = session.get_artifact_size(name)
        print(f"  - {name} ({size} bytes)")

    # -------------------------------------------------------------------------
    # Reading Artifacts
    # -------------------------------------------------------------------------
    print("\n3. Reading Artifacts")
    print("-" * 40)

    # Read text artifact
    input_data = session.read_artifact("input.txt")
    print(f"input.txt content: '{input_data.decode()}'")

    # Read JSON artifact
    config_data = session.read_artifact("config.json")
    loaded_config = json.loads(config_data.decode())
    print(f"config.json parsed: {loaded_config}")

    # -------------------------------------------------------------------------
    # Updating Artifacts
    # -------------------------------------------------------------------------
    print("\n4. Updating Artifacts")
    print("-" * 40)

    old_size = session.get_artifact_size("input.txt")
    session.write_artifact("input.txt", b"Updated content with more text!")
    new_size = session.get_artifact_size("input.txt")

    print(f"input.txt size: {old_size} -> {new_size} bytes")
    print(f"New content: '{session.read_artifact('input.txt').decode()}'")

    # -------------------------------------------------------------------------
    # Artifacts in Trajectory
    # -------------------------------------------------------------------------
    print("\n5. Artifacts in Trajectory")
    print("-" * 40)

    print("Artifact-related trajectory entries:")
    for entry in session.get_trajectory():
        if entry.entry_type.value == "artifact_written":
            is_update = entry.content.get("is_update", False)
            action = "Updated" if is_update else "Created"
            print(f"  [{entry.seq_num}] {action}: {entry.content['name']} ({entry.content['size']} bytes)")

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    print("\n6. Artifacts Survive Save/Load")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix=".kaizen", delete=False) as f:
        session_path = f.name

    session.save(session_path)
    print(f"Saved session to: {session_path}")

    restored = Session.load(session_path)
    print(f"Loaded session")

    print(f"\nRestored artifacts:")
    for name in restored.list_artifacts():
        size = restored.get_artifact_size(name)
        print(f"  - {name} ({size} bytes)")

    # Verify content
    assert restored.read_artifact("input.txt") == session.read_artifact("input.txt")
    assert restored.read_artifact("config.json") == session.read_artifact("config.json")
    print("\n✓ All artifact content verified!")

    # Cleanup
    Path(session_path).unlink()

    # -------------------------------------------------------------------------
    # Size Limits
    # -------------------------------------------------------------------------
    print("\n7. Artifact Size Limits")
    print("-" * 40)

    # Create session with small limit
    small_session = Session(max_artifact_size=100)
    print(f"Max artifact size: {small_session.max_artifact_size} bytes")

    # This should work
    small_session.write_artifact("small.txt", b"x" * 50)
    print("Created 50-byte artifact: OK")

    # This should fail
    try:
        small_session.write_artifact("large.txt", b"x" * 200)
        print("Created 200-byte artifact: OK")
    except ValueError as e:
        print(f"Creating 200-byte artifact: REJECTED")
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Sample complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
