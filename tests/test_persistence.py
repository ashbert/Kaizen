"""
Tests for Session persistence (save/load).

This module tests the SQLite-based persistence functionality:
- save() to write session to file
- load() to restore session from file
- Roundtrip preservation of all data

The key invariant is:

    save/load preserves all session data.

This means state, trajectory, and artifacts should all survive
the roundtrip without any data loss or corruption.
"""

from pathlib import Path

import pytest

from kaizen.session import Session
from kaizen.types import EntryType


class TestSaveBasics:
    """Tests for basic save functionality."""

    def test_save_creates_file(self, temp_session_path: Path) -> None:
        """Verify save() creates a file at the specified path."""
        session = Session()
        session.save(temp_session_path)

        assert temp_session_path.exists()

    def test_save_string_path(self, temp_dir: Path) -> None:
        """Verify save() accepts string paths."""
        session = Session()
        path = str(temp_dir / "session.trace")
        session.save(path)

        assert Path(path).exists()

    def test_save_overwrites_existing(self, temp_session_path: Path) -> None:
        """Verify save() overwrites existing file."""
        session1 = Session()
        session1.set("version", 1)
        session1.save(temp_session_path)

        session2 = Session()
        session2.set("version", 2)
        session2.save(temp_session_path)

        # Loading should give version 2
        loaded = Session.load(temp_session_path)
        assert loaded.get("version") == 2


class TestLoadBasics:
    """Tests for basic load functionality."""

    def test_load_returns_session(self, temp_session_path: Path) -> None:
        """Verify load() returns a Session instance."""
        session = Session()
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert isinstance(loaded, Session)

    def test_load_string_path(self, temp_session_path: Path) -> None:
        """Verify load() accepts string paths."""
        session = Session()
        session.save(temp_session_path)

        loaded = Session.load(str(temp_session_path))
        assert isinstance(loaded, Session)

    def test_load_nonexistent_raises(self, temp_session_path: Path) -> None:
        """Verify load() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            Session.load(temp_session_path)


class TestRoundtripSessionId:
    """Tests for session ID preservation."""

    def test_session_id_preserved(self, temp_session_path: Path) -> None:
        """Verify session ID survives roundtrip."""
        session = Session(session_id="my-custom-id-123")
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.session_id == "my-custom-id-123"


class TestRoundtripState:
    """Tests for state preservation."""

    def test_empty_state_preserved(self, temp_session_path: Path) -> None:
        """Verify empty state survives roundtrip."""
        session = Session()
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.get_all_state() == {}

    def test_single_value_preserved(self, temp_session_path: Path) -> None:
        """Verify single value survives roundtrip."""
        session = Session()
        session.set("key", "value")
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.get("key") == "value"

    def test_multiple_values_preserved(self, temp_session_path: Path) -> None:
        """Verify multiple values survive roundtrip."""
        session = Session()
        session.set("string", "hello")
        session.set("number", 42)
        session.set("float", 3.14)
        session.set("bool", True)
        session.set("null", None)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.get("string") == "hello"
        assert loaded.get("number") == 42
        assert loaded.get("float") == 3.14
        assert loaded.get("bool") is True
        assert loaded.get("null") is None

    def test_complex_nested_state_preserved(
        self, temp_session_path: Path, sample_state_data: dict
    ) -> None:
        """Verify complex nested state survives roundtrip."""
        session = Session()
        session.set("data", sample_state_data)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.get("data") == sample_state_data

    def test_state_version_preserved(self, temp_session_path: Path) -> None:
        """Verify state version survives roundtrip."""
        session = Session()
        session.set("a", 1)
        session.set("b", 2)
        session.set("c", 3)
        original_version = session.get_state_version()
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.get_state_version() == original_version


class TestRoundtripTrajectory:
    """Tests for trajectory preservation."""

    def test_trajectory_entries_preserved(self, temp_session_path: Path) -> None:
        """Verify trajectory entries survive roundtrip."""
        session = Session()
        session.append("agent1", EntryType.AGENT_INVOKED, {"step": 1})
        session.append("agent1", EntryType.AGENT_COMPLETED, {"step": 2})
        session.append("agent2", EntryType.STATE_SET, {"key": "value"})
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        trajectory = loaded.get_trajectory()

        # Should have session_created + 3 user entries + session_saved + session_loaded
        user_entries = [e for e in trajectory if e.agent_id != "system"]
        assert len(user_entries) == 3

    def test_trajectory_sequence_numbers_preserved(
        self, temp_session_path: Path
    ) -> None:
        """Verify trajectory sequence numbers survive roundtrip."""
        session = Session()
        for i in range(5):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        trajectory = loaded.get_trajectory()

        # Sequence numbers should be consecutive (accounting for save/load entries)
        seq_nums = [e.seq_num for e in trajectory]
        for i in range(1, len(seq_nums)):
            assert seq_nums[i] == seq_nums[i - 1] + 1

    def test_trajectory_timestamps_preserved(self, temp_session_path: Path) -> None:
        """Verify trajectory timestamps survive roundtrip."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        original_trajectory = session.get_trajectory()
        original_timestamp = original_trajectory[0].timestamp
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        loaded_trajectory = loaded.get_trajectory()

        # Find the matching entry
        matching = [e for e in loaded_trajectory if e.seq_num == 1]
        assert len(matching) == 1
        loaded_timestamp = matching[0].timestamp

        # Timestamps should match (within microsecond precision)
        delta = abs((loaded_timestamp - original_timestamp).total_seconds())
        assert delta < 0.001

    def test_trajectory_content_preserved(self, temp_session_path: Path) -> None:
        """Verify trajectory entry content survives roundtrip."""
        session = Session()
        complex_content = {
            "input": {"text": "hello", "options": ["a", "b"]},
            "output": {"result": "world"},
            "metrics": {"time_ms": 123.45},
        }
        session.append("agent", EntryType.AGENT_COMPLETED, complex_content)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        # Find the agent entry (not system)
        user_entries = [
            e for e in loaded.get_trajectory()
            if e.agent_id == "agent"
        ]
        assert len(user_entries) == 1
        assert user_entries[0].content == complex_content

    def test_trajectory_entry_types_preserved(self, temp_session_path: Path) -> None:
        """Verify trajectory entry types survive roundtrip."""
        session = Session()
        session.append("agent", EntryType.AGENT_INVOKED, {})
        session.append("agent", EntryType.AGENT_COMPLETED, {})
        session.append("agent", EntryType.AGENT_FAILED, {})
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        user_entries = [
            e for e in loaded.get_trajectory()
            if e.agent_id == "agent"
        ]

        assert user_entries[0].entry_type == EntryType.AGENT_INVOKED
        assert user_entries[1].entry_type == EntryType.AGENT_COMPLETED
        assert user_entries[2].entry_type == EntryType.AGENT_FAILED


class TestRoundtripArtifacts:
    """Tests for artifact preservation."""

    def test_empty_artifacts_preserved(self, temp_session_path: Path) -> None:
        """Verify session with no artifacts survives roundtrip."""
        session = Session()
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.list_artifacts() == []

    def test_single_artifact_preserved(self, temp_session_path: Path) -> None:
        """Verify single artifact survives roundtrip."""
        session = Session()
        session.write_artifact("test.txt", b"Hello, world!")
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.list_artifacts() == ["test.txt"]
        assert loaded.read_artifact("test.txt") == b"Hello, world!"

    def test_multiple_artifacts_preserved(self, temp_session_path: Path) -> None:
        """Verify multiple artifacts survive roundtrip."""
        session = Session()
        session.write_artifact("a.txt", b"AAA")
        session.write_artifact("b.txt", b"BBB")
        session.write_artifact("c.txt", b"CCC")
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.list_artifacts() == ["a.txt", "b.txt", "c.txt"]
        assert loaded.read_artifact("a.txt") == b"AAA"
        assert loaded.read_artifact("b.txt") == b"BBB"
        assert loaded.read_artifact("c.txt") == b"CCC"

    def test_binary_artifact_preserved(self, temp_session_path: Path) -> None:
        """Verify binary artifacts survive roundtrip."""
        session = Session()
        binary_data = bytes(range(256)) * 10
        session.write_artifact("binary.bin", binary_data)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.read_artifact("binary.bin") == binary_data

    def test_large_artifact_preserved(
        self, temp_session_path: Path, large_artifact_data: bytes
    ) -> None:
        """Verify large artifacts survive roundtrip."""
        session = Session()
        session.write_artifact("large.bin", large_artifact_data)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.read_artifact("large.bin") == large_artifact_data


class TestRoundtripConfig:
    """Tests for configuration preservation."""

    def test_max_artifact_size_preserved(self, temp_session_path: Path) -> None:
        """Verify max_artifact_size survives roundtrip."""
        session = Session(max_artifact_size=12345)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        assert loaded.max_artifact_size == 12345


class TestLifecycleTrajectoryEntries:
    """Tests for trajectory entries created during save/load."""

    def test_save_creates_trajectory_entry(self, temp_session_path: Path) -> None:
        """Verify save() adds a SESSION_SAVED entry."""
        session = Session()
        session.save(temp_session_path)

        # Get trajectory before save (from the saved file)
        loaded = Session.load(temp_session_path)
        trajectory = loaded.get_trajectory()

        # Should have SESSION_SAVED entry
        saved_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.SESSION_SAVED
        ]
        assert len(saved_entries) == 1
        assert str(temp_session_path) in saved_entries[0].content["path"]

    def test_load_creates_trajectory_entry(self, temp_session_path: Path) -> None:
        """Verify load() adds a SESSION_LOADED entry."""
        session = Session()
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        trajectory = loaded.get_trajectory()

        # Should have SESSION_LOADED entry
        loaded_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.SESSION_LOADED
        ]
        assert len(loaded_entries) == 1


class TestResumability:
    """Tests for session resumability after load."""

    def test_can_modify_state_after_load(self, temp_session_path: Path) -> None:
        """Verify session can be modified after loading."""
        session = Session()
        session.set("counter", 1)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        loaded.set("counter", 2)
        loaded.set("new_key", "new_value")

        assert loaded.get("counter") == 2
        assert loaded.get("new_key") == "new_value"

    def test_can_append_trajectory_after_load(self, temp_session_path: Path) -> None:
        """Verify trajectory can be appended after loading."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"phase": 1})
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        seq = loaded.append("agent", EntryType.AGENT_COMPLETED, {"phase": 2})

        # Should get next sequence number
        assert seq > 1

    def test_can_add_artifacts_after_load(self, temp_session_path: Path) -> None:
        """Verify artifacts can be added after loading."""
        session = Session()
        session.write_artifact("first.txt", b"first")
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        loaded.write_artifact("second.txt", b"second")

        assert "first.txt" in loaded.list_artifacts()
        assert "second.txt" in loaded.list_artifacts()

    def test_can_save_again_after_load(self, temp_session_path: Path) -> None:
        """Verify session can be saved again after loading."""
        session = Session()
        session.set("value", 1)
        session.save(temp_session_path)

        loaded = Session.load(temp_session_path)
        loaded.set("value", 2)
        loaded.save(temp_session_path)

        reloaded = Session.load(temp_session_path)
        assert reloaded.get("value") == 2


class TestFullWorkflowRoundtrip:
    """Tests for complete workflow roundtrips."""

    def test_complex_session_roundtrip(self, temp_session_path: Path) -> None:
        """
        Verify a complex session with all features survives roundtrip.

        This is the integration test that checks everything together.
        """
        session = Session(session_id="complex-test-session")

        # Add state
        session.set("input_text", "hello world")
        session.set("config", {"mode": "test", "verbose": True})
        session.set("results", [1, 2, 3])

        # Add trajectory
        session.append("agent1", EntryType.AGENT_INVOKED, {"action": "process"})
        session.append("agent1", EntryType.STATE_SET, {"key": "input_text"})
        session.append("agent1", EntryType.AGENT_COMPLETED, {"output": "HELLO WORLD"})

        # Add artifacts
        session.write_artifact("input.txt", b"hello world")
        session.write_artifact("output.txt", b"HELLO WORLD")

        # Save
        session.save(temp_session_path)

        # Load and verify everything
        loaded = Session.load(temp_session_path)

        # Verify session ID
        assert loaded.session_id == "complex-test-session"

        # Verify state
        assert loaded.get("input_text") == "hello world"
        assert loaded.get("config") == {"mode": "test", "verbose": True}
        assert loaded.get("results") == [1, 2, 3]

        # Verify trajectory (find user entries)
        user_entries = [
            e for e in loaded.get_trajectory()
            if e.agent_id == "agent1"
        ]
        assert len(user_entries) == 3
        assert user_entries[0].entry_type == EntryType.AGENT_INVOKED
        assert user_entries[2].content["output"] == "HELLO WORLD"

        # Verify artifacts
        assert loaded.list_artifacts() == ["input.txt", "output.txt"]
        assert loaded.read_artifact("input.txt") == b"hello world"
        assert loaded.read_artifact("output.txt") == b"HELLO WORLD"
