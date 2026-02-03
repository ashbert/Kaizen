"""
Tests for Session snapshot functionality.

This module tests the snapshot_for_agent() method which provides
agents with a read-only view of session state. The key invariant is:

    Snapshots are copies, never live references.

This means agents cannot accidentally (or intentionally) modify the
session state by mutating values in their snapshot.
"""

import pytest

from kaizen.session import Session
from kaizen.types import EntryType


class TestSnapshotCreation:
    """Tests for snapshot creation."""

    def test_snapshot_contains_session_id(self) -> None:
        """Verify snapshot includes session ID."""
        session = Session(session_id="test-session-123")
        snapshot = session.snapshot_for_agent("agent")

        assert snapshot["session_id"] == "test-session-123"

    def test_snapshot_contains_state(self) -> None:
        """Verify snapshot includes current state."""
        session = Session()
        session.set("key1", "value1")
        session.set("key2", 42)

        snapshot = session.snapshot_for_agent("agent")

        assert snapshot["state"]["key1"] == "value1"
        assert snapshot["state"]["key2"] == 42

    def test_snapshot_contains_state_version(self) -> None:
        """Verify snapshot includes state version."""
        session = Session()
        session.set("key", "value")
        session.set("key", "updated")

        snapshot = session.snapshot_for_agent("agent")

        assert snapshot["state_version"] == 2

    def test_snapshot_contains_trajectory(self) -> None:
        """Verify snapshot includes trajectory entries."""
        session = Session()
        session.append("agent1", EntryType.AGENT_COMPLETED, {"n": 1})
        session.append("agent2", EntryType.AGENT_COMPLETED, {"n": 2})

        snapshot = session.snapshot_for_agent("agent")

        # Trajectory entries should be dictionaries (serialized)
        assert isinstance(snapshot["trajectory"], list)
        assert len(snapshot["trajectory"]) == 3  # session_created + 2

    def test_snapshot_contains_artifact_list(self) -> None:
        """Verify snapshot includes artifact names (not data)."""
        session = Session()
        session.write_artifact("file1.txt", b"data1")
        session.write_artifact("file2.txt", b"data2")

        snapshot = session.snapshot_for_agent("agent")

        # Should list names, not data
        assert "file1.txt" in snapshot["artifacts"]
        assert "file2.txt" in snapshot["artifacts"]

    def test_snapshot_contains_timestamp(self) -> None:
        """Verify snapshot includes creation timestamp."""
        session = Session()
        snapshot = session.snapshot_for_agent("agent")

        assert "snapshot_time" in snapshot
        # Should be ISO format string
        assert "T" in snapshot["snapshot_time"]

    def test_snapshot_contains_trajectory_total_length(self) -> None:
        """Verify snapshot includes total trajectory length."""
        session = Session()
        for i in range(10):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})

        snapshot = session.snapshot_for_agent("agent", depth=3)

        # Even with depth=3, should report total length
        assert snapshot["trajectory_total_length"] == 11  # session_created + 10


class TestSnapshotDepth:
    """Tests for snapshot trajectory depth parameter."""

    def test_snapshot_default_depth_is_10(self) -> None:
        """Verify default depth is 10 entries."""
        session = Session()
        for i in range(20):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})

        snapshot = session.snapshot_for_agent("agent")

        # Should have last 10 entries
        assert len(snapshot["trajectory"]) == 10

    def test_snapshot_custom_depth(self) -> None:
        """Verify custom depth is respected."""
        session = Session()
        for i in range(20):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})

        snapshot = session.snapshot_for_agent("agent", depth=5)

        assert len(snapshot["trajectory"]) == 5

    def test_snapshot_depth_larger_than_trajectory(self) -> None:
        """Verify depth larger than trajectory returns all entries."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        snapshot = session.snapshot_for_agent("agent", depth=100)

        # Should return all 2 entries (session_created + 1)
        assert len(snapshot["trajectory"]) == 2

    def test_snapshot_depth_zero(self) -> None:
        """Verify depth=0 returns no trajectory entries."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        snapshot = session.snapshot_for_agent("agent", depth=0)

        assert snapshot["trajectory"] == []
        # But should still report total length
        assert snapshot["trajectory_total_length"] == 2


class TestSnapshotIsolation:
    """Tests for snapshot isolation invariant."""

    def test_mutating_snapshot_state_does_not_affect_session(self) -> None:
        """
        Verify mutating snapshot state doesn't affect session.

        This is the core isolation invariant.
        """
        session = Session()
        session.set("data", {"nested": {"key": "original"}})

        snapshot = session.snapshot_for_agent("agent")

        # Mutate the snapshot
        snapshot["state"]["data"]["nested"]["key"] = "MUTATED"
        snapshot["state"]["new_key"] = "new_value"

        # Session state should be unchanged
        assert session.get("data")["nested"]["key"] == "original"
        assert session.get("new_key") is None

    def test_mutating_snapshot_trajectory_does_not_affect_session(self) -> None:
        """Verify mutating snapshot trajectory doesn't affect session."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        snapshot = session.snapshot_for_agent("agent")

        # Mutate the snapshot trajectory
        snapshot["trajectory"].append({"fake": "entry"})
        snapshot["trajectory"][0]["content"] = {"MUTATED": True}

        # Session trajectory should be unchanged
        trajectory = session.get_trajectory()
        assert len(trajectory) == 2
        assert trajectory[0].content.get("MUTATED") is None

    def test_mutating_snapshot_artifacts_does_not_affect_session(self) -> None:
        """Verify mutating snapshot artifacts list doesn't affect session."""
        session = Session()
        session.write_artifact("test.txt", b"data")

        snapshot = session.snapshot_for_agent("agent")

        # Mutate the snapshot artifacts list
        snapshot["artifacts"].append("fake.txt")
        snapshot["artifacts"].remove("test.txt")

        # Session artifacts should be unchanged
        assert session.list_artifacts() == ["test.txt"]

    def test_multiple_snapshots_are_independent(self) -> None:
        """Verify multiple snapshots are independent copies."""
        session = Session()
        session.set("data", {"value": "original"})

        snap1 = session.snapshot_for_agent("agent1")
        snap2 = session.snapshot_for_agent("agent2")

        # Mutate snap1
        snap1["state"]["data"]["value"] = "mutated_in_snap1"

        # snap2 should be unaffected
        assert snap2["state"]["data"]["value"] == "original"

        # Session should be unaffected
        assert session.get("data")["value"] == "original"

    def test_snapshot_after_mutation_reflects_current_state(self) -> None:
        """Verify new snapshots reflect current session state."""
        session = Session()
        session.set("counter", 1)

        snap1 = session.snapshot_for_agent("agent")
        assert snap1["state"]["counter"] == 1

        # Update session
        session.set("counter", 2)

        # New snapshot should have updated value
        snap2 = session.snapshot_for_agent("agent")
        assert snap2["state"]["counter"] == 2

        # Old snapshot should be unchanged
        assert snap1["state"]["counter"] == 1


class TestSnapshotAgentId:
    """Tests for agent_id parameter in snapshots."""

    def test_snapshot_works_with_any_agent_id(self) -> None:
        """Verify snapshot can be created for any agent_id."""
        session = Session()
        session.set("key", "value")

        # All these should work
        snap1 = session.snapshot_for_agent("agent_1")
        snap2 = session.snapshot_for_agent("my-custom-agent")
        snap3 = session.snapshot_for_agent("Agent With Spaces")

        assert snap1["state"]["key"] == "value"
        assert snap2["state"]["key"] == "value"
        assert snap3["state"]["key"] == "value"


class TestSnapshotTrajectoryFormat:
    """Tests for trajectory format in snapshots."""

    def test_trajectory_entries_are_dicts(self) -> None:
        """Verify trajectory entries are serialized to dicts."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"result": "ok"})

        snapshot = session.snapshot_for_agent("agent")

        for entry in snapshot["trajectory"]:
            assert isinstance(entry, dict)
            assert "seq_num" in entry
            assert "timestamp" in entry
            assert "agent_id" in entry
            assert "entry_type" in entry
            assert "content" in entry

    def test_trajectory_timestamp_is_iso_string(self) -> None:
        """Verify trajectory timestamps are ISO format strings."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        snapshot = session.snapshot_for_agent("agent")

        for entry in snapshot["trajectory"]:
            # ISO format contains T separator
            assert "T" in entry["timestamp"]
            # Should be parseable
            from datetime import datetime
            datetime.fromisoformat(entry["timestamp"])

    def test_trajectory_entry_type_is_string(self) -> None:
        """Verify entry_type is string value, not enum."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        snapshot = session.snapshot_for_agent("agent")

        for entry in snapshot["trajectory"]:
            assert isinstance(entry["entry_type"], str)
