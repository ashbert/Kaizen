"""
Tests for Session trajectory management.

This module tests the trajectory subsystem of the Session class:
- append() operation
- get_trajectory() with limits
- Sequence number auto-increment
- Timestamp generation
- Append-only invariant

The trajectory is an append-only log that records all actions in a session.
These tests verify the core invariants:
- Entries are strictly ordered by sequence number
- Sequence numbers are monotonically increasing
- Entries cannot be modified or deleted
- Each entry has a UTC timestamp
"""

from datetime import datetime, timezone, timedelta

import pytest

from trace.session import Session
from trace.types import EntryType


class TestTrajectoryAppend:
    """Tests for trajectory append operation."""

    def test_append_returns_sequence_number(self) -> None:
        """Verify append() returns the assigned sequence number."""
        session = Session()
        # First user entry will have seq_num 2 (after session_created)
        seq = session.append(
            agent_id="test_agent",
            entry_type=EntryType.AGENT_COMPLETED,
            content={"action": "test"},
        )
        assert seq == 2

    def test_append_sequence_numbers_increment(self) -> None:
        """Verify sequence numbers auto-increment."""
        session = Session()

        seq1 = session.append("agent", EntryType.AGENT_INVOKED, {"n": 1})
        seq2 = session.append("agent", EntryType.AGENT_COMPLETED, {"n": 2})
        seq3 = session.append("agent", EntryType.AGENT_INVOKED, {"n": 3})

        assert seq2 == seq1 + 1
        assert seq3 == seq2 + 1

    def test_append_creates_entry_with_correct_fields(self) -> None:
        """Verify appended entry has all expected fields."""
        session = Session()
        before = datetime.now(timezone.utc)

        session.append(
            agent_id="my_agent",
            entry_type=EntryType.STATE_SET,
            content={"key": "test", "value": 42},
        )

        after = datetime.now(timezone.utc)

        # Get the last entry
        trajectory = session.get_trajectory()
        entry = trajectory[-1]

        assert entry.agent_id == "my_agent"
        assert entry.entry_type == EntryType.STATE_SET
        assert entry.content == {"key": "test", "value": 42}
        assert before <= entry.timestamp <= after
        assert entry.timestamp.tzinfo is not None  # UTC timezone

    def test_append_validates_content_json_serializable(self) -> None:
        """Verify append() rejects non-JSON-serializable content."""
        session = Session()

        with pytest.raises(ValueError, match="JSON-serializable"):
            session.append(
                agent_id="agent",
                entry_type=EntryType.AGENT_COMPLETED,
                content={"func": lambda x: x},  # Not serializable
            )

    def test_append_stores_copy_of_content(self) -> None:
        """Verify append() stores a copy, not reference."""
        session = Session()
        content = {"data": [1, 2, 3]}

        session.append("agent", EntryType.AGENT_COMPLETED, content)

        # Mutate original
        content["data"].append(999)
        content["new_key"] = "new_value"

        # Entry should be unchanged
        entry = session.get_trajectory()[-1]
        assert entry.content == {"data": [1, 2, 3]}


class TestTrajectoryRetrieval:
    """Tests for trajectory retrieval operations."""

    def test_get_trajectory_returns_all_entries(self) -> None:
        """Verify get_trajectory() returns all entries when no limit."""
        session = Session()

        # Add several entries
        for i in range(5):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})

        trajectory = session.get_trajectory()

        # Should have session_created + 5 user entries
        assert len(trajectory) == 6

    def test_get_trajectory_with_limit(self) -> None:
        """Verify get_trajectory(limit) returns most recent N entries."""
        session = Session()

        # Add several entries
        for i in range(10):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})

        # Get last 3
        recent = session.get_trajectory(limit=3)

        assert len(recent) == 3
        # Should be the last 3 entries
        assert recent[0].content["n"] == 7
        assert recent[1].content["n"] == 8
        assert recent[2].content["n"] == 9

    def test_get_trajectory_limit_larger_than_length(self) -> None:
        """Verify limit larger than trajectory returns all entries."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        trajectory = session.get_trajectory(limit=100)

        # Should return all (2 entries: session_created + 1)
        assert len(trajectory) == 2

    def test_get_trajectory_limit_zero_returns_empty(self) -> None:
        """Verify limit=0 returns empty list."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        trajectory = session.get_trajectory(limit=0)
        assert trajectory == []

    def test_get_trajectory_negative_limit_returns_empty(self) -> None:
        """Verify negative limit returns empty list."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        trajectory = session.get_trajectory(limit=-5)
        assert trajectory == []

    def test_get_trajectory_preserves_order(self) -> None:
        """Verify entries are returned in chronological order."""
        session = Session()

        for i in range(5):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})

        trajectory = session.get_trajectory()

        # Sequence numbers should be strictly increasing
        seq_nums = [e.seq_num for e in trajectory]
        assert seq_nums == sorted(seq_nums)

        # Timestamps should be non-decreasing
        timestamps = [e.timestamp for e in trajectory]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_get_trajectory_length(self) -> None:
        """Verify get_trajectory_length() returns correct count."""
        session = Session()
        assert session.get_trajectory_length() == 1  # session_created

        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})
        assert session.get_trajectory_length() == 2

        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 2})
        assert session.get_trajectory_length() == 3


class TestTrajectoryOrdering:
    """Tests for trajectory ordering invariants."""

    def test_sequence_numbers_start_at_one(self) -> None:
        """Verify sequence numbers are 1-indexed for readability."""
        session = Session()
        trajectory = session.get_trajectory()

        assert trajectory[0].seq_num == 1

    def test_sequence_numbers_are_consecutive(self) -> None:
        """Verify sequence numbers have no gaps."""
        session = Session()

        for i in range(10):
            session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})

        trajectory = session.get_trajectory()
        seq_nums = [e.seq_num for e in trajectory]

        # Should be [1, 2, 3, ..., 11]
        assert seq_nums == list(range(1, 12))

    def test_sequence_numbers_monotonically_increasing(self) -> None:
        """
        Verify sequence numbers never decrease.

        This is a core invariant for ordering.
        """
        session = Session()

        prev_seq = 0
        for i in range(100):
            seq = session.append("agent", EntryType.AGENT_COMPLETED, {"n": i})
            assert seq > prev_seq
            prev_seq = seq

    def test_timestamps_have_utc_timezone(self) -> None:
        """Verify all timestamps are UTC."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        for entry in session.get_trajectory():
            assert entry.timestamp.tzinfo == timezone.utc


class TestTrajectoryImmutability:
    """Tests for trajectory append-only invariant."""

    def test_trajectory_entry_is_immutable(self) -> None:
        """Verify trajectory entries cannot be modified."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {"n": 1})

        entry = session.get_trajectory()[-1]

        # Attempting to modify should raise
        with pytest.raises(AttributeError):
            entry.content = {"modified": True}  # type: ignore

        with pytest.raises(AttributeError):
            entry.seq_num = 999  # type: ignore

    def test_no_delete_method(self) -> None:
        """Verify there's no way to delete trajectory entries."""
        session = Session()

        # Session should not have delete methods for trajectory
        assert not hasattr(session, "delete_trajectory")
        assert not hasattr(session, "remove_entry")
        assert not hasattr(session, "clear_trajectory")

    def test_no_update_method(self) -> None:
        """Verify there's no way to update trajectory entries."""
        session = Session()

        # Session should not have update methods for trajectory
        assert not hasattr(session, "update_entry")
        assert not hasattr(session, "modify_trajectory")


class TestTrajectoryEntryTypes:
    """Tests for various trajectory entry types."""

    def test_all_entry_types_can_be_appended(self) -> None:
        """Verify all EntryType values can be used in append()."""
        session = Session()

        for entry_type in EntryType:
            # Skip session lifecycle types that are system-managed
            if entry_type in (
                EntryType.SESSION_CREATED,
                EntryType.SESSION_LOADED,
                EntryType.SESSION_SAVED,
            ):
                continue

            session.append(
                agent_id="test",
                entry_type=entry_type,
                content={"type": entry_type.value},
            )

        # Should have session_created + all other types
        trajectory = session.get_trajectory()
        assert len(trajectory) > 1

    def test_agent_attributed_entries(self) -> None:
        """Verify entries from different agents are properly attributed."""
        session = Session()

        session.append("agent_1", EntryType.AGENT_INVOKED, {"step": 1})
        session.append("agent_2", EntryType.AGENT_COMPLETED, {"step": 2})
        session.append("agent_1", EntryType.AGENT_COMPLETED, {"step": 3})

        trajectory = session.get_trajectory()
        user_entries = [e for e in trajectory if e.agent_id != "system"]

        assert len(user_entries) == 3
        assert user_entries[0].agent_id == "agent_1"
        assert user_entries[1].agent_id == "agent_2"
        assert user_entries[2].agent_id == "agent_1"


class TestTrajectoryContent:
    """Tests for trajectory entry content handling."""

    def test_content_can_be_empty_dict(self) -> None:
        """Verify content can be an empty dictionary."""
        session = Session()
        session.append("agent", EntryType.AGENT_COMPLETED, {})

        entry = session.get_trajectory()[-1]
        assert entry.content == {}

    def test_content_preserves_nested_structure(self) -> None:
        """Verify complex nested content is preserved."""
        session = Session()
        complex_content = {
            "input": {"text": "hello", "options": ["a", "b"]},
            "output": {"result": "world"},
            "metrics": {"time_ms": 123.45},
            "flags": [True, False],
            "nullable": None,
        }

        session.append("agent", EntryType.AGENT_COMPLETED, complex_content)

        entry = session.get_trajectory()[-1]
        assert entry.content == complex_content

    def test_content_types(self) -> None:
        """Verify various JSON types in content are preserved."""
        session = Session()

        content = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool_true": True,
            "bool_false": False,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": "value"},
        }

        session.append("agent", EntryType.AGENT_COMPLETED, content)

        entry = session.get_trajectory()[-1]
        assert entry.content["string"] == "hello"
        assert entry.content["int"] == 42
        assert entry.content["float"] == 3.14
        assert entry.content["bool_true"] is True
        assert entry.content["bool_false"] is False
        assert entry.content["null"] is None
        assert entry.content["array"] == [1, 2, 3]
        assert entry.content["object"] == {"nested": "value"}
