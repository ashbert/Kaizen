"""
Tests for Session state management.

This module tests the state subsystem of the Session class:
- get() / set() operations
- State versioning (monotonic increments)
- JSON serialization of values
- Isolation (mutations don't affect internal state)

The state is a versioned key-value store that forms the "working memory"
of a session. These tests verify the core invariants:
- State version increments on every set()
- State version is monotonically increasing
- Values are JSON-serializable
- External mutations don't affect internal state
"""

import pytest
from trace.session import Session


class TestSessionCreation:
    """Tests for Session initialization."""

    def test_create_session_generates_uuid(self) -> None:
        """
        Verify new session gets a unique UUID.

        Sessions should have globally unique identifiers to prevent
        collision when saving/loading multiple sessions.
        """
        session = Session()

        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        assert len(session.session_id) == 36
        assert session.session_id.count("-") == 4

    def test_create_session_with_custom_id(self) -> None:
        """Verify session can be created with a custom ID."""
        session = Session(session_id="custom-session-123")
        assert session.session_id == "custom-session-123"

    def test_create_multiple_sessions_have_different_ids(self) -> None:
        """Verify each new session has a unique ID."""
        sessions = [Session() for _ in range(10)]
        ids = [s.session_id for s in sessions]

        # All IDs should be unique
        assert len(set(ids)) == 10

    def test_session_initial_state_version_is_zero(self) -> None:
        """
        Verify new session starts with state version 0.

        A fresh session with no set() calls should have version 0.
        """
        session = Session()
        assert session.get_state_version() == 0

    def test_session_default_artifact_size(self) -> None:
        """Verify default max artifact size is 100MB."""
        session = Session()
        assert session.max_artifact_size == 100 * 1024 * 1024

    def test_session_custom_artifact_size(self) -> None:
        """Verify custom max artifact size is respected."""
        session = Session(max_artifact_size=1024)
        assert session.max_artifact_size == 1024

    def test_session_creates_trajectory_entry_on_init(self) -> None:
        """
        Verify session creation is recorded in trajectory.

        The SESSION_CREATED entry provides an audit trail of when
        and how the session was initialized.
        """
        session = Session()
        trajectory = session.get_trajectory()

        assert len(trajectory) == 1
        entry = trajectory[0]
        assert entry.entry_type.value == "session_created"
        assert entry.agent_id == "system"
        assert entry.content["session_id"] == session.session_id


class TestStateGetSet:
    """Tests for state get/set operations."""

    def test_set_and_get_string(self) -> None:
        """Verify string values can be stored and retrieved."""
        session = Session()
        session.set("message", "hello world")
        assert session.get("message") == "hello world"

    def test_set_and_get_integer(self) -> None:
        """Verify integer values can be stored and retrieved."""
        session = Session()
        session.set("count", 42)
        assert session.get("count") == 42

    def test_set_and_get_float(self) -> None:
        """Verify float values can be stored and retrieved."""
        session = Session()
        session.set("ratio", 3.14159)
        assert session.get("ratio") == 3.14159

    def test_set_and_get_boolean(self) -> None:
        """Verify boolean values can be stored and retrieved."""
        session = Session()
        session.set("enabled", True)
        session.set("disabled", False)
        assert session.get("enabled") is True
        assert session.get("disabled") is False

    def test_set_and_get_none(self) -> None:
        """Verify None values can be stored and retrieved."""
        session = Session()
        session.set("empty", None)
        assert session.get("empty") is None

    def test_set_and_get_list(self) -> None:
        """Verify list values can be stored and retrieved."""
        session = Session()
        session.set("items", [1, 2, 3, "four"])
        assert session.get("items") == [1, 2, 3, "four"]

    def test_set_and_get_dict(self) -> None:
        """Verify dict values can be stored and retrieved."""
        session = Session()
        session.set("metadata", {"author": "test", "version": 1})
        assert session.get("metadata") == {"author": "test", "version": 1}

    def test_set_and_get_nested_structure(self) -> None:
        """Verify complex nested structures can be stored and retrieved."""
        session = Session()
        complex_value = {
            "users": [
                {"name": "Alice", "scores": [95, 87, 92]},
                {"name": "Bob", "scores": [88, 91, 85]},
            ],
            "metadata": {"timestamp": "2024-01-01", "version": 2},
            "flags": [True, False, None],
        }
        session.set("data", complex_value)
        assert session.get("data") == complex_value

    def test_get_nonexistent_key_returns_none(self) -> None:
        """Verify get() returns None for missing keys by default."""
        session = Session()
        assert session.get("nonexistent") is None

    def test_get_nonexistent_key_with_default(self) -> None:
        """Verify get() returns custom default for missing keys."""
        session = Session()
        assert session.get("missing", default="fallback") == "fallback"
        assert session.get("missing", default=42) == 42
        assert session.get("missing", default=[]) == []

    def test_set_overwrites_existing_value(self) -> None:
        """Verify set() overwrites existing values."""
        session = Session()
        session.set("key", "first")
        session.set("key", "second")
        assert session.get("key") == "second"

    def test_set_empty_key_raises_error(self) -> None:
        """Verify set() rejects empty keys."""
        session = Session()
        with pytest.raises(ValueError, match="non-empty string"):
            session.set("", "value")

    def test_set_non_string_key_raises_error(self) -> None:
        """Verify set() rejects non-string keys."""
        session = Session()
        with pytest.raises(ValueError):
            session.set(123, "value")  # type: ignore

    def test_set_non_json_serializable_raises_error(self) -> None:
        """Verify set() rejects non-JSON-serializable values."""
        session = Session()

        # Functions are not JSON-serializable
        with pytest.raises(ValueError, match="JSON-serializable"):
            session.set("func", lambda x: x)

        # Custom objects are not JSON-serializable
        class CustomObject:
            pass

        with pytest.raises(ValueError, match="JSON-serializable"):
            session.set("obj", CustomObject())


class TestStateVersioning:
    """Tests for state version behavior."""

    def test_version_starts_at_zero(self) -> None:
        """Verify version is 0 before any set() calls."""
        session = Session()
        assert session.get_state_version() == 0

    def test_version_increments_on_set(self) -> None:
        """Verify version increments by 1 on each set()."""
        session = Session()

        assert session.get_state_version() == 0

        session.set("a", 1)
        assert session.get_state_version() == 1

        session.set("b", 2)
        assert session.get_state_version() == 2

        session.set("c", 3)
        assert session.get_state_version() == 3

    def test_version_increments_on_overwrite(self) -> None:
        """Verify version increments even when overwriting same key."""
        session = Session()

        session.set("key", "value1")
        v1 = session.get_state_version()

        session.set("key", "value2")
        v2 = session.get_state_version()

        assert v2 == v1 + 1

    def test_set_returns_new_version(self) -> None:
        """Verify set() returns the new version number."""
        session = Session()

        v1 = session.set("a", 1)
        v2 = session.set("b", 2)
        v3 = session.set("c", 3)

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_version_is_monotonically_increasing(self) -> None:
        """
        Verify version only increases, never decreases.

        This is a core invariant: the version number provides
        a total ordering of state changes.
        """
        session = Session()
        versions = []

        for i in range(100):
            v = session.set(f"key_{i}", i)
            versions.append(v)

        # Each version should be greater than the previous
        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1]

    def test_get_does_not_change_version(self) -> None:
        """Verify get() is read-only and doesn't change version."""
        session = Session()
        session.set("key", "value")
        v1 = session.get_state_version()

        # Multiple get() calls
        session.get("key")
        session.get("nonexistent")
        session.get("key", default="fallback")

        v2 = session.get_state_version()
        assert v2 == v1


class TestStateIsolation:
    """Tests for state isolation invariant."""

    def test_get_returns_copy_not_reference(self) -> None:
        """
        Verify get() returns a copy, not a reference.

        Mutating the returned value should not affect the session state.
        """
        session = Session()
        original = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        session.set("data", original)

        # Get the value and mutate it
        retrieved = session.get("data")
        retrieved["nested"]["key"] = "MUTATED"
        retrieved["list"].append(999)
        retrieved["new_key"] = "new_value"

        # Original state should be unchanged
        stored = session.get("data")
        assert stored["nested"]["key"] == "value"
        assert stored["list"] == [1, 2, 3]
        assert "new_key" not in stored

    def test_set_stores_copy_not_reference(self) -> None:
        """
        Verify set() stores a copy, not a reference.

        Mutating the original value after set() should not affect state.
        """
        session = Session()
        original = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        session.set("data", original)

        # Mutate the original after set()
        original["nested"]["key"] = "MUTATED"
        original["list"].append(999)

        # State should have the original values
        stored = session.get("data")
        assert stored["nested"]["key"] == "value"
        assert stored["list"] == [1, 2, 3]

    def test_multiple_gets_return_independent_copies(self) -> None:
        """Verify each get() returns an independent copy."""
        session = Session()
        session.set("data", {"key": "value"})

        copy1 = session.get("data")
        copy2 = session.get("data")

        # Mutate one copy
        copy1["key"] = "mutated"

        # Other copy should be unaffected
        assert copy2["key"] == "value"

        # State should be unaffected
        assert session.get("data")["key"] == "value"


class TestGetAllState:
    """Tests for get_all_state() method."""

    def test_get_all_state_empty_session(self) -> None:
        """Verify get_all_state() returns empty dict for new session."""
        session = Session()
        assert session.get_all_state() == {}

    def test_get_all_state_with_data(self) -> None:
        """Verify get_all_state() returns all key-value pairs."""
        session = Session()
        session.set("a", 1)
        session.set("b", "two")
        session.set("c", [3])

        state = session.get_all_state()
        assert state == {"a": 1, "b": "two", "c": [3]}

    def test_get_all_state_returns_copy(self) -> None:
        """Verify get_all_state() returns a copy, not reference."""
        session = Session()
        session.set("data", {"key": "value"})

        state = session.get_all_state()
        state["data"]["key"] = "MUTATED"
        state["new_key"] = "new_value"

        # State should be unaffected
        assert session.get("data")["key"] == "value"
        assert session.get("new_key") is None


class TestStateTrajectoryRecording:
    """Tests for trajectory recording of state changes."""

    def test_set_records_trajectory_entry(self) -> None:
        """Verify set() creates a trajectory entry."""
        session = Session()
        initial_len = session.get_trajectory_length()

        session.set("key", "value")

        assert session.get_trajectory_length() == initial_len + 1

    def test_set_trajectory_entry_content(self) -> None:
        """Verify set() trajectory entry has correct content."""
        session = Session()
        session.set("key", "new_value")

        trajectory = session.get_trajectory()
        # Find the state_set entry (skip session_created)
        state_entries = [
            e for e in trajectory if e.entry_type.value == "state_set"
        ]
        assert len(state_entries) == 1

        entry = state_entries[0]
        assert entry.agent_id == "system"
        assert entry.content["key"] == "key"
        assert entry.content["old_value"] is None
        assert entry.content["new_value"] == "new_value"
        assert entry.content["state_version"] == 1

    def test_set_records_old_value_on_overwrite(self) -> None:
        """Verify set() records old value when overwriting."""
        session = Session()
        session.set("key", "first")
        session.set("key", "second")

        trajectory = session.get_trajectory()
        state_entries = [
            e for e in trajectory if e.entry_type.value == "state_set"
        ]
        assert len(state_entries) == 2

        # First set: old_value should be None
        assert state_entries[0].content["old_value"] is None
        assert state_entries[0].content["new_value"] == "first"

        # Second set: old_value should be "first"
        assert state_entries[1].content["old_value"] == "first"
        assert state_entries[1].content["new_value"] == "second"
