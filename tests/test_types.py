"""
Tests for core data types in Kaizen.

This module tests the fundamental data structures:
- TrajectoryEntry: Immutable trajectory records
- InvokeResult: Agent invocation results
- CapabilityCall: Capability invocation requests
- AgentInfo: Agent metadata
- EntryType and ErrorCode enums

Tests verify:
1. Correct construction and field access
2. Immutability constraints (frozen dataclasses)
3. Validation logic in __post_init__
4. Serialization/deserialization roundtrips
5. Factory methods work correctly
"""

from datetime import datetime, timezone, timedelta
import pytest

from kaizen.types import (
    TrajectoryEntry,
    InvokeResult,
    CapabilityCall,
    AgentInfo,
    EntryType,
    ErrorCode,
)


# =============================================================================
# ENTRY TYPE ENUM TESTS
# =============================================================================


class TestEntryType:
    """Tests for the EntryType enum."""

    def test_entry_type_values_are_strings(self) -> None:
        """
        Verify EntryType values are strings for JSON serialization.

        The enum inherits from str, so .value should return the string
        directly usable in JSON.
        """
        assert EntryType.AGENT_INVOKED.value == "agent_invoked"
        assert EntryType.STATE_SET.value == "state_set"
        assert EntryType.SESSION_CREATED.value == "session_created"

    def test_entry_type_string_conversion(self) -> None:
        """
        Verify EntryType can be compared with strings directly.

        Since it inherits from str, it should be comparable to string values.
        The .value attribute gives the raw string.
        """
        entry_type = EntryType.AGENT_COMPLETED
        # Direct comparison with string works due to str inheritance
        assert entry_type == "agent_completed"
        # Use .value for explicit string conversion
        assert entry_type.value == "agent_completed"

    def test_entry_type_from_string(self) -> None:
        """
        Verify EntryType can be constructed from string values.

        This is important for deserialization from JSON/database.
        """
        assert EntryType("agent_invoked") == EntryType.AGENT_INVOKED
        assert EntryType("state_set") == EntryType.STATE_SET

    def test_all_entry_types_defined(self) -> None:
        """
        Verify all expected entry types are defined.

        This serves as documentation and catches accidental deletions.
        """
        expected_types = {
            "agent_invoked", "agent_completed", "agent_failed",
            "state_set", "state_deleted",
            "artifact_written", "artifact_deleted",
            "session_created", "session_loaded", "session_saved",
            "plan_created", "plan_step_started",
            "user_input", "system_note",
        }
        actual_types = {e.value for e in EntryType}
        assert actual_types == expected_types


# =============================================================================
# ERROR CODE ENUM TESTS
# =============================================================================


class TestErrorCode:
    """Tests for the ErrorCode enum."""

    def test_error_code_values_are_strings(self) -> None:
        """Verify ErrorCode values are strings."""
        assert ErrorCode.AGENT_NOT_FOUND.value == "agent_not_found"
        assert ErrorCode.PERSIST_SAVE_FAILED.value == "persist_save_failed"

    def test_error_code_from_string(self) -> None:
        """Verify ErrorCode can be constructed from string values."""
        assert ErrorCode("agent_not_found") == ErrorCode.AGENT_NOT_FOUND

    def test_error_code_categories_exist(self) -> None:
        """
        Verify error codes exist for each major category.

        Categories: session, agent, dispatch, plan, persist, artifact
        """
        # Check at least one error code exists per category
        session_codes = [e for e in ErrorCode if e.value.startswith("session_")]
        agent_codes = [e for e in ErrorCode if e.value.startswith("agent_")]
        dispatch_codes = [e for e in ErrorCode if e.value.startswith("dispatch_")]
        plan_codes = [e for e in ErrorCode if e.value.startswith("plan_")]
        persist_codes = [e for e in ErrorCode if e.value.startswith("persist_")]
        artifact_codes = [e for e in ErrorCode if e.value.startswith("artifact_")]

        assert len(session_codes) >= 1
        assert len(agent_codes) >= 1
        assert len(dispatch_codes) >= 1
        assert len(plan_codes) >= 1
        assert len(persist_codes) >= 1
        assert len(artifact_codes) >= 1


# =============================================================================
# TRAJECTORY ENTRY TESTS
# =============================================================================


class TestTrajectoryEntry:
    """Tests for TrajectoryEntry dataclass."""

    @pytest.fixture
    def valid_entry(self) -> TrajectoryEntry:
        """Create a valid trajectory entry for testing."""
        return TrajectoryEntry(
            seq_num=1,
            timestamp=datetime.now(timezone.utc),
            agent_id="test_agent",
            entry_type=EntryType.AGENT_COMPLETED,
            content={"action": "test", "result": "success"},
        )

    def test_create_valid_entry(self, valid_entry: TrajectoryEntry) -> None:
        """Verify a valid entry can be created with all fields."""
        assert valid_entry.seq_num == 1
        assert valid_entry.agent_id == "test_agent"
        assert valid_entry.entry_type == EntryType.AGENT_COMPLETED
        assert valid_entry.content == {"action": "test", "result": "success"}

    def test_entry_is_immutable(self, valid_entry: TrajectoryEntry) -> None:
        """
        Verify TrajectoryEntry is immutable (frozen).

        This enforces the append-only invariant - entries cannot be modified.
        """
        with pytest.raises(AttributeError):
            valid_entry.seq_num = 2  # type: ignore

        with pytest.raises(AttributeError):
            valid_entry.content = {}  # type: ignore

    def test_seq_num_must_be_positive(self) -> None:
        """Verify seq_num validation rejects non-positive values."""
        with pytest.raises(ValueError, match="seq_num must be >= 1"):
            TrajectoryEntry(
                seq_num=0,
                timestamp=datetime.now(timezone.utc),
                agent_id="test",
                entry_type=EntryType.AGENT_COMPLETED,
                content={},
            )

        with pytest.raises(ValueError, match="seq_num must be >= 1"):
            TrajectoryEntry(
                seq_num=-1,
                timestamp=datetime.now(timezone.utc),
                agent_id="test",
                entry_type=EntryType.AGENT_COMPLETED,
                content={},
            )

    def test_agent_id_cannot_be_empty(self) -> None:
        """Verify agent_id validation rejects empty strings."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            TrajectoryEntry(
                seq_num=1,
                timestamp=datetime.now(timezone.utc),
                agent_id="",
                entry_type=EntryType.AGENT_COMPLETED,
                content={},
            )

        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            TrajectoryEntry(
                seq_num=1,
                timestamp=datetime.now(timezone.utc),
                agent_id="   ",  # whitespace only
                entry_type=EntryType.AGENT_COMPLETED,
                content={},
            )

    def test_timestamp_must_be_timezone_aware(self) -> None:
        """
        Verify timestamp validation requires timezone info.

        Naive datetimes (without timezone) are rejected to ensure
        consistent UTC timestamps throughout the system.
        """
        naive_timestamp = datetime.now()  # No timezone!

        with pytest.raises(ValueError, match="timezone-aware"):
            TrajectoryEntry(
                seq_num=1,
                timestamp=naive_timestamp,
                agent_id="test",
                entry_type=EntryType.AGENT_COMPLETED,
                content={},
            )

    def test_to_dict_serialization(self, valid_entry: TrajectoryEntry) -> None:
        """Verify to_dict produces a JSON-serializable dictionary."""
        data = valid_entry.to_dict()

        assert data["seq_num"] == 1
        assert data["agent_id"] == "test_agent"
        assert data["entry_type"] == "agent_completed"  # String, not enum
        assert data["content"] == {"action": "test", "result": "success"}
        # Timestamp should be ISO format string
        assert isinstance(data["timestamp"], str)
        assert "T" in data["timestamp"]  # ISO format contains T

    def test_from_dict_deserialization(self) -> None:
        """Verify from_dict correctly reconstructs an entry."""
        timestamp = datetime.now(timezone.utc)
        data = {
            "seq_num": 42,
            "timestamp": timestamp.isoformat(),
            "agent_id": "loaded_agent",
            "entry_type": "state_set",
            "content": {"key": "value"},
        }

        entry = TrajectoryEntry.from_dict(data)

        assert entry.seq_num == 42
        assert entry.agent_id == "loaded_agent"
        assert entry.entry_type == EntryType.STATE_SET
        assert entry.content == {"key": "value"}
        # Timestamps should match (allowing for microsecond precision)
        assert abs((entry.timestamp - timestamp).total_seconds()) < 0.001

    def test_roundtrip_serialization(self, valid_entry: TrajectoryEntry) -> None:
        """Verify to_dict -> from_dict preserves all data."""
        data = valid_entry.to_dict()
        restored = TrajectoryEntry.from_dict(data)

        assert restored.seq_num == valid_entry.seq_num
        assert restored.agent_id == valid_entry.agent_id
        assert restored.entry_type == valid_entry.entry_type
        assert restored.content == valid_entry.content
        # Check timestamp (may have slight precision differences)
        delta = abs((restored.timestamp - valid_entry.timestamp).total_seconds())
        assert delta < 0.001

    def test_content_can_be_complex_nested_structure(self) -> None:
        """Verify content can hold complex nested JSON structures."""
        complex_content = {
            "action": "transform",
            "input": {"text": "hello", "options": ["a", "b"]},
            "output": {"result": "HELLO", "metadata": {"length": 5}},
            "metrics": [1.5, 2.0, 3.14],
            "flags": {"verbose": True, "debug": False},
            "nullable": None,
        }

        entry = TrajectoryEntry(
            seq_num=1,
            timestamp=datetime.now(timezone.utc),
            agent_id="test",
            entry_type=EntryType.AGENT_COMPLETED,
            content=complex_content,
        )

        assert entry.content == complex_content

        # Verify roundtrip preserves complex structure
        restored = TrajectoryEntry.from_dict(entry.to_dict())
        assert restored.content == complex_content


# =============================================================================
# INVOKE RESULT TESTS
# =============================================================================


class TestInvokeResult:
    """Tests for InvokeResult dataclass."""

    def test_create_success_result_directly(self) -> None:
        """Verify successful result can be created directly."""
        result = InvokeResult(
            success=True,
            result={"output": "test"},
            error=None,
            agent_id="test_agent",
            capability="test_cap",
        )

        assert result.success is True
        assert result.result == {"output": "test"}
        assert result.error is None
        assert result.agent_id == "test_agent"
        assert result.capability == "test_cap"

    def test_create_failure_result_directly(self) -> None:
        """Verify failed result can be created directly."""
        result = InvokeResult(
            success=False,
            result=None,
            error={"error_code": "test_error", "message": "Something went wrong"},
            agent_id="test_agent",
            capability="test_cap",
        )

        assert result.success is False
        assert result.result is None
        assert result.error is not None
        assert result.error["error_code"] == "test_error"

    def test_result_is_immutable(self) -> None:
        """Verify InvokeResult is immutable."""
        result = InvokeResult.ok(
            result="test",
            agent_id="agent",
            capability="cap",
        )

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore

    def test_success_result_cannot_have_error(self) -> None:
        """Verify validation rejects success=True with error info."""
        with pytest.raises(ValueError, match="should not have error"):
            InvokeResult(
                success=True,
                result="test",
                error={"message": "oops"},  # Invalid!
                agent_id="agent",
                capability="cap",
            )

    def test_failed_result_must_have_error(self) -> None:
        """Verify validation requires error info when success=False."""
        with pytest.raises(ValueError, match="must have error"):
            InvokeResult(
                success=False,
                result=None,
                error=None,  # Invalid!
                agent_id="agent",
                capability="cap",
            )

    def test_ok_factory_method(self) -> None:
        """Verify InvokeResult.ok() creates correct success result."""
        result = InvokeResult.ok(
            result={"data": [1, 2, 3]},
            agent_id="my_agent",
            capability="process",
        )

        assert result.success is True
        assert result.result == {"data": [1, 2, 3]}
        assert result.error is None
        assert result.agent_id == "my_agent"
        assert result.capability == "process"

    def test_fail_factory_method(self) -> None:
        """Verify InvokeResult.fail() creates correct failure result."""
        result = InvokeResult.fail(
            error_code=ErrorCode.AGENT_INVALID_PARAMS,
            message="Missing required parameter",
            agent_id="my_agent",
            capability="process",
        )

        assert result.success is False
        assert result.result is None
        assert result.error is not None
        assert result.error["error_code"] == "agent_invalid_params"
        assert result.error["message"] == "Missing required parameter"
        assert result.agent_id == "my_agent"
        assert result.capability == "process"

    def test_fail_with_details(self) -> None:
        """Verify InvokeResult.fail() can include additional details."""
        result = InvokeResult.fail(
            error_code=ErrorCode.AGENT_INVOCATION_FAILED,
            message="Execution failed",
            agent_id="agent",
            capability="cap",
            details={"exception": "ValueError", "traceback": "..."},
        )

        assert result.error is not None
        assert result.error["details"]["exception"] == "ValueError"

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict produces JSON-serializable output."""
        result = InvokeResult.ok(
            result={"processed": True},
            agent_id="agent",
            capability="cap",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["result"] == {"processed": True}
        assert data["error"] is None
        assert data["agent_id"] == "agent"
        assert data["capability"] == "cap"

    def test_result_can_be_any_json_type(self) -> None:
        """Verify result field accepts any JSON-serializable value."""
        # String
        r1 = InvokeResult.ok(result="hello", agent_id="a", capability="c")
        assert r1.result == "hello"

        # Number
        r2 = InvokeResult.ok(result=42, agent_id="a", capability="c")
        assert r2.result == 42

        # List
        r3 = InvokeResult.ok(result=[1, 2, 3], agent_id="a", capability="c")
        assert r3.result == [1, 2, 3]

        # None (valid JSON null)
        r4 = InvokeResult.ok(result=None, agent_id="a", capability="c")
        assert r4.result is None

        # Boolean
        r5 = InvokeResult.ok(result=True, agent_id="a", capability="c")
        assert r5.result is True


# =============================================================================
# CAPABILITY CALL TESTS
# =============================================================================


class TestCapabilityCall:
    """Tests for CapabilityCall dataclass."""

    def test_create_with_params(self) -> None:
        """Verify CapabilityCall can be created with parameters."""
        call = CapabilityCall(
            capability="reverse",
            params={"key": "text", "option": True},
        )

        assert call.capability == "reverse"
        assert call.params == {"key": "text", "option": True}

    def test_create_without_params(self) -> None:
        """Verify CapabilityCall can be created with default empty params."""
        call = CapabilityCall(capability="status")

        assert call.capability == "status"
        assert call.params == {}

    def test_call_is_immutable(self) -> None:
        """Verify CapabilityCall is immutable."""
        call = CapabilityCall(capability="test", params={})

        with pytest.raises(AttributeError):
            call.capability = "other"  # type: ignore

    def test_capability_cannot_be_empty(self) -> None:
        """Verify validation rejects empty capability name."""
        with pytest.raises(ValueError, match="capability cannot be empty"):
            CapabilityCall(capability="")

        with pytest.raises(ValueError, match="capability cannot be empty"):
            CapabilityCall(capability="   ")

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict produces correct output."""
        call = CapabilityCall(
            capability="transform",
            params={"input": "hello"},
        )

        data = call.to_dict()

        assert data["capability"] == "transform"
        assert data["params"] == {"input": "hello"}

    def test_from_dict_with_params(self) -> None:
        """Verify from_dict correctly creates call with params."""
        data = {
            "capability": "process",
            "params": {"x": 1, "y": 2},
        }

        call = CapabilityCall.from_dict(data)

        assert call.capability == "process"
        assert call.params == {"x": 1, "y": 2}

    def test_from_dict_without_params(self) -> None:
        """Verify from_dict handles missing params field."""
        data = {"capability": "status"}

        call = CapabilityCall.from_dict(data)

        assert call.capability == "status"
        assert call.params == {}

    def test_roundtrip_serialization(self) -> None:
        """Verify to_dict -> from_dict preserves data."""
        original = CapabilityCall(
            capability="complex",
            params={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )

        restored = CapabilityCall.from_dict(original.to_dict())

        assert restored.capability == original.capability
        assert restored.params == original.params


# =============================================================================
# AGENT INFO TESTS
# =============================================================================


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_create_full_agent_info(self) -> None:
        """Verify AgentInfo can be created with all fields."""
        info = AgentInfo(
            agent_id="reverse_v1",
            name="Reverse Agent",
            version="1.0.0",
            capabilities=["reverse", "mirror"],
            description="Reverses text",
        )

        assert info.agent_id == "reverse_v1"
        assert info.name == "Reverse Agent"
        assert info.version == "1.0.0"
        assert info.capabilities == ["reverse", "mirror"]
        assert info.description == "Reverses text"

    def test_create_without_description(self) -> None:
        """Verify AgentInfo can be created without description."""
        info = AgentInfo(
            agent_id="test",
            name="Test Agent",
            version="1.0.0",
            capabilities=["test"],
        )

        assert info.description == ""

    def test_info_is_immutable(self) -> None:
        """Verify AgentInfo is immutable."""
        info = AgentInfo(
            agent_id="test",
            name="Test",
            version="1.0",
            capabilities=["test"],
        )

        with pytest.raises(AttributeError):
            info.agent_id = "other"  # type: ignore

    def test_agent_id_cannot_be_empty(self) -> None:
        """Verify validation rejects empty agent_id."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            AgentInfo(
                agent_id="",
                name="Test",
                version="1.0",
                capabilities=["test"],
            )

    def test_name_cannot_be_empty(self) -> None:
        """Verify validation rejects empty name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            AgentInfo(
                agent_id="test",
                name="",
                version="1.0",
                capabilities=["test"],
            )

    def test_capabilities_cannot_be_empty(self) -> None:
        """Verify validation rejects empty capabilities list."""
        with pytest.raises(ValueError, match="capabilities cannot be empty"):
            AgentInfo(
                agent_id="test",
                name="Test",
                version="1.0",
                capabilities=[],
            )

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict produces correct output."""
        info = AgentInfo(
            agent_id="test_id",
            name="Test Agent",
            version="2.0.0",
            capabilities=["a", "b"],
            description="Does testing",
        )

        data = info.to_dict()

        assert data["agent_id"] == "test_id"
        assert data["name"] == "Test Agent"
        assert data["version"] == "2.0.0"
        assert data["capabilities"] == ["a", "b"]
        assert data["description"] == "Does testing"
