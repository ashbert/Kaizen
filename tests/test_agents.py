"""
Tests for Agent protocol and toy agents.

This module tests:
- Agent abstract base class
- AgentProtocol (structural typing)
- ReverseAgent implementation
- UppercaseAgent implementation

Tests verify:
- Protocol compliance
- info() returns correct metadata
- invoke() succeeds with valid inputs
- invoke() fails appropriately with invalid inputs
- Trajectory entries are created
- Session state is properly modified
"""

import pytest

from kaizen.agent import Agent, AgentProtocol
from kaizen.agents import ReverseAgent, UppercaseAgent
from kaizen.session import Session
from kaizen.types import AgentInfo, InvokeResult, EntryType, ErrorCode


# =============================================================================
# AGENT PROTOCOL TESTS
# =============================================================================


class TestAgentProtocol:
    """Tests for the AgentProtocol structural typing."""

    def test_reverse_agent_is_agent_protocol(self) -> None:
        """Verify ReverseAgent satisfies AgentProtocol."""
        agent = ReverseAgent()
        assert isinstance(agent, AgentProtocol)

    def test_uppercase_agent_is_agent_protocol(self) -> None:
        """Verify UppercaseAgent satisfies AgentProtocol."""
        agent = UppercaseAgent()
        assert isinstance(agent, AgentProtocol)

    def test_agent_subclass_is_agent_protocol(self) -> None:
        """Verify any Agent subclass satisfies AgentProtocol."""
        # Both toy agents inherit from Agent
        assert isinstance(ReverseAgent(), Agent)
        assert isinstance(UppercaseAgent(), Agent)


# =============================================================================
# REVERSE AGENT TESTS
# =============================================================================


class TestReverseAgentInfo:
    """Tests for ReverseAgent.info()."""

    def test_info_returns_agent_info(self) -> None:
        """Verify info() returns AgentInfo instance."""
        agent = ReverseAgent()
        info = agent.info()
        assert isinstance(info, AgentInfo)

    def test_info_has_correct_agent_id(self) -> None:
        """Verify info contains correct agent_id."""
        agent = ReverseAgent()
        info = agent.info()
        assert info.agent_id == "reverse_agent_v1"

    def test_info_has_correct_name(self) -> None:
        """Verify info contains correct name."""
        agent = ReverseAgent()
        info = agent.info()
        assert info.name == "Reverse Agent"

    def test_info_has_version(self) -> None:
        """Verify info contains version."""
        agent = ReverseAgent()
        info = agent.info()
        assert info.version == "1.0.0"

    def test_info_has_reverse_capability(self) -> None:
        """Verify info declares the reverse capability."""
        agent = ReverseAgent()
        info = agent.info()
        assert "reverse" in info.capabilities

    def test_info_has_description(self) -> None:
        """Verify info contains description."""
        agent = ReverseAgent()
        info = agent.info()
        assert info.description != ""


class TestReverseAgentInvoke:
    """Tests for ReverseAgent.invoke()."""

    def test_invoke_reverses_string(self) -> None:
        """Verify invoke() reverses the string."""
        session = Session()
        session.set("text", "hello")

        agent = ReverseAgent()
        result = agent.invoke("reverse", session, {"key": "text"})

        assert result.success is True
        assert session.get("text") == "olleh"

    def test_invoke_returns_original_and_reversed(self) -> None:
        """Verify invoke() result contains both values."""
        session = Session()
        session.set("text", "hello")

        agent = ReverseAgent()
        result = agent.invoke("reverse", session, {"key": "text"})

        assert result.result["original"] == "hello"
        assert result.result["reversed"] == "olleh"

    def test_invoke_with_empty_string(self) -> None:
        """Verify invoke() handles empty string."""
        session = Session()
        session.set("text", "")

        agent = ReverseAgent()
        result = agent.invoke("reverse", session, {"key": "text"})

        assert result.success is True
        assert result.result["reversed"] == ""

    def test_invoke_with_unicode(self) -> None:
        """Verify invoke() handles unicode strings."""
        session = Session()
        session.set("text", "hello 世界 🌍")

        agent = ReverseAgent()
        result = agent.invoke("reverse", session, {"key": "text"})

        assert result.success is True
        assert result.result["reversed"] == "🌍 界世 olleh"

    def test_invoke_unknown_capability_fails(self) -> None:
        """Verify invoke() fails for unknown capability."""
        session = Session()
        agent = ReverseAgent()

        result = agent.invoke("unknown", session, {})

        assert result.success is False
        assert result.error is not None
        assert result.error["error_code"] == ErrorCode.AGENT_CAPABILITY_NOT_FOUND.value

    def test_invoke_missing_key_param_fails(self) -> None:
        """Verify invoke() fails when key param is missing."""
        session = Session()
        agent = ReverseAgent()

        result = agent.invoke("reverse", session, {})

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_INVALID_PARAMS.value
        assert "key" in result.error["message"]

    def test_invoke_nonexistent_key_fails(self) -> None:
        """Verify invoke() fails when key doesn't exist in state."""
        session = Session()
        agent = ReverseAgent()

        result = agent.invoke("reverse", session, {"key": "nonexistent"})

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_INVALID_PARAMS.value

    def test_invoke_non_string_value_fails(self) -> None:
        """Verify invoke() fails when value is not a string."""
        session = Session()
        session.set("number", 42)

        agent = ReverseAgent()
        result = agent.invoke("reverse", session, {"key": "number"})

        assert result.success is False
        assert "string" in result.error["message"].lower()


class TestReverseAgentTrajectory:
    """Tests for ReverseAgent trajectory recording."""

    def test_invoke_creates_trajectory_entries(self) -> None:
        """Verify invoke() creates trajectory entries."""
        session = Session()
        session.set("text", "hello")
        initial_len = session.get_trajectory_length()

        agent = ReverseAgent()
        agent.invoke("reverse", session, {"key": "text"})

        # Should have created AGENT_INVOKED and AGENT_COMPLETED entries
        # plus any state change entries
        assert session.get_trajectory_length() > initial_len

    def test_invoke_records_agent_invoked(self) -> None:
        """Verify AGENT_INVOKED entry is recorded."""
        session = Session()
        session.set("text", "hello")

        agent = ReverseAgent()
        agent.invoke("reverse", session, {"key": "text"})

        trajectory = session.get_trajectory()
        invoked_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.AGENT_INVOKED
            and e.agent_id == "reverse_agent_v1"
        ]

        assert len(invoked_entries) == 1
        assert invoked_entries[0].content["capability"] == "reverse"

    def test_invoke_records_agent_completed(self) -> None:
        """Verify AGENT_COMPLETED entry is recorded."""
        session = Session()
        session.set("text", "hello")

        agent = ReverseAgent()
        agent.invoke("reverse", session, {"key": "text"})

        trajectory = session.get_trajectory()
        completed_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.AGENT_COMPLETED
            and e.agent_id == "reverse_agent_v1"
        ]

        assert len(completed_entries) == 1
        assert completed_entries[0].content["original"] == "hello"
        assert completed_entries[0].content["reversed"] == "olleh"


# =============================================================================
# UPPERCASE AGENT TESTS
# =============================================================================


class TestUppercaseAgentInfo:
    """Tests for UppercaseAgent.info()."""

    def test_info_returns_agent_info(self) -> None:
        """Verify info() returns AgentInfo instance."""
        agent = UppercaseAgent()
        info = agent.info()
        assert isinstance(info, AgentInfo)

    def test_info_has_correct_agent_id(self) -> None:
        """Verify info contains correct agent_id."""
        agent = UppercaseAgent()
        info = agent.info()
        assert info.agent_id == "uppercase_agent_v1"

    def test_info_has_uppercase_capability(self) -> None:
        """Verify info declares the uppercase capability."""
        agent = UppercaseAgent()
        info = agent.info()
        assert "uppercase" in info.capabilities


class TestUppercaseAgentInvoke:
    """Tests for UppercaseAgent.invoke()."""

    def test_invoke_uppercases_string(self) -> None:
        """Verify invoke() uppercases the string."""
        session = Session()
        session.set("text", "hello")

        agent = UppercaseAgent()
        result = agent.invoke("uppercase", session, {"key": "text"})

        assert result.success is True
        assert session.get("text") == "HELLO"

    def test_invoke_returns_original_and_uppercased(self) -> None:
        """Verify invoke() result contains both values."""
        session = Session()
        session.set("text", "hello")

        agent = UppercaseAgent()
        result = agent.invoke("uppercase", session, {"key": "text"})

        assert result.result["original"] == "hello"
        assert result.result["uppercased"] == "HELLO"

    def test_invoke_with_mixed_case(self) -> None:
        """Verify invoke() handles mixed case."""
        session = Session()
        session.set("text", "HeLLo WoRLd")

        agent = UppercaseAgent()
        result = agent.invoke("uppercase", session, {"key": "text"})

        assert result.success is True
        assert result.result["uppercased"] == "HELLO WORLD"

    def test_invoke_with_already_uppercase(self) -> None:
        """Verify invoke() handles already uppercase string."""
        session = Session()
        session.set("text", "HELLO")

        agent = UppercaseAgent()
        result = agent.invoke("uppercase", session, {"key": "text"})

        assert result.success is True
        assert result.result["uppercased"] == "HELLO"

    def test_invoke_unknown_capability_fails(self) -> None:
        """Verify invoke() fails for unknown capability."""
        session = Session()
        agent = UppercaseAgent()

        result = agent.invoke("unknown", session, {})

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_CAPABILITY_NOT_FOUND.value

    def test_invoke_missing_key_param_fails(self) -> None:
        """Verify invoke() fails when key param is missing."""
        session = Session()
        agent = UppercaseAgent()

        result = agent.invoke("uppercase", session, {})

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_INVALID_PARAMS.value


class TestUppercaseAgentTrajectory:
    """Tests for UppercaseAgent trajectory recording."""

    def test_invoke_records_agent_completed(self) -> None:
        """Verify AGENT_COMPLETED entry is recorded."""
        session = Session()
        session.set("text", "hello")

        agent = UppercaseAgent()
        agent.invoke("uppercase", session, {"key": "text"})

        trajectory = session.get_trajectory()
        completed_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.AGENT_COMPLETED
            and e.agent_id == "uppercase_agent_v1"
        ]

        assert len(completed_entries) == 1
        assert completed_entries[0].content["uppercased"] == "HELLO"


# =============================================================================
# AGENT CHAINING TESTS
# =============================================================================


class TestAgentChaining:
    """Tests for chaining multiple agents."""

    def test_reverse_then_uppercase(self) -> None:
        """Verify agents can be chained: reverse then uppercase."""
        session = Session()
        session.set("text", "hello")

        reverse = ReverseAgent()
        uppercase = UppercaseAgent()

        # Chain: hello -> olleh -> OLLEH
        reverse.invoke("reverse", session, {"key": "text"})
        uppercase.invoke("uppercase", session, {"key": "text"})

        assert session.get("text") == "OLLEH"

    def test_uppercase_then_reverse(self) -> None:
        """Verify agents can be chained: uppercase then reverse."""
        session = Session()
        session.set("text", "hello")

        reverse = ReverseAgent()
        uppercase = UppercaseAgent()

        # Chain: hello -> HELLO -> OLLEH
        uppercase.invoke("uppercase", session, {"key": "text"})
        reverse.invoke("reverse", session, {"key": "text"})

        assert session.get("text") == "OLLEH"

    def test_chained_trajectory_records_all_agents(self) -> None:
        """Verify trajectory records actions from all chained agents."""
        session = Session()
        session.set("text", "hello")

        reverse = ReverseAgent()
        uppercase = UppercaseAgent()

        reverse.invoke("reverse", session, {"key": "text"})
        uppercase.invoke("uppercase", session, {"key": "text"})

        trajectory = session.get_trajectory()

        # Should have entries from both agents
        reverse_entries = [
            e for e in trajectory if e.agent_id == "reverse_agent_v1"
        ]
        uppercase_entries = [
            e for e in trajectory if e.agent_id == "uppercase_agent_v1"
        ]

        assert len(reverse_entries) >= 2  # INVOKED + COMPLETED
        assert len(uppercase_entries) >= 2  # INVOKED + COMPLETED


# =============================================================================
# AGENT HELPER METHOD TESTS
# =============================================================================


class TestAgentHelperMethods:
    """Tests for Agent helper methods."""

    def test_unknown_capability_helper(self) -> None:
        """Verify _unknown_capability returns proper result."""
        agent = ReverseAgent()
        result = agent._unknown_capability("fake_cap")

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_CAPABILITY_NOT_FOUND.value
        assert "fake_cap" in result.error["message"]
        assert result.agent_id == "reverse_agent_v1"
        assert result.capability == "fake_cap"

    def test_invalid_params_helper(self) -> None:
        """Verify _invalid_params returns proper result."""
        agent = ReverseAgent()
        result = agent._invalid_params(
            "reverse",
            "Missing key",
            details={"hint": "provide key param"},
        )

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_INVALID_PARAMS.value
        assert "Missing key" in result.error["message"]
        assert result.error["details"]["hint"] == "provide key param"

    def test_invocation_failed_helper(self) -> None:
        """Verify _invocation_failed returns proper result."""
        agent = ReverseAgent()
        result = agent._invocation_failed(
            "reverse",
            "Something went wrong",
            details={"exception": "ValueError"},
        )

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_INVOCATION_FAILED.value
        assert "Something went wrong" in result.error["message"]

    def test_agent_repr(self) -> None:
        """Verify __repr__ returns useful string."""
        agent = ReverseAgent()
        repr_str = repr(agent)

        assert "ReverseAgent" in repr_str
        assert "reverse_agent_v1" in repr_str
        assert "reverse" in repr_str
