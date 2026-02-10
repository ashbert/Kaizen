"""
Tests for the Dispatcher.

This module tests the Dispatcher which routes capability calls to
registered agents and executes them sequentially.

Tests verify:
- Agent registration and unregistration
- Capability lookup and introspection
- Sequential execution of calls
- Fail-fast behavior on errors
- Trajectory recording of dispatch steps
- DispatchResult aggregation
"""

import pytest

from kaizen.dispatcher import Dispatcher, DispatchResult
from kaizen.session import Session
from kaizen.agents import ReverseAgent, UppercaseAgent
from kaizen.types import CapabilityCall, InvokeResult, ErrorCode, EntryType


# =============================================================================
# DISPATCH RESULT TESTS
# =============================================================================


class TestDispatchResult:
    """Tests for DispatchResult class."""

    def test_success_when_all_succeed(self) -> None:
        """Verify success=True when all results succeed."""
        results = [
            InvokeResult.ok(result="a", agent_id="a", capability="c"),
            InvokeResult.ok(result="b", agent_id="b", capability="c"),
        ]
        dr = DispatchResult(results)

        assert dr.success is True
        assert dr.failed_at is None
        assert dr.error is None

    def test_failure_when_any_fails(self) -> None:
        """Verify success=False when any result fails."""
        results = [
            InvokeResult.ok(result="a", agent_id="a", capability="c"),
            InvokeResult.fail(
                ErrorCode.AGENT_INVOCATION_FAILED, "error", "b", "c"
            ),
        ]
        dr = DispatchResult(results)

        assert dr.success is False
        assert dr.failed_at == 1
        assert dr.error is not None

    def test_failed_at_is_first_failure(self) -> None:
        """Verify failed_at points to first failure."""
        results = [
            InvokeResult.ok(result="a", agent_id="a", capability="c"),
            InvokeResult.fail(
                ErrorCode.AGENT_INVOCATION_FAILED, "first", "b", "c"
            ),
            InvokeResult.fail(
                ErrorCode.AGENT_INVOCATION_FAILED, "second", "c", "c"
            ),
        ]
        dr = DispatchResult(results)

        assert dr.failed_at == 1
        assert "first" in dr.error["message"]

    def test_executed_count(self) -> None:
        """Verify executed_count returns number of results."""
        results = [
            InvokeResult.ok(result="a", agent_id="a", capability="c"),
            InvokeResult.ok(result="b", agent_id="b", capability="c"),
        ]
        dr = DispatchResult(results)

        assert dr.executed_count == 2

    def test_empty_results_is_success(self) -> None:
        """Verify empty results list is considered success."""
        dr = DispatchResult([])

        assert dr.success is True
        assert dr.executed_count == 0

    def test_repr_success(self) -> None:
        """Verify __repr__ for success case."""
        results = [InvokeResult.ok(result="a", agent_id="a", capability="c")]
        dr = DispatchResult(results)

        repr_str = repr(dr)
        assert "success=True" in repr_str
        assert "executed=1" in repr_str

    def test_repr_failure(self) -> None:
        """Verify __repr__ for failure case."""
        results = [
            InvokeResult.fail(
                ErrorCode.AGENT_INVOCATION_FAILED, "error", "a", "c"
            )
        ]
        dr = DispatchResult(results)

        repr_str = repr(dr)
        assert "success=False" in repr_str
        assert "failed_at=0" in repr_str


# =============================================================================
# DISPATCHER REGISTRATION TESTS
# =============================================================================


class TestDispatcherRegistration:
    """Tests for agent registration."""

    def test_register_agent(self) -> None:
        """Verify agent can be registered."""
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        assert dispatcher.has_capability("reverse")

    def test_register_multiple_agents(self) -> None:
        """Verify multiple agents can be registered."""
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        assert dispatcher.has_capability("reverse")
        assert dispatcher.has_capability("uppercase")

    def test_register_overwrites_capability(self) -> None:
        """Verify registering same capability overwrites."""
        dispatcher = Dispatcher()

        # Create two agents with same capability
        agent1 = ReverseAgent()
        agent2 = ReverseAgent()  # Same capability

        dispatcher.register(agent1)
        dispatcher.register(agent2)

        # Should have the second agent
        agent = dispatcher.get_agent_for_capability("reverse")
        assert agent is agent2

    def test_unregister_agent(self) -> None:
        """Verify agent can be unregistered."""
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        result = dispatcher.unregister("reverse_agent_v1")

        assert result is True
        assert not dispatcher.has_capability("reverse")

    def test_unregister_nonexistent_returns_false(self) -> None:
        """Verify unregistering nonexistent agent returns False."""
        dispatcher = Dispatcher()

        result = dispatcher.unregister("nonexistent")

        assert result is False


# =============================================================================
# DISPATCHER INTROSPECTION TESTS
# =============================================================================


class TestDispatcherIntrospection:
    """Tests for dispatcher introspection methods."""

    def test_get_capabilities_empty(self) -> None:
        """Verify get_capabilities returns empty list initially."""
        dispatcher = Dispatcher()
        assert dispatcher.get_capabilities() == []

    def test_get_capabilities_returns_sorted(self) -> None:
        """Verify get_capabilities returns sorted list."""
        dispatcher = Dispatcher()
        dispatcher.register(UppercaseAgent())  # 'uppercase'
        dispatcher.register(ReverseAgent())    # 'reverse'

        caps = dispatcher.get_capabilities()
        assert caps == ["reverse", "uppercase"]

    def test_get_agent_for_capability(self) -> None:
        """Verify get_agent_for_capability returns correct agent."""
        dispatcher = Dispatcher()
        agent = ReverseAgent()
        dispatcher.register(agent)

        result = dispatcher.get_agent_for_capability("reverse")
        assert result is agent

    def test_get_agent_for_unknown_capability(self) -> None:
        """Verify get_agent_for_capability returns None for unknown."""
        dispatcher = Dispatcher()

        result = dispatcher.get_agent_for_capability("unknown")
        assert result is None

    def test_get_registered_agents(self) -> None:
        """Verify get_registered_agents returns all agent info."""
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        agents = dispatcher.get_registered_agents()

        assert len(agents) == 2
        agent_ids = {a.agent_id for a in agents}
        assert "reverse_agent_v1" in agent_ids
        assert "uppercase_agent_v1" in agent_ids

    def test_has_capability_true(self) -> None:
        """Verify has_capability returns True for registered."""
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        assert dispatcher.has_capability("reverse") is True

    def test_has_capability_false(self) -> None:
        """Verify has_capability returns False for unregistered."""
        dispatcher = Dispatcher()

        assert dispatcher.has_capability("unknown") is False

    def test_repr(self) -> None:
        """Verify __repr__ returns useful string."""
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        repr_str = repr(dispatcher)
        assert "Dispatcher" in repr_str
        assert "reverse" in repr_str


# =============================================================================
# DISPATCH SEQUENCE TESTS
# =============================================================================


class TestDispatchSequence:
    """Tests for dispatch_sequence execution."""

    def test_dispatch_single_call(self) -> None:
        """Verify single call is executed."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        calls = [CapabilityCall("reverse", {"key": "text"})]
        result = dispatcher.dispatch_sequence(calls, session)

        assert result.success is True
        assert session.get("text") == "olleh"

    def test_dispatch_multiple_calls(self) -> None:
        """Verify multiple calls are executed in order."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        calls = [
            CapabilityCall("reverse", {"key": "text"}),
            CapabilityCall("uppercase", {"key": "text"}),
        ]
        result = dispatcher.dispatch_sequence(calls, session)

        assert result.success is True
        assert result.executed_count == 2
        # hello -> olleh -> OLLEH
        assert session.get("text") == "OLLEH"

    def test_dispatch_accepts_dict_calls(self) -> None:
        """Verify dispatch accepts dict format for calls."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        calls = [{"capability": "reverse", "params": {"key": "text"}}]
        result = dispatcher.dispatch_sequence(calls, session)

        assert result.success is True
        assert session.get("text") == "olleh"

    def test_dispatch_mixed_call_formats(self) -> None:
        """Verify dispatch accepts mixed CapabilityCall and dict."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        calls = [
            CapabilityCall("reverse", {"key": "text"}),
            {"capability": "uppercase", "params": {"key": "text"}},
        ]
        result = dispatcher.dispatch_sequence(calls, session)

        assert result.success is True
        assert session.get("text") == "OLLEH"

    def test_dispatch_empty_calls(self) -> None:
        """Verify empty call list returns success."""
        session = Session()
        dispatcher = Dispatcher()

        result = dispatcher.dispatch_sequence([], session)

        assert result.success is True
        assert result.executed_count == 0


class TestDispatchFailFast:
    """Tests for fail-fast behavior."""

    def test_stops_on_first_failure(self) -> None:
        """Verify execution stops on first failure."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        calls = [
            CapabilityCall("reverse", {"key": "text"}),
            CapabilityCall("reverse", {"key": "nonexistent"}),  # Will fail
            CapabilityCall("uppercase", {"key": "text"}),  # Never reached
        ]
        result = dispatcher.dispatch_sequence(calls, session)

        assert result.success is False
        assert result.failed_at == 1
        assert result.executed_count == 2  # Only 2 were executed

        # Third call was never executed, so text is still "olleh"
        assert session.get("text") == "olleh"

    def test_unknown_capability_fails(self) -> None:
        """Verify unknown capability causes failure."""
        session = Session()
        dispatcher = Dispatcher()

        calls = [CapabilityCall("unknown", {})]
        result = dispatcher.dispatch_sequence(calls, session)

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.DISPATCH_NO_AGENT_FOR_CAPABILITY.value

    def test_agent_failure_propagates(self) -> None:
        """Verify agent failure propagates through dispatch."""
        session = Session()
        # No value set for "text" - agent will fail

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        calls = [CapabilityCall("reverse", {"key": "text"})]
        result = dispatcher.dispatch_sequence(calls, session)

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_INVALID_PARAMS.value


class TestDispatchTrajectory:
    """Tests for trajectory recording during dispatch."""

    def test_dispatch_records_plan_step_started(self) -> None:
        """Verify PLAN_STEP_STARTED entries are created."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        calls = [CapabilityCall("reverse", {"key": "text"})]
        dispatcher.dispatch_sequence(calls, session)

        trajectory = session.get_trajectory()
        step_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_STEP_STARTED
        ]

        assert len(step_entries) == 1
        assert step_entries[0].agent_id == "dispatcher"
        assert step_entries[0].content["capability"] == "reverse"
        assert step_entries[0].content["step_index"] == 0

    def test_dispatch_records_multiple_steps(self) -> None:
        """Verify each call gets a PLAN_STEP_STARTED entry."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        calls = [
            CapabilityCall("reverse", {"key": "text"}),
            CapabilityCall("uppercase", {"key": "text"}),
        ]
        dispatcher.dispatch_sequence(calls, session)

        trajectory = session.get_trajectory()
        step_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_STEP_STARTED
        ]

        assert len(step_entries) == 2
        assert step_entries[0].content["step_index"] == 0
        assert step_entries[1].content["step_index"] == 1


# =============================================================================
# DISPATCH SINGLE TESTS
# =============================================================================


class TestDispatchSingle:
    """Tests for dispatch_single convenience method."""

    def test_dispatch_single_success(self) -> None:
        """Verify dispatch_single executes and returns result."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        result = dispatcher.dispatch_single("reverse", session, {"key": "text"})

        assert result.success is True
        assert result.result["reversed"] == "olleh"

    def test_dispatch_single_with_no_params(self) -> None:
        """Verify dispatch_single works with no params."""
        session = Session()
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        # Will fail because no key, but shouldn't crash
        result = dispatcher.dispatch_single("reverse", session)

        assert result.success is False

    def test_dispatch_single_unknown_capability(self) -> None:
        """Verify dispatch_single handles unknown capability."""
        session = Session()
        dispatcher = Dispatcher()

        result = dispatcher.dispatch_single("unknown", session, {})

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.DISPATCH_NO_AGENT_FOR_CAPABILITY.value


# =============================================================================
# EDGE CASES
# =============================================================================


class TestDispatcherEdgeCases:
    """Tests for edge cases and error handling."""

    def test_agent_exception_is_caught(self) -> None:
        """Verify agent exceptions are caught and converted to failures."""
        from kaizen.agent import Agent
        from kaizen.types import AgentInfo

        class BrokenAgent(Agent):
            def info(self) -> AgentInfo:
                return AgentInfo(
                    agent_id="broken",
                    name="Broken Agent",
                    version="1.0.0",
                    capabilities=["crash"],
                )

            def invoke(self, capability, session, params):
                raise RuntimeError("I always crash!")

        session = Session()
        dispatcher = Dispatcher()
        dispatcher.register(BrokenAgent())

        result = dispatcher.dispatch_single("crash", session, {})

        assert result.success is False
        assert result.error["error_code"] == ErrorCode.AGENT_INVOCATION_FAILED.value
        assert "RuntimeError" in result.error["message"]

    def test_dispatch_preserves_session_state_on_failure(self) -> None:
        """Verify partial state changes are preserved on failure."""
        session = Session()
        session.set("text", "hello")
        session.set("counter", 0)

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        # First call succeeds, second fails
        calls = [
            CapabilityCall("reverse", {"key": "text"}),      # Succeeds
            CapabilityCall("reverse", {"key": "missing"}),   # Fails
        ]
        dispatcher.dispatch_sequence(calls, session)

        # First change should be preserved
        assert session.get("text") == "olleh"
        assert session.get("counter") == 0


# =============================================================================
# COMPLETED INDICES TESTS
# =============================================================================


class TestDispatchResultCompletedIndices:
    """Tests for DispatchResult.completed_indices property."""

    def test_all_successful(self) -> None:
        """Verify completed_indices returns all indices when all succeed."""
        results = [
            InvokeResult.ok(result="a", agent_id="a", capability="c1"),
            InvokeResult.ok(result="b", agent_id="b", capability="c2"),
        ]
        dr = DispatchResult(results)
        assert dr.completed_indices == [0, 1]

    def test_partial_failure(self) -> None:
        """Verify completed_indices excludes failed steps."""
        results = [
            InvokeResult.ok(result="a", agent_id="a", capability="c1"),
            InvokeResult.fail(ErrorCode.AGENT_INVOCATION_FAILED, "err", "b", "c2"),
        ]
        dr = DispatchResult(results)
        assert dr.completed_indices == [0]

    def test_empty(self) -> None:
        """Verify completed_indices is empty for empty results."""
        dr = DispatchResult([])
        assert dr.completed_indices == []


# =============================================================================
# PLAN_STEP_COMPLETED RECORDING TESTS
# =============================================================================


class TestPlanStepCompletedRecording:
    """Tests for PLAN_STEP_COMPLETED trajectory entries."""

    def test_records_plan_step_completed_on_success(self) -> None:
        """Verify PLAN_STEP_COMPLETED is recorded for successful steps."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        calls = [CapabilityCall("reverse", {"key": "text"})]
        dispatcher.dispatch_sequence(calls, session)

        trajectory = session.get_trajectory()
        completed_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_STEP_COMPLETED
        ]
        assert len(completed_entries) == 1
        assert completed_entries[0].content["step_index"] == 0
        assert completed_entries[0].content["capability"] == "reverse"
        assert completed_entries[0].content["success"] is True

    def test_records_plan_step_completed_on_failure(self) -> None:
        """Verify PLAN_STEP_COMPLETED is recorded for failed steps."""
        session = Session()

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        # No value set for "text" — agent will fail
        calls = [CapabilityCall("reverse", {"key": "text"})]
        dispatcher.dispatch_sequence(calls, session)

        trajectory = session.get_trajectory()
        completed_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_STEP_COMPLETED
        ]
        assert len(completed_entries) == 1
        assert completed_entries[0].content["success"] is False

    def test_records_plan_step_completed_for_unknown_capability(self) -> None:
        """Verify PLAN_STEP_COMPLETED recorded when no agent found."""
        session = Session()
        dispatcher = Dispatcher()

        calls = [CapabilityCall("unknown", {})]
        dispatcher.dispatch_sequence(calls, session)

        trajectory = session.get_trajectory()
        completed_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_STEP_COMPLETED
        ]
        assert len(completed_entries) == 1
        assert completed_entries[0].content["success"] is False
        assert completed_entries[0].content["capability"] == "unknown"

    def test_records_completed_for_each_step(self) -> None:
        """Verify each step gets its own PLAN_STEP_COMPLETED entry."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        calls = [
            CapabilityCall("reverse", {"key": "text"}),
            CapabilityCall("uppercase", {"key": "text"}),
        ]
        dispatcher.dispatch_sequence(calls, session)

        trajectory = session.get_trajectory()
        completed_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_STEP_COMPLETED
        ]
        assert len(completed_entries) == 2
        assert completed_entries[0].content["step_index"] == 0
        assert completed_entries[1].content["step_index"] == 1


# =============================================================================
# RESUME SEQUENCE TESTS
# =============================================================================


class TestResumeSequence:
    """Tests for Dispatcher.resume_sequence()."""

    def test_resume_skips_completed_steps(self) -> None:
        """Verify resume_sequence skips already-completed steps."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        # First run: execute only step 0
        first_calls = [CapabilityCall("reverse", {"key": "text"})]
        dispatcher.dispatch_sequence(first_calls, session)

        # text is now "olleh", and trajectory has PLAN_STEP_COMPLETED for
        # (step_index=0, capability="reverse")

        # Resume the full 2-step sequence: should skip step 0, execute step 1
        full_calls = [
            CapabilityCall("reverse", {"key": "text"}),
            CapabilityCall("uppercase", {"key": "text"}),
        ]
        result = dispatcher.resume_sequence(full_calls, session)

        assert result.success is True
        assert result.executed_count == 2
        # Step 0 should be synthetic (resumed)
        assert result.results[0].result.get("resumed") is True
        # Step 1 should have executed — text goes from "olleh" -> "OLLEH"
        assert session.get("text") == "OLLEH"

    def test_resume_all_completed(self) -> None:
        """Verify resume_sequence returns immediately if all steps done."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        calls = [
            CapabilityCall("reverse", {"key": "text"}),
            CapabilityCall("uppercase", {"key": "text"}),
        ]
        # Full run
        dispatcher.dispatch_sequence(calls, session)
        assert session.get("text") == "OLLEH"

        # Resume should skip everything
        result = dispatcher.resume_sequence(calls, session)

        assert result.success is True
        assert result.executed_count == 2
        assert result.results[0].result.get("resumed") is True
        assert result.results[1].result.get("resumed") is True

    def test_resume_none_completed(self) -> None:
        """Verify resume_sequence executes all if none completed."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        calls = [CapabilityCall("reverse", {"key": "text"})]
        result = dispatcher.resume_sequence(calls, session)

        assert result.success is True
        assert session.get("text") == "olleh"
        # Should not be a resumed result
        assert result.results[0].result.get("resumed") is not True

    def test_resume_mismatched_capability_reexecutes(self) -> None:
        """Verify changed capability at same index gets re-executed."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        # First run: reverse at index 0
        first_calls = [CapabilityCall("reverse", {"key": "text"})]
        dispatcher.dispatch_sequence(first_calls, session)
        # text = "olleh"

        # Resume with DIFFERENT capability at index 0
        session.set("text", "hello")
        new_calls = [CapabilityCall("uppercase", {"key": "text"})]
        result = dispatcher.resume_sequence(new_calls, session)

        assert result.success is True
        # Should have executed uppercase (not skipped)
        assert session.get("text") == "HELLO"
        assert result.results[0].result.get("resumed") is not True

    def test_resume_accepts_dict_calls(self) -> None:
        """Verify resume_sequence accepts dict format calls."""
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        calls = [{"capability": "reverse", "params": {"key": "text"}}]
        result = dispatcher.resume_sequence(calls, session)

        assert result.success is True
        assert session.get("text") == "olleh"
