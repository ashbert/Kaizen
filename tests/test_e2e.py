"""
End-to-end integration tests for Kaizen.

This module tests the complete workflow as specified in the V1 documentation:
    User: "Reverse this text and then uppercase it"
    → Planner emits ordered calls
    → Dispatcher executes
    → Trajectory records actions
    → Session saved and resumable

These tests verify the success criteria:
    "You can replay, inspect, resume, and reproduce any run from a single
    session file. If this is true, V1 is complete."
"""

from pathlib import Path

import pytest

from kaizen.session import Session
from kaizen.dispatcher import Dispatcher
from kaizen.planner import Planner
from kaizen.agents import ReverseAgent, UppercaseAgent
from kaizen.llm.base import LLMProvider, LLMResponse
from kaizen.types import CapabilityCall, EntryType


# =============================================================================
# MOCK LLM FOR DETERMINISTIC TESTS
# =============================================================================


class DeterministicLLMProvider(LLMProvider):
    """
    LLM provider with pre-programmed responses for deterministic testing.

    This allows end-to-end tests to run without a real LLM server.
    """

    def __init__(self):
        """Initialize with mapping of inputs to outputs."""
        self._responses = {
            # Standard test case from spec
            "reverse this text and then uppercase it": '''[
                {"capability": "reverse", "params": {"key": "text"}},
                {"capability": "uppercase", "params": {"key": "text"}}
            ]''',
            "reverse the text": '[{"capability": "reverse", "params": {"key": "text"}}]',
            "uppercase the text": '[{"capability": "uppercase", "params": {"key": "text"}}]',
            "do nothing": '[]',
        }
        self._default = '[]'

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        """Return pre-programmed response."""
        prompt_lower = prompt.lower().strip()
        response = self._responses.get(prompt_lower, self._default)
        return LLMResponse(text=response, model="deterministic-test")

    @property
    def model_name(self) -> str:
        return "deterministic-test"


# =============================================================================
# FULL WORKFLOW TESTS
# =============================================================================


class TestExampleWorkflow:
    """
    Tests for the example workflow from the spec.

    "Reverse this text and then uppercase it" → planner emits ordered calls
    → dispatcher executes → trajectory records actions → session saved
    """

    def test_example_workflow(self, temp_session_path: Path) -> None:
        """
        Test the complete workflow from the specification.

        This is the primary success criteria test.
        """
        # === SETUP ===
        # Create session with input
        session = Session()
        session.set("text", "hello world")

        # Create dispatcher with agents
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        # Create planner with LLM
        llm = DeterministicLLMProvider()
        planner = Planner(llm, capabilities=dispatcher.get_capabilities())

        # === PLAN ===
        # Convert user input to capability calls
        plan_result = planner.plan(
            "Reverse this text and then uppercase it",
            session=session,
        )

        assert plan_result.success is True
        assert len(plan_result.calls) == 2
        assert plan_result.calls[0].capability == "reverse"
        assert plan_result.calls[1].capability == "uppercase"

        # === EXECUTE ===
        # Dispatch the planned calls
        dispatch_result = dispatcher.dispatch_sequence(plan_result.calls, session)

        assert dispatch_result.success is True
        assert dispatch_result.executed_count == 2

        # Verify final state
        # hello world -> dlrow olleh -> DLROW OLLEH
        assert session.get("text") == "DLROW OLLEH"

        # === PERSIST ===
        # Save session
        session.save(temp_session_path)

        # === RESUME ===
        # Load and verify
        restored = Session.load(temp_session_path)

        assert restored.get("text") == "DLROW OLLEH"
        assert restored.session_id == session.session_id

    def test_workflow_trajectory_is_complete(self, temp_session_path: Path) -> None:
        """
        Verify the trajectory captures the complete execution history.

        The trajectory should allow full replay and inspection.
        """
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        llm = DeterministicLLMProvider()
        planner = Planner(llm, capabilities=dispatcher.get_capabilities())

        # Execute workflow
        plan_result = planner.plan("reverse this text and then uppercase it", session)
        dispatcher.dispatch_sequence(plan_result.calls, session)
        session.save(temp_session_path)

        # Load and inspect trajectory
        restored = Session.load(temp_session_path)
        trajectory = restored.get_trajectory()

        # Find all entry types
        entry_types = {e.entry_type for e in trajectory}

        # Should have these entry types from the workflow:
        assert EntryType.SESSION_CREATED in entry_types
        assert EntryType.STATE_SET in entry_types
        assert EntryType.PLAN_CREATED in entry_types
        assert EntryType.PLAN_STEP_STARTED in entry_types
        assert EntryType.AGENT_INVOKED in entry_types
        assert EntryType.AGENT_COMPLETED in entry_types
        assert EntryType.SESSION_SAVED in entry_types
        assert EntryType.SESSION_LOADED in entry_types


class TestResumeAfterFailure:
    """Tests for resume-after-failure scenarios."""

    def test_resume_partial_execution(self, temp_session_path: Path) -> None:
        """
        Verify session can resume after partial execution failure.

        Simulate a workflow where execution stops mid-way and verify
        we can resume from the saved state.
        """
        # === FIRST EXECUTION (partial) ===
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())

        # Execute only reverse
        dispatcher.dispatch_single("reverse", session, {"key": "text"})

        # "Save" at this point (simulating crash/interrupt)
        session.save(temp_session_path)

        assert session.get("text") == "olleh"

        # === RESUME ===
        restored = Session.load(temp_session_path)

        # State should be preserved
        assert restored.get("text") == "olleh"

        # Now register uppercase and continue
        dispatcher2 = Dispatcher()
        dispatcher2.register(UppercaseAgent())

        dispatcher2.dispatch_single("uppercase", restored, {"key": "text"})

        assert restored.get("text") == "OLLEH"

        # Save final state
        restored.save(temp_session_path)

        # Verify complete workflow
        final = Session.load(temp_session_path)
        assert final.get("text") == "OLLEH"

    def test_resume_preserves_trajectory_continuity(
        self, temp_session_path: Path
    ) -> None:
        """
        Verify trajectory sequence numbers are continuous across save/load.
        """
        session = Session()
        session.set("counter", 0)
        session.save(temp_session_path)

        # Resume and add more
        restored = Session.load(temp_session_path)
        restored.set("counter", 1)
        restored.save(temp_session_path)

        # Resume again
        restored2 = Session.load(temp_session_path)
        restored2.set("counter", 2)

        # Check trajectory is continuous
        trajectory = restored2.get_trajectory()
        seq_nums = [e.seq_num for e in trajectory]

        # Should be consecutive
        for i in range(1, len(seq_nums)):
            assert seq_nums[i] == seq_nums[i - 1] + 1


class TestReplayAndInspection:
    """Tests for replay and inspection capabilities."""

    def test_can_inspect_all_state_changes(self, temp_session_path: Path) -> None:
        """
        Verify all state changes can be inspected from trajectory.
        """
        session = Session()
        session.set("text", "a")
        session.set("text", "b")
        session.set("text", "c")
        session.save(temp_session_path)

        restored = Session.load(temp_session_path)
        trajectory = restored.get_trajectory()

        # Find all state changes
        state_changes = [
            e for e in trajectory
            if e.entry_type == EntryType.STATE_SET
            and e.content.get("key") == "text"
        ]

        # Should capture all changes with old and new values
        assert len(state_changes) == 3
        assert state_changes[0].content["new_value"] == "a"
        assert state_changes[1].content["old_value"] == "a"
        assert state_changes[1].content["new_value"] == "b"
        assert state_changes[2].content["old_value"] == "b"
        assert state_changes[2].content["new_value"] == "c"

    def test_can_identify_agent_actions(self, temp_session_path: Path) -> None:
        """
        Verify agent actions can be identified and attributed.
        """
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        dispatcher.dispatch_single("reverse", session, {"key": "text"})
        dispatcher.dispatch_single("uppercase", session, {"key": "text"})
        session.save(temp_session_path)

        restored = Session.load(temp_session_path)
        trajectory = restored.get_trajectory()

        # Find agent completions
        agent_actions = [
            e for e in trajectory
            if e.entry_type == EntryType.AGENT_COMPLETED
        ]

        # Should have 2 agent actions with proper attribution
        assert len(agent_actions) == 2

        agents = [e.agent_id for e in agent_actions]
        assert "reverse_agent_v1" in agents
        assert "uppercase_agent_v1" in agents

    def test_can_reconstruct_execution_order(self, temp_session_path: Path) -> None:
        """
        Verify execution order can be reconstructed from trajectory.
        """
        session = Session()
        session.set("text", "hello")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        # Execute in specific order
        dispatcher.dispatch_single("reverse", session, {"key": "text"})
        dispatcher.dispatch_single("uppercase", session, {"key": "text"})
        session.save(temp_session_path)

        restored = Session.load(temp_session_path)
        trajectory = restored.get_trajectory()

        # Find agent invocations in order
        invocations = [
            e for e in trajectory
            if e.entry_type == EntryType.AGENT_INVOKED
        ]

        # Order should be preserved (reverse before uppercase)
        assert len(invocations) == 2
        assert invocations[0].agent_id == "reverse_agent_v1"
        assert invocations[1].agent_id == "uppercase_agent_v1"

        # Sequence numbers should be increasing
        assert invocations[0].seq_num < invocations[1].seq_num


class TestReproducibility:
    """Tests for reproducibility of session execution."""

    def test_same_input_produces_same_plan(self) -> None:
        """
        Verify same user input produces same plan (deterministic planning).
        """
        llm = DeterministicLLMProvider()
        planner = Planner(llm, capabilities=["reverse", "uppercase"])

        # Generate plan twice
        result1 = planner.plan("reverse this text and then uppercase it")
        result2 = planner.plan("reverse this text and then uppercase it")

        # Should be identical
        assert len(result1.calls) == len(result2.calls)
        for c1, c2 in zip(result1.calls, result2.calls):
            assert c1.capability == c2.capability
            assert c1.params == c2.params

    def test_same_calls_produce_same_result(self) -> None:
        """
        Verify same capability calls produce same state changes.
        """
        # Run 1
        session1 = Session()
        session1.set("text", "hello")
        dispatcher1 = Dispatcher()
        dispatcher1.register(ReverseAgent())
        dispatcher1.dispatch_single("reverse", session1, {"key": "text"})

        # Run 2
        session2 = Session()
        session2.set("text", "hello")
        dispatcher2 = Dispatcher()
        dispatcher2.register(ReverseAgent())
        dispatcher2.dispatch_single("reverse", session2, {"key": "text"})

        # Results should be identical
        assert session1.get("text") == session2.get("text")


class TestArtifactWorkflow:
    """Tests for workflows involving artifacts."""

    def test_artifact_preserved_through_workflow(
        self, temp_session_path: Path
    ) -> None:
        """
        Verify artifacts are preserved through save/load cycles.
        """
        session = Session()
        session.set("text", "hello")
        session.write_artifact("input.txt", b"hello world")

        # Execute some work
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.dispatch_single("reverse", session, {"key": "text"})

        # Add result artifact
        session.write_artifact("output.txt", session.get("text").encode())
        session.save(temp_session_path)

        # Resume and verify
        restored = Session.load(temp_session_path)

        assert restored.read_artifact("input.txt") == b"hello world"
        assert restored.read_artifact("output.txt") == b"olleh"


# =============================================================================
# INTEGRATION TEST WITH REAL LLM (OPTIONAL)
# =============================================================================


@pytest.mark.integration
class TestRealLLMWorkflow:
    """Integration tests with real LLM (requires Ollama)."""

    def test_complete_workflow_with_ollama(self, temp_session_path: Path) -> None:
        """
        Test complete workflow with real Ollama LLM.
        """
        from kaizen.llm import OllamaProvider

        llm = OllamaProvider()
        if not llm.is_available():
            pytest.skip("Ollama not available")

        session = Session()
        session.set("text", "hello world")

        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        planner = Planner(llm, capabilities=dispatcher.get_capabilities())

        # Plan
        plan_result = planner.plan(
            "Reverse the text and then make it uppercase",
            session=session,
        )

        print(f"Plan: {plan_result.calls}")

        if plan_result.success and plan_result.calls:
            # Execute
            dispatch_result = dispatcher.dispatch_sequence(
                plan_result.calls, session
            )
            print(f"Dispatch: {dispatch_result}")
            print(f"Final text: {session.get('text')}")

            # Save
            session.save(temp_session_path)

            # Restore and verify
            restored = Session.load(temp_session_path)
            print(f"Restored text: {restored.get('text')}")
