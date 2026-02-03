"""
Tests for the Planner.

This module tests the Planner which converts natural language user input
into capability call sequences using an LLM.

Tests use a mock LLM provider to avoid requiring a running Ollama server.
Integration tests with real LLM are marked with @pytest.mark.integration.
"""

import pytest

from trace.planner import Planner, PlanResult
from trace.session import Session
from trace.llm.base import LLMProvider, LLMResponse, LLMError
from trace.types import CapabilityCall, ErrorCode, EntryType


# =============================================================================
# MOCK LLM PROVIDER
# =============================================================================


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns pre-configured responses for testing the planner's
    parsing and validation logic.
    """

    def __init__(self, response: str = "[]", model: str = "mock-model"):
        """
        Initialize with a canned response.

        Args:
            response: The text to return from complete().
            model: Model name to report.
        """
        self._response = response
        self._model = model
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.call_count = 0

    def set_response(self, response: str) -> None:
        """Set the response to return."""
        self._response = response

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return the configured response."""
        self.last_prompt = prompt
        self.last_system = system
        self.call_count += 1
        return LLMResponse(text=self._response, model=self._model)

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model


class FailingLLMProvider(LLMProvider):
    """Mock provider that always fails."""

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        raise LLMError("Mock failure", "mock")

    @property
    def model_name(self) -> str:
        return "failing-model"


# =============================================================================
# PLAN RESULT TESTS
# =============================================================================


class TestPlanResult:
    """Tests for PlanResult class."""

    def test_ok_result(self) -> None:
        """Verify PlanResult.ok() creates success result."""
        calls = [CapabilityCall("test", {"key": "value"})]
        result = PlanResult.ok(calls=calls)

        assert result.success is True
        assert result.calls == calls
        assert result.error is None

    def test_fail_result(self) -> None:
        """Verify PlanResult.fail() creates failure result."""
        result = PlanResult.fail(
            error_code=ErrorCode.PLAN_GENERATION_FAILED,
            message="Test failure",
        )

        assert result.success is False
        assert result.calls == []
        assert result.error is not None
        assert result.error["error_code"] == "plan_generation_failed"

    def test_fail_with_details(self) -> None:
        """Verify PlanResult.fail() includes details."""
        result = PlanResult.fail(
            error_code=ErrorCode.PLAN_INVALID_FORMAT,
            message="Parse error",
            details={"line": 5},
        )

        assert result.error["details"]["line"] == 5

    def test_repr_success(self) -> None:
        """Verify __repr__ for success."""
        calls = [CapabilityCall("test", {})]
        result = PlanResult.ok(calls=calls)

        repr_str = repr(result)
        assert "success=True" in repr_str
        assert "test" in repr_str

    def test_repr_failure(self) -> None:
        """Verify __repr__ for failure."""
        result = PlanResult.fail(ErrorCode.PLAN_GENERATION_FAILED, "Error")

        repr_str = repr(result)
        assert "success=False" in repr_str


# =============================================================================
# PLANNER CONFIGURATION TESTS
# =============================================================================


class TestPlannerConfig:
    """Tests for Planner configuration."""

    def test_create_planner(self) -> None:
        """Verify Planner can be created."""
        provider = MockLLMProvider()
        planner = Planner(provider)

        assert planner.capabilities == []

    def test_create_with_capabilities(self) -> None:
        """Verify Planner can be created with initial capabilities."""
        provider = MockLLMProvider()
        planner = Planner(provider, capabilities=["a", "b"])

        assert planner.capabilities == ["a", "b"]

    def test_set_capabilities(self) -> None:
        """Verify set_capabilities replaces capabilities."""
        provider = MockLLMProvider()
        planner = Planner(provider)

        planner.set_capabilities(["x", "y", "z"])

        assert planner.capabilities == ["x", "y", "z"]

    def test_add_capability(self) -> None:
        """Verify add_capability adds to list."""
        provider = MockLLMProvider()
        planner = Planner(provider, capabilities=["a"])

        planner.add_capability("b")

        assert "b" in planner.capabilities

    def test_add_capability_no_duplicates(self) -> None:
        """Verify add_capability doesn't add duplicates."""
        provider = MockLLMProvider()
        planner = Planner(provider, capabilities=["a"])

        planner.add_capability("a")  # Already exists

        assert planner.capabilities == ["a"]

    def test_capabilities_returns_copy(self) -> None:
        """Verify capabilities property returns a copy."""
        provider = MockLLMProvider()
        planner = Planner(provider, capabilities=["a"])

        caps = planner.capabilities
        caps.append("modified")

        assert "modified" not in planner.capabilities

    def test_repr(self) -> None:
        """Verify __repr__ returns useful string."""
        provider = MockLLMProvider()
        planner = Planner(provider, capabilities=["test"])

        repr_str = repr(planner)
        assert "Planner" in repr_str
        assert "mock-model" in repr_str


# =============================================================================
# PLANNER PLAN TESTS
# =============================================================================


class TestPlannerPlan:
    """Tests for Planner.plan() method."""

    def test_plan_no_capabilities_fails(self) -> None:
        """Verify plan fails when no capabilities registered."""
        provider = MockLLMProvider()
        planner = Planner(provider)

        result = planner.plan("do something")

        assert result.success is False
        assert result.error["error_code"] == "plan_generation_failed"
        assert "No capabilities" in result.error["message"]

    def test_plan_passes_prompt_to_llm(self) -> None:
        """Verify plan passes user input to LLM."""
        provider = MockLLMProvider(response="[]")
        planner = Planner(provider, capabilities=["test"])

        planner.plan("my user input")

        assert provider.last_prompt == "my user input"

    def test_plan_includes_capabilities_in_system(self) -> None:
        """Verify plan includes capabilities in system prompt."""
        provider = MockLLMProvider(response="[]")
        planner = Planner(provider, capabilities=["reverse", "uppercase"])

        planner.plan("test")

        assert "reverse" in provider.last_system
        assert "uppercase" in provider.last_system

    def test_plan_parses_valid_json(self) -> None:
        """Verify plan parses valid JSON response."""
        response = '[{"capability": "reverse", "params": {"key": "text"}}]'
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["reverse"])

        result = planner.plan("reverse the text")

        assert result.success is True
        assert len(result.calls) == 1
        assert result.calls[0].capability == "reverse"
        assert result.calls[0].params == {"key": "text"}

    def test_plan_parses_multiple_calls(self) -> None:
        """Verify plan parses multiple capability calls."""
        response = '''[
            {"capability": "reverse", "params": {"key": "text"}},
            {"capability": "uppercase", "params": {"key": "text"}}
        ]'''
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["reverse", "uppercase"])

        result = planner.plan("reverse and uppercase")

        assert result.success is True
        assert len(result.calls) == 2
        assert result.calls[0].capability == "reverse"
        assert result.calls[1].capability == "uppercase"

    def test_plan_handles_empty_array(self) -> None:
        """Verify plan handles empty response."""
        provider = MockLLMProvider(response="[]")
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("do nothing")

        assert result.success is True
        assert result.calls == []

    def test_plan_handles_json_with_extra_text(self) -> None:
        """Verify plan extracts JSON from text with extra content."""
        response = '''Here's the plan:
        [{"capability": "test", "params": {}}]
        That should work!'''
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("test")

        assert result.success is True
        assert len(result.calls) == 1

    def test_plan_handles_missing_params(self) -> None:
        """Verify plan handles calls without params field."""
        response = '[{"capability": "test"}]'
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("test")

        assert result.success is True
        assert result.calls[0].params == {}


class TestPlannerValidation:
    """Tests for plan validation."""

    def test_plan_rejects_unknown_capability(self) -> None:
        """Verify plan fails for unknown capabilities."""
        response = '[{"capability": "unknown", "params": {}}]'
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("test")

        assert result.success is False
        assert result.error["error_code"] == "plan_invalid_format"
        assert "Unknown capability" in result.error["message"]

    def test_plan_rejects_invalid_json(self) -> None:
        """Verify plan fails for invalid JSON."""
        response = "not valid json at all"
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("test")

        assert result.success is False
        assert result.error["error_code"] == "plan_invalid_format"

    def test_plan_rejects_non_array_json(self) -> None:
        """Verify plan fails for non-array JSON."""
        response = '{"capability": "test"}'  # Object, not array
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("test")

        assert result.success is False
        assert "array" in result.error["message"].lower()

    def test_plan_rejects_missing_capability_field(self) -> None:
        """Verify plan fails when capability field is missing."""
        response = '[{"params": {"key": "text"}}]'  # Missing capability
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("test")

        assert result.success is False
        assert "capability" in result.error["message"].lower()


class TestPlannerLLMErrors:
    """Tests for LLM error handling."""

    def test_plan_handles_llm_error(self) -> None:
        """Verify plan handles LLM errors."""
        provider = FailingLLMProvider()
        planner = Planner(provider, capabilities=["test"])

        result = planner.plan("test")

        assert result.success is False
        assert result.error["error_code"] == "plan_llm_error"


class TestPlannerTrajectory:
    """Tests for trajectory recording."""

    def test_plan_records_to_session(self) -> None:
        """Verify plan records to session trajectory."""
        response = '[{"capability": "test", "params": {}}]'
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])
        session = Session()

        planner.plan("test input", session=session)

        trajectory = session.get_trajectory()
        plan_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_CREATED
        ]

        assert len(plan_entries) == 1
        assert plan_entries[0].agent_id == "planner"
        assert plan_entries[0].content["user_input"] == "test input"

    def test_plan_records_calls_to_session(self) -> None:
        """Verify plan records generated calls."""
        response = '[{"capability": "reverse", "params": {"key": "text"}}]'
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["reverse"])
        session = Session()

        planner.plan("reverse", session=session)

        trajectory = session.get_trajectory()
        plan_entries = [
            e for e in trajectory
            if e.entry_type == EntryType.PLAN_CREATED
        ]

        assert len(plan_entries[0].content["calls"]) == 1
        assert plan_entries[0].content["calls"][0]["capability"] == "reverse"

    def test_plan_no_trajectory_without_session(self) -> None:
        """Verify plan doesn't require session."""
        response = '[{"capability": "test", "params": {}}]'
        provider = MockLLMProvider(response=response)
        planner = Planner(provider, capabilities=["test"])

        # Should not raise
        result = planner.plan("test")

        assert result.success is True


# =============================================================================
# INTEGRATION TEST (OPTIONAL)
# =============================================================================


@pytest.mark.integration
class TestPlannerIntegration:
    """Integration tests with real LLM."""

    def test_plan_with_ollama(self) -> None:
        """Test planning with real Ollama (if available)."""
        from trace.llm import OllamaProvider

        provider = OllamaProvider()
        if not provider.is_available():
            pytest.skip("Ollama not available")

        planner = Planner(provider, capabilities=["reverse", "uppercase"])

        result = planner.plan("Reverse the text and then make it uppercase")

        # The LLM should understand this and return appropriate calls
        assert result.success is True
        # We can't guarantee exact output, but it should have some calls
        print(f"LLM plan: {result.calls}")
