"""
Planner for Trace.

The Planner converts natural language user input into an ordered list of
capability calls that can be executed by the Dispatcher. It uses an LLM
to understand the user's intent and map it to available capabilities.

Design decisions:
1. LLM-based planning - uses an LLM provider for natural language understanding
2. Structured output - prompts the LLM to return JSON capability calls
3. Capability-aware - the planner knows what capabilities are available
4. Validation - validates the plan before returning it

The Planner does NOT:
- Execute capabilities (that's the Dispatcher's job)
- Handle ambiguity interactively (returns best-effort plan)
- Learn from feedback (no reinforcement in V1)

Example usage:
    from trace.llm import OllamaProvider
    from trace.planner import Planner

    provider = OllamaProvider()
    planner = Planner(provider)

    # Register available capabilities
    planner.set_capabilities(["reverse", "uppercase", "lowercase"])

    # Generate a plan from user input
    result = planner.plan("Reverse the text and make it uppercase")
    # Returns: [CapabilityCall("reverse", {...}), CapabilityCall("uppercase", {...})]
"""

import json
import re
from typing import Any

from trace.llm.base import LLMProvider, LLMError
from trace.types import CapabilityCall, ErrorCode, EntryType

# Import Session for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trace.session import Session


# =============================================================================
# PLAN RESULT
# =============================================================================


class PlanResult:
    """
    Result of planning operation.

    Attributes:
        success: Whether planning succeeded.
        calls: List of capability calls (if success=True).
        error: Error information (if success=False).
        raw_response: The raw LLM response text.
    """

    def __init__(
        self,
        success: bool,
        calls: list[CapabilityCall] | None = None,
        error: dict[str, Any] | None = None,
        raw_response: str | None = None,
    ) -> None:
        """
        Initialize a plan result.

        Args:
            success: Whether planning succeeded.
            calls: List of capability calls (required if success=True).
            error: Error information (required if success=False).
            raw_response: The raw LLM response text.
        """
        self.success = success
        self.calls = calls or []
        self.error = error
        self.raw_response = raw_response

    @classmethod
    def ok(
        cls,
        calls: list[CapabilityCall],
        raw_response: str | None = None,
    ) -> "PlanResult":
        """Create a successful plan result."""
        return cls(success=True, calls=calls, raw_response=raw_response)

    @classmethod
    def fail(
        cls,
        error_code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        raw_response: str | None = None,
    ) -> "PlanResult":
        """Create a failed plan result."""
        error = {
            "error_code": error_code.value,
            "message": message,
        }
        if details:
            error["details"] = details
        return cls(success=False, error=error, raw_response=raw_response)

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.success:
            caps = [c.capability for c in self.calls]
            return f"PlanResult(success=True, calls={caps})"
        return f"PlanResult(success=False, error={self.error})"


# =============================================================================
# PLANNER
# =============================================================================


# System prompt template for the LLM
SYSTEM_PROMPT = """You are a planning assistant that converts user requests into a sequence of capability calls.

Available capabilities:
{capabilities}

Your task:
1. Understand what the user wants to do
2. Break it down into a sequence of capability calls
3. Return ONLY a JSON array of capability calls

Each capability call must have this format:
{{"capability": "capability_name", "params": {{"key": "text"}}}}

The "key" parameter specifies which state key to operate on. Use "text" as the default key.

Rules:
- Return ONLY valid JSON, no other text
- The JSON must be an array of capability call objects
- Execute capabilities in the order they should be performed
- If the request doesn't match any capabilities, return an empty array []

Example user input: "reverse the text and make it uppercase"
Example output: [{{"capability": "reverse", "params": {{"key": "text"}}}}, {{"capability": "uppercase", "params": {{"key": "text"}}}}]

Remember: Return ONLY the JSON array, nothing else."""


class Planner:
    """
    Converts user input into capability call sequences using an LLM.

    The Planner uses an LLM to understand natural language requests and
    map them to available capabilities. It produces an ordered list of
    CapabilityCall objects that can be executed by the Dispatcher.

    Attributes:
        provider: The LLM provider to use for planning.
        capabilities: List of available capability names.

    Example:
        provider = OllamaProvider()
        planner = Planner(provider)
        planner.set_capabilities(["reverse", "uppercase"])

        result = planner.plan("reverse and uppercase the text")
        if result.success:
            for call in result.calls:
                print(f"Call: {call.capability}({call.params})")
    """

    def __init__(
        self,
        provider: LLMProvider,
        capabilities: list[str] | None = None,
    ) -> None:
        """
        Initialize the planner.

        Args:
            provider: The LLM provider to use.
            capabilities: Optional initial list of capabilities.
        """
        self._provider = provider
        self._capabilities: list[str] = capabilities or []

    @property
    def capabilities(self) -> list[str]:
        """Get the list of available capabilities."""
        return self._capabilities.copy()

    def set_capabilities(self, capabilities: list[str]) -> None:
        """
        Set the available capabilities.

        Args:
            capabilities: List of capability names.
        """
        self._capabilities = list(capabilities)

    def add_capability(self, capability: str) -> None:
        """
        Add a capability to the available list.

        Args:
            capability: Capability name to add.
        """
        if capability not in self._capabilities:
            self._capabilities.append(capability)

    def plan(
        self,
        user_input: str,
        session: "Session | None" = None,
    ) -> PlanResult:
        """
        Generate a plan from user input.

        Uses the LLM to convert natural language into capability calls.
        Optionally records the plan in the session's trajectory.

        Args:
            user_input: The user's natural language request.
            session: Optional session to record the plan in.

        Returns:
            PlanResult: Success with calls, or failure with error.
        """
        # Validate we have capabilities
        if not self._capabilities:
            return PlanResult.fail(
                error_code=ErrorCode.PLAN_GENERATION_FAILED,
                message="No capabilities available. Register agents with the dispatcher first.",
            )

        # Build the system prompt with available capabilities
        capabilities_text = "\n".join(f"- {cap}" for cap in self._capabilities)
        system_prompt = SYSTEM_PROMPT.format(capabilities=capabilities_text)

        # Call the LLM
        try:
            response = self._provider.complete(
                prompt=user_input,
                system=system_prompt,
            )
        except LLMError as e:
            return PlanResult.fail(
                error_code=ErrorCode.PLAN_LLM_ERROR,
                message=f"LLM error: {e.message}",
                details=e.details,
            )

        # Parse the response
        raw_text = response.text.strip()

        try:
            calls = self._parse_response(raw_text)
        except ValueError as e:
            return PlanResult.fail(
                error_code=ErrorCode.PLAN_INVALID_FORMAT,
                message=f"Failed to parse LLM response: {e}",
                details={"raw_response": raw_text},
                raw_response=raw_text,
            )

        # Validate the calls
        validation_error = self._validate_calls(calls)
        if validation_error:
            return PlanResult.fail(
                error_code=ErrorCode.PLAN_INVALID_FORMAT,
                message=validation_error,
                details={"calls": [c.to_dict() for c in calls]},
                raw_response=raw_text,
            )

        # Record in session if provided
        if session is not None:
            session.append(
                agent_id="planner",
                entry_type=EntryType.PLAN_CREATED,
                content={
                    "user_input": user_input,
                    "calls": [c.to_dict() for c in calls],
                    "model": self._provider.model_name,
                },
            )

        return PlanResult.ok(calls=calls, raw_response=raw_text)

    def _parse_response(self, text: str) -> list[CapabilityCall]:
        """
        Parse the LLM response into capability calls.

        The LLM should return a JSON array, but we need to handle
        cases where it includes extra text.

        Args:
            text: The raw LLM response text.

        Returns:
            list[CapabilityCall]: Parsed capability calls.

        Raises:
            ValueError: If parsing fails.
        """
        # Try to find JSON array in the response
        # The LLM might include extra text before/after
        json_match = re.search(r'\[.*\]', text, re.DOTALL)

        if not json_match:
            # Maybe it's empty or just says something like "no capabilities needed"
            if any(word in text.lower() for word in ["empty", "none", "no ", "[]"]):
                return []
            raise ValueError(f"No JSON array found in response: {text[:200]}")

        json_text = json_match.group(0)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        # Convert to CapabilityCall objects
        calls = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not an object: {item}")

            if "capability" not in item:
                raise ValueError(f"Item {i} missing 'capability' field")

            calls.append(CapabilityCall(
                capability=item["capability"],
                params=item.get("params", {}),
            ))

        return calls

    def _validate_calls(self, calls: list[CapabilityCall]) -> str | None:
        """
        Validate that all calls use known capabilities.

        Args:
            calls: List of capability calls to validate.

        Returns:
            Error message if validation fails, None if valid.
        """
        for call in calls:
            if call.capability not in self._capabilities:
                return (
                    f"Unknown capability '{call.capability}'. "
                    f"Available: {self._capabilities}"
                )
        return None

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Planner(provider={self._provider.model_name}, "
            f"capabilities={self._capabilities})"
        )
