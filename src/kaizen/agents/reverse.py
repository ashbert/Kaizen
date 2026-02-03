"""
Reverse Agent - A toy agent that reverses text.

This agent demonstrates the Agent protocol by implementing a simple
text reversal capability. It serves as a reference implementation
and is used in tests and examples.

Capability: "reverse"
    Reverses a string value in session state.

    Parameters:
        key (str): The state key containing the text to reverse.
                   The reversed text is written back to the same key.

    Result:
        original (str): The original text before reversal.
        reversed (str): The text after reversal.

Example:
    session.set("text", "hello")
    agent.invoke("reverse", session, {"key": "text"})
    # session.get("text") == "olleh"
"""

from typing import Any

from kaizen.agent import Agent
from kaizen.types import AgentInfo, InvokeResult, EntryType

# Import Session for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kaizen.session import Session


class ReverseAgent(Agent):
    """
    Agent that reverses text stored in session state.

    This is a simple demonstration agent that shows how to:
    - Implement the Agent protocol
    - Read and write session state
    - Record actions in the trajectory
    - Handle errors with appropriate InvokeResult

    The agent is stateless - all data lives in the session.
    """

    def info(self) -> AgentInfo:
        """
        Return metadata about this agent.

        Returns:
            AgentInfo with agent details and capabilities.
        """
        return AgentInfo(
            agent_id="reverse_agent_v1",
            name="Reverse Agent",
            version="1.0.0",
            capabilities=["reverse"],
            description="Reverses text stored in session state",
        )

    def invoke(
        self,
        capability: str,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        """
        Execute the reverse capability.

        Args:
            capability: Must be "reverse".
            session: Session containing the text to reverse.
            params: Must contain "key" - the state key with text.

        Returns:
            InvokeResult with original and reversed text on success,
            or error details on failure.
        """
        # Validate capability
        if capability != "reverse":
            return self._unknown_capability(capability)

        # Validate params
        if "key" not in params:
            return self._invalid_params(
                capability,
                "Missing required parameter: key",
                details={"required": ["key"], "received": list(params.keys())},
            )

        key = params["key"]

        # Validate key is a string
        if not isinstance(key, str):
            return self._invalid_params(
                capability,
                f"Parameter 'key' must be a string, got {type(key).__name__}",
            )

        # Get the value from session state
        value = session.get(key)

        # Validate value exists
        if value is None:
            return self._invalid_params(
                capability,
                f"No value found at key '{key}'",
            )

        # Validate value is a string
        if not isinstance(value, str):
            return self._invalid_params(
                capability,
                f"Value at '{key}' must be a string, got {type(value).__name__}",
            )

        # -----------------------------------------------------------------
        # Execute the capability: reverse the string
        # -----------------------------------------------------------------
        original = value
        reversed_text = value[::-1]

        # Record the action in trajectory BEFORE modifying state
        # This ensures the trajectory reflects the intent even if
        # the state modification fails
        agent_info = self.info()
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_INVOKED,
            content={
                "capability": capability,
                "params": params,
                "input_value": original,
            },
        )

        # Update session state with reversed text
        session.set(key, reversed_text)

        # Record successful completion
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": capability,
                "original": original,
                "reversed": reversed_text,
            },
        )

        # Return success result
        return InvokeResult.ok(
            result={
                "original": original,
                "reversed": reversed_text,
            },
            agent_id=agent_info.agent_id,
            capability=capability,
        )
