"""
Uppercase Agent - A toy agent that converts text to uppercase.

This agent demonstrates the Agent protocol by implementing a simple
text transformation capability. It serves as a reference implementation
and is used in tests and examples.

Capability: "uppercase"
    Converts a string value in session state to uppercase.

    Parameters:
        key (str): The state key containing the text to uppercase.
                   The uppercased text is written back to the same key.

    Result:
        original (str): The original text before transformation.
        uppercased (str): The text after transformation.

Example:
    session.set("text", "hello")
    agent.invoke("uppercase", session, {"key": "text"})
    # session.get("text") == "HELLO"
"""

from typing import Any

from trace.agent import Agent
from trace.types import AgentInfo, InvokeResult, EntryType

# Import Session for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trace.session import Session


class UppercaseAgent(Agent):
    """
    Agent that converts text to uppercase in session state.

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
            agent_id="uppercase_agent_v1",
            name="Uppercase Agent",
            version="1.0.0",
            capabilities=["uppercase"],
            description="Converts text to uppercase in session state",
        )

    def invoke(
        self,
        capability: str,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        """
        Execute the uppercase capability.

        Args:
            capability: Must be "uppercase".
            session: Session containing the text to uppercase.
            params: Must contain "key" - the state key with text.

        Returns:
            InvokeResult with original and uppercased text on success,
            or error details on failure.
        """
        # Validate capability
        if capability != "uppercase":
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
        # Execute the capability: uppercase the string
        # -----------------------------------------------------------------
        original = value
        uppercased = value.upper()

        # Record the action in trajectory BEFORE modifying state
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

        # Update session state with uppercased text
        session.set(key, uppercased)

        # Record successful completion
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": capability,
                "original": original,
                "uppercased": uppercased,
            },
        )

        # Return success result
        return InvokeResult.ok(
            result={
                "original": original,
                "uppercased": uppercased,
            },
            agent_id=agent_info.agent_id,
            capability=capability,
        )
