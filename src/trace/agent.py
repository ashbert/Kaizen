"""
Agent protocol for Trace.

This module defines the Agent abstract base class that all agents must
implement. An agent is a callable unit with declared capabilities that
can read/write session state and append to the trajectory.

The agent protocol is intentionally minimal:
1. info() - Return metadata about the agent (ID, name, version, capabilities)
2. invoke() - Execute a capability with given parameters

Agents should be:
- Stateless: All state lives in the Session
- Deterministic: Same inputs should produce same outputs
- Transparent: All actions should be recorded in the trajectory

Example implementation:

    class ReverseAgent(Agent):
        def info(self) -> AgentInfo:
            return AgentInfo(
                agent_id="reverse_v1",
                name="Reverse Agent",
                version="1.0.0",
                capabilities=["reverse"],
                description="Reverses text in session state"
            )

        def invoke(
            self,
            capability: str,
            session: Session,
            params: dict[str, Any]
        ) -> InvokeResult:
            if capability != "reverse":
                return InvokeResult.fail(...)

            # Do work, modify session, return result
            ...
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from trace.types import AgentInfo, InvokeResult, ErrorCode


# Using TYPE_CHECKING to avoid circular imports
# Session is only needed for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trace.session import Session


# =============================================================================
# AGENT PROTOCOL (STRUCTURAL TYPING)
# =============================================================================


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol definition for agents (structural typing).

    This protocol enables duck typing - any class with info() and invoke()
    methods with the right signatures can be used as an agent, even without
    inheriting from Agent.

    Use this for type hints when you want to accept any agent-like object:

        def process(agent: AgentProtocol, session: Session) -> None:
            info = agent.info()
            result = agent.invoke("capability", session, {})
    """

    def info(self) -> AgentInfo:
        """Return metadata about the agent."""
        ...

    def invoke(
        self,
        capability: str,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        """Execute a capability."""
        ...


# =============================================================================
# AGENT ABSTRACT BASE CLASS
# =============================================================================


class Agent(ABC):
    """
    Abstract base class for Trace agents.

    Agents are the callable units in Trace that perform work. Each agent
    declares one or more capabilities and can be invoked to execute them.

    When implementing an agent:

    1. Override info() to return agent metadata including capabilities
    2. Override invoke() to handle capability execution
    3. Use session.get/set for state access
    4. Use session.append for trajectory logging
    5. Return InvokeResult.ok() on success, InvokeResult.fail() on error

    Thread Safety:
        Agents should be stateless and thread-safe. All mutable state
        should live in the Session, not the agent instance.

    Example:
        class MyAgent(Agent):
            def info(self) -> AgentInfo:
                return AgentInfo(
                    agent_id="my_agent_v1",
                    name="My Agent",
                    version="1.0.0",
                    capabilities=["my_capability"],
                )

            def invoke(
                self,
                capability: str,
                session: Session,
                params: dict[str, Any]
            ) -> InvokeResult:
                if capability == "my_capability":
                    # Do work...
                    return InvokeResult.ok(
                        result={"done": True},
                        agent_id=self.info().agent_id,
                        capability=capability,
                    )
                return self._unknown_capability(capability)
    """

    @abstractmethod
    def info(self) -> AgentInfo:
        """
        Return metadata about this agent.

        This method should return an AgentInfo containing:
        - agent_id: Unique identifier for this agent
        - name: Human-readable name
        - version: Version string (semver recommended)
        - capabilities: List of capability names this agent handles
        - description: Optional description

        The returned info should be consistent across calls (agents are
        expected to have static metadata).

        Returns:
            AgentInfo: Metadata about this agent.
        """
        pass

    @abstractmethod
    def invoke(
        self,
        capability: str,
        session: "Session",
        params: dict[str, Any],
    ) -> InvokeResult:
        """
        Execute a capability.

        This is the main entry point for agent execution. The agent should:
        1. Validate the capability name
        2. Validate and extract parameters
        3. Read necessary state from session
        4. Perform the capability's work
        5. Update session state with results
        6. Append trajectory entries to record actions
        7. Return success or failure result

        Args:
            capability: Name of the capability to execute. Should be one
                       of the capabilities returned by info().
            session: The session to operate on. Agents can read/write state,
                    append to trajectory, and access artifacts.
            params: Parameters for the capability. Structure depends on
                   the specific capability.

        Returns:
            InvokeResult: Success or failure result with:
                - success=True: result field contains return data
                - success=False: error field contains error details

        Note:
            Agents should NOT raise exceptions for normal errors. Instead,
            return InvokeResult.fail() with appropriate error details.
            Exceptions should only be raised for truly unexpected conditions.
        """
        pass

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _unknown_capability(self, capability: str) -> InvokeResult:
        """
        Create a failure result for an unknown capability.

        Convenience method for handling unknown capability requests.

        Args:
            capability: The unknown capability name.

        Returns:
            InvokeResult: Failure result with appropriate error.
        """
        info = self.info()
        return InvokeResult.fail(
            error_code=ErrorCode.AGENT_CAPABILITY_NOT_FOUND,
            message=f"Unknown capability '{capability}'. "
                   f"Available: {info.capabilities}",
            agent_id=info.agent_id,
            capability=capability,
        )

    def _invalid_params(
        self,
        capability: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> InvokeResult:
        """
        Create a failure result for invalid parameters.

        Convenience method for parameter validation failures.

        Args:
            capability: The capability that was invoked.
            message: Description of what's wrong with the parameters.
            details: Optional additional details.

        Returns:
            InvokeResult: Failure result with appropriate error.
        """
        return InvokeResult.fail(
            error_code=ErrorCode.AGENT_INVALID_PARAMS,
            message=message,
            agent_id=self.info().agent_id,
            capability=capability,
            details=details,
        )

    def _invocation_failed(
        self,
        capability: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> InvokeResult:
        """
        Create a failure result for invocation failure.

        Convenience method for when capability execution fails.

        Args:
            capability: The capability that was invoked.
            message: Description of what went wrong.
            details: Optional additional details (e.g., exception info).

        Returns:
            InvokeResult: Failure result with appropriate error.
        """
        return InvokeResult.fail(
            error_code=ErrorCode.AGENT_INVOCATION_FAILED,
            message=message,
            agent_id=self.info().agent_id,
            capability=capability,
            details=details,
        )

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        info = self.info()
        return f"{self.__class__.__name__}(id={info.agent_id}, caps={info.capabilities})"
