"""
Dispatcher for Kaizen.

The Dispatcher routes capability calls to registered agents and executes
them sequentially. It is the orchestration layer that connects the Planner's
output to actual agent execution.

Design decisions:
1. Sequential execution only (V1) - no parallelism
2. Stop on first failure - return error immediately
3. Agents registered by capability - one agent per capability
4. All execution recorded in trajectory

The Dispatcher does NOT:
- Execute in parallel (deferred to V2)
- Retry failed operations
- Handle timeouts (agents should handle internally)
- Provide transaction/rollback semantics

Example usage:
    dispatcher = Dispatcher()
    dispatcher.register(ReverseAgent())
    dispatcher.register(UppercaseAgent())

    calls = [
        CapabilityCall("reverse", {"key": "text"}),
        CapabilityCall("uppercase", {"key": "text"}),
    ]

    results = dispatcher.dispatch_sequence(calls, session)
"""

from typing import Any

from kaizen.agent import Agent, AgentProtocol
from kaizen.types import (
    AgentInfo,
    CapabilityCall,
    InvokeResult,
    ErrorCode,
    EntryType,
)

# Import Session for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kaizen.session import Session


# =============================================================================
# DISPATCH RESULT
# =============================================================================


class DispatchResult:
    """
    Result of dispatching a sequence of capability calls.

    This aggregates the results of all executed calls and provides
    convenient access to success/failure status.

    Attributes:
        results: List of InvokeResult from each executed call.
        success: True if ALL calls succeeded, False if any failed.
        failed_at: Index of the first failed call (None if all succeeded).
        error: The error from the first failed call (None if all succeeded).
    """

    def __init__(self, results: list[InvokeResult]) -> None:
        """
        Initialize with list of results.

        Args:
            results: List of InvokeResult from executed calls.
        """
        self._results = results

        # Find first failure (if any)
        self._failed_at: int | None = None
        for i, result in enumerate(results):
            if not result.success:
                self._failed_at = i
                break

    @property
    def results(self) -> list[InvokeResult]:
        """Get all execution results."""
        return self._results

    @property
    def success(self) -> bool:
        """True if all calls succeeded."""
        return self._failed_at is None

    @property
    def failed_at(self) -> int | None:
        """Index of first failed call, or None if all succeeded."""
        return self._failed_at

    @property
    def error(self) -> dict[str, Any] | None:
        """Error from first failed call, or None if all succeeded."""
        if self._failed_at is None:
            return None
        return self._results[self._failed_at].error

    @property
    def executed_count(self) -> int:
        """Number of calls that were executed."""
        return len(self._results)

    @property
    def completed_indices(self) -> list[int]:
        """Indices of calls that completed successfully."""
        return [i for i, r in enumerate(self._results) if r.success]

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.success:
            return f"DispatchResult(success=True, executed={self.executed_count})"
        return (
            f"DispatchResult(success=False, failed_at={self.failed_at}, "
            f"error_code={self.error['error_code'] if self.error else None})"
        )


# =============================================================================
# DISPATCHER
# =============================================================================


class Dispatcher:
    """
    Routes capability calls to agents and executes them sequentially.

    The Dispatcher maintains a registry of agents indexed by capability.
    When dispatch_sequence() is called, it looks up the appropriate agent
    for each capability and invokes it in order.

    Registration:
        Agents are registered by calling register(agent). The dispatcher
        extracts capabilities from agent.info() and maps each capability
        to the agent. If a capability is already registered, it is
        overwritten (last registration wins).

    Execution:
        dispatch_sequence() executes calls in order. If any call fails,
        execution stops immediately and the error is returned. This is
        "fail-fast" behavior for easier debugging.

    Thread Safety:
        The Dispatcher itself is not thread-safe. For concurrent access,
        external synchronization is required. However, since execution
        is sequential, this is rarely needed.

    Example:
        dispatcher = Dispatcher()
        dispatcher.register(ReverseAgent())
        dispatcher.register(UppercaseAgent())

        calls = [
            CapabilityCall("reverse", {"key": "text"}),
            CapabilityCall("uppercase", {"key": "text"}),
        ]

        result = dispatcher.dispatch_sequence(calls, session)
        if result.success:
            print("All calls succeeded!")
        else:
            print(f"Failed at call {result.failed_at}: {result.error}")
    """

    def __init__(self) -> None:
        """Initialize an empty dispatcher."""
        # Map from capability name to agent
        # Each capability maps to exactly one agent
        self._capability_to_agent: dict[str, AgentProtocol] = {}

        # Map from agent_id to AgentInfo for introspection
        self._agent_info: dict[str, AgentInfo] = {}

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(self, agent: AgentProtocol) -> None:
        """
        Register an agent with the dispatcher.

        The agent's capabilities are extracted from agent.info() and
        each capability is mapped to the agent. If a capability is
        already registered to another agent, it is overwritten.

        Args:
            agent: The agent to register. Must implement AgentProtocol.

        Example:
            dispatcher = Dispatcher()
            dispatcher.register(ReverseAgent())
            dispatcher.register(UppercaseAgent())
        """
        info = agent.info()

        # Store agent info for introspection
        self._agent_info[info.agent_id] = info

        # Register each capability
        for capability in info.capabilities:
            self._capability_to_agent[capability] = agent

    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent by its ID.

        Removes the agent and all its capability mappings.

        Args:
            agent_id: The ID of the agent to unregister.

        Returns:
            bool: True if agent was found and removed, False if not found.
        """
        if agent_id not in self._agent_info:
            return False

        info = self._agent_info[agent_id]

        # Remove capability mappings for this agent
        for capability in info.capabilities:
            if capability in self._capability_to_agent:
                # Only remove if still mapped to this agent
                # (another agent may have overwritten it)
                current = self._capability_to_agent[capability]
                if current.info().agent_id == agent_id:
                    del self._capability_to_agent[capability]

        # Remove agent info
        del self._agent_info[agent_id]

        return True

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def get_capabilities(self) -> list[str]:
        """
        Get list of all registered capabilities.

        Returns:
            list[str]: Sorted list of capability names.
        """
        return sorted(self._capability_to_agent.keys())

    def get_agent_for_capability(self, capability: str) -> AgentProtocol | None:
        """
        Get the agent registered for a capability.

        Args:
            capability: The capability name.

        Returns:
            The registered agent, or None if capability is not registered.
        """
        return self._capability_to_agent.get(capability)

    def get_registered_agents(self) -> list[AgentInfo]:
        """
        Get info for all registered agents.

        Returns:
            list[AgentInfo]: Info for each registered agent.
        """
        return list(self._agent_info.values())

    def has_capability(self, capability: str) -> bool:
        """
        Check if a capability is registered.

        Args:
            capability: The capability name.

        Returns:
            bool: True if capability is registered.
        """
        return capability in self._capability_to_agent

    # =========================================================================
    # DISPATCH
    # =========================================================================

    def dispatch_sequence(
        self,
        calls: list[CapabilityCall] | list[dict[str, Any]],
        session: "Session",
    ) -> DispatchResult:
        """
        Execute a sequence of capability calls.

        Calls are executed in order. If any call fails, execution stops
        immediately and the partial results are returned. This is
        "fail-fast" behavior.

        Each call is:
        1. Validated (capability must be registered)
        2. Recorded in trajectory (PLAN_STEP_STARTED)
        3. Executed via agent.invoke()
        4. Result collected

        Args:
            calls: List of CapabilityCall objects or dicts with
                   'capability' and 'params' keys.
            session: The session to operate on.

        Returns:
            DispatchResult: Contains all results and success/failure info.

        Example:
            calls = [
                CapabilityCall("reverse", {"key": "text"}),
                {"capability": "uppercase", "params": {"key": "text"}},
            ]
            result = dispatcher.dispatch_sequence(calls, session)
        """
        results: list[InvokeResult] = []

        for i, call in enumerate(calls):
            # Normalize call to CapabilityCall if it's a dict
            if isinstance(call, dict):
                call = CapabilityCall.from_dict(call)

            # Record that we're starting this step
            session.append(
                agent_id="dispatcher",
                entry_type=EntryType.PLAN_STEP_STARTED,
                content={
                    "step_index": i,
                    "capability": call.capability,
                    "params": call.params,
                },
            )

            # Look up the agent for this capability
            agent = self._capability_to_agent.get(call.capability)

            if agent is None:
                # No agent registered for this capability
                error_result = InvokeResult.fail(
                    error_code=ErrorCode.DISPATCH_NO_AGENT_FOR_CAPABILITY,
                    message=f"No agent registered for capability '{call.capability}'",
                    agent_id="dispatcher",
                    capability=call.capability,
                    details={
                        "available_capabilities": self.get_capabilities(),
                        "step_index": i,
                    },
                )
                results.append(error_result)

                # Record step completion (failed)
                session.append(
                    agent_id="dispatcher",
                    entry_type=EntryType.PLAN_STEP_COMPLETED,
                    content={
                        "step_index": i,
                        "capability": call.capability,
                        "success": False,
                        "result_summary": str(error_result.error)[:200],
                    },
                )

                # Fail fast - stop execution
                return DispatchResult(results)

            # Execute the capability
            try:
                result = agent.invoke(call.capability, session, call.params)
            except Exception as e:
                # Agent raised an exception (shouldn't happen, but handle it)
                result = InvokeResult.fail(
                    error_code=ErrorCode.AGENT_INVOCATION_FAILED,
                    message=f"Agent raised exception: {type(e).__name__}: {e}",
                    agent_id=agent.info().agent_id,
                    capability=call.capability,
                    details={
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "step_index": i,
                    },
                )

            results.append(result)

            # Record step completion
            session.append(
                agent_id="dispatcher",
                entry_type=EntryType.PLAN_STEP_COMPLETED,
                content={
                    "step_index": i,
                    "capability": call.capability,
                    "success": result.success,
                    "result_summary": str(result.result)[:200] if result.success else str(result.error)[:200],
                },
            )

            # If this call failed, stop execution (fail-fast)
            if not result.success:
                return DispatchResult(results)

        # All calls succeeded
        return DispatchResult(results)

    def dispatch_single(
        self,
        capability: str,
        session: "Session",
        params: dict[str, Any] | None = None,
    ) -> InvokeResult:
        """
        Execute a single capability call.

        Convenience method for executing one call without creating
        a CapabilityCall object.

        Args:
            capability: The capability to invoke.
            session: The session to operate on.
            params: Parameters for the capability (default: empty dict).

        Returns:
            InvokeResult: The result of the invocation.

        Example:
            result = dispatcher.dispatch_single("reverse", session, {"key": "text"})
        """
        call = CapabilityCall(capability, params or {})
        dispatch_result = self.dispatch_sequence([call], session)
        return dispatch_result.results[0]

    # =========================================================================
    # RESUME
    # =========================================================================

    def resume_sequence(
        self,
        calls: list[CapabilityCall] | list[dict[str, Any]],
        session: "Session",
    ) -> DispatchResult:
        """
        Resume a previously interrupted sequence of capability calls.

        Scans the session trajectory for PLAN_STEP_COMPLETED entries with
        success=True, determines which steps already succeeded, and skips
        them. Resumes execution from the first incomplete step.

        Matching is done by (step_index, capability) pairs to handle cases
        where the calls list may have been reordered or modified since the
        last run. A step is considered completed only if both its index AND
        capability name match a successful PLAN_STEP_COMPLETED entry.

        Args:
            calls: List of CapabilityCall objects or dicts with
                   'capability' and 'params' keys.
            session: The session to operate on.

        Returns:
            DispatchResult: Contains results for all steps (both previously
            completed and newly executed). Previously completed steps are
            represented as synthetic InvokeResult.ok entries.
        """
        # Normalize all calls to CapabilityCall
        normalized: list[CapabilityCall] = []
        for call in calls:
            if isinstance(call, dict):
                normalized.append(CapabilityCall.from_dict(call))
            else:
                normalized.append(call)

        # Scan trajectory for successful PLAN_STEP_COMPLETED entries
        completed: set[tuple[int, str]] = set()
        for entry in session.get_trajectory():
            if entry.entry_type == EntryType.PLAN_STEP_COMPLETED:
                content = entry.content
                if content.get("success"):
                    completed.add((content["step_index"], content["capability"]))

        # Execute steps, skipping already-completed ones
        results: list[InvokeResult] = []

        for i, call in enumerate(normalized):
            if (i, call.capability) in completed:
                # This step already succeeded — create a synthetic result
                results.append(InvokeResult.ok(
                    result={"resumed": True, "step_index": i},
                    agent_id="dispatcher",
                    capability=call.capability,
                ))
                continue

            # Record that we're starting this step
            session.append(
                agent_id="dispatcher",
                entry_type=EntryType.PLAN_STEP_STARTED,
                content={
                    "step_index": i,
                    "capability": call.capability,
                    "params": call.params,
                },
            )

            # Look up the agent for this capability
            agent = self._capability_to_agent.get(call.capability)

            if agent is None:
                error_result = InvokeResult.fail(
                    error_code=ErrorCode.DISPATCH_NO_AGENT_FOR_CAPABILITY,
                    message=f"No agent registered for capability '{call.capability}'",
                    agent_id="dispatcher",
                    capability=call.capability,
                    details={
                        "available_capabilities": self.get_capabilities(),
                        "step_index": i,
                    },
                )
                results.append(error_result)

                session.append(
                    agent_id="dispatcher",
                    entry_type=EntryType.PLAN_STEP_COMPLETED,
                    content={
                        "step_index": i,
                        "capability": call.capability,
                        "success": False,
                        "result_summary": str(error_result.error)[:200],
                    },
                )
                return DispatchResult(results)

            # Execute the capability
            try:
                result = agent.invoke(call.capability, session, call.params)
            except Exception as e:
                result = InvokeResult.fail(
                    error_code=ErrorCode.AGENT_INVOCATION_FAILED,
                    message=f"Agent raised exception: {type(e).__name__}: {e}",
                    agent_id=agent.info().agent_id,
                    capability=call.capability,
                    details={
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "step_index": i,
                    },
                )

            results.append(result)

            # Record step completion
            session.append(
                agent_id="dispatcher",
                entry_type=EntryType.PLAN_STEP_COMPLETED,
                content={
                    "step_index": i,
                    "capability": call.capability,
                    "success": result.success,
                    "result_summary": str(result.result)[:200] if result.success else str(result.error)[:200],
                },
            )

            if not result.success:
                return DispatchResult(results)

        return DispatchResult(results)

    # =========================================================================
    # DEBUGGING
    # =========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        caps = self.get_capabilities()
        agents = len(self._agent_info)
        return f"Dispatcher(agents={agents}, capabilities={caps})"
