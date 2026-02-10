"""
Core data types for Kaizen.

This module defines the fundamental data structures used throughout the Kaizen
session substrate. These types are designed to be:

1. Immutable where appropriate (trajectory entries should never change)
2. Serializable to JSON for persistence
3. Self-documenting with clear field definitions

The types defined here form the "vocabulary" of the system - they are the
building blocks that sessions, agents, and dispatchers communicate with.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# =============================================================================
# ENUMS
# =============================================================================


class EntryType(str, Enum):
    """
    Classification of trajectory entries.

    Each entry in the trajectory has a type that indicates what kind of
    action or event it represents. This allows for filtering, analysis,
    and replay of specific entry types.

    The enum inherits from str to ensure JSON serialization produces
    readable string values rather than integer codes.
    """

    # Agent-related entries
    AGENT_INVOKED = "agent_invoked"      # An agent capability was called
    AGENT_COMPLETED = "agent_completed"  # An agent finished execution
    AGENT_FAILED = "agent_failed"        # An agent encountered an error

    # State-related entries
    STATE_SET = "state_set"              # A state key was set/updated
    STATE_DELETED = "state_deleted"      # A state key was removed

    # Artifact-related entries
    ARTIFACT_WRITTEN = "artifact_written"   # An artifact was stored
    ARTIFACT_DELETED = "artifact_deleted"   # An artifact was removed

    # Session lifecycle entries
    SESSION_CREATED = "session_created"     # Session was initialized
    SESSION_LOADED = "session_loaded"       # Session was loaded from disk
    SESSION_SAVED = "session_saved"         # Session was saved to disk

    # Planning entries
    PLAN_CREATED = "plan_created"           # Planner generated a plan
    PLAN_STEP_STARTED = "plan_step_started" # A plan step began execution
    PLAN_STEP_COMPLETED = "plan_step_completed" # A plan step finished execution

    # User/system entries
    USER_INPUT = "user_input"               # User provided input
    SYSTEM_NOTE = "system_note"             # System-generated note/log


class ErrorCode(str, Enum):
    """
    Standard error codes for Kaizen operations.

    Using an enum ensures consistent error codes across the codebase
    and makes it easy to handle specific error conditions programmatically.

    Error codes are organized by category:
    - SESSION_*: Session-level errors
    - AGENT_*: Agent execution errors
    - DISPATCH_*: Dispatcher errors
    - PLAN_*: Planner errors
    - PERSIST_*: Persistence/storage errors
    """

    # Session errors (1xx equivalent)
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_CORRUPTED = "session_corrupted"
    SESSION_INVALID_STATE = "session_invalid_state"

    # Agent errors (2xx equivalent)
    AGENT_NOT_FOUND = "agent_not_found"
    AGENT_CAPABILITY_NOT_FOUND = "agent_capability_not_found"
    AGENT_INVOCATION_FAILED = "agent_invocation_failed"
    AGENT_INVALID_PARAMS = "agent_invalid_params"
    AGENT_TIMEOUT = "agent_timeout"

    # Dispatcher errors (3xx equivalent)
    DISPATCH_SEQUENCE_FAILED = "dispatch_sequence_failed"
    DISPATCH_NO_AGENT_FOR_CAPABILITY = "dispatch_no_agent_for_capability"

    # Planner errors (4xx equivalent)
    PLAN_GENERATION_FAILED = "plan_generation_failed"
    PLAN_INVALID_FORMAT = "plan_invalid_format"
    PLAN_LLM_ERROR = "plan_llm_error"

    # Persistence errors (5xx equivalent)
    PERSIST_SAVE_FAILED = "persist_save_failed"
    PERSIST_LOAD_FAILED = "persist_load_failed"
    PERSIST_FILE_NOT_FOUND = "persist_file_not_found"
    PERSIST_SCHEMA_MISMATCH = "persist_schema_mismatch"

    # Artifact errors (6xx equivalent)
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_TOO_LARGE = "artifact_too_large"
    ARTIFACT_INVALID_NAME = "artifact_invalid_name"

    # Generic errors
    UNKNOWN_ERROR = "unknown_error"
    VALIDATION_ERROR = "validation_error"


# =============================================================================
# TRAJECTORY ENTRY
# =============================================================================


@dataclass(frozen=True)
class TrajectoryEntry:
    """
    An immutable record of an action or event in a session's trajectory.

    The trajectory is the append-only log that records everything that happens
    in a session. Each entry captures:

    - WHEN: timestamp of when the entry was created
    - WHO: which agent (or system) created the entry
    - WHAT: the type of entry and its content

    Entries are immutable (frozen=True) to enforce the append-only invariant.
    Once an entry is created, it cannot be modified - only new entries can
    be appended.

    Attributes:
        seq_num: Monotonically increasing sequence number (1-indexed).
                 Assigned by the session when the entry is appended.
        timestamp: UTC timestamp when the entry was created.
        agent_id: Identifier of the agent that created this entry.
                  Use "system" for system-generated entries.
        entry_type: Classification of the entry (see EntryType enum).
        content: Arbitrary JSON-serializable content specific to the entry type.
                 The structure depends on entry_type.

    Example:
        entry = TrajectoryEntry(
            seq_num=1,
            timestamp=datetime.now(timezone.utc),
            agent_id="reverse_agent",
            entry_type=EntryType.AGENT_COMPLETED,
            content={"input": "hello", "output": "olleh"}
        )
    """

    seq_num: int
    timestamp: datetime
    agent_id: str
    entry_type: EntryType
    content: dict[str, Any]

    def __post_init__(self) -> None:
        """
        Validate the entry after initialization.

        Since the dataclass is frozen, we can't modify values here,
        but we can raise errors for invalid data.
        """
        # Validate seq_num is positive
        if self.seq_num < 1:
            raise ValueError(f"seq_num must be >= 1, got {self.seq_num}")

        # Validate agent_id is not empty
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")

        # Validate timestamp has timezone info (should be UTC)
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (use UTC)")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the entry to a JSON-serializable dictionary.

        This is used for persistence and for creating snapshots.
        The timestamp is serialized as an ISO format string.

        Returns:
            dict: JSON-serializable representation of the entry.
        """
        return {
            "seq_num": self.seq_num,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "entry_type": self.entry_type.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryEntry":
        """
        Create a TrajectoryEntry from a dictionary.

        This is used when loading entries from persistence.

        Args:
            data: Dictionary with entry fields. The timestamp should be
                  an ISO format string.

        Returns:
            TrajectoryEntry: Reconstructed entry.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If field values are invalid.
        """
        return cls(
            seq_num=data["seq_num"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_id=data["agent_id"],
            entry_type=EntryType(data["entry_type"]),
            content=data["content"],
        )


# =============================================================================
# INVOKE RESULT
# =============================================================================


@dataclass(frozen=True)
class InvokeResult:
    """
    The result of invoking an agent capability.

    This is returned by Agent.invoke() and collected by the Dispatcher.
    It captures whether the invocation succeeded and carries either the
    result data or error information.

    The class is immutable (frozen=True) to prevent accidental modification
    of results after they're returned.

    Attributes:
        success: True if the capability executed without errors.
        result: The return value from the capability (if success=True).
                Can be any JSON-serializable value.
        error: Error information (if success=False). Contains error_code
               and message for debugging.
        agent_id: The ID of the agent that was invoked.
        capability: The capability that was invoked.

    Example (success):
        result = InvokeResult(
            success=True,
            result={"reversed": "olleh"},
            error=None,
            agent_id="reverse_agent",
            capability="reverse"
        )

    Example (failure):
        result = InvokeResult(
            success=False,
            result=None,
            error={
                "error_code": ErrorCode.AGENT_INVALID_PARAMS.value,
                "message": "Missing required parameter: text"
            },
            agent_id="reverse_agent",
            capability="reverse"
        )
    """

    success: bool
    result: Any  # JSON-serializable, present if success=True
    error: dict[str, Any] | None  # Present if success=False
    agent_id: str
    capability: str

    def __post_init__(self) -> None:
        """
        Validate the result after initialization.

        Ensures that:
        - Successful results don't have error info
        - Failed results have error info
        """
        if self.success and self.error is not None:
            raise ValueError("Successful result should not have error info")

        if not self.success and self.error is None:
            raise ValueError("Failed result must have error info")

    @classmethod
    def ok(
        cls,
        result: Any,
        agent_id: str,
        capability: str,
    ) -> "InvokeResult":
        """
        Factory method to create a successful result.

        This is the preferred way to create success results as it
        ensures the error field is properly set to None.

        Args:
            result: The return value from the capability.
            agent_id: The ID of the agent that was invoked.
            capability: The capability that was invoked.

        Returns:
            InvokeResult: A successful result.
        """
        return cls(
            success=True,
            result=result,
            error=None,
            agent_id=agent_id,
            capability=capability,
        )

    @classmethod
    def fail(
        cls,
        error_code: ErrorCode,
        message: str,
        agent_id: str,
        capability: str,
        details: dict[str, Any] | None = None,
    ) -> "InvokeResult":
        """
        Factory method to create a failed result.

        This is the preferred way to create failure results as it
        ensures consistent error structure.

        Args:
            error_code: The error code from ErrorCode enum.
            message: Human-readable error message.
            agent_id: The ID of the agent that was invoked.
            capability: The capability that was invoked.
            details: Optional additional error details for debugging.

        Returns:
            InvokeResult: A failed result.
        """
        error = {
            "error_code": error_code.value,
            "message": message,
        }
        if details:
            error["details"] = details

        return cls(
            success=False,
            result=None,
            error=error,
            agent_id=agent_id,
            capability=capability,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to a JSON-serializable dictionary.

        Returns:
            dict: JSON-serializable representation of the result.
        """
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "agent_id": self.agent_id,
            "capability": self.capability,
        }


# =============================================================================
# CAPABILITY CALL
# =============================================================================


@dataclass(frozen=True)
class CapabilityCall:
    """
    A request to invoke a specific capability with parameters.

    This is the unit of work that the Planner produces and the Dispatcher
    consumes. It specifies what capability to call and with what parameters.

    Attributes:
        capability: The name of the capability to invoke (e.g., "reverse").
        params: Parameters to pass to the capability. Must be JSON-serializable.

    Example:
        call = CapabilityCall(
            capability="reverse",
            params={"key": "text"}
        )
    """

    capability: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the call after initialization."""
        if not self.capability or not self.capability.strip():
            raise ValueError("capability cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Returns:
            dict: Dictionary with capability and params.
        """
        return {
            "capability": self.capability,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityCall":
        """
        Create a CapabilityCall from a dictionary.

        Args:
            data: Dictionary with 'capability' and optional 'params'.

        Returns:
            CapabilityCall: The constructed call.
        """
        return cls(
            capability=data["capability"],
            params=data.get("params", {}),
        )


# =============================================================================
# AGENT INFO
# =============================================================================


@dataclass(frozen=True)
class AgentInfo:
    """
    Metadata about an agent.

    This is returned by Agent.info() and provides information about
    the agent's identity and capabilities. It's used by the Dispatcher
    to route capability calls to the appropriate agent.

    Attributes:
        agent_id: Unique identifier for the agent instance.
        name: Human-readable name of the agent.
        version: Version string for the agent (semver recommended).
        capabilities: List of capability names this agent can handle.
        description: Optional description of what the agent does.

    Example:
        info = AgentInfo(
            agent_id="reverse_agent_v1",
            name="Reverse Agent",
            version="1.0.0",
            capabilities=["reverse"],
            description="Reverses text stored in session state"
        )
    """

    agent_id: str
    name: str
    version: str
    capabilities: list[str]
    description: str = ""

    def __post_init__(self) -> None:
        """Validate agent info after initialization."""
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")

        if not self.name or not self.name.strip():
            raise ValueError("name cannot be empty")

        if not self.capabilities:
            raise ValueError("capabilities cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Returns:
            dict: Dictionary representation.
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "description": self.description,
        }
