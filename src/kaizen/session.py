"""
Session implementation for Kaizen.

The Session is the central data structure in Kaizen - it represents a single
unit of execution and persistence. A session contains:

1. State: A versioned key-value store for any JSON-serializable data
2. Trajectory: An append-only log of all actions and events
3. Artifacts: Binary blob storage for files and other data

This module implements the Session class with all subsystems. The design
priorities are:

- Auditability: Everything is recorded in the trajectory
- Reproducibility: Sessions can be replayed deterministically
- Resume-after-failure: Sessions persist to disk and can be restored
- Isolation: Snapshots are copies, never live references

Key Invariants:
- Trajectory is append-only and strictly ordered
- State version is monotonically increasing
- Snapshots are deep copies, never references
- Save/load roundtrip preserves all data
"""

import copy
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaizen.types import TrajectoryEntry, EntryType, ErrorCode


# =============================================================================
# CONFIGURATION
# =============================================================================


# Default maximum artifact size in bytes (100MB)
DEFAULT_MAX_ARTIFACT_SIZE = 100 * 1024 * 1024

# Current schema version for persistence format
SCHEMA_VERSION = 1


# =============================================================================
# SESSION CLASS
# =============================================================================


class Session:
    """
    A session is the unit of execution and persistence in Kaizen.

    Sessions are portable and resumable - they can be saved to disk and
    restored later with full fidelity. All operations are recorded in
    the trajectory for auditability.

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        external synchronization is required.

    Attributes:
        session_id: Unique identifier for this session (UUID).
        max_artifact_size: Maximum allowed artifact size in bytes.

    Example:
        # Create a new session
        session = Session()

        # Work with state
        session.set("input", "hello world")
        text = session.get("input")
        version = session.get_state_version()

        # Save and restore
        session.save("my_session.kaizen")
        restored = Session.load("my_session.kaizen")
    """

    def __init__(
        self,
        session_id: str | None = None,
        max_artifact_size: int = DEFAULT_MAX_ARTIFACT_SIZE,
    ) -> None:
        """
        Initialize a new session.

        Args:
            session_id: Optional session ID. If not provided, a UUID is generated.
            max_artifact_size: Maximum artifact size in bytes. Default is 100MB.
        """
        # Generate a unique session ID if not provided
        # Using UUID4 for random unique identifiers
        self._session_id = session_id or str(uuid.uuid4())

        # Configuration
        self._max_artifact_size = max_artifact_size

        # -----------------------------------------------------------------
        # State Storage
        # -----------------------------------------------------------------
        # The state is a simple key-value store. Keys are strings,
        # values can be any JSON-serializable data.
        self._state: dict[str, Any] = {}

        # State version starts at 0 and increments on every set() call.
        # This provides a monotonically increasing version number for
        # optimistic concurrency control and change detection.
        self._state_version: int = 0

        # -----------------------------------------------------------------
        # Trajectory Storage
        # -----------------------------------------------------------------
        # The trajectory is an append-only list of entries.
        # Each entry has a sequence number starting at 1.
        self._trajectory: list[TrajectoryEntry] = []

        # Next sequence number for trajectory entries.
        # Starts at 1 (not 0) for human readability.
        self._next_seq_num: int = 1

        # -----------------------------------------------------------------
        # Artifact Storage
        # -----------------------------------------------------------------
        # Artifacts are stored as binary blobs keyed by name.
        # In-memory storage for now; persisted to SQLite on save().
        self._artifacts: dict[str, bytes] = {}

        # -----------------------------------------------------------------
        # Session Lifecycle
        # -----------------------------------------------------------------
        # Record session creation in trajectory
        self._append_internal(
            agent_id="system",
            entry_type=EntryType.SESSION_CREATED,
            content={
                "session_id": self._session_id,
                "max_artifact_size": self._max_artifact_size,
                "schema_version": SCHEMA_VERSION,
            },
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def session_id(self) -> str:
        """
        Get the unique identifier for this session.

        Returns:
            str: The session ID (UUID format).
        """
        return self._session_id

    @property
    def max_artifact_size(self) -> int:
        """
        Get the maximum allowed artifact size in bytes.

        Returns:
            int: Maximum artifact size.
        """
        return self._max_artifact_size

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the session state.

        This is a read-only operation that does not modify the state
        or increment the version number.

        Args:
            key: The key to look up.
            default: Value to return if key is not found. Defaults to None.

        Returns:
            The value associated with the key, or the default if not found.
            Note: Returns a deep copy to prevent external mutation.
        """
        if key not in self._state:
            return default

        # Return a deep copy to maintain isolation invariant.
        # External code cannot modify internal state by mutating
        # the returned value.
        return copy.deepcopy(self._state[key])

    def set(self, key: str, value: Any) -> int:
        """
        Set a value in the session state.

        This operation:
        1. Stores a deep copy of the value (isolation)
        2. Increments the state version (monotonic versioning)
        3. Records the change in the trajectory (auditability)

        Args:
            key: The key to set. Must be a non-empty string.
            value: The value to store. Must be JSON-serializable.

        Returns:
            int: The new state version after this change.

        Raises:
            ValueError: If the key is empty or value is not JSON-serializable.

        Example:
            version = session.set("count", 42)
            print(f"State is now at version {version}")
        """
        # Validate key
        if not key or not isinstance(key, str):
            raise ValueError("Key must be a non-empty string")

        # Validate that value is JSON-serializable by attempting serialization.
        # This catches non-serializable types early with a clear error.
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Value must be JSON-serializable: {e}") from e

        # Store old value for trajectory (None if key didn't exist)
        old_value = self._state.get(key)

        # Store a deep copy to maintain isolation.
        # External code cannot modify internal state by mutating
        # the original value after set().
        self._state[key] = copy.deepcopy(value)

        # Increment version number (monotonic)
        self._state_version += 1

        # Record the state change in trajectory
        self._append_internal(
            agent_id="system",
            entry_type=EntryType.STATE_SET,
            content={
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "state_version": self._state_version,
            },
        )

        return self._state_version

    def get_state_version(self) -> int:
        """
        Get the current state version number.

        The version starts at 0 for a new session and increments by 1
        on every set() call. This provides:
        - Change detection: Compare versions to detect changes
        - Ordering: Higher version means more recent state
        - Optimistic concurrency: Check version before updates

        Returns:
            int: The current state version (0 if no set() calls yet).
        """
        return self._state_version

    def get_all_state(self) -> dict[str, Any]:
        """
        Get a copy of all state key-value pairs.

        This is useful for debugging and for creating snapshots.
        Returns a deep copy to maintain isolation.

        Returns:
            dict: Deep copy of all state data.
        """
        return copy.deepcopy(self._state)

    # =========================================================================
    # TRAJECTORY MANAGEMENT
    # =========================================================================

    def append(
        self,
        agent_id: str,
        entry_type: EntryType,
        content: dict[str, Any],
    ) -> int:
        """
        Append an entry to the trajectory.

        This is the public interface for agents to record their actions.
        Each entry gets:
        - An auto-incrementing sequence number (1-indexed)
        - A UTC timestamp
        - The provided agent_id, entry_type, and content

        Args:
            agent_id: ID of the agent creating this entry.
            entry_type: Classification of the entry.
            content: Structured content (must be JSON-serializable).

        Returns:
            int: The sequence number assigned to this entry.

        Raises:
            ValueError: If agent_id is empty or content is not serializable.
        """
        return self._append_internal(agent_id, entry_type, content)

    def _append_internal(
        self,
        agent_id: str,
        entry_type: EntryType,
        content: dict[str, Any],
    ) -> int:
        """
        Internal append method (shared by public append and internal operations).

        This is separated to allow internal operations (like state changes)
        to record trajectory entries without exposing internal implementation.

        Args:
            agent_id: ID of the agent creating this entry.
            entry_type: Classification of the entry.
            content: Structured content.

        Returns:
            int: The sequence number assigned to this entry.
        """
        # Validate content is JSON-serializable
        try:
            json.dumps(content)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Content must be JSON-serializable: {e}") from e

        # Create the entry with current timestamp and next sequence number
        entry = TrajectoryEntry(
            seq_num=self._next_seq_num,
            timestamp=datetime.now(timezone.utc),
            agent_id=agent_id,
            entry_type=entry_type,
            content=copy.deepcopy(content),  # Store a copy for isolation
        )

        # Append to trajectory (the only mutation allowed)
        self._trajectory.append(entry)

        # Increment sequence number for next entry
        seq_num = self._next_seq_num
        self._next_seq_num += 1

        return seq_num

    def get_trajectory(self, limit: int | None = None) -> list[TrajectoryEntry]:
        """
        Get trajectory entries.

        Returns entries in chronological order (oldest first).
        If limit is specified, returns only the most recent N entries.

        Args:
            limit: Maximum number of entries to return. If None, returns all.
                   If specified, returns the N most recent entries.

        Returns:
            list[TrajectoryEntry]: List of trajectory entries.
            Note: Returns copies of entries to maintain isolation.
        """
        if limit is None:
            # Return all entries (as copies)
            return list(self._trajectory)

        if limit <= 0:
            return []

        # Return most recent N entries
        # Slice from the end to get most recent, preserving chronological order
        return list(self._trajectory[-limit:])

    def get_trajectory_length(self) -> int:
        """
        Get the number of entries in the trajectory.

        Returns:
            int: Number of trajectory entries.
        """
        return len(self._trajectory)

    # =========================================================================
    # ARTIFACT MANAGEMENT
    # =========================================================================

    def write_artifact(self, name: str, data: bytes) -> None:
        """
        Store a binary artifact in the session.

        Artifacts are binary blobs identified by name. They can be used
        to store files, images, or any other binary data.

        Args:
            name: Name/identifier for the artifact. Must be non-empty.
            data: Binary data to store.

        Raises:
            ValueError: If name is empty or data exceeds max_artifact_size.
            TypeError: If data is not bytes.
        """
        # Validate name
        if not name or not isinstance(name, str):
            raise ValueError("Artifact name must be a non-empty string")

        # Validate data type
        if not isinstance(data, bytes):
            raise TypeError(f"Artifact data must be bytes, got {type(data).__name__}")

        # Validate size
        if len(data) > self._max_artifact_size:
            raise ValueError(
                f"Artifact size ({len(data)} bytes) exceeds maximum "
                f"({self._max_artifact_size} bytes)"
            )

        # Check if this is a new artifact or an update
        is_update = name in self._artifacts
        old_size = len(self._artifacts[name]) if is_update else None

        # Store the artifact
        self._artifacts[name] = data

        # Record in trajectory
        self._append_internal(
            agent_id="system",
            entry_type=EntryType.ARTIFACT_WRITTEN,
            content={
                "name": name,
                "size": len(data),
                "is_update": is_update,
                "old_size": old_size,
            },
        )

    def read_artifact(self, name: str) -> bytes:
        """
        Read a binary artifact from the session.

        Args:
            name: Name of the artifact to read.

        Returns:
            bytes: The artifact data.

        Raises:
            KeyError: If the artifact does not exist.
        """
        if name not in self._artifacts:
            raise KeyError(f"Artifact not found: {name}")

        # Return the actual bytes (no deep copy needed for immutable bytes)
        return self._artifacts[name]

    def list_artifacts(self) -> list[str]:
        """
        List all artifact names in the session.

        Returns:
            list[str]: Names of all stored artifacts, sorted alphabetically.
        """
        return sorted(self._artifacts.keys())

    def get_artifact_size(self, name: str) -> int:
        """
        Get the size of an artifact in bytes.

        Args:
            name: Name of the artifact.

        Returns:
            int: Size in bytes.

        Raises:
            KeyError: If the artifact does not exist.
        """
        if name not in self._artifacts:
            raise KeyError(f"Artifact not found: {name}")

        return len(self._artifacts[name])

    # =========================================================================
    # SNAPSHOTS / VIEWS
    # =========================================================================

    def snapshot_for_agent(
        self,
        agent_id: str,
        depth: int = 10,
    ) -> dict[str, Any]:
        """
        Create a snapshot of session state for an agent.

        This provides agents with a read-only view of the session that
        includes recent trajectory, current state, and artifact list.

        The snapshot is a DEEP COPY - agents cannot modify the session
        by mutating the snapshot. This enforces the isolation invariant.

        Args:
            agent_id: ID of the agent requesting the snapshot.
                      Used for logging/auditing.
            depth: Number of recent trajectory entries to include.
                   Default is 10.

        Returns:
            dict: Snapshot containing:
                - session_id: The session ID
                - state: Deep copy of all state
                - state_version: Current state version
                - trajectory: Recent trajectory entries (as dicts)
                - artifacts: List of artifact names
                - snapshot_time: When the snapshot was created
        """
        # Get recent trajectory entries as dictionaries
        recent_entries = self.get_trajectory(limit=depth)
        trajectory_dicts = [entry.to_dict() for entry in recent_entries]

        # Create the snapshot (all values are deep copies)
        snapshot = {
            "session_id": self._session_id,
            "state": copy.deepcopy(self._state),
            "state_version": self._state_version,
            "trajectory": trajectory_dicts,
            "artifacts": self.list_artifacts(),
            "snapshot_time": datetime.now(timezone.utc).isoformat(),
            "trajectory_total_length": self.get_trajectory_length(),
        }

        return snapshot

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """
        Save the session to a SQLite database file.

        The session is persisted in a single SQLite file containing:
        - Metadata (session_id, version, config)
        - State (all key-value pairs)
        - Trajectory (all entries)
        - Artifacts (all binary blobs)

        The save operation is atomic - either all data is written or none.

        Args:
            path: Path to save the session file. Will be overwritten if exists.

        Raises:
            IOError: If the file cannot be written.
        """
        path = Path(path)

        # Record save in trajectory before saving
        self._append_internal(
            agent_id="system",
            entry_type=EntryType.SESSION_SAVED,
            content={"path": str(path)},
        )

        # Create/overwrite the database file
        # Using WAL mode for better concurrency, even though we're single-threaded
        conn = sqlite3.connect(path)
        try:
            # Enable foreign keys and WAL mode
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")

            # Create schema
            self._create_schema(conn)

            # Save all data within a transaction
            with conn:
                self._save_metadata(conn)
                self._save_state(conn)
                self._save_trajectory(conn)
                self._save_artifacts(conn)

        finally:
            conn.close()

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        """
        Load a session from a SQLite database file.

        This is a class method that returns a new Session instance
        populated with the saved data.

        Args:
            path: Path to the session file.

        Returns:
            Session: A new session instance with restored data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is corrupted or schema mismatch.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        conn = sqlite3.connect(path)
        try:
            # Load metadata first to get config
            session_id, max_artifact_size, state_version = cls._load_metadata(conn)

            # Create session instance (this will add a SESSION_CREATED entry)
            session = cls(
                session_id=session_id,
                max_artifact_size=max_artifact_size,
            )

            # Clear the auto-created trajectory entry
            # We'll restore the original trajectory
            session._trajectory = []
            session._next_seq_num = 1

            # Restore state
            session._state = cls._load_state(conn)
            session._state_version = state_version

            # Restore trajectory
            session._trajectory = cls._load_trajectory(conn)
            if session._trajectory:
                session._next_seq_num = session._trajectory[-1].seq_num + 1

            # Restore artifacts
            session._artifacts = cls._load_artifacts(conn)

            # Record the load in trajectory
            session._append_internal(
                agent_id="system",
                entry_type=EntryType.SESSION_LOADED,
                content={"path": str(path)},
            )

            return session

        finally:
            conn.close()

    # =========================================================================
    # PERSISTENCE HELPERS (PRIVATE)
    # =========================================================================

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create the database schema for session persistence."""
        conn.executescript("""
            -- Metadata table (single row)
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- State table (key-value pairs)
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );

            -- Trajectory table (append-only log)
            CREATE TABLE IF NOT EXISTS trajectory (
                seq_num INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                content_json TEXT NOT NULL
            );

            -- Artifacts table (binary blobs)
            CREATE TABLE IF NOT EXISTS artifacts (
                name TEXT PRIMARY KEY,
                data BLOB NOT NULL
            );
        """)

    def _save_metadata(self, conn: sqlite3.Connection) -> None:
        """Save session metadata."""
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "session_id": self._session_id,
            "max_artifact_size": str(self._max_artifact_size),
            "state_version": str(self._state_version),
        }
        conn.execute("DELETE FROM metadata")
        conn.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            metadata.items(),
        )

    def _save_state(self, conn: sqlite3.Connection) -> None:
        """Save session state."""
        conn.execute("DELETE FROM state")
        conn.executemany(
            "INSERT INTO state (key, value_json) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in self._state.items()],
        )

    def _save_trajectory(self, conn: sqlite3.Connection) -> None:
        """Save trajectory entries."""
        conn.execute("DELETE FROM trajectory")
        conn.executemany(
            """INSERT INTO trajectory
               (seq_num, timestamp, agent_id, entry_type, content_json)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    e.seq_num,
                    e.timestamp.isoformat(),
                    e.agent_id,
                    e.entry_type.value,
                    json.dumps(e.content),
                )
                for e in self._trajectory
            ],
        )

    def _save_artifacts(self, conn: sqlite3.Connection) -> None:
        """Save artifacts."""
        conn.execute("DELETE FROM artifacts")
        conn.executemany(
            "INSERT INTO artifacts (name, data) VALUES (?, ?)",
            self._artifacts.items(),
        )

    @classmethod
    def _load_metadata(
        cls, conn: sqlite3.Connection
    ) -> tuple[str, int, int]:
        """
        Load session metadata.

        Returns:
            Tuple of (session_id, max_artifact_size, state_version)
        """
        cursor = conn.execute("SELECT key, value FROM metadata")
        metadata = dict(cursor.fetchall())

        # Validate schema version
        schema_version = int(metadata.get("schema_version", "0"))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: file has {schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )

        return (
            metadata["session_id"],
            int(metadata["max_artifact_size"]),
            int(metadata["state_version"]),
        )

    @classmethod
    def _load_state(cls, conn: sqlite3.Connection) -> dict[str, Any]:
        """Load session state."""
        cursor = conn.execute("SELECT key, value_json FROM state")
        return {key: json.loads(value_json) for key, value_json in cursor.fetchall()}

    @classmethod
    def _load_trajectory(cls, conn: sqlite3.Connection) -> list[TrajectoryEntry]:
        """Load trajectory entries."""
        cursor = conn.execute(
            """SELECT seq_num, timestamp, agent_id, entry_type, content_json
               FROM trajectory ORDER BY seq_num"""
        )
        entries = []
        for row in cursor.fetchall():
            seq_num, timestamp, agent_id, entry_type, content_json = row
            entries.append(
                TrajectoryEntry(
                    seq_num=seq_num,
                    timestamp=datetime.fromisoformat(timestamp),
                    agent_id=agent_id,
                    entry_type=EntryType(entry_type),
                    content=json.loads(content_json),
                )
            )
        return entries

    @classmethod
    def _load_artifacts(cls, conn: sqlite3.Connection) -> dict[str, bytes]:
        """Load artifacts."""
        cursor = conn.execute("SELECT name, data FROM artifacts")
        return {name: data for name, data in cursor.fetchall()}

    # =========================================================================
    # DEBUGGING / INTROSPECTION
    # =========================================================================

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return (
            f"Session(id={self._session_id[:8]}..., "
            f"state_keys={len(self._state)}, "
            f"trajectory_len={len(self._trajectory)}, "
            f"artifacts={len(self._artifacts)})"
        )
