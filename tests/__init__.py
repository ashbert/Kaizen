"""
Test suite for Trace.

This package contains all tests for the Trace session substrate.
Tests are organized by module:

- test_session_state.py: State management tests (get/set, versioning)
- test_session_trajectory.py: Trajectory tests (append-only, ordering)
- test_session_artifacts.py: Artifact storage tests
- test_session_snapshots.py: Snapshot isolation tests
- test_persistence.py: SQLite save/load roundtrip tests
- test_agents.py: Agent protocol and toy agent tests
- test_dispatcher.py: Dispatcher sequential execution tests
- test_planner.py: Planner LLM integration tests
- test_e2e.py: End-to-end workflow tests

Run tests with: pytest
Run with coverage: pytest --cov=trace
"""
