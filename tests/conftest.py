"""
Pytest configuration and shared fixtures for Trace tests.

This module provides common test fixtures and configuration that can be
used across all test modules. Fixtures defined here are automatically
available to all tests without explicit imports.

Fixtures:
- temp_session_path: Provides a temporary file path for session persistence tests
- sample_trajectory_entries: Sample trajectory data for testing
- sample_state_data: Sample state data for testing
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def temp_session_path() -> Generator[Path, None, None]:
    """
    Provides a temporary file path for session persistence tests.

    The file is automatically cleaned up after the test completes,
    whether it passes or fails.

    Yields:
        Path: A path to a temporary .trace file that can be used for
              save/load operations.

    Example:
        def test_save_load(temp_session_path):
            session = Session()
            session.save(temp_session_path)
            loaded = Session.load(temp_session_path)
    """
    # Create a temporary directory that will be cleaned up automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        # Yield a path within the temp directory
        # The .trace extension is conventional for trace session files
        yield Path(tmpdir) / "test_session.trace"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Provides a temporary directory for tests that need multiple files.

    Yields:
        Path: Path to a temporary directory that will be cleaned up
              after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_state_data() -> dict:
    """
    Provides sample state data for testing.

    Returns a dictionary with various JSON-compatible value types
    to test serialization and state management.

    Returns:
        dict: Sample state data including strings, numbers, lists,
              nested objects, booleans, and null values.
    """
    return {
        "text": "hello world",
        "count": 42,
        "ratio": 3.14159,
        "tags": ["foo", "bar", "baz"],
        "metadata": {
            "author": "test",
            "version": 1,
            "nested": {"deep": "value"},
        },
        "enabled": True,
        "disabled": False,
        "empty": None,
    }


@pytest.fixture
def sample_trajectory_content() -> list[dict]:
    """
    Provides sample trajectory entry content for testing.

    Returns a list of dictionaries representing typical trajectory
    entry content from different agent operations.

    Returns:
        list[dict]: Sample trajectory entry content.
    """
    return [
        {
            "action": "reverse",
            "input": "hello",
            "output": "olleh",
        },
        {
            "action": "uppercase",
            "input": "olleh",
            "output": "OLLEH",
        },
        {
            "action": "state_change",
            "key": "result",
            "old_value": None,
            "new_value": "OLLEH",
        },
    ]


@pytest.fixture
def sample_artifact_data() -> bytes:
    """
    Provides sample binary artifact data for testing.

    Returns:
        bytes: Sample binary data representing an artifact.
    """
    return b"This is sample artifact content for testing.\n" * 100


@pytest.fixture
def large_artifact_data() -> bytes:
    """
    Provides large artifact data for testing size limits.

    This generates data close to but under the default 100MB limit.
    Note: This fixture creates 1MB of data for practical test speed.

    Returns:
        bytes: 1MB of sample data.
    """
    # 1MB of data - enough to test handling of larger artifacts
    # without making tests too slow
    return b"X" * (1024 * 1024)


# =============================================================================
# MARKER CONFIGURATION
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """
    Register custom markers for the test suite.

    This ensures that custom markers are properly documented and don't
    trigger warnings when used.
    """
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may require external services)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running",
    )
