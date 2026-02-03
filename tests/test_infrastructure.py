"""
Infrastructure tests for Trace.

These tests verify that the test infrastructure itself is working correctly.
They serve as a sanity check that pytest is configured properly and that
the project structure is valid.

These tests should always pass and provide confidence that the testing
framework is operational before writing substantive tests.
"""

import sys
from pathlib import Path

import pytest


class TestProjectStructure:
    """Tests that verify the project structure is correct."""

    def test_trace_package_importable(self) -> None:
        """
        Verify that the trace package can be imported.

        This confirms that:
        1. The src directory is in the Python path (via pyproject.toml)
        2. The trace package has a valid __init__.py
        3. Basic package metadata is accessible
        """
        import trace

        # Verify version is defined
        assert hasattr(trace, "__version__")
        assert trace.__version__ == "0.1.0"

    def test_trace_subpackages_importable(self) -> None:
        """
        Verify that trace subpackages can be imported.

        This ensures the package structure is valid and all
        __init__.py files are in place.
        """
        # These imports should not raise ImportError
        import trace.llm
        import trace.agents

        # Verify the modules are actually loaded
        assert trace.llm is not None
        assert trace.agents is not None

    def test_src_in_python_path(self) -> None:
        """
        Verify that the src directory is properly configured in sys.path.

        This is essential for the package to be importable during tests
        without installation.
        """
        # Find paths that contain 'trace' as a package
        trace_paths = [p for p in sys.path if "trace" in str(p).lower()]

        # At minimum, the test should be able to import trace
        # (which we verified above), but let's also check the path setup
        import trace
        trace_file = trace.__file__

        assert trace_file is not None
        assert "src/trace" in trace_file or "src\\trace" in trace_file


class TestFixtures:
    """Tests that verify pytest fixtures are working correctly."""

    def test_temp_session_path_fixture(self, temp_session_path: Path) -> None:
        """
        Verify the temp_session_path fixture provides a valid path.

        The fixture should:
        1. Provide a Path object
        2. Have the .trace extension
        3. Be in a directory that exists
        4. Not exist yet (it's a path for the test to create)
        """
        assert isinstance(temp_session_path, Path)
        assert temp_session_path.suffix == ".trace"
        assert temp_session_path.parent.exists()
        assert not temp_session_path.exists()

    def test_temp_dir_fixture(self, temp_dir: Path) -> None:
        """
        Verify the temp_dir fixture provides a valid directory.

        The fixture should provide an existing, empty directory.
        """
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_sample_state_data_fixture(self, sample_state_data: dict) -> None:
        """
        Verify the sample_state_data fixture provides expected data.

        The fixture should include various JSON-compatible types
        for thorough testing of state serialization.
        """
        assert isinstance(sample_state_data, dict)

        # Check for various types
        assert isinstance(sample_state_data["text"], str)
        assert isinstance(sample_state_data["count"], int)
        assert isinstance(sample_state_data["ratio"], float)
        assert isinstance(sample_state_data["tags"], list)
        assert isinstance(sample_state_data["metadata"], dict)
        assert isinstance(sample_state_data["enabled"], bool)
        assert sample_state_data["empty"] is None

    def test_sample_trajectory_content_fixture(
        self, sample_trajectory_content: list[dict]
    ) -> None:
        """
        Verify the sample_trajectory_content fixture provides expected data.
        """
        assert isinstance(sample_trajectory_content, list)
        assert len(sample_trajectory_content) > 0
        assert all(isinstance(entry, dict) for entry in sample_trajectory_content)

    def test_sample_artifact_data_fixture(self, sample_artifact_data: bytes) -> None:
        """
        Verify the sample_artifact_data fixture provides bytes.
        """
        assert isinstance(sample_artifact_data, bytes)
        assert len(sample_artifact_data) > 0

    def test_large_artifact_data_fixture(self, large_artifact_data: bytes) -> None:
        """
        Verify the large_artifact_data fixture provides appropriately sized data.
        """
        assert isinstance(large_artifact_data, bytes)
        # Should be 1MB
        assert len(large_artifact_data) == 1024 * 1024


class TestMarkers:
    """Tests that verify custom pytest markers are registered."""

    @pytest.mark.integration
    def test_integration_marker_registered(self) -> None:
        """
        Verify the integration marker is registered and usable.

        This test is marked with @pytest.mark.integration to verify
        the marker doesn't trigger warnings.
        """
        # This test just needs to run without marker warnings
        assert True

    @pytest.mark.slow
    def test_slow_marker_registered(self) -> None:
        """
        Verify the slow marker is registered and usable.

        This test is marked with @pytest.mark.slow to verify
        the marker doesn't trigger warnings.
        """
        # This test just needs to run without marker warnings
        assert True
