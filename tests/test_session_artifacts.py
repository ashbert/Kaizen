"""
Tests for Session artifact management.

This module tests the artifact subsystem of the Session class:
- write_artifact() / read_artifact() operations
- list_artifacts() enumeration
- Size limits and validation
- Trajectory recording of artifact operations

Artifacts are binary blobs stored in the session, useful for files,
images, or any other binary data that agents need to work with.
"""

import pytest

from trace.session import Session, DEFAULT_MAX_ARTIFACT_SIZE


class TestArtifactWriteRead:
    """Tests for artifact write and read operations."""

    def test_write_and_read_artifact(self) -> None:
        """Verify basic write/read roundtrip."""
        session = Session()
        data = b"Hello, artifact world!"

        session.write_artifact("test.txt", data)
        retrieved = session.read_artifact("test.txt")

        assert retrieved == data

    def test_write_artifact_binary_data(self) -> None:
        """Verify binary data (non-text) can be stored."""
        session = Session()
        # Binary data with null bytes and non-UTF8 sequences
        data = bytes(range(256)) * 100

        session.write_artifact("binary.bin", data)
        retrieved = session.read_artifact("binary.bin")

        assert retrieved == data

    def test_write_artifact_empty_data(self) -> None:
        """Verify empty bytes can be stored."""
        session = Session()

        session.write_artifact("empty.txt", b"")
        retrieved = session.read_artifact("empty.txt")

        assert retrieved == b""

    def test_write_artifact_overwrites_existing(self) -> None:
        """Verify writing to same name overwrites."""
        session = Session()

        session.write_artifact("file.txt", b"first")
        session.write_artifact("file.txt", b"second")

        assert session.read_artifact("file.txt") == b"second"

    def test_read_nonexistent_artifact_raises(self) -> None:
        """Verify reading missing artifact raises KeyError."""
        session = Session()

        with pytest.raises(KeyError, match="not found"):
            session.read_artifact("nonexistent.txt")

    def test_write_artifact_empty_name_raises(self) -> None:
        """Verify empty artifact name is rejected."""
        session = Session()

        with pytest.raises(ValueError, match="non-empty string"):
            session.write_artifact("", b"data")

    def test_write_artifact_non_string_name_raises(self) -> None:
        """Verify non-string artifact name is rejected."""
        session = Session()

        with pytest.raises(ValueError):
            session.write_artifact(123, b"data")  # type: ignore

    def test_write_artifact_non_bytes_data_raises(self) -> None:
        """Verify non-bytes data is rejected."""
        session = Session()

        with pytest.raises(TypeError, match="must be bytes"):
            session.write_artifact("test.txt", "string data")  # type: ignore

        with pytest.raises(TypeError, match="must be bytes"):
            session.write_artifact("test.txt", [1, 2, 3])  # type: ignore


class TestArtifactSizeLimits:
    """Tests for artifact size limit enforcement."""

    def test_default_max_size_is_100mb(self) -> None:
        """Verify default max artifact size is 100MB."""
        assert DEFAULT_MAX_ARTIFACT_SIZE == 100 * 1024 * 1024

    def test_artifact_within_limit_accepted(self) -> None:
        """Verify artifacts within limit are accepted."""
        session = Session(max_artifact_size=1000)
        data = b"X" * 1000

        # Should not raise
        session.write_artifact("test.bin", data)
        assert session.read_artifact("test.bin") == data

    def test_artifact_exceeding_limit_rejected(self) -> None:
        """Verify artifacts exceeding limit are rejected."""
        session = Session(max_artifact_size=1000)
        data = b"X" * 1001

        with pytest.raises(ValueError, match="exceeds maximum"):
            session.write_artifact("test.bin", data)

    def test_artifact_at_exact_limit_accepted(self) -> None:
        """Verify artifacts at exactly the limit are accepted."""
        session = Session(max_artifact_size=1000)
        data = b"X" * 1000

        session.write_artifact("test.bin", data)
        assert len(session.read_artifact("test.bin")) == 1000

    def test_custom_artifact_size_limit(self) -> None:
        """Verify custom size limit is respected."""
        session = Session(max_artifact_size=500)

        # 500 bytes should work
        session.write_artifact("ok.bin", b"X" * 500)

        # 501 bytes should fail
        with pytest.raises(ValueError, match="exceeds maximum"):
            session.write_artifact("fail.bin", b"X" * 501)

    def test_large_artifact_near_default_limit(self, large_artifact_data: bytes) -> None:
        """Verify large artifacts work within limit."""
        session = Session()

        # 1MB should be fine with 100MB limit
        session.write_artifact("large.bin", large_artifact_data)
        retrieved = session.read_artifact("large.bin")

        assert len(retrieved) == 1024 * 1024


class TestArtifactListing:
    """Tests for artifact listing."""

    def test_list_artifacts_empty(self) -> None:
        """Verify empty session has no artifacts."""
        session = Session()
        assert session.list_artifacts() == []

    def test_list_artifacts_single(self) -> None:
        """Verify listing with single artifact."""
        session = Session()
        session.write_artifact("test.txt", b"data")

        assert session.list_artifacts() == ["test.txt"]

    def test_list_artifacts_multiple(self) -> None:
        """Verify listing with multiple artifacts."""
        session = Session()
        session.write_artifact("c.txt", b"c")
        session.write_artifact("a.txt", b"a")
        session.write_artifact("b.txt", b"b")

        # Should be sorted alphabetically
        assert session.list_artifacts() == ["a.txt", "b.txt", "c.txt"]

    def test_list_artifacts_after_overwrite(self) -> None:
        """Verify listing after overwriting an artifact."""
        session = Session()
        session.write_artifact("test.txt", b"first")
        session.write_artifact("test.txt", b"second")

        # Should still have just one artifact
        assert session.list_artifacts() == ["test.txt"]


class TestArtifactSize:
    """Tests for artifact size retrieval."""

    def test_get_artifact_size(self) -> None:
        """Verify artifact size is correctly reported."""
        session = Session()
        data = b"Hello, world!"
        session.write_artifact("test.txt", data)

        assert session.get_artifact_size("test.txt") == len(data)

    def test_get_artifact_size_empty(self) -> None:
        """Verify size of empty artifact is 0."""
        session = Session()
        session.write_artifact("empty.txt", b"")

        assert session.get_artifact_size("empty.txt") == 0

    def test_get_artifact_size_nonexistent_raises(self) -> None:
        """Verify getting size of missing artifact raises KeyError."""
        session = Session()

        with pytest.raises(KeyError, match="not found"):
            session.get_artifact_size("nonexistent.txt")


class TestArtifactTrajectoryRecording:
    """Tests for trajectory recording of artifact operations."""

    def test_write_artifact_creates_trajectory_entry(self) -> None:
        """Verify write_artifact creates a trajectory entry."""
        session = Session()
        initial_len = session.get_trajectory_length()

        session.write_artifact("test.txt", b"data")

        assert session.get_trajectory_length() == initial_len + 1

    def test_write_artifact_trajectory_content(self) -> None:
        """Verify trajectory entry content for artifact write."""
        session = Session()
        session.write_artifact("test.txt", b"hello")

        trajectory = session.get_trajectory()
        artifact_entries = [
            e for e in trajectory if e.entry_type.value == "artifact_written"
        ]

        assert len(artifact_entries) == 1
        entry = artifact_entries[0]

        assert entry.agent_id == "system"
        assert entry.content["name"] == "test.txt"
        assert entry.content["size"] == 5
        assert entry.content["is_update"] is False
        assert entry.content["old_size"] is None

    def test_overwrite_artifact_records_update(self) -> None:
        """Verify overwriting artifact records as update."""
        session = Session()
        session.write_artifact("test.txt", b"first")
        session.write_artifact("test.txt", b"second value")

        trajectory = session.get_trajectory()
        artifact_entries = [
            e for e in trajectory if e.entry_type.value == "artifact_written"
        ]

        assert len(artifact_entries) == 2

        # Second entry should indicate update
        entry = artifact_entries[1]
        assert entry.content["is_update"] is True
        assert entry.content["old_size"] == 5  # len("first")
        assert entry.content["size"] == 12  # len("second value")


class TestArtifactNaming:
    """Tests for artifact naming."""

    def test_artifact_with_path_like_name(self) -> None:
        """Verify path-like artifact names work."""
        session = Session()

        session.write_artifact("data/files/test.txt", b"content")
        assert session.read_artifact("data/files/test.txt") == b"content"

    def test_artifact_with_special_characters(self) -> None:
        """Verify artifact names with special characters work."""
        session = Session()

        names = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.multiple.dots.txt",
            "UPPERCASE.TXT",
            "MixedCase.Txt",
        ]

        for name in names:
            session.write_artifact(name, name.encode())

        for name in names:
            assert session.read_artifact(name) == name.encode()

        # All should be listed
        listed = session.list_artifacts()
        assert len(listed) == len(names)

    def test_artifact_names_are_case_sensitive(self) -> None:
        """Verify artifact names are case-sensitive."""
        session = Session()

        session.write_artifact("Test.txt", b"uppercase T")
        session.write_artifact("test.txt", b"lowercase t")

        assert session.read_artifact("Test.txt") == b"uppercase T"
        assert session.read_artifact("test.txt") == b"lowercase t"
        assert len(session.list_artifacts()) == 2
