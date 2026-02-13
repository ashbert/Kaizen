from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kaizen.session import Session
from demo.py_to_go import run_demo


class _DummyResult:
    def __init__(self, success: bool, result: dict | None = None, error: dict | None = None):
        self.success = success
        self.result = result or {}
        self.error = error


class _DummyDispatcher:
    def __init__(self):
        self._tests = 0
        self._fixes = 0

    def dispatch_single(self, capability: str, session: Session, params: dict) -> _DummyResult:
        _ = params
        if capability == "run_tests":
            self._tests += 1
            return _DummyResult(success=False, error={"message": "failed"})
        if capability == "fix":
            self._fixes += 1
            return _DummyResult(success=True, result={"files_fixed": 0})
        raise ValueError(f"unexpected capability {capability}")


def test_build_failure_signature_normalizes_line_numbers() -> None:
    a = "pkg/file.go:10:2: undefined: Foo"
    b = "pkg/file.go:99:7: undefined: Foo"
    assert run_demo._build_failure_signature(a) == run_demo._build_failure_signature(b)


def test_trailing_repeat_count() -> None:
    assert run_demo._trailing_repeat_count([]) == 0
    assert run_demo._trailing_repeat_count(["a", "a", "a"]) == 3
    assert run_demo._trailing_repeat_count(["a", "b", "b"]) == 2


def test_toolsmith_trigger_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIZEN_TOOLSMITH_ENABLED", "1")
    assert run_demo._toolsmith_trigger_reason(1, 8, 0, ["sig-1"]) == "fixer_no_changes"
    assert run_demo._toolsmith_trigger_reason(3, 8, 2, ["x", "x"]) == "repeat_signature"
    assert run_demo._toolsmith_trigger_reason(6, 8, 2, ["x", "y"]) == "late_budget"


def test_toolsmith_trigger_reason_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAIZEN_TOOLSMITH_ENABLED", raising=False)
    assert run_demo._toolsmith_trigger_reason(6, 8, 0, ["x"]) is None


def test_invoke_toolsmith_ingests_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    artifact_dir = workspace / ".toolsmith" / "runs" / "run-123"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "final.json").write_text('{"result":{"status":"improved"}}')
    (artifact_dir / "summary.txt").write_text("status=improved")

    payload = {
        "status": "improved",
        "run_id": "run-123",
        "result": {"artifact_dir": str(artifact_dir)},
    }

    class FakeProc:
        returncode = 10
        stdout = json.dumps(payload)
        stderr = ""

    session = Session()
    session.set("go_output_path", str(workspace))

    monkeypatch.setenv("KAIZEN_TOOLSMITH_MAX_INVOCATIONS", "2")
    monkeypatch.setenv("KAIZEN_TOOLSMITH_TIMEOUT_SECONDS", "120")

    with patch("demo.py_to_go.run_demo.subprocess.run", return_value=FakeProc()):
        result = run_demo._invoke_toolsmith(
            session=session,
            iteration=3,
            max_iter=8,
            label="Source compilation fix",
            trigger_reason="repeat_signature",
        )

    assert result["invoked"] is True
    assert result["status"] == "improved"
    assert session.get("toolsmith_runs")[0]["toolsmith_run_id"] == "run-123"
    artifacts = set(session.list_artifacts())
    assert "toolsmith_run_01.json" in artifacts
    assert "toolsmith_run_01_final.json" in artifacts
    assert "toolsmith_run_01_summary.txt" in artifacts


def test_run_fix_loop_invokes_toolsmith_on_no_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session()
    session.set("go_output_path", "/tmp")
    session.set("last_test_output", "pkg/file.go:10:2: undefined: Foo")
    session.write_artifact("go_test_output_01.log", b"pkg/file.go:10:2: undefined: Foo")
    session.set("last_test_artifact", "go_test_output_01.log")

    calls: list[str] = []

    def _fake_invoke(**kwargs):
        calls.append(str(kwargs["trigger_reason"]))
        return {"invoked": False}

    monkeypatch.setenv("KAIZEN_TOOLSMITH_ENABLED", "1")
    monkeypatch.setattr(run_demo, "_invoke_toolsmith", _fake_invoke)

    dispatcher = _DummyDispatcher()
    ok = run_demo.run_fix_loop(
        dispatcher=dispatcher,  # type: ignore[arg-type]
        session=session,
        step_num=7,
        label="Source compilation fix",
        max_iter=2,
    )

    assert ok is False
    assert calls
    assert calls[0] == "fixer_no_changes"
