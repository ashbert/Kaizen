"""
Test Runner Agent for Python to Go Conversion.

This agent runs Go tests in the output directory and captures results.
It updates session state with pass/fail status and stores test output as artifacts.

Capability: "run_tests"
    Runs `go test ./...` in the Go output directory.

    Parameters: None

    Session State Updates:
        - tests_passed: Boolean indicating if all tests passed
        - last_test_run: Timestamp of last test run
        - test_run_count: Number of test runs performed
        - last_test_output: Summary of test output

    Artifacts Created:
        - go_test_output_{run_count}.log: Full test output

    Trajectory Entries:
        - AGENT_INVOKED: When tests start
        - AGENT_COMPLETED: With test results summary
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaizen import Agent, AgentInfo, InvokeResult, Session
from kaizen.types import EntryType


class TestRunnerAgent(Agent):
    """
    Agent that runs Go tests and captures results.

    The test runner executes `go test ./...` in the Go output directory,
    captures stdout/stderr, and updates session state with the results.
    """

    def info(self) -> AgentInfo:
        """Return agent metadata."""
        return AgentInfo(
            agent_id="test_runner_agent_v1",
            name="Test Runner Agent",
            version="1.0.0",
            capabilities=["run_tests"],
            description="Runs Go tests and captures results",
        )

    def invoke(
        self,
        capability: str,
        session: Session,
        params: dict[str, Any],
    ) -> InvokeResult:
        """Execute the run_tests capability."""
        if capability != "run_tests":
            return self._unknown_capability(capability)

        return self._run_tests(session)

    def _run_tests(self, session: Session) -> InvokeResult:
        """
        Run Go tests in the output directory.

        Executes `go test ./...` and captures all output.
        """
        agent_info = self.info()

        # Get output path from session
        go_output = session.get("go_output_path")
        if not go_output:
            return self._invalid_params(
                "run_tests",
                "Missing go_output_path in session state",
            )

        go_output_path = Path(go_output)
        if not go_output_path.exists():
            return self._invalid_params(
                "run_tests",
                f"Go output directory does not exist: {go_output}",
            )

        # Increment test run count
        run_count = session.get("test_run_count", 0) + 1
        session.set("test_run_count", run_count)

        # Record invocation
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_INVOKED,
            content={
                "capability": "run_tests",
                "run_count": run_count,
                "go_output_path": go_output,
            },
        )

        # First, resolve dependencies
        tidy_result = self._run_go_command(
            ["go", "mod", "tidy"],
            go_output_path,
        )

        # Then build to catch compile errors
        build_result = self._run_go_command(
            ["go", "build", "./..."],
            go_output_path,
        )

        # Then run tests
        test_result = self._run_go_command(
            ["go", "test", "./...", "-v"],
            go_output_path,
        )

        # Combine outputs
        full_output = f"=== TIDY OUTPUT ===\n{tidy_result['output']}\n\n"
        full_output += f"=== BUILD OUTPUT ===\n{build_result['output']}\n\n"
        full_output += f"=== TEST OUTPUT ===\n{test_result['output']}"

        # Determine overall success
        # Tests pass if both build and test succeed (tidy warnings are ok)
        tests_passed = build_result["success"] and test_result["success"]

        # Save output as artifact
        artifact_name = f"go_test_output_{run_count:02d}.log"
        session.write_artifact(artifact_name, full_output.encode("utf-8"))

        # Update session state
        timestamp = datetime.now(timezone.utc).isoformat()
        session.set("tests_passed", tests_passed)
        session.set("last_test_run", timestamp)
        session.set("last_test_output", self._summarize_output(full_output, tests_passed))
        session.set("last_test_artifact", artifact_name)

        # Update status if tests pass
        if tests_passed:
            session.set("status", "completed")

        # Record completion
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": "run_tests",
                "run_count": run_count,
                "tests_passed": tests_passed,
                "build_success": build_result["success"],
                "test_success": test_result["success"],
                "artifact": artifact_name,
                "summary": self._summarize_output(full_output, tests_passed),
            },
        )

        return InvokeResult.ok(
            result={
                "tests_passed": tests_passed,
                "run_count": run_count,
                "artifact": artifact_name,
            },
            agent_id=agent_info.agent_id,
            capability="run_tests",
        )

    def _run_go_command(
        self,
        cmd: list[str],
        cwd: Path,
    ) -> dict[str, Any]:
        """
        Run a Go command and capture output.

        Args:
            cmd: Command and arguments
            cwd: Working directory

        Returns:
            Dict with 'success', 'output', 'return_code'
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            output = result.stdout
            if result.stderr:
                output += f"\n\nSTDERR:\n{result.stderr}"

            return {
                "success": result.returncode == 0,
                "output": output,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "Command timed out after 120 seconds",
                "return_code": -1,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": f"Command not found: {cmd[0]}. Is Go installed?",
                "return_code": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Error running command: {e}",
                "return_code": -1,
            }

    def _summarize_output(self, output: str, passed: bool) -> str:
        """
        Create a brief summary of test output.

        Args:
            output: Full test output
            passed: Whether tests passed

        Returns:
            Brief summary string
        """
        if passed:
            return "All tests passed"

        # Extract error lines for summary
        lines = output.split("\n")
        error_lines = []

        for line in lines:
            # Look for compile errors
            if ": " in line and (".go:" in line or "undefined" in line.lower()):
                error_lines.append(line.strip())
            # Look for test failures
            elif line.strip().startswith("---") and "FAIL" in line:
                error_lines.append(line.strip())
            elif "cannot find package" in line.lower():
                error_lines.append(line.strip())

        if error_lines:
            # Return first few error lines
            summary = "; ".join(error_lines[:5])
            if len(error_lines) > 5:
                summary += f" ... and {len(error_lines) - 5} more errors"
            return summary

        return "Tests failed - see artifact for details"
