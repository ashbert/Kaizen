"""
Fixer Agent for Python to Go Conversion.

This agent analyzes test failures and applies minimal fixes to the Go code.
It uses LLM to understand errors and generate appropriate fixes.

Capability: "fix"
    Analyzes test failures and applies fixes to Go files.

    Parameters: None (reads from session state)

    Session State Updates:
        - fix_count: Incremented after each fix attempt
        - fixes_applied: List of applied fixes

    Artifacts Created:
        - fix_{count}.patch: Description of fixes applied

    Trajectory Entries:
        - AGENT_INVOKED: When fix attempt starts
        - AGENT_COMPLETED: With details of fixes applied
"""

import re
from pathlib import Path
from typing import Any

from kaizen import Agent, AgentInfo, InvokeResult, Session
from kaizen.types import EntryType
from kaizen.llm import LLMProvider, OllamaProvider


# System prompt for fixing Go code
FIX_SYSTEM_PROMPT = """You are an expert Go programmer fixing compilation and test errors.

Your task is to fix the Go code based on the error messages provided.

Rules:
1. Make minimal, targeted fixes - don't rewrite large sections
2. Fix one error at a time if there are multiple
3. Preserve the overall structure and logic
4. Add missing imports if needed
5. Fix type mismatches with proper conversions
6. Fix undefined references by adding declarations or imports
7. Output ONLY the complete fixed Go file, no explanations
8. Start directly with the package declaration

Common fixes:
- Missing imports: add to import block
- Undefined types: check if type should be from another package
- Method signature mismatches: adjust parameters/returns
- Missing interface implementations: add required methods
- Syntax errors: fix brackets, semicolons, etc."""

FIX_USER_PROMPT = """Fix the following Go file based on the error messages.

File path: {file_path}

Current Go code:
```go
{go_code}
```

Error messages:
```
{errors}
```

Generate the complete fixed Go file."""


class FixerAgent(Agent):
    """
    Agent that fixes Go code based on test/build failures.

    The fixer reads test output, identifies errors, and uses LLM
    to generate and apply fixes to the affected files.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        """
        Initialize the fixer agent.

        Args:
            llm_provider: Optional LLM provider. If not provided, creates
                         a default OllamaProvider.
        """
        self._llm = llm_provider or OllamaProvider(
            model="llama3.1:8b",
            timeout=300.0,
        )

    def info(self) -> AgentInfo:
        """Return agent metadata."""
        return AgentInfo(
            agent_id="fixer_agent_v1",
            name="Fixer Agent",
            version="1.0.0",
            capabilities=["fix"],
            description="LLM-powered Go code fixer for test failures",
        )

    def invoke(
        self,
        capability: str,
        session: Session,
        params: dict[str, Any],
    ) -> InvokeResult:
        """Execute the fix capability."""
        if capability != "fix":
            return self._unknown_capability(capability)

        return self._apply_fixes(session)

    def _apply_fixes(self, session: Session) -> InvokeResult:
        """
        Analyze errors and apply fixes.

        Reads the last test output and attempts to fix identified errors.
        """
        agent_info = self.info()

        # Get test output
        test_artifact = session.get("last_test_artifact")
        if not test_artifact:
            return self._invalid_params(
                "fix",
                "No test output available. Run tests first.",
            )

        try:
            test_output = session.read_artifact(test_artifact).decode("utf-8")
        except KeyError:
            return self._invalid_params(
                "fix",
                f"Test artifact not found: {test_artifact}",
            )

        # Get output path
        go_output = session.get("go_output_path")
        if not go_output:
            return self._invalid_params(
                "fix",
                "Missing go_output_path in session state",
            )

        # Increment fix count
        fix_count = session.get("fix_count", 0) + 1
        session.set("fix_count", fix_count)

        # Record invocation
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_INVOKED,
            content={
                "capability": "fix",
                "fix_count": fix_count,
                "test_artifact": test_artifact,
            },
        )

        # Parse errors to identify files needing fixes
        errors_by_file = self._parse_errors(test_output, go_output)

        if not errors_by_file:
            # No specific file errors found - might be missing packages
            session.append(
                agent_id=agent_info.agent_id,
                entry_type=EntryType.AGENT_COMPLETED,
                content={
                    "capability": "fix",
                    "fix_count": fix_count,
                    "files_fixed": 0,
                    "note": "No specific file errors identified",
                },
            )
            return InvokeResult.ok(
                result={
                    "files_fixed": 0,
                    "note": "Could not identify specific files to fix",
                },
                agent_id=agent_info.agent_id,
                capability="fix",
            )

        # Apply fixes to each file
        fixes_applied = []
        for file_path, errors in errors_by_file.items():
            fix_result = self._fix_file(file_path, errors, session)
            if fix_result:
                fixes_applied.append(fix_result)

        # Update session state
        all_fixes = session.get("fixes_applied", [])
        all_fixes.extend(fixes_applied)
        session.set("fixes_applied", all_fixes)

        # Save fix summary as artifact
        fix_summary = self._create_fix_summary(fixes_applied)
        artifact_name = f"fix_{fix_count:02d}.txt"
        session.write_artifact(artifact_name, fix_summary.encode("utf-8"))

        # Record completion
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": "fix",
                "fix_count": fix_count,
                "files_fixed": len(fixes_applied),
                "files": [f["file"] for f in fixes_applied],
                "artifact": artifact_name,
            },
        )

        return InvokeResult.ok(
            result={
                "files_fixed": len(fixes_applied),
                "files": [f["file"] for f in fixes_applied],
            },
            agent_id=agent_info.agent_id,
            capability="fix",
        )

    def _parse_errors(
        self,
        output: str,
        go_output: str,
    ) -> dict[str, list[str]]:
        """
        Parse test/build output to identify errors by file.

        Args:
            output: Test/build output text
            go_output: Base path of Go output directory

        Returns:
            Dict mapping file paths to list of error messages
        """
        errors_by_file: dict[str, list[str]] = {}
        go_output_path = Path(go_output)

        # Pattern for Go compiler errors: file.go:line:col: message
        error_pattern = re.compile(r'([^\s:]+\.go):(\d+):(\d+):\s*(.+)')

        # Pattern for package errors
        package_pattern = re.compile(r'package\s+(\S+):\s*(.+)')

        for line in output.split("\n"):
            line = line.strip()

            # Match file-specific errors
            match = error_pattern.search(line)
            if match:
                file_name = match.group(1)
                error_msg = line

                # Find the full path
                full_path = self._find_file(file_name, go_output_path)
                if full_path:
                    if full_path not in errors_by_file:
                        errors_by_file[full_path] = []
                    errors_by_file[full_path].append(error_msg)
                continue

            # Match undefined/undeclared errors that mention a file
            if "undefined:" in line.lower() or "undeclared" in line.lower():
                # Try to extract file from context
                for word in line.split():
                    if word.endswith(".go") or ".go:" in word:
                        file_name = word.split(":")[0]
                        full_path = self._find_file(file_name, go_output_path)
                        if full_path:
                            if full_path not in errors_by_file:
                                errors_by_file[full_path] = []
                            errors_by_file[full_path].append(line)
                        break

        return errors_by_file

    def _find_file(self, file_name: str, base_path: Path) -> str | None:
        """
        Find full path for a file name within the output directory.

        Args:
            file_name: File name (may include partial path)
            base_path: Base directory to search

        Returns:
            Full path if found, None otherwise
        """
        # Clean up file name
        file_name = file_name.strip().rstrip(":")

        # Try direct match first
        direct = base_path / file_name
        if direct.exists():
            return str(direct)

        # Search recursively
        for go_file in base_path.rglob("*.go"):
            if go_file.name == Path(file_name).name:
                return str(go_file)
            if str(go_file).endswith(file_name):
                return str(go_file)

        return None

    def _fix_file(
        self,
        file_path: str,
        errors: list[str],
        session: Session,
    ) -> dict[str, Any] | None:
        """
        Apply LLM-generated fix to a single file.

        Args:
            file_path: Path to the Go file
            errors: List of error messages for this file
            session: Kaizen session

        Returns:
            Dict with fix details, or None if fix failed
        """
        path = Path(file_path)
        if not path.exists():
            return None

        # Read current content
        original_code = path.read_text()

        # Build fix prompt
        errors_text = "\n".join(errors[:10])  # Limit errors to avoid huge prompts
        user_prompt = FIX_USER_PROMPT.format(
            file_path=file_path,
            go_code=original_code,
            errors=errors_text,
        )

        # Call LLM for fix
        try:
            response = self._llm.complete(
                prompt=user_prompt,
                system=FIX_SYSTEM_PROMPT,
            )
            fixed_code = response.text.strip()

            # Clean up response
            fixed_code = self._clean_go_code(fixed_code)

        except Exception as e:
            return {
                "file": file_path,
                "status": "failed",
                "error": str(e),
            }

        # Verify the fix is different
        if fixed_code == original_code:
            return {
                "file": file_path,
                "status": "unchanged",
                "note": "LLM returned identical code",
            }

        # Write fixed code
        path.write_text(fixed_code)

        return {
            "file": file_path,
            "status": "fixed",
            "errors_addressed": len(errors),
            "original_lines": len(original_code.split("\n")),
            "fixed_lines": len(fixed_code.split("\n")),
        }

    def _clean_go_code(self, code: str) -> str:
        """
        Clean up LLM response to extract pure Go code.

        Removes markdown code blocks and other artifacts.
        """
        # Remove markdown code blocks
        if code.startswith("```go"):
            code = code[5:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        # Ensure it starts with package declaration
        lines = code.strip().split("\n")
        result_lines = []
        found_package = False

        for line in lines:
            if line.strip().startswith("package "):
                found_package = True
            if found_package:
                result_lines.append(line)

        if not result_lines:
            return code.strip()

        return "\n".join(result_lines)

    def _create_fix_summary(self, fixes: list[dict[str, Any]]) -> str:
        """
        Create a summary of applied fixes.

        Args:
            fixes: List of fix result dictionaries

        Returns:
            Summary text
        """
        lines = ["Fix Summary", "=" * 40, ""]

        for fix in fixes:
            lines.append(f"File: {fix['file']}")
            lines.append(f"  Status: {fix['status']}")
            if fix['status'] == 'fixed':
                lines.append(f"  Errors addressed: {fix.get('errors_addressed', 'N/A')}")
                lines.append(f"  Lines: {fix.get('original_lines', '?')} -> {fix.get('fixed_lines', '?')}")
            elif 'error' in fix:
                lines.append(f"  Error: {fix['error']}")
            elif 'note' in fix:
                lines.append(f"  Note: {fix['note']}")
            lines.append("")

        return "\n".join(lines)
