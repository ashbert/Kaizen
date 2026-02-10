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

# Pattern for self-import lines: "kaizen/pkg" or alias "kaizen/pkg"
_SELF_IMPORT_TMPL = r'^\s*(?:\w+\s+)?"kaizen/{pkg}"\s*$'

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
- Syntax errors: fix brackets, semicolons, etc.

Important:
- Internal package imports use "kaizen/..." (e.g. "kaizen/types", "kaizen/session"), NOT "github.com/kaizen/..."
- The Go module is named "kaizen" — all internal cross-package imports start with "kaizen/"
- You CANNOT define methods on interface types in Go. If you see "invalid receiver type X (pointer or
  interface type)", the fix is to REMOVE those method implementations entirely — interfaces only declare
  method signatures, they don't have implementations
- If "undefined: pkg.Symbol" appears, check the provided package context files for the correct symbol names
- NEVER import the package you are in. If the file is "package llm", do NOT import "kaizen/llm".
  Types from other files in the same package are available directly — no import needed.
- If "multiple-value X in single-value context" appears, use a two-value assignment: `val, err := X(...)`.
  If the function is being used where only an error is needed (like returning from a session method),
  replace it with `errors.New("message")` or `fmt.Errorf("message: %v", details)` instead.
- If a struct field and method have the same name, rename the field to lowercase (unexported) — e.g.
  `BaseURL string` becomes `baseURL string` and access it via the method `BaseURL()` instead.
- Types from sibling files in the SAME package are used DIRECTLY without import or package prefix.
  For example, if LLMResponse is defined in llm/provider.go and you're fixing llm/ollama.go,
  use `LLMResponse` not `types.LLMResponse`.
- For test file fixes (*_test.go): use "testing" package, func TestXxx(t *testing.T) pattern,
  t.Errorf/t.Fatalf for assertions, t.TempDir() for temp dirs. Test files share the package
  with the code they test — types are available directly.
- SQLite: use "database/sql" with blank import `_ "modernc.org/sqlite"`.
  Open with `sql.Open("sqlite", path)`. Do NOT use sqlite3.Open() or any sqlite3 package.
  Use db.Exec(), db.Query(), db.QueryRow(). Use "?" for placeholders.
- "no new variables on left side of :=" means the variable already exists in scope — use `=` not `:=`.
  For example: `_, err := stmt.Exec(...)` should be `_, err = stmt.Exec(...)` if err was declared earlier."""

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
{sibling_context}
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

            # Match import cycle errors: "imports kaizen/pkg from file.go: import cycle"
            if "import cycle" in line.lower():
                cycle_match = re.search(r'from\s+(\S+\.go)', line)
                if cycle_match:
                    file_name = cycle_match.group(1)
                    full_path = self._find_file(file_name, go_output_path)
                    if full_path:
                        if full_path not in errors_by_file:
                            errors_by_file[full_path] = []
                        errors_by_file[full_path].append(line)

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

        # Read related Go snapshots from session for cross-file context
        sibling_context = self._get_session_context(session, file_path, errors)

        # Build fix prompt
        errors_text = "\n".join(errors[:10])  # Limit errors to avoid huge prompts
        user_prompt = FIX_USER_PROMPT.format(
            file_path=file_path,
            go_code=original_code,
            errors=errors_text,
            sibling_context=sibling_context,
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

            # Strip self-imports deterministically
            pkg = Path(file_path).parent.name
            if pkg:
                self_re = re.compile(
                    _SELF_IMPORT_TMPL.format(pkg=re.escape(pkg))
                )
                fixed_code = "\n".join(
                    l for l in fixed_code.split("\n")
                    if not self_re.match(l)
                )
                fixed_code = re.sub(r'import\s*\(\s*\)', '', fixed_code)
                fixed_code = re.sub(r'\n{3,}', '\n\n', fixed_code)

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

    def _get_session_context(
        self,
        session: Session,
        file_path: str,
        errors: list[str],
    ) -> str:
        """
        Read Go snapshots from session artifacts for cross-file context.

        Includes:
        - Sibling files in the same package (same-package types)
        - Files from imported packages referenced in errors (cross-package types)
        """
        artifacts = session.list_artifacts()
        plan = session.get("conversion_plan", [])

        # Build a map: go_target → snapshot artifact name
        target_to_artifact: dict[str, str] = {}
        for step in plan:
            art_name = f"{step['step_name']}.go.snapshot"
            if art_name in artifacts:
                target_to_artifact[step.get("go_target", "")] = art_name

        # Determine which package this file is in
        file_pkg = Path(file_path).parent.name  # e.g. "llm", "types", "session"

        same_pkg_parts = []
        cross_pkg_parts = []

        for go_target, art_name in target_to_artifact.items():
            target_pkg = Path(go_target).parent.name  # package dir name
            target_file = Path(go_target).name

            # Skip the file being fixed
            if go_target.endswith(Path(file_path).name) and target_pkg == file_pkg:
                continue

            try:
                content = session.read_artifact(art_name).decode("utf-8")
            except Exception:
                continue

            if target_pkg == file_pkg:
                # Same package — types available directly without import
                same_pkg_parts.append(
                    f"Same package file ({target_file}):\n```go\n{content}\n```"
                )
            else:
                # Different package — include if referenced in errors
                # or always include types package (commonly needed)
                include = target_pkg == "types"
                if not include:
                    for error in errors:
                        if f"{target_pkg}." in error:
                            include = True
                            break
                if include:
                    cross_pkg_parts.append(
                        f"Imported package file ({go_target}):\n```go\n{content}\n```"
                    )

        parts = []
        if same_pkg_parts:
            parts.append(
                "The following files are in the SAME Go package. "
                "Types and functions defined here are available directly "
                "without importing.\n\n" + "\n\n".join(same_pkg_parts)
            )
        if cross_pkg_parts:
            parts.append(
                "The following files are from imported packages. "
                "Use the EXACT type/constant names defined here:\n\n"
                + "\n\n".join(cross_pkg_parts)
            )

        if not parts:
            return ""

        return "\n" + "\n\n".join(parts) + "\n"

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
