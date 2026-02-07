"""
Converter Agent for Python to Go Conversion.

This agent performs LLM-powered conversion of Python source files to idiomatic Go.
It reads from the Python source directory and writes to the Go output directory.

Capability: "convert"
    Converts a single Python file to Go based on the conversion plan.

    Parameters:
        step_index (int): Index into the conversion_plan to process

    Session State Updates:
        - converted_modules: List updated with completed module name
        - current_step_index: Updated after successful conversion

    Artifacts Created:
        - {module_name}.go.snapshot: Full content of generated Go file
        - {module_name}.go.prompt: The prompt sent to the LLM (for debugging)

    Trajectory Entries:
        - AGENT_INVOKED: When conversion starts
        - AGENT_COMPLETED: With conversion details and file paths
"""

import os
from pathlib import Path
from typing import Any

from kaizen import Agent, AgentInfo, InvokeResult, Session
from kaizen.types import EntryType
from kaizen.llm import LLMProvider, OllamaProvider


# System prompt for Python to Go conversion
CONVERSION_SYSTEM_PROMPT = """You are an expert Go programmer converting Python code to idiomatic Go.

Rules:
1. Produce compilable, idiomatic Go code
2. Use proper Go naming conventions (CamelCase for exported, camelCase for private)
3. Convert Python classes to Go structs with methods
4. Convert Python exceptions to Go error returns
5. Convert Python ABC/Protocol to Go interfaces
6. Use appropriate Go types (string, int, float64, bool, []T, map[K]V)
7. Add proper package declaration based on the target file path
8. Include necessary imports
9. Convert Python docstrings to Go doc comments
10. Use pointer receivers for methods that modify state
11. Handle None/null as nil or zero values appropriately
12. Convert Python dataclasses to Go structs
13. Convert Python enums to Go const blocks with iota or string constants

Output ONLY the Go code, no explanations or markdown code blocks.
Start directly with the package declaration."""

CONVERSION_USER_PROMPT = """Convert this Python file to idiomatic Go.

Target Go file path: {go_path}
Package name should be: {package_name}

Python source code:
```python
{python_code}
```

Additional context from mapping:
- This is part of the Kaizen agentic session substrate
- The Go module is named "kaizen"
- Other packages in the module: types, session, agent, dispatcher, planner, llm, agents

Generate the complete Go file with all necessary imports."""


class ConverterAgent(Agent):
    """
    Agent that converts Python files to Go using LLM.

    The converter reads Python source code, sends it to an LLM with
    conversion instructions, and writes the resulting Go code to the
    output directory.

    Each invocation handles one file from the conversion plan.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        """
        Initialize the converter agent.

        Args:
            llm_provider: Optional LLM provider. If not provided, creates
                         a default OllamaProvider.
        """
        self._llm = llm_provider or OllamaProvider(
            model="llama3.1:8b",
            timeout=300.0,  # 5 minutes for complex conversions
        )

    def info(self) -> AgentInfo:
        """Return agent metadata."""
        return AgentInfo(
            agent_id="converter_agent_v1",
            name="Converter Agent",
            version="1.0.0",
            capabilities=["convert"],
            description="LLM-powered Python to Go converter",
        )

    def invoke(
        self,
        capability: str,
        session: Session,
        params: dict[str, Any],
    ) -> InvokeResult:
        """Execute the convert capability."""
        if capability != "convert":
            return self._unknown_capability(capability)

        step_index = params.get("step_index")
        if step_index is None:
            return self._invalid_params(
                "convert",
                "Missing required parameter: step_index",
            )

        return self._convert_step(session, step_index)

    def _convert_step(self, session: Session, step_index: int) -> InvokeResult:
        """
        Convert a single step from the plan.

        Args:
            session: The Kaizen session
            step_index: Index into conversion_plan
        """
        agent_info = self.info()

        # Get the conversion plan
        plan = session.get("conversion_plan", [])
        if not plan:
            return self._invalid_params(
                "convert",
                "No conversion_plan in session state. Run planner first.",
            )

        if step_index < 0 or step_index >= len(plan):
            return self._invalid_params(
                "convert",
                f"step_index {step_index} out of range (0-{len(plan)-1})",
            )

        step = plan[step_index]

        # Record invocation
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_INVOKED,
            content={
                "capability": "convert",
                "step_index": step_index,
                "step_name": step["step_name"],
                "python_source": step["python_source"],
                "go_target": step["go_target"],
            },
        )

        # Read Python source
        py_path = Path(step["python_full_path"])
        if not py_path.exists():
            return self._invocation_failed(
                "convert",
                f"Python source file not found: {py_path}",
            )

        python_code = py_path.read_text()

        # Determine Go package name from target path
        go_target = step["go_target"]
        package_name = Path(go_target).parent.name
        if not package_name or package_name == ".":
            package_name = "main"

        # Build the conversion prompt
        user_prompt = CONVERSION_USER_PROMPT.format(
            go_path=go_target,
            package_name=package_name,
            python_code=python_code,
        )

        # Save prompt as artifact for debugging
        prompt_artifact_name = f"{step['step_name']}.prompt.txt"
        session.write_artifact(
            prompt_artifact_name,
            user_prompt.encode("utf-8"),
        )

        # Call LLM for conversion
        try:
            response = self._llm.complete(
                prompt=user_prompt,
                system=CONVERSION_SYSTEM_PROMPT,
            )
            go_code = response.text.strip()

            # Clean up the response - remove markdown code blocks if present
            go_code = self._clean_go_code(go_code)

        except Exception as e:
            session.append(
                agent_id=agent_info.agent_id,
                entry_type=EntryType.AGENT_FAILED,
                content={
                    "capability": "convert",
                    "step_index": step_index,
                    "error": str(e),
                },
            )
            return self._invocation_failed(
                "convert",
                f"LLM conversion failed: {e}",
            )

        # Create output directory if needed
        go_full_path = Path(step["go_full_path"])
        go_full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write Go file
        go_full_path.write_text(go_code)

        # Save Go code as artifact
        snapshot_artifact_name = f"{step['step_name']}.go.snapshot"
        session.write_artifact(
            snapshot_artifact_name,
            go_code.encode("utf-8"),
        )

        # Update plan step status
        plan[step_index]["status"] = "completed"
        session.set("conversion_plan", plan)

        # Update converted modules list
        converted = session.get("converted_modules", [])
        converted.append(step["step_name"])
        session.set("converted_modules", converted)

        # Update current step index
        session.set("current_step_index", step_index + 1)

        # Record completion
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": "convert",
                "step_index": step_index,
                "step_name": step["step_name"],
                "go_file_written": str(go_full_path),
                "go_code_lines": len(go_code.split("\n")),
                "artifacts_created": [prompt_artifact_name, snapshot_artifact_name],
            },
        )

        return InvokeResult.ok(
            result={
                "step_name": step["step_name"],
                "go_file": str(go_full_path),
                "lines": len(go_code.split("\n")),
            },
            agent_id=agent_info.agent_id,
            capability="convert",
        )

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
            # If no package found, return original (cleaned) code
            return code.strip()

        return "\n".join(result_lines)
