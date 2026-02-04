"""
Planner Agent for Python to Go Conversion.

This agent analyzes the Python source repository and mapping file to generate
an ordered conversion plan. It does not modify any files - only produces the plan.

Capability: "plan"
    Generates a conversion plan based on mapping.md and source analysis.

    Parameters: None

    Session State Updates:
        - conversion_plan: List of step dictionaries with file mappings
        - planning_complete: Boolean indicating planning finished

    Artifacts Read:
        - mapping.md: Python to Go file mapping

    Trajectory Entries:
        - AGENT_INVOKED: When planning starts
        - AGENT_COMPLETED: With plan summary and reasoning
"""

import os
from pathlib import Path
from typing import Any

from kaizen import Agent, AgentInfo, InvokeResult, Session
from kaizen.types import EntryType


class PlannerAgent(Agent):
    """
    Agent that generates a conversion plan from Python to Go.

    The planner reads the mapping file and scans the source repository
    to produce an ordered list of conversion steps. Each step specifies
    which Python file to convert and where to write the Go output.

    The plan respects dependency order to ensure files are converted
    in a sequence that allows incremental compilation.
    """

    def info(self) -> AgentInfo:
        """Return agent metadata."""
        return AgentInfo(
            agent_id="planner_agent_v1",
            name="Planner Agent",
            version="1.0.0",
            capabilities=["plan"],
            description="Generates ordered conversion plan from Python to Go",
        )

    def invoke(
        self,
        capability: str,
        session: Session,
        params: dict[str, Any],
    ) -> InvokeResult:
        """Execute the plan capability."""
        if capability != "plan":
            return self._unknown_capability(capability)

        return self._generate_plan(session)

    def _generate_plan(self, session: Session) -> InvokeResult:
        """
        Generate the conversion plan.

        Reads the mapping file, analyzes the source structure, and produces
        an ordered list of conversion steps.
        """
        agent_info = self.info()

        # Record that we're starting planning
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_INVOKED,
            content={
                "capability": "plan",
                "action": "Generating conversion plan",
            },
        )

        # Get paths from session state
        python_repo = session.get("python_repo_clone")
        go_output = session.get("go_output_path")

        if not python_repo or not go_output:
            return self._invalid_params(
                "plan",
                "Missing python_repo_clone or go_output_path in session state",
            )

        # Read mapping from artifact or file
        mapping_content = self._get_mapping(session)
        if mapping_content is None:
            return self._invocation_failed(
                "plan",
                "Could not read mapping.md artifact or file",
            )

        # Parse the mapping to extract file pairs
        file_mappings = self._parse_mapping(mapping_content)

        # Verify source files exist
        source_dir = Path(python_repo) / "src" / "kaizen"
        verified_mappings = []

        for py_file, go_file in file_mappings:
            # Construct full Python path
            py_path = Path(python_repo) / py_file
            if py_path.exists():
                verified_mappings.append({
                    "step_name": Path(go_file).stem,
                    "python_source": py_file,
                    "go_target": go_file,
                    "python_full_path": str(py_path),
                    "go_full_path": str(Path(go_output) / go_file),
                    "status": "pending",
                })
            else:
                # Log missing file but continue
                session.append(
                    agent_id=agent_info.agent_id,
                    entry_type=EntryType.SYSTEM_NOTE,
                    content={
                        "warning": f"Source file not found: {py_file}",
                    },
                )

        # Create the conversion plan with dependency order
        # Order matters: types first, then session, then agents, etc.
        ordered_plan = self._order_by_dependencies(verified_mappings)

        # Store plan in session state
        session.set("conversion_plan", ordered_plan)
        session.set("planning_complete", True)
        session.set("current_step_index", 0)
        session.set("converted_modules", [])
        session.set("status", "converting")

        # Record completion with reasoning
        session.append(
            agent_id=agent_info.agent_id,
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": "plan",
                "total_steps": len(ordered_plan),
                "step_names": [s["step_name"] for s in ordered_plan],
                "reasoning": "Plan generated based on dependency order: "
                            "types → session → agent → llm → dispatcher → planner → agents",
            },
        )

        return InvokeResult.ok(
            result={
                "plan_steps": len(ordered_plan),
                "steps": [s["step_name"] for s in ordered_plan],
            },
            agent_id=agent_info.agent_id,
            capability="plan",
        )

    def _get_mapping(self, session: Session) -> str | None:
        """
        Get mapping content from artifact or file.

        First tries to read from session artifact, then falls back to file.
        """
        # Try artifact first
        try:
            mapping_bytes = session.read_artifact("mapping.md")
            return mapping_bytes.decode("utf-8")
        except KeyError:
            pass

        # Fall back to file in demo directory
        mapping_path = Path(__file__).parent.parent / "mapping.md"
        if mapping_path.exists():
            return mapping_path.read_text()

        return None

    def _parse_mapping(self, content: str) -> list[tuple[str, str]]:
        """
        Parse mapping.md to extract Python → Go file pairs.

        Returns list of (python_file, go_file) tuples.
        """
        mappings = []

        for line in content.split("\n"):
            line = line.strip()

            # Look for table rows (start with |)
            if not line.startswith("|"):
                continue

            # Split by | and get parts
            parts = [p.strip() for p in line.split("|")]
            # Filter out empty parts and header separators (---)
            parts = [p for p in parts if p and not p.startswith("-")]

            if len(parts) < 2:
                continue

            py_file = parts[0].strip("`")
            go_file = parts[1].strip("`")

            # Skip header rows
            if py_file in ("Python Source", "Python", "Source"):
                continue

            # Only include actual file paths
            if py_file.endswith(".py") and go_file.endswith(".go"):
                mappings.append((py_file, go_file))

        return mappings

    def _order_by_dependencies(
        self,
        mappings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Order mappings by dependency (types first, then session, etc.).

        This ensures files are converted in an order that allows
        incremental compilation.
        """
        # Define priority order (lower = earlier)
        priority = {
            "types": 1,
            "session": 2,
            "agent": 3,
            "provider": 4,  # llm/provider.go
            "ollama": 5,    # llm/ollama.go
            "dispatcher": 6,
            "planner": 7,
            "reverse": 8,
            "uppercase": 9,
        }

        def get_priority(mapping: dict[str, Any]) -> int:
            name = mapping["step_name"]
            return priority.get(name, 100)

        return sorted(mappings, key=get_priority)
