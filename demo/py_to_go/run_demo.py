#!/usr/bin/env python3
"""
Python to Go Conversion Demo - Main Orchestration Script

This script demonstrates Kaizen's capabilities by using it to orchestrate
a Python-to-Go port of the Kaizen codebase itself.

The demo:
1. Clones the Kaizen repo into a temporary source directory
2. Creates a separate temporary Go output directory
3. Uses Kaizen session for state management and auditability
4. Runs agents: Planner → Converter → TestRunner → Fixer
5. Saves all artifacts and trajectory to the session file

Prerequisites:
- Ollama running with llama3.1:8b model
- Go toolchain installed
- Git installed

Usage:
    python demo/py_to_go/run_demo.py

The demo creates temporary directories that must be manually removed
after inspection. Cleanup commands are printed at the end.
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # For demo package
sys.path.insert(0, str(project_root / "src"))  # For kaizen package

from kaizen import Session, Dispatcher
from kaizen.types import EntryType
from kaizen.llm import OllamaProvider

# Import demo agents
from demo.py_to_go.agents.planner import PlannerAgent
from demo.py_to_go.agents.converter import ConverterAgent
from demo.py_to_go.agents.test_runner import TestRunnerAgent
from demo.py_to_go.agents.fixer import FixerAgent


# Configuration
REPO_URL = "https://github.com/ashbert/Kaizen.git"
GO_MODULE_NAME = "kaizen"
MAX_FIX_ITERATIONS = 5
SESSION_FILE = Path(__file__).parent / "py_to_go.kaizen"


def print_banner(text: str) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_step(step: int, text: str) -> None:
    """Print a step indicator."""
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def check_prerequisites() -> bool:
    """Check that required tools are available."""
    print_step(0, "Checking prerequisites")

    # Check git
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("  ✓ Git is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ✗ Git is not available")
        return False

    # Check go
    try:
        subprocess.run(["go", "version"], capture_output=True, check=True)
        print("  ✓ Go is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ✗ Go is not available")
        return False

    # Check Ollama
    try:
        llm = OllamaProvider()
        if llm.is_available():
            print("  ✓ Ollama is available")
        else:
            print("  ✗ Ollama is not running")
            print("    Start it with: ollama serve")
            return False
    except Exception as e:
        print(f"  ✗ Ollama check failed: {e}")
        return False

    return True


def create_temp_directories() -> tuple[str, str]:
    """Create temporary directories for source and output."""
    src_tmp = tempfile.mkdtemp(prefix="kaizen_src_")
    out_tmp = tempfile.mkdtemp(prefix="kaizen_go_out_")
    return src_tmp, out_tmp


def clone_repository(src_tmp: str) -> bool:
    """Clone the Kaizen repository."""
    print_step(1, "Cloning Kaizen repository")
    print(f"  Target: {src_tmp}")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, src_tmp],
            check=True,
            capture_output=True,
        )
        print("  ✓ Repository cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Clone failed: {e.stderr.decode()}")
        return False


def initialize_go_module(out_tmp: str) -> bool:
    """Initialize Go module in output directory."""
    print_step(2, "Initializing Go module")
    print(f"  Target: {out_tmp}")

    try:
        subprocess.run(
            ["go", "mod", "init", GO_MODULE_NAME],
            cwd=out_tmp,
            check=True,
            capture_output=True,
        )
        print(f"  ✓ Go module '{GO_MODULE_NAME}' initialized")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Go mod init failed: {e.stderr.decode()}")
        return False


def setup_session(src_tmp: str, out_tmp: str) -> Session:
    """Create or load the Kaizen session."""
    print_step(3, "Setting up Kaizen session")

    # Check for existing session
    if SESSION_FILE.exists():
        print(f"  Loading existing session: {SESSION_FILE}")
        session = Session.load(str(SESSION_FILE))
        print(f"  ✓ Session loaded (ID: {session.session_id[:8]}...)")

        # Update paths for this run (temp dirs are new)
        session.set("python_repo_clone", src_tmp)
        session.set("go_output_path", out_tmp)
        return session

    # Create new session
    print("  Creating new session")
    session = Session()

    # Seed initial state
    session.set("source_language", "python")
    session.set("target_language", "go")
    session.set("python_repo_clone", src_tmp)
    session.set("go_output_path", out_tmp)
    session.set("goal", "Port Kaizen from Python to Go with all tests passing")
    session.set("status", "planning")
    session.set("demo_started", datetime.now(timezone.utc).isoformat())

    # Attach mapping.md as artifact
    mapping_path = Path(__file__).parent / "mapping.md"
    if mapping_path.exists():
        mapping_content = mapping_path.read_bytes()
        session.write_artifact("mapping.md", mapping_content)
        print("  ✓ Attached mapping.md artifact")

    # Save initial session
    session.save(str(SESSION_FILE))
    print(f"  ✓ Session created (ID: {session.session_id[:8]}...)")
    print(f"  ✓ Session saved to: {SESSION_FILE}")

    return session


def setup_dispatcher(llm: OllamaProvider) -> Dispatcher:
    """Create and configure the dispatcher with all agents."""
    print_step(4, "Setting up dispatcher and agents")

    dispatcher = Dispatcher()

    # Register all agents
    dispatcher.register(PlannerAgent())
    dispatcher.register(ConverterAgent(llm_provider=llm))
    dispatcher.register(TestRunnerAgent())
    dispatcher.register(FixerAgent(llm_provider=llm))

    print(f"  ✓ Registered capabilities: {dispatcher.get_capabilities()}")
    return dispatcher


def run_planning(dispatcher: Dispatcher, session: Session) -> bool:
    """Run the planner agent."""
    print_step(5, "Running Planner agent")

    result = dispatcher.dispatch_single("plan", session, {})

    if result.success:
        plan = session.get("conversion_plan", [])
        print(f"  ✓ Plan generated with {len(plan)} steps:")
        for i, step in enumerate(plan):
            print(f"      {i+1}. {step['step_name']}: {step['python_source']} → {step['go_target']}")
        session.save(str(SESSION_FILE))
        return True
    else:
        print(f"  ✗ Planning failed: {result.error}")
        return False


def run_conversion(dispatcher: Dispatcher, session: Session) -> bool:
    """Run the converter agent for each plan step."""
    print_step(6, "Running Converter agent")

    plan = session.get("conversion_plan", [])

    for i, step in enumerate(plan):
        print(f"\n  Converting [{i+1}/{len(plan)}]: {step['step_name']}")

        result = dispatcher.dispatch_single("convert", session, {"step_index": i})

        if result.success:
            print(f"    ✓ Converted to: {result.result.get('go_file', 'N/A')}")
            print(f"      Lines: {result.result.get('lines', 'N/A')}")
        else:
            print(f"    ✗ Conversion failed: {result.error}")
            # Continue with other files even if one fails

        # Save after each step
        session.save(str(SESSION_FILE))

    converted = session.get("converted_modules", [])
    print(f"\n  ✓ Conversion complete: {len(converted)}/{len(plan)} modules")
    return len(converted) > 0


def run_test_fix_loop(dispatcher: Dispatcher, session: Session) -> bool:
    """Run tests and fix loop."""
    print_step(7, "Running Test/Fix loop")

    for iteration in range(1, MAX_FIX_ITERATIONS + 1):
        print(f"\n  --- Iteration {iteration}/{MAX_FIX_ITERATIONS} ---")

        # Run tests
        print("  Running tests...")
        test_result = dispatcher.dispatch_single("run_tests", session, {})

        if test_result.success and session.get("tests_passed", False):
            print("  ✓ All tests passed!")
            session.save(str(SESSION_FILE))
            return True

        # Show test summary
        summary = session.get("last_test_output", "No output")
        print(f"  ✗ Tests failed: {summary[:200]}")

        if iteration == MAX_FIX_ITERATIONS:
            print(f"  ✗ Max iterations ({MAX_FIX_ITERATIONS}) reached")
            break

        # Run fixer
        print("  Running fixer...")
        fix_result = dispatcher.dispatch_single("fix", session, {})

        if fix_result.success:
            files_fixed = fix_result.result.get("files_fixed", 0)
            print(f"    ✓ Fixed {files_fixed} file(s)")
        else:
            print(f"    ✗ Fix failed: {fix_result.error}")

        session.save(str(SESSION_FILE))

    return False


def print_summary(session: Session, src_tmp: str, out_tmp: str, success: bool) -> None:
    """Print final summary and cleanup instructions."""
    print_banner("Demo Complete")

    print(f"\nStatus: {'SUCCESS' if success else 'INCOMPLETE'}")
    print(f"Session file: {SESSION_FILE}")

    # Print trajectory summary
    trajectory = session.get_trajectory()
    print(f"\nTrajectory entries: {len(trajectory)}")

    agent_counts: dict[str, int] = {}
    for entry in trajectory:
        agent = entry.agent_id
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    print("  By agent:")
    for agent, count in sorted(agent_counts.items()):
        print(f"    {agent}: {count}")

    # Print artifacts
    artifacts = session.list_artifacts()
    print(f"\nArtifacts stored: {len(artifacts)}")
    for name in artifacts[:10]:
        size = session.get_artifact_size(name)
        print(f"    {name} ({size} bytes)")
    if len(artifacts) > 10:
        print(f"    ... and {len(artifacts) - 10} more")

    # Print cleanup instructions
    print("\n" + "-" * 60)
    print("TEMPORARY DIRECTORIES (inspect before removing):")
    print(f"  Source clone: {src_tmp}")
    print(f"  Go output:    {out_tmp}")
    print("\nTo clean up, run:")
    print(f"  rm -rf {src_tmp} {out_tmp}")
    print("-" * 60)


def main() -> int:
    """Main entry point."""
    print_banner("Kaizen Python → Go Conversion Demo")
    print(f"Start time: {datetime.now(timezone.utc).isoformat()}")

    # Check prerequisites
    if not check_prerequisites():
        print("\n✗ Prerequisites not met. Exiting.")
        return 1

    # Create temporary directories
    src_tmp, out_tmp = create_temp_directories()
    print(f"\nTemporary directories created:")
    print(f"  Source: {src_tmp}")
    print(f"  Output: {out_tmp}")

    success = False

    try:
        # Clone repository
        if not clone_repository(src_tmp):
            return 1

        # Initialize Go module
        if not initialize_go_module(out_tmp):
            return 1

        # Set up session
        session = setup_session(src_tmp, out_tmp)

        # Set up LLM provider (shared across agents)
        llm = OllamaProvider(model="llama3.1:8b", timeout=300.0)

        # Set up dispatcher
        dispatcher = setup_dispatcher(llm)

        # Check if we need to plan
        if not session.get("planning_complete", False):
            if not run_planning(dispatcher, session):
                return 1

        # Run conversion
        if session.get("status") in ("planning", "converting"):
            if not run_conversion(dispatcher, session):
                print("\n✗ No modules converted. Check errors above.")
                return 1

        # Run test/fix loop
        success = run_test_fix_loop(dispatcher, session)

        # Final save
        session.set("demo_completed", datetime.now(timezone.utc).isoformat())
        session.set("demo_success", success)
        session.save(str(SESSION_FILE))

    except KeyboardInterrupt:
        print("\n\n✗ Demo interrupted by user")
        return 130

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Always print summary with cleanup instructions
        try:
            print_summary(session, src_tmp, out_tmp, success)
        except NameError:
            print(f"\nTo clean up temp directories, run:")
            print(f"  rm -rf {src_tmp} {out_tmp}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
