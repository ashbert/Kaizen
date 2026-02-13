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
import re
import sys
import subprocess
import tempfile
import hashlib
import json
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # For demo package
sys.path.insert(0, str(project_root / "src"))  # For kaizen package

from kaizen import Session, Dispatcher
from kaizen.types import EntryType
from kaizen.llm import LLMProvider, OllamaProvider, OpenAICompatProvider

# Import demo agents
from demo.py_to_go.agents.planner import PlannerAgent
from demo.py_to_go.agents.converter import ConverterAgent
from demo.py_to_go.agents.test_runner import TestRunnerAgent
from demo.py_to_go.agents.fixer import FixerAgent


# Configuration
REPO_URL = "https://github.com/ashbert/Kaizen.git"
GO_MODULE_NAME = "kaizen"
MAX_FIX_ITERATIONS = 8
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

    # Check LLM provider
    if os.environ.get("KAIZEN_MODEL_URL"):
        print(f"  ✓ Using remote LLM: {os.environ['KAIZEN_MODEL_URL']}")
    else:
        try:
            llm = OllamaProvider()
            if llm.is_available():
                print("  ✓ Ollama is available")
            else:
                print("  ✗ Ollama is not running")
                print("    Start it with: ollama serve")
                print("    Or set KAIZEN_MODEL_URL for a remote endpoint")
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
    """Clone the Kaizen repository (falls back to local source)."""
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
        print(f"  ⚠ Clone failed: {e.stderr.decode().strip()}")
        # Fall back to local source + tests
        import shutil
        local_src = project_root / "src"
        local_tests = project_root / "tests"
        if local_src.exists():
            shutil.copytree(str(local_src), str(Path(src_tmp) / "src"))
            if local_tests.exists():
                shutil.copytree(str(local_tests), str(Path(src_tmp) / "tests"))
            print("  ✓ Using local source as fallback")
            return True
        print("  ✗ No local source available either")
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

    workspace = os.environ.get("KAIZEN_WORKSPACE")

    # Always start a fresh session for the demo
    # (Previous sessions have baked-in temp paths that won't exist)
    if SESSION_FILE.exists():
        print(f"  Removing stale session: {SESSION_FILE}")
        SESSION_FILE.unlink()

    # If workspace is set, use subdirs inside it
    if workspace:
        src_dir = str(Path(workspace) / "src")
        out_dir = str(Path(workspace) / "out")
        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        src_tmp = src_dir
        out_tmp = out_dir
        print(f"  Using workspace: {workspace}")

    # Create new session
    print("  Creating new session")
    session = Session(workspace_path=workspace)

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


def setup_dispatcher(llm: LLMProvider) -> Dispatcher:
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


def _get_session_go_context(session: Session, current_step: str) -> str:
    """Read already-converted Go snapshots from session for LLM context."""
    artifacts = session.list_artifacts()
    plan = session.get("conversion_plan", [])
    parts = []
    for name in artifacts:
        if not name.endswith(".go.snapshot"):
            continue
        module_name = name.replace(".go.snapshot", "")
        if module_name == current_step:
            continue
        try:
            content = session.read_artifact(name).decode("utf-8")
            go_path = module_name
            for step in plan:
                if step.get("step_name") == module_name:
                    go_path = step.get("go_target", module_name)
                    break
            parts.append(f"Already converted ({go_path}):\n```go\n{content}\n```")
        except Exception:
            continue
    if not parts:
        return ""
    return (
        "\nAlready-converted Go files — use matching types, interfaces, "
        "and constants exactly as defined here:\n\n"
        + "\n\n".join(parts) + "\n"
    )


def _strip_self_imports(go_code: str, package_name: str) -> str:
    """Remove self-imports from Go code.

    The LLM sometimes generates imports like `import "kaizen/llm"` inside a file
    that is already `package llm`. This creates import cycles. We strip them
    deterministically since prompt-based fixes are unreliable.
    """
    # Match both plain and aliased imports: "kaizen/pkg" or alias "kaizen/pkg"
    self_import_re = re.compile(
        rf'^\s*(?:\w+\s+)?"kaizen/{re.escape(package_name)}"\s*$'
    )
    lines = go_code.split("\n")
    filtered = []
    for line in lines:
        if self_import_re.match(line):
            continue
        filtered.append(line)

    # Clean up empty import blocks: import (\n)
    result = "\n".join(filtered)
    result = re.sub(r'import\s*\(\s*\)', '', result)
    # Remove leftover blank lines from removed imports (collapse triple+ newlines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


def _export_struct_fields(go_code: str) -> str:
    """Make all struct fields exported (uppercase first letter) and update references.

    The LLM sometimes generates unexported (lowercase) struct fields,
    which can't be accessed from other packages. Since all Kaizen types
    are used cross-package, we export all fields and update all references.
    """
    # Collect method names to avoid field/method naming collisions
    method_names: set[str] = set()
    for m in re.finditer(r'func\s+\([^)]+\)\s+(\w+)\s*\(', go_code):
        method_names.add(m.group(1))

    # First pass: collect unexported struct field names and fix definitions
    renames: dict[str, str] = {}
    lines = go_code.split("\n")
    result_lines = []
    in_struct = False
    for line in lines:
        stripped = line.strip()
        if stripped.endswith("struct {"):
            in_struct = True
            result_lines.append(line)
            continue
        if in_struct:
            if stripped == "}":
                in_struct = False
                result_lines.append(line)
                continue
            if not stripped or stripped.startswith("//"):
                result_lines.append(line)
                continue
            field_match = re.match(r'^(\s+)([a-z]\w*)(\s+\S.*)', line)
            if field_match:
                indent = field_match.group(1)
                old_name = field_match.group(2)
                rest = field_match.group(3)
                new_name = old_name[0].upper() + old_name[1:]
                # Skip if exporting would collide with a method name
                if new_name in method_names:
                    result_lines.append(line)
                    continue
                renames[old_name] = new_name
                result_lines.append(f"{indent}{new_name}{rest}")
                continue
        result_lines.append(line)

    if not renames:
        return go_code

    go_code = "\n".join(result_lines)

    # Second pass: update field accesses (.fieldName) and struct literal keys (fieldName:)
    for old_name, new_name in renames.items():
        # Field access: .fieldName → .FieldName
        go_code = re.sub(
            rf'\.{re.escape(old_name)}\b',
            f'.{new_name}',
            go_code,
        )
        # Struct literal keys: `\tfieldName:` → `\tFieldName:`
        # Only match when the key is indented (struct literal context)
        go_code = re.sub(
            rf'^(\s+){re.escape(old_name)}:',
            rf'\g<1>{new_name}:',
            go_code,
            flags=re.MULTILINE,
        )

    return go_code


def _export_all_struct_fields(go_output_path: str) -> int:
    """Export struct fields in all .go files (cross-file aware).

    Phase 1: Collect ALL struct field names (exported AND unexported) from all files
    Phase 2: Export fields in definitions and fix lowercase references across all files
    """
    go_files = list(Path(go_output_path).rglob("*.go"))

    # Collect method names to avoid collisions
    all_method_names: set[str] = set()
    for go_file in go_files:
        code = go_file.read_text()
        for m in re.finditer(r'func\s+\([^)]+\)\s+(\w+)\s*\(', code):
            all_method_names.add(m.group(1))

    # Phase 1: Collect lowercase→Uppercase mappings from ALL struct fields
    # Include BOTH: unexported fields (to be exported) AND already-exported fields
    # (to fix cross-file references that still use lowercase names)
    all_renames: dict[str, str] = {}

    for go_file in go_files:
        code = go_file.read_text()
        in_struct = False
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.endswith("struct {"):
                in_struct = True
                continue
            if in_struct:
                if stripped == "}":
                    in_struct = False
                    continue
                if not stripped or stripped.startswith("//"):
                    continue
                # Match EXPORTED field: e.g. "AgentID string" → map agentID->AgentID
                exp_match = re.match(r'^\s+([A-Z]\w*)\s+\S', line)
                if exp_match:
                    exported_name = exp_match.group(1)
                    lower_name = exported_name[0].lower() + exported_name[1:]
                    if lower_name != exported_name:  # skip single-letter
                        all_renames[lower_name] = exported_name
                    continue
                # Match UNEXPORTED field: e.g. "agentID string" → map agentID->AgentID
                unexp_match = re.match(r'^\s+([a-z]\w*)\s+\S', line)
                if unexp_match:
                    old_name = unexp_match.group(1)
                    new_name = old_name[0].upper() + old_name[1:]
                    if new_name not in all_method_names:
                        all_renames[old_name] = new_name

    if not all_renames:
        return 0

    # Phase 2: Export field definitions + fix references across ALL files
    modified = 0
    for go_file in go_files:
        code = go_file.read_text()
        # Collect method names for THIS file to avoid collisions
        file_methods: set[str] = set()
        for m in re.finditer(r'func\s+\([^)]+\)\s+(\w+)\s*\(', code):
            file_methods.add(m.group(1))

        # Export unexported field definitions in struct blocks (per-file)
        new_code = _export_struct_fields(code)
        # Fix all lowercase field references across files,
        # but skip renames that would collide with a method in THIS file
        for old_name, new_name in all_renames.items():
            if new_name in file_methods:
                continue  # Skip: would collide with a method in this file
            # Field access: .fieldName → .FieldName
            new_code = re.sub(rf'\.{re.escape(old_name)}\b', f'.{new_name}', new_code)
            # Struct literal keys: fieldName: → FieldName:
            new_code = re.sub(
                rf'^(\s+){re.escape(old_name)}:',
                rf'\g<1>{new_name}:',
                new_code,
                flags=re.MULTILINE,
            )
        if new_code != code:
            go_file.write_text(new_code)
            modified += 1
    return modified


def _is_test_step(step: dict) -> bool:
    """Check if a plan step is a test file (not source)."""
    return step.get("go_target", "").endswith("_test.go") or step.get("step_name") == "testutil"


def _convert_steps(
    steps: list[tuple[int, dict]],
    session: Session,
    llm: LLMProvider,
    plan: list[dict],
    label: str,
) -> list[str]:
    """Convert a list of plan steps sequentially. Returns list of converted module names."""
    from demo.py_to_go.agents.converter import CONVERSION_SYSTEM_PROMPT, CONVERSION_USER_PROMPT
    import time as _time

    converted = session.get("converted_modules", [])

    for i, step in steps:
        py_path = Path(step["python_full_path"])
        if not py_path.exists():
            print(f"    ✗ [{step['step_name']}] File not found: {py_path}")
            continue

        python_code = py_path.read_text()
        package_name = Path(step["go_target"]).parent.name
        if not package_name or package_name == ".":
            package_name = "main"

        ctx = _get_session_go_context(session, step["step_name"])
        prompt = CONVERSION_USER_PROMPT.format(
            go_path=step["go_target"],
            package_name=package_name,
            python_code=python_code,
            converted_context=ctx,
        )

        # Retry up to 4 times on transient errors
        last_err = None
        go_code = None
        for attempt in range(5):
            try:
                print(f"      [{step['step_name']}] sending request (attempt {attempt+1})...", flush=True)
                response = llm.complete(prompt=prompt, system=CONVERSION_SYSTEM_PROMPT)
                go_code = response.text.strip()
                # Clean markdown fences
                if go_code.startswith("```go"):
                    go_code = go_code[5:]
                elif go_code.startswith("```"):
                    go_code = go_code[3:]
                if go_code.endswith("```"):
                    go_code = go_code[:-3]
                go_code = go_code.strip()
                go_code = _strip_self_imports(go_code, package_name)
                last_err = None
                break
            except Exception as e:
                last_err = e
                _time.sleep(2 ** attempt)

        if last_err:
            print(f"    ✗ [{step['step_name']}]: {last_err}")
            continue

        # Write Go file
        go_path = Path(step["go_full_path"])
        go_path.parent.mkdir(parents=True, exist_ok=True)
        go_path.write_text(go_code)

        # Store result in session
        session.write_artifact(f"{step['step_name']}.go.snapshot", go_code.encode("utf-8"))
        plan[i]["status"] = "completed"
        converted.append(step["step_name"])
        session.append(
            agent_id="converter_agent_v1",
            entry_type=EntryType.AGENT_COMPLETED,
            content={
                "capability": "convert",
                "step_index": i,
                "step_name": step["step_name"],
                "go_file_written": step["go_full_path"],
                "go_code_lines": len(go_code.split("\n")),
            },
        )
        session.save(str(SESSION_FILE))

        lines = len(go_code.split("\n"))
        print(f"    ✓ [{step['step_name']}]: {lines} lines")
        _time.sleep(2)  # Brief pause between conversions

    session.set("conversion_plan", plan)
    session.set("converted_modules", converted)
    return converted


def run_source_conversion(dispatcher: Dispatcher, session: Session, llm: LLMProvider) -> bool:
    """Convert source files (not tests) in waves."""
    print_step(6, "Running source conversion")

    plan = session.get("conversion_plan", [])
    source_steps = [(i, step) for i, step in enumerate(plan) if not _is_test_step(step)]

    # Wave 1: types and session (foundational)
    foundational = ["types", "session"]
    wave1 = [(i, s) for i, s in source_steps if s["step_name"] in foundational]
    wave2 = [(i, s) for i, s in source_steps if s["step_name"] not in foundational]

    if wave1:
        print(f"  Wave 1: Converting {len(wave1)} foundational package(s)...")
        _convert_steps(wave1, session, llm, plan, "foundational")

    if wave2:
        print(f"  Wave 2: Converting {len(wave2)} remaining source modules...")
        _convert_steps(wave2, session, llm, plan, "source")

    # Post-conversion deterministic fixups
    go_output = session.get("go_output_path", "")
    if go_output:
        n_imports = _strip_all_self_imports(go_output)
        n_fields = _export_all_struct_fields(go_output)
        if n_imports or n_fields:
            print(f"  Post-fix: stripped self-imports from {n_imports}, exported fields in {n_fields} file(s)")

    session.save(str(SESSION_FILE))

    converted = session.get("converted_modules", [])
    source_count = len([s for _, s in source_steps])
    converted_source = len([c for c in converted if not c.endswith("_test") and c != "testutil"])
    print(f"\n  ✓ Source conversion complete: {converted_source}/{source_count} modules")
    return converted_source > 0


def run_test_conversion(dispatcher: Dispatcher, session: Session, llm: LLMProvider) -> bool:
    """Convert test files after source compiles."""
    print_step(8, "Running test file conversion")

    plan = session.get("conversion_plan", [])
    test_steps = [(i, step) for i, step in enumerate(plan) if _is_test_step(step)]

    if not test_steps:
        print("  No test files in conversion plan")
        return True

    print(f"  Converting {len(test_steps)} test file(s) sequentially...")
    _convert_steps(test_steps, session, llm, plan, "test")

    # Post-conversion fixups for test files too
    go_output = session.get("go_output_path", "")
    if go_output:
        n_imports = _strip_all_self_imports(go_output)
        n_fields = _export_all_struct_fields(go_output)
        if n_imports or n_fields:
            print(f"  Post-fix: stripped self-imports from {n_imports}, exported fields in {n_fields} file(s)")

    session.save(str(SESSION_FILE))

    converted = session.get("converted_modules", [])
    test_names = {s["step_name"] for _, s in test_steps}
    converted_tests = len([c for c in converted if c in test_names])
    print(f"\n  ✓ Test conversion complete: {converted_tests}/{len(test_steps)} files")
    return converted_tests > 0


def _fix_walrus_errors(go_output_path: str, test_output: str) -> int:
    """Deterministically fix 'no new variables on left side of :=' errors.

    When Go reports this error, we know the exact file and line. We just
    change ':=' to '=' on that line. This is a common LLM mistake that
    the fixer often fails to correct.

    Returns the number of lines fixed.
    """
    # Pattern: file.go:line:col: no new variables on left side of :=
    pattern = re.compile(r'(\S+\.go):(\d+):\d+:\s*no new variables on left side of :=')
    fixes_by_file: dict[str, list[int]] = {}

    for match in pattern.finditer(test_output):
        file_ref = match.group(1)
        line_num = int(match.group(2))
        # Resolve the file path
        full_path = Path(go_output_path) / file_ref
        if not full_path.exists():
            # Try searching
            for f in Path(go_output_path).rglob(Path(file_ref).name):
                if str(f).endswith(file_ref):
                    full_path = f
                    break
        if full_path.exists():
            key = str(full_path)
            if key not in fixes_by_file:
                fixes_by_file[key] = []
            fixes_by_file[key].append(line_num)

    total_fixed = 0
    for file_path, line_nums in fixes_by_file.items():
        lines = Path(file_path).read_text().split("\n")
        changed = False
        for ln in set(line_nums):
            idx = ln - 1  # 0-based
            if 0 <= idx < len(lines) and ":=" in lines[idx]:
                lines[idx] = lines[idx].replace(":=", "=", 1)
                changed = True
                total_fixed += 1
        if changed:
            Path(file_path).write_text("\n".join(lines))

    return total_fixed


def _fix_unused_imports(go_output_path: str, test_output: str) -> int:
    """Deterministically remove unused imports reported by the Go compiler.

    Returns the number of imports removed.
    """
    # Pattern: file.go:line:col: "pkg" imported and not used
    pattern = re.compile(r'(\S+\.go):(\d+):\d+:\s*"([^"]+)"\s+imported and not used')
    removes_by_file: dict[str, set[int]] = {}

    for match in pattern.finditer(test_output):
        file_ref = match.group(1)
        line_num = int(match.group(2))
        full_path = Path(go_output_path) / file_ref
        if not full_path.exists():
            for f in Path(go_output_path).rglob(Path(file_ref).name):
                if str(f).endswith(file_ref):
                    full_path = f
                    break
        if full_path.exists():
            key = str(full_path)
            if key not in removes_by_file:
                removes_by_file[key] = set()
            removes_by_file[key].add(line_num)

    total_removed = 0
    for file_path, line_nums in removes_by_file.items():
        lines = Path(file_path).read_text().split("\n")
        new_lines = []
        for i, line in enumerate(lines):
            if (i + 1) in line_nums:
                total_removed += 1
                continue  # Skip this import line
            new_lines.append(line)
        if total_removed:
            # Clean up empty import blocks
            code = "\n".join(new_lines)
            code = re.sub(r'import\s*\(\s*\)', '', code)
            code = re.sub(r'\n{3,}', '\n\n', code)
            Path(file_path).write_text(code)

    return total_removed


def _fix_missing_imports(go_output_path: str, test_output: str) -> int:
    """Deterministically add missing standard library imports.

    When Go reports 'undefined: net' or similar, and the identifier matches
    a known stdlib package used as a qualifier (e.g. net.Error), we add the
    missing import. This fixes the most common fixer oscillation pattern.

    Returns the number of imports added.
    """
    # Known Go stdlib packages that are commonly used as qualifiers
    _STDLIB_PACKAGES = {
        "net", "os", "io", "fmt", "log", "time", "sync", "sort",
        "bytes", "strings", "strconv", "errors", "context", "regexp",
        "math", "path", "filepath", "bufio", "crypto", "hash",
        "encoding", "json", "xml", "http", "url", "sql",
    }

    # Pattern: file.go:line:col: undefined: <name>
    pattern = re.compile(r'(\S+\.go):\d+:\d+:\s*undefined:\s*(\w+)')
    adds_by_file: dict[str, set[str]] = {}

    for match in pattern.finditer(test_output):
        file_ref = match.group(1)
        undefined_name = match.group(2)
        if undefined_name not in _STDLIB_PACKAGES:
            continue
        full_path = Path(go_output_path) / file_ref
        if not full_path.exists():
            for f in Path(go_output_path).rglob(Path(file_ref).name):
                if str(f).endswith(file_ref):
                    full_path = f
                    break
        if full_path.exists():
            key = str(full_path)
            if key not in adds_by_file:
                adds_by_file[key] = set()
            adds_by_file[key].add(undefined_name)

    total_added = 0
    for file_path, pkg_names in adds_by_file.items():
        code = Path(file_path).read_text()
        for pkg in pkg_names:
            import_str = f'"{pkg}"'
            # Special cases: net/http, path/filepath, encoding/json, etc.
            if pkg == "http":
                import_str = '"net/http"'
            elif pkg == "filepath":
                import_str = '"path/filepath"'
            elif pkg == "json":
                import_str = '"encoding/json"'
            elif pkg == "xml":
                import_str = '"encoding/xml"'
            elif pkg == "url":
                import_str = '"net/url"'
            elif pkg == "sql":
                import_str = '"database/sql"'

            # Skip if already imported
            if import_str in code:
                continue

            # Add to existing import block or create one
            if re.search(r'import\s*\(', code):
                code = re.sub(r'(import\s*\()', rf'\1\n\t{import_str}', code, count=1)
            else:
                # Add after package line
                code = re.sub(
                    r'(package\s+\w+\s*\n)',
                    rf'\1\nimport {import_str}\n',
                    code,
                    count=1,
                )
            total_added += 1
        if total_added:
            Path(file_path).write_text(code)

    return total_added


def _strip_all_self_imports(go_output_path: str) -> int:
    """Strip self-imports from all .go files in the output directory.

    Returns the number of files modified.
    """
    modified = 0
    for go_file in Path(go_output_path).rglob("*.go"):
        pkg_name = go_file.parent.name
        if not pkg_name:
            continue
        code = go_file.read_text()
        cleaned = _strip_self_imports(code, pkg_name)
        if cleaned != code:
            go_file.write_text(cleaned)
            modified += 1
    return modified


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_last_test_output(session: Session) -> str:
    """Read raw output from the latest test artifact."""
    test_artifact = session.get("last_test_artifact", "")
    if not test_artifact:
        return ""
    try:
        return session.read_artifact(test_artifact).decode("utf-8")
    except Exception:
        return ""


def _build_failure_signature(output: str) -> str:
    """Build a stable digest from compiler/test failure lines."""
    if not output.strip():
        return "empty"

    normalized_lines: list[str] = []
    for raw in output.splitlines():
        line = raw.strip().lower()
        if not line:
            continue
        if ".go:" in line or "undefined:" in line or "imported and not used" in line:
            line = re.sub(r":[0-9]+:[0-9]+", ":<line>:<col>", line)
            line = re.sub(r"[0-9]+", "<num>", line)
            normalized_lines.append(line)

    if not normalized_lines:
        normalized_lines = [re.sub(r"[0-9]+", "<num>", output.strip().lower())]

    normalized_lines.sort()
    payload = "\n".join(normalized_lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trailing_repeat_count(items: list[str]) -> int:
    """Count trailing repeats of the most recent value."""
    if not items:
        return 0
    last = items[-1]
    count = 0
    for value in reversed(items):
        if value != last:
            break
        count += 1
    return count


def _toolsmith_trigger_reason(
    iteration: int,
    max_iter: int,
    files_fixed: int,
    signature_history: list[str],
) -> str | None:
    """Return trigger reason for Toolsmith fallback, if any."""
    if not _env_flag("KAIZEN_TOOLSMITH_ENABLED", default=False):
        return None

    if files_fixed == 0:
        return "fixer_no_changes"

    if _trailing_repeat_count(signature_history) >= 2:
        return "repeat_signature"

    late_budget_threshold = max(1, (max_iter * 3 + 4) // 5)  # ceil(60% * max_iter)
    if iteration >= late_budget_threshold:
        return "late_budget"

    return None


def _ingest_toolsmith_artifacts(
    session: Session,
    artifact_dir: str,
    invocation_index: int,
) -> list[str]:
    """Copy selected Toolsmith artifact files into Kaizen session artifacts."""
    copied: list[str] = []
    root = Path(artifact_dir)
    if not root.exists():
        return copied

    names = [
        "final.json",
        "attempts.json",
        "baseline.log",
        "final.log",
        "summary.txt",
        "metadata.json",
        "policy.json",
        "request.json",
        "baseline_signature.json",
        "final_signature.json",
    ]
    for name in names:
        path = root / name
        if not path.exists():
            continue
        artifact_name = f"toolsmith_run_{invocation_index:02d}_{name}"
        session.write_artifact(artifact_name, path.read_bytes())
        copied.append(artifact_name)

    return copied


def _invoke_toolsmith(
    session: Session,
    iteration: int,
    max_iter: int,
    label: str,
    trigger_reason: str,
) -> dict[str, object]:
    """Run Toolsmith CLI as an optional fallback and ingest its outputs."""
    if session.get("toolsmith_unavailable", False):
        return {"invoked": False, "reason": "unavailable"}

    runs = session.get("toolsmith_runs", [])
    if not isinstance(runs, list):
        runs = []

    max_invocations = max(1, _env_int("KAIZEN_TOOLSMITH_MAX_INVOCATIONS", 2))
    if len(runs) >= max_invocations:
        return {"invoked": False, "reason": "budget_exhausted"}

    go_output = session.get("go_output_path", "")
    if not go_output:
        return {"invoked": False, "reason": "missing_workspace"}

    cmd = os.environ.get("KAIZEN_TOOLSMITH_CMD", "toolsmith")
    validator = os.environ.get(
        "KAIZEN_TOOLSMITH_VALIDATOR",
        "go build ./... && go test ./... -v",
    )
    timeout_seconds = max(30, _env_int("KAIZEN_TOOLSMITH_TIMEOUT_SECONDS", 900))
    invocation_index = len(runs) + 1

    metadata = {
        "session_id": session.session_id,
        "fix_iteration": iteration,
        "max_iterations": max_iter,
        "phase": label,
        "trigger_reason": trigger_reason,
    }
    artifact_root = str(Path(go_output) / ".toolsmith")
    run_id = f"{session.session_id[:8]}-i{iteration:02d}-ts{invocation_index:02d}"

    argv = shlex.split(cmd) + [
        "fix",
        "--workspace",
        go_output,
        "--validator",
        validator,
        "--backend",
        "auto",
        "--host",
        "kaizen",
        "--output",
        "json",
        "--run-id",
        run_id,
        "--artifact-root",
        artifact_root,
        "--metadata-json",
        json.dumps(metadata, separators=(",", ":")),
    ]

    session.append(
        agent_id="toolsmith_fallback",
        entry_type=EntryType.SYSTEM_NOTE,
        content={
            "event": "toolsmith_invoked",
            "invocation": invocation_index,
            "trigger_reason": trigger_reason,
            "run_id": run_id,
            "command": argv,
        },
    )

    try:
        proc = subprocess.run(
            argv,
            cwd=go_output,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        session.set("toolsmith_unavailable", True)
        session.append(
            agent_id="toolsmith_fallback",
            entry_type=EntryType.SYSTEM_NOTE,
            content={
                "event": "toolsmith_unavailable",
                "reason": "command_not_found",
                "command": cmd,
            },
        )
        return {"invoked": False, "reason": "command_not_found"}
    except subprocess.TimeoutExpired:
        return {"invoked": False, "reason": "timeout"}

    stdout_name = f"toolsmith_run_{invocation_index:02d}.stdout.log"
    stderr_name = f"toolsmith_run_{invocation_index:02d}.stderr.log"
    session.write_artifact(stdout_name, proc.stdout.encode("utf-8"))
    session.write_artifact(stderr_name, proc.stderr.encode("utf-8"))

    parsed_payload: dict[str, object] | None = None
    json_artifact = ""
    try:
        parsed = json.loads(proc.stdout) if proc.stdout.strip() else {}
        if isinstance(parsed, dict):
            parsed_payload = parsed
            json_artifact = f"toolsmith_run_{invocation_index:02d}.json"
            session.write_artifact(
                json_artifact,
                json.dumps(parsed_payload, indent=2, sort_keys=True).encode("utf-8"),
            )
    except json.JSONDecodeError:
        parsed_payload = None

    status = "failed"
    toolsmith_run_id = ""
    artifact_dir = ""
    copied_artifacts: list[str] = []
    if parsed_payload is not None:
        status = str(parsed_payload.get("status", "failed"))
        toolsmith_run_id = str(parsed_payload.get("run_id", ""))
        result = parsed_payload.get("result", {})
        if isinstance(result, dict):
            artifact_dir = str(result.get("artifact_dir", ""))
        if artifact_dir:
            copied_artifacts = _ingest_toolsmith_artifacts(
                session=session,
                artifact_dir=artifact_dir,
                invocation_index=invocation_index,
            )

    record = {
        "invocation": invocation_index,
        "trigger_reason": trigger_reason,
        "iteration": iteration,
        "status": status,
        "toolsmith_run_id": toolsmith_run_id,
        "exit_code": proc.returncode,
        "stdout_artifact": stdout_name,
        "stderr_artifact": stderr_name,
        "json_artifact": json_artifact,
        "copied_artifacts": copied_artifacts,
    }
    runs.append(record)
    session.set("toolsmith_runs", runs)

    session.append(
        agent_id="toolsmith_fallback",
        entry_type=EntryType.AGENT_COMPLETED,
        content={
            "event": "toolsmith_result",
            "invocation": invocation_index,
            "trigger_reason": trigger_reason,
            "status": status,
            "toolsmith_run_id": toolsmith_run_id,
            "exit_code": proc.returncode,
            "artifact_dir": artifact_dir,
            "artifacts": [stdout_name, stderr_name, json_artifact] + copied_artifacts,
        },
    )

    return {
        "invoked": True,
        "status": status,
        "run_id": toolsmith_run_id,
        "artifact_dir": artifact_dir,
        "record": record,
    }


def run_fix_loop(dispatcher: Dispatcher, session: Session, step_num: int, label: str, max_iter: int = 8) -> bool:
    """Run build/test and fix loop.

    Args:
        step_num: Step number for display
        label: Label for the step (e.g. "Source compilation fix" or "Test fix")
        max_iter: Maximum fix iterations
    """
    print_step(step_num, f"Running {label} loop")
    signature_history: list[str] = []

    for iteration in range(1, max_iter + 1):
        print(f"\n  --- Iteration {iteration}/{max_iter} ---")

        # Run tests (which includes go build + go test)
        print("  Running tests...")
        test_result = dispatcher.dispatch_single("run_tests", session, {})

        if test_result.success and session.get("tests_passed", False):
            print("  ✓ All tests passed!")
            session.save(str(SESSION_FILE))
            return True

        # Show test summary
        summary = session.get("last_test_output", "No output")
        print(f"  ✗ Tests failed: {summary[:200]}")

        raw_test_output = _read_last_test_output(session)
        signature_history.append(_build_failure_signature(raw_test_output))

        if iteration == max_iter:
            trigger_reason = _toolsmith_trigger_reason(
                iteration=iteration,
                max_iter=max_iter,
                files_fixed=0,
                signature_history=signature_history,
            )
            if trigger_reason is not None:
                ts = _invoke_toolsmith(
                    session=session,
                    iteration=iteration,
                    max_iter=max_iter,
                    label=label,
                    trigger_reason=trigger_reason,
                )
                if ts.get("invoked"):
                    ts_status = str(ts.get("status", "unknown"))
                    print(f"    ✓ Toolsmith fallback run status: {ts_status}")
                    if ts_status == "passed":
                        print("    ✓ Re-running tests after Toolsmith fallback...")
                        confirm = dispatcher.dispatch_single("run_tests", session, {})
                        if confirm.success and session.get("tests_passed", False):
                            print("  ✓ All tests passed after Toolsmith fallback!")
                            session.save(str(SESSION_FILE))
                            return True
            print(f"  ✗ Max iterations ({max_iter}) reached")
            break

        # Run fixer
        print("  Running fixer...")
        fix_result = dispatcher.dispatch_single("fix", session, {})

        files_fixed = 0
        if fix_result.success:
            files_fixed = int(fix_result.result.get("files_fixed", 0))
            print(f"    ✓ Fixed {files_fixed} file(s)")
        else:
            print(f"    ✗ Fix failed: {fix_result.error}")

        # Deterministic post-fixes
        go_output = session.get("go_output_path", "")
        if go_output:
            # Read the raw test output for deterministic fixes
            test_artifact = session.get("last_test_artifact", "")
            raw_test_output = ""
            if test_artifact:
                try:
                    raw_test_output = session.read_artifact(test_artifact).decode("utf-8")
                except Exception:
                    pass

            n_walrus = _fix_walrus_errors(go_output, raw_test_output)
            n_unused = _fix_unused_imports(go_output, raw_test_output)
            n_missing = _fix_missing_imports(go_output, raw_test_output)
            n_imports = _strip_all_self_imports(go_output)
            n_fields = _export_all_struct_fields(go_output)

            fixes = []
            if n_walrus:
                fixes.append(f"{n_walrus} :=→= fix(es)")
            if n_unused:
                fixes.append(f"{n_unused} unused import(s)")
            if n_missing:
                fixes.append(f"{n_missing} missing import(s)")
            if n_imports:
                fixes.append(f"{n_imports} self-import(s)")
            if n_fields:
                fixes.append(f"{n_fields} field export(s)")
            if fixes:
                print(f"    ✓ Post-fix: {', '.join(fixes)}")

        trigger_reason = _toolsmith_trigger_reason(
            iteration=iteration,
            max_iter=max_iter,
            files_fixed=files_fixed,
            signature_history=signature_history,
        )
        if trigger_reason is not None:
            ts = _invoke_toolsmith(
                session=session,
                iteration=iteration,
                max_iter=max_iter,
                label=label,
                trigger_reason=trigger_reason,
            )
            if ts.get("invoked"):
                ts_status = str(ts.get("status", "unknown"))
                print(f"    ✓ Toolsmith fallback run status: {ts_status}")
                if ts_status == "passed":
                    print("    ✓ Re-running tests after Toolsmith fallback...")
                    confirm = dispatcher.dispatch_single("run_tests", session, {})
                    if confirm.success and session.get("tests_passed", False):
                        print("  ✓ All tests passed after Toolsmith fallback!")
                        session.save(str(SESSION_FILE))
                        return True

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
        model_url = os.environ.get("KAIZEN_MODEL_URL")
        if model_url:
            llm: LLMProvider = OpenAICompatProvider(
                base_url=model_url,
                model=os.environ.get("KAIZEN_MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct"),
                api_key=os.environ.get("KAIZEN_API_KEY"),
                endpoint=os.environ.get("KAIZEN_ENDPOINT", ""),
                timeout=600.0,
                max_tokens=8192,
            )
        else:
            llm = OllamaProvider(model="llama3.1:8b", timeout=300.0)

        # Set up dispatcher
        dispatcher = setup_dispatcher(llm)

        # Check if we need to plan
        if not session.get("planning_complete", False):
            if not run_planning(dispatcher, session):
                return 1

        # Phase 1: Convert source files
        if session.get("status") in ("planning", "converting"):
            if not run_source_conversion(dispatcher, session, llm):
                print("\n✗ No source modules converted. Check errors above.")
                return 1

        # Phase 2: Fix source compilation
        source_ok = run_fix_loop(dispatcher, session, step_num=7, label="Source compilation fix")

        if source_ok:
            # Phase 3: Convert test files
            run_test_conversion(dispatcher, session, llm)

            # Phase 4: Fix test failures
            success = run_fix_loop(dispatcher, session, step_num=9, label="Test fix")
        else:
            success = False

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
