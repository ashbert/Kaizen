# Trace

**Agentic Session Substrate** — A minimal, deterministic substrate for agent workflows.

## Core Idea

> A session is an **append-only trajectory + versioned state + artifacts**.

Trace provides the foundational infrastructure for building auditable, reproducible, and resumable agent workflows. Every action is recorded, every state change is versioned, and sessions can be saved to disk and resumed later.

## Key Properties

| Property | Description |
|----------|-------------|
| **Auditability** | Every action is recorded in an append-only trajectory with timestamps and agent attribution |
| **Reproducibility** | Sessions can be replayed deterministically from saved files |
| **Resume-after-failure** | Sessions persist to disk and can be restored at any point |
| **Simple orchestration** | Sequential capability dispatch with fail-fast error handling |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd trace

# Create virtual environment (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

For LLM-based planning, install and run [Ollama](https://ollama.ai/):

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the default model
ollama pull llama3.1:8b

# Start Ollama server
ollama serve
```

## Quick Start

```python
from trace import Session, Dispatcher
from trace.agents import ReverseAgent, UppercaseAgent

# Create a session
session = Session()
session.set("text", "hello world")

# Set up dispatcher with agents
dispatcher = Dispatcher()
dispatcher.register(ReverseAgent())
dispatcher.register(UppercaseAgent())

# Execute capabilities
dispatcher.dispatch_single("reverse", session, {"key": "text"})
dispatcher.dispatch_single("uppercase", session, {"key": "text"})

print(session.get("text"))  # "DLROW OLLEH"

# Save for later
session.save("my_session.trace")

# Resume anytime
restored = Session.load("my_session.trace")
print(restored.get("text"))  # "DLROW OLLEH"
```

## Architecture

### Session

The **Session** is the central data structure containing:

- **State**: Versioned key-value store (any JSON-serializable value)
- **Trajectory**: Append-only log of all actions and events
- **Artifacts**: Binary blob storage (files, images, etc.)

```python
session = Session()

# State management
session.set("input", "hello")
value = session.get("input")
version = session.get_state_version()

# Trajectory
session.append("my_agent", EntryType.AGENT_COMPLETED, {"result": "done"})
entries = session.get_trajectory(limit=10)

# Artifacts
session.write_artifact("output.txt", b"result data")
data = session.read_artifact("output.txt")

# Persistence
session.save("session.trace")
restored = Session.load("session.trace")
```

### Agent

**Agents** are callable units with declared capabilities:

```python
from trace import Agent, AgentInfo, InvokeResult
from trace.types import EntryType

class MyAgent(Agent):
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="my_agent_v1",
            name="My Agent",
            version="1.0.0",
            capabilities=["my_capability"],
            description="Does something useful",
        )

    def invoke(self, capability, session, params) -> InvokeResult:
        if capability != "my_capability":
            return self._unknown_capability(capability)

        # Read state
        value = session.get(params.get("key", "text"))

        # Do work
        result = do_something(value)

        # Write state
        session.set(params["key"], result)

        # Record in trajectory
        session.append(
            self.info().agent_id,
            EntryType.AGENT_COMPLETED,
            {"input": value, "output": result},
        )

        return InvokeResult.ok(
            result={"output": result},
            agent_id=self.info().agent_id,
            capability=capability,
        )
```

### Dispatcher

The **Dispatcher** routes capability calls to agents:

```python
from trace import Dispatcher
from trace.types import CapabilityCall

dispatcher = Dispatcher()
dispatcher.register(ReverseAgent())
dispatcher.register(UppercaseAgent())

# Execute a sequence of calls
calls = [
    CapabilityCall("reverse", {"key": "text"}),
    CapabilityCall("uppercase", {"key": "text"}),
]
result = dispatcher.dispatch_sequence(calls, session)

if result.success:
    print("All calls succeeded!")
else:
    print(f"Failed at step {result.failed_at}: {result.error}")
```

### Planner

The **Planner** uses an LLM to convert natural language to capability calls:

```python
from trace import Planner
from trace.llm import OllamaProvider

# Create planner with LLM
llm = OllamaProvider(model="llama3.1:8b")
planner = Planner(llm, capabilities=["reverse", "uppercase"])

# Generate plan from natural language
plan = planner.plan("Reverse the text and make it uppercase", session)

if plan.success:
    print(f"Generated {len(plan.calls)} capability calls")
    dispatcher.dispatch_sequence(plan.calls, session)
```

## Complete Workflow Example

```python
from trace import Session, Dispatcher, Planner
from trace.agents import ReverseAgent, UppercaseAgent
from trace.llm import OllamaProvider

# 1. Create session with initial state
session = Session()
session.set("text", "hello world")

# 2. Set up agents and dispatcher
dispatcher = Dispatcher()
dispatcher.register(ReverseAgent())
dispatcher.register(UppercaseAgent())

# 3. Plan using LLM
llm = OllamaProvider()
planner = Planner(llm, capabilities=dispatcher.get_capabilities())
plan = planner.plan("Reverse this text and then uppercase it", session)

# 4. Execute the plan
if plan.success:
    result = dispatcher.dispatch_sequence(plan.calls, session)
    print(f"Result: {session.get('text')}")  # "DLROW OLLEH"

# 5. Save session
session.save("workflow.trace")

# 6. Later: Resume and inspect
restored = Session.load("workflow.trace")
for entry in restored.get_trajectory():
    print(f"[{entry.seq_num}] {entry.agent_id}: {entry.entry_type.value}")
```

## Trajectory & Auditability

Every action is recorded in the trajectory:

```python
session = Session()
session.set("x", 1)
session.set("x", 2)

for entry in session.get_trajectory():
    print(f"{entry.seq_num}: {entry.entry_type.value} by {entry.agent_id}")
    print(f"   Content: {entry.content}")
    print(f"   Time: {entry.timestamp}")
```

Output:
```
1: session_created by system
   Content: {'session_id': '...', 'max_artifact_size': 104857600, 'schema_version': 1}
   Time: 2024-01-15 10:30:00+00:00
2: state_set by system
   Content: {'key': 'x', 'old_value': None, 'new_value': 1, 'state_version': 1}
   Time: 2024-01-15 10:30:01+00:00
3: state_set by system
   Content: {'key': 'x', 'old_value': 1, 'new_value': 2, 'state_version': 2}
   Time: 2024-01-15 10:30:02+00:00
```

## Snapshots

Agents receive isolated snapshots that cannot modify the session:

```python
snapshot = session.snapshot_for_agent("my_agent", depth=10)

# Snapshot contains:
# - session_id
# - state (deep copy)
# - state_version
# - trajectory (recent entries as dicts)
# - artifacts (list of names)
# - snapshot_time
```

## Persistence

Sessions are stored in SQLite for portability:

```python
# Save
session.save("my_session.trace")

# Load
restored = Session.load("my_session.trace")

# All data is preserved:
assert restored.session_id == session.session_id
assert restored.get("key") == session.get("key")
assert restored.get_state_version() == session.get_state_version()
```

## Error Handling

The dispatcher stops on first failure (fail-fast):

```python
result = dispatcher.dispatch_sequence(calls, session)

if not result.success:
    print(f"Failed at step {result.failed_at}")
    print(f"Error code: {result.error['error_code']}")
    print(f"Message: {result.error['message']}")

    # Partial results are available
    for i, r in enumerate(result.results):
        status = "✓" if r.success else "✗"
        print(f"  {status} Step {i}: {r.capability}")
```

## Configuration

### Artifact Size Limits

```python
# Default: 100MB
session = Session(max_artifact_size=100 * 1024 * 1024)

# Custom limit
session = Session(max_artifact_size=10 * 1024 * 1024)  # 10MB
```

### LLM Configuration

```python
from trace.llm import OllamaProvider

# Default (localhost, llama3.1:8b)
llm = OllamaProvider()

# Custom model
llm = OllamaProvider(model="mistral:7b")

# Remote server
llm = OllamaProvider(
    base_url="http://gpu-server:11434",
    model="llama3.1:70b",
    timeout=300,
)
```

## Project Structure

```
trace/
├── src/trace/
│   ├── __init__.py       # Package exports
│   ├── types.py          # Core types (TrajectoryEntry, InvokeResult, etc.)
│   ├── session.py        # Session class
│   ├── agent.py          # Agent protocol
│   ├── dispatcher.py     # Dispatcher
│   ├── planner.py        # LLM-based planner
│   ├── agents/           # Built-in agents
│   │   ├── reverse.py
│   │   └── uppercase.py
│   └── llm/              # LLM providers
│       ├── base.py
│       └── ollama.py
├── tests/                # Test suite (324 tests)
├── samples/              # Example scripts
└── pyproject.toml
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trace

# Run specific test file
pytest tests/test_session_state.py

# Skip integration tests (no Ollama required)
pytest -m "not integration"
```

## Design Philosophy

From the specification:

> Optimize for **clarity over cleverness**, **determinism over throughput**, and **extendability over completeness**.

## What's NOT in V1

The following are explicitly deferred to future versions:

- Distributed sessions
- Automatic parallelization
- Logical clocks
- VM isolation
- LLM-based compaction
- Pessimistic locking
- Checkpoints
- Remote agents

## License

MIT
