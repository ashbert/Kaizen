# Kaizen

**Agentic Session Substrate** — A minimal, deterministic substrate for agent workflows.

> A session is an **append-only trajectory + versioned state + artifacts**.

Kaizen provides infrastructure for auditable, reproducible, and resumable agent workflows.

## Installation

```bash
pip install -e ".[dev]"
```

For LLM-based planning:
```bash
brew install ollama && ollama pull llama3.1:8b && ollama serve
```

## Quick Start

```python
from kaizen import Session, Dispatcher
from kaizen.agents import ReverseAgent, UppercaseAgent

session = Session()
session.set("text", "hello world")

dispatcher = Dispatcher()
dispatcher.register(ReverseAgent())
dispatcher.register(UppercaseAgent())

dispatcher.dispatch_single("reverse", session, {"key": "text"})
dispatcher.dispatch_single("uppercase", session, {"key": "text"})

print(session.get("text"))  # "DLROW OLLEH"

session.save("my_session.kaizen")
```

## Core Components

### Session
Central data structure: versioned state + append-only trajectory + artifact storage.

```python
session = Session()
session.set("key", "value")           # State
session.append(agent_id, type, data)  # Trajectory
session.write_artifact("file", data)  # Artifacts
session.save("session.kaizen")        # Persistence
```

### Agent
Callable unit with declared capabilities:

```python
class MyAgent(Agent):
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="my_agent_v1",
            capabilities=["my_capability"],
            ...
        )

    def invoke(self, capability, session, params) -> InvokeResult:
        # Read state, do work, write state, return result
        ...
```

### Dispatcher
Routes capability calls to registered agents:

```python
dispatcher = Dispatcher()
dispatcher.register(MyAgent())
dispatcher.dispatch_single("my_capability", session, params)
```

### Planner
LLM-powered natural language to capability calls:

```python
from kaizen import Planner
from kaizen.llm import OllamaProvider

planner = Planner(OllamaProvider(), capabilities=["reverse", "uppercase"])
plan = planner.plan("Reverse and uppercase the text", session)
dispatcher.dispatch_sequence(plan.calls, session)
```

## Demo: Python-to-Go Conversion

The `demo/py_to_go/` directory contains a self-hosted demo that uses Kaizen to orchestrate LLM-powered conversion of the Kaizen codebase from Python to Go.

```bash
python demo/py_to_go/run_demo.py
```

## Tests

```bash
pytest                    # All tests
pytest -m "not integration"  # Skip Ollama-dependent tests
```

## License

MIT
