# Trace

**Agentic Session Substrate** - A minimal, deterministic substrate for agent workflows.

## Core Concept

A session is an **append-only trajectory + versioned state + artifacts**.

Trace focuses on:
- **Auditability**: Every action is recorded with full attribution
- **Reproducibility**: Sessions can be replayed deterministically
- **Resume-after-failure**: Sessions are persistent and resumable
- **Simple multi-agent orchestration**: Sequential capability dispatch

## Installation

```bash
pip install trace
```

For development:

```bash
git clone <repo>
cd trace
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

# Execute capabilities in sequence
results = dispatcher.dispatch_sequence([
    {"capability": "reverse", "params": {"key": "text"}},
    {"capability": "uppercase", "params": {"key": "text"}},
], session)

# Save session for later
session.save("my_session.trace")

# Resume later
restored = Session.load("my_session.trace")
```

## Architecture

### Session
The unit of execution and persistence. Contains:
- **State**: Versioned key-value store (any JSON value)
- **Trajectory**: Append-only log of all actions
- **Artifacts**: Binary blob storage

### Agent
Callable unit with declared capabilities. Agents can:
- Read/write session state
- Append entries to trajectory
- Store/retrieve artifacts

### Dispatcher
Routes capability calls to registered agents and executes them sequentially.

### Planner
Uses an LLM to convert user input into an ordered list of capability calls.

## License

MIT
