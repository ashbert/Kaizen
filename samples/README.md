# Kaizen Samples

This directory contains example scripts demonstrating how to use Kaizen.

## Prerequisites

```bash
# From the project root directory
cd kaizen

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install kaizen in development mode
pip install -e ".[dev]"
```

For samples that use the LLM planner (Sample 03), you also need Ollama:

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull the default model (in another terminal)
ollama pull llama3.1:8b
```

## Running Samples

```bash
# Run from the project root directory
python samples/01_basic_session.py
python samples/02_agents_and_dispatcher.py
python samples/03_complete_workflow.py  # Requires Ollama
python samples/04_custom_agent.py
python samples/05_artifacts.py
```

## Sample Descriptions

### 01_basic_session.py
**No external dependencies**

Demonstrates fundamental Session operations:
- Creating a session
- Getting and setting state
- State versioning
- Saving and loading sessions

### 02_agents_and_dispatcher.py
**No external dependencies**

Demonstrates agents and the dispatcher:
- Using built-in agents (ReverseAgent, UppercaseAgent)
- Registering agents with the Dispatcher
- Executing single and multiple capability calls
- Error handling with fail-fast behavior

### 03_complete_workflow.py
**Requires Ollama**

Demonstrates the full Kaizen workflow:
1. Create a session with initial state
2. Set up agents and dispatcher
3. Use LLM planner to convert natural language to capability calls
4. Execute the plan
5. Save and resume the session
6. Inspect the trajectory

### 04_custom_agent.py
**No external dependencies**

Demonstrates how to create custom agents:
- Implementing the Agent protocol
- Defining capabilities
- Reading and writing session state
- Recording actions in the trajectory
- Proper error handling

### 05_artifacts.py
**No external dependencies**

Demonstrates artifact management:
- Writing binary artifacts to a session
- Reading artifacts back
- Listing artifacts
- Artifacts surviving save/load cycles
- Size limits

## Quick Reference

### Session

```python
from kaizen import Session

session = Session()
session.set("key", "value")
value = session.get("key")
session.save("session.kaizen")
restored = Session.load("session.kaizen")
```

### Dispatcher

```python
from kaizen import Dispatcher
from kaizen.agents import ReverseAgent

dispatcher = Dispatcher()
dispatcher.register(ReverseAgent())
result = dispatcher.dispatch_single("reverse", session, {"key": "text"})
```

### Planner (requires Ollama)

```python
from kaizen import Planner
from kaizen.llm import OllamaProvider

llm = OllamaProvider()
planner = Planner(llm, capabilities=["reverse", "uppercase"])
plan = planner.plan("reverse and uppercase the text", session)
```

### Custom Agent

```python
from kaizen import Agent, AgentInfo, InvokeResult

class MyAgent(Agent):
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="my_agent_v1",
            name="My Agent",
            version="1.0.0",
            capabilities=["my_cap"],
        )

    def invoke(self, capability, session, params) -> InvokeResult:
        # Implementation here
        return InvokeResult.ok(result={}, agent_id="my_agent_v1", capability=capability)
```
