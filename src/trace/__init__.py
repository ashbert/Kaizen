"""
Trace: Agentic Session Substrate

A minimal, deterministic substrate for agent workflows built around a single core idea:
A session is an append-only trajectory + versioned state + artifacts.

This module provides the core building blocks for:
- Session management (state, trajectory, artifacts)
- Agent protocol and execution
- Dispatcher for sequential capability execution
- Planner for LLM-based workflow planning

Example usage:
    from trace import Session, Dispatcher
    from trace.agents import ReverseAgent, UppercaseAgent

    # Create a new session
    session = Session()

    # Register agents with dispatcher
    dispatcher = Dispatcher()
    dispatcher.register(ReverseAgent())
    dispatcher.register(UppercaseAgent())

    # Execute capability calls
    calls = [
        {"capability": "reverse", "params": {"text": "hello"}},
        {"capability": "uppercase", "params": {}},
    ]
    results = dispatcher.dispatch_sequence(calls, session)

    # Save session for later
    session.save("my_session.trace")

For more information, see the project documentation.
"""

__version__ = "0.1.0"

# Core exports will be added as modules are implemented:
# - Session (session.py)
# - Agent, AgentInfo, InvokeResult (agent.py)
# - Dispatcher (dispatcher.py)
# - Planner (planner.py)
