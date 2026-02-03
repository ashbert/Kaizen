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
    from trace import Session, Dispatcher, Planner
    from trace.agents import ReverseAgent, UppercaseAgent
    from trace.llm import OllamaProvider

    # Create a new session
    session = Session()
    session.set("text", "hello world")

    # Register agents with dispatcher
    dispatcher = Dispatcher()
    dispatcher.register(ReverseAgent())
    dispatcher.register(UppercaseAgent())

    # Plan using LLM
    llm = OllamaProvider()
    planner = Planner(llm, capabilities=dispatcher.get_capabilities())
    plan = planner.plan("reverse the text and uppercase it", session)

    # Execute the plan
    result = dispatcher.dispatch_sequence(plan.calls, session)

    # Save session for later
    session.save("my_session.trace")

    # Resume later
    restored = Session.load("my_session.trace")

For more information, see the project documentation.
"""

__version__ = "0.1.0"

# Core types
from trace.types import (
    TrajectoryEntry,
    InvokeResult,
    CapabilityCall,
    AgentInfo,
    EntryType,
    ErrorCode,
)

# Session
from trace.session import Session

# Agent protocol
from trace.agent import Agent, AgentProtocol

# Dispatcher
from trace.dispatcher import Dispatcher, DispatchResult

# Planner
from trace.planner import Planner, PlanResult

__all__ = [
    # Version
    "__version__",
    # Types
    "TrajectoryEntry",
    "InvokeResult",
    "CapabilityCall",
    "AgentInfo",
    "EntryType",
    "ErrorCode",
    # Session
    "Session",
    # Agent
    "Agent",
    "AgentProtocol",
    # Dispatcher
    "Dispatcher",
    "DispatchResult",
    # Planner
    "Planner",
    "PlanResult",
]
