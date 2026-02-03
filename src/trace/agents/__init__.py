"""
Built-in agents for Trace.

This module contains example/toy agents that demonstrate the Agent protocol
and can be used for testing the session substrate.

Available agents:
- ReverseAgent: Reverses text stored in session state
- UppercaseAgent: Converts text in session state to uppercase

These agents serve as reference implementations for building custom agents.
"""

from trace.agents.reverse import ReverseAgent
from trace.agents.uppercase import UppercaseAgent

__all__ = ["ReverseAgent", "UppercaseAgent"]
