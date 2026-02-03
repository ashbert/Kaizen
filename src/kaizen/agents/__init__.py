"""
Built-in agents for Kaizen.

This module contains example/toy agents that demonstrate the Agent protocol
and can be used for testing the session substrate.

Available agents:
- ReverseAgent: Reverses text stored in session state
- UppercaseAgent: Converts text in session state to uppercase

These agents serve as reference implementations for building custom agents.
"""

from kaizen.agents.reverse import ReverseAgent
from kaizen.agents.uppercase import UppercaseAgent

__all__ = ["ReverseAgent", "UppercaseAgent"]
