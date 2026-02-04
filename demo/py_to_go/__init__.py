"""
Python to Go Conversion Demo.

This demo uses Kaizen to orchestrate a Python-to-Go port of the Kaizen
codebase itself, demonstrating:
- Session state management
- Agent orchestration via Dispatcher
- Trajectory logging for auditability
- Artifact storage for diffs and logs
- LLM-powered code conversion

The demo uses temporary directories to avoid mutating the original repo.
"""
