# Python → Go Mapping

This document defines the mapping between Python source files and their
corresponding Go implementations. The conversion follows Go conventions:
- Package per directory
- Exported names start with uppercase
- Interfaces for protocols
- Structs for data classes

## Core Types

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `src/kaizen/types.py` | `types/types.go` | EntryType, ErrorCode enums → const blocks; dataclasses → structs |

## Session

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `src/kaizen/session.py` | `session/session.go` | Session class → Session struct with methods |

## Agent Protocol

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `src/kaizen/agent.py` | `agent/agent.go` | ABC → interface; helper methods as package functions |

## Dispatcher

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `src/kaizen/dispatcher.py` | `dispatcher/dispatcher.go` | Dispatcher class → Dispatcher struct |

## Planner

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `src/kaizen/planner.py` | `planner/planner.go` | Planner class → Planner struct |

## LLM Providers

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `src/kaizen/llm/base.py` | `llm/provider.go` | LLMProvider ABC → LLMProvider interface |
| `src/kaizen/llm/ollama.py` | `llm/ollama.go` | OllamaProvider → OllamaProvider struct |

## Built-in Agents

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `src/kaizen/agents/reverse.py` | `agents/reverse.go` | ReverseAgent implementation |
| `src/kaizen/agents/uppercase.py` | `agents/uppercase.go` | UppercaseAgent implementation |

## Tests — Shared Helpers

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `tests/conftest.py` | `testutil/testutil.go` | Shared pytest fixtures → Go test helper functions |

## Tests — Core Types

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `tests/test_types.py` | `types/types_test.go` | EntryType, InvokeResult, CapabilityCall, AgentInfo tests |

## Tests — Session

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `tests/test_session_state.py` | `session/state_test.go` | State get/set, versioning, isolation |
| `tests/test_session_trajectory.py` | `session/trajectory_test.go` | Trajectory append, ordering, limits |
| `tests/test_session_artifacts.py` | `session/artifacts_test.go` | Artifact write/read/list/size |
| `tests/test_session_snapshots.py` | `session/snapshots_test.go` | Agent snapshot views |
| `tests/test_persistence.py` | `session/persistence_test.go` | SQLite save/load roundtrip |

## Tests — Agents and Dispatcher

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `tests/test_agents.py` | `agents/agents_test.go` | ReverseAgent and UppercaseAgent tests |
| `tests/test_dispatcher.py` | `dispatcher/dispatcher_test.go` | Dispatcher registration and dispatch tests |

## Tests — LLM

| Python Source | Go Target | Notes |
|--------------|-----------|-------|
| `tests/test_llm.py` | `llm/llm_test.go` | LLM provider interface and mock tests |

## Conversion Order

The recommended conversion order (dependencies first):

**Source (Phase 1):**
1. `types/types.go` - Core types with no internal dependencies
2. `session/session.go` - Uses types
3. `agent/agent.go` - Uses types, references Session
4. `llm/provider.go` - LLM interface, uses types
5. `llm/ollama.go` - Implements LLM interface
6. `dispatcher/dispatcher.go` - Uses agent, types, session
7. `planner/planner.go` - Uses llm, types, session
8. `agents/reverse.go` - Implements Agent interface
9. `agents/uppercase.go` - Implements Agent interface

**Tests (Phase 2 — after source compiles):**
10. `testutil/testutil.go` - Shared test helpers
11. `types/types_test.go` - Type tests
12. `session/state_test.go` - Session state tests
13. `session/trajectory_test.go` - Trajectory tests
14. `session/artifacts_test.go` - Artifact tests
15. `session/snapshots_test.go` - Snapshot tests
16. `session/persistence_test.go` - Persistence tests
17. `agents/agents_test.go` - Agent tests
18. `dispatcher/dispatcher_test.go` - Dispatcher tests
19. `llm/llm_test.go` - LLM tests

## Go Idioms to Apply

- Use `error` return values instead of exceptions
- Use `context.Context` for cancellation where appropriate
- Use `encoding/json` for serialization
- Use `database/sql` with `modernc.org/sqlite` for SQLite (pure Go)
- Use `time.Time` with UTC for timestamps
- Use pointer receivers for methods that modify state
- Use value receivers for read-only methods

## Go Testing Idioms

- Use `func TestXxx(t *testing.T)` pattern
- Use `t.Run("subtest", func(t *testing.T) { ... })` for subtests
- Use `t.Errorf` / `t.Fatalf` instead of assert
- Use `t.TempDir()` for temporary directories
- Use `t.Helper()` in test helper functions
- Convert `pytest.raises(XError)` to checking `err != nil` and error types
- Convert pytest fixtures to helper functions returning test data
