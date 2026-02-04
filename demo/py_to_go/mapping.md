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

## Conversion Order

The recommended conversion order (dependencies first):

1. `types/types.go` - Core types with no internal dependencies
2. `session/session.go` - Uses types
3. `agent/agent.go` - Uses types, references Session
4. `llm/provider.go` - LLM interface, uses types
5. `llm/ollama.go` - Implements LLM interface
6. `dispatcher/dispatcher.go` - Uses agent, types, session
7. `planner/planner.go` - Uses llm, types, session
8. `agents/reverse.go` - Implements Agent interface
9. `agents/uppercase.go` - Implements Agent interface

## Go Idioms to Apply

- Use `error` return values instead of exceptions
- Use `context.Context` for cancellation where appropriate
- Use `encoding/json` for serialization
- Use `database/sql` with `modernc.org/sqlite` for SQLite (pure Go)
- Use `time.Time` with UTC for timestamps
- Use pointer receivers for methods that modify state
- Use value receivers for read-only methods
