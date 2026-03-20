# Week 9: Agent Frameworks

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

Agents apply LLMs iteratively, using tool calls to acquire information and take actions. This week covers the ReAct pattern, function calling, multi-agent architectures, and LangGraph.

**Core concepts:**
- ReAct (Reasoning + Acting) agent loop
- Tool use / function calling
- Planning and task decomposition
- Multi-agent coordination: supervisor + worker patterns
- Memory: in-context, external, and episodic

---

## Architecture Diagrams

### ReAct Agent Loop

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  LLM: Generate Thought + Action              │
│                                              │
│  Thought: "I need to look up X"              │
│  Action: search("X")                         │
└───────────────────┬─────────────────────────┘
                    │ (parse action)
                    ▼
           Tool Executor
           (search / code / API)
                    │
                    ▼
           Observation: "X is..."
                    │
                    ▼ (append to context)
           [Loop until Action=Answer]
                    │
                    ▼
             Final Answer
```

### Multi-Agent Supervisor Pattern

```
User Task
    │
    ▼
┌──────────────────────────────┐
│         SUPERVISOR           │
│  - Breaks task into subtasks │
│  - Routes to specialist      │
│  - Aggregates results        │
│  - Checks completion         │
└───┬──────────┬───────────────┘
    │          │          │
    ▼          ▼          ▼
┌───────┐  ┌──────┐  ┌────────┐
│Researcher│ │Coder │  │ Writer │
│  Agent │  │ Agent│  │  Agent │
└───────┘  └──────┘  └────────┘
```

---

## Setup

```bash
pip install langchain langchain-community openai anthropic
```

---

## Running the Implementations

```bash
python react_agent_from_scratch.py    # ReAct loop from scratch
python multi_agent_system.py          # Supervisor + worker pattern
```

---

## Key Takeaways

| Concept | Core Insight |
|---|---|
| ReAct loop | Thought → Action → Observation, repeated until Answer |
| Function calling | Model outputs structured JSON; orchestrator executes |
| Tool design | Tools should be narrow, deterministic, well-documented |
| Multi-agent | Decompose complex tasks; route subtasks to specialists |
| Memory | In-context (fast, limited), vector store (scalable, fuzzy), KV store (exact) |

---

## References

- [ReAct: Synergizing Reasoning and Acting in LLMs](https://arxiv.org/abs/2210.03629) — Yao et al., 2022
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Schick et al., 2023
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) — Wu et al., 2023
