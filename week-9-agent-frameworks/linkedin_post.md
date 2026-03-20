# Week 9 LinkedIn Post: Agent Frameworks

---

I gave a language model a calculator, a web search tool, and a code executor this week.

Then I watched it get confused, recover, and eventually solve a problem I didn't expect it to solve.

**Week 9: Agent Frameworks — ReAct, tool use, and multi-agent systems.**

---

**💡 Aha #1 — ReAct is the simplest possible agent loop**

ReAct (Reasoning + Acting) interleaves reasoning traces with tool calls in the same generation:

```
Thought: I need to find the current population of France.
Action: search("France population 2024")
Observation: France has approximately 68 million people.
Thought: Now I can calculate the population density.
Action: calculator("68000000 / 551695")
Observation: 123.3
Answer: France's population density is approximately 123 people per km².
```

The model generates "Thought" and "Action" tokens just like any other text. The orchestrator intercepts "Action" lines, executes the tool, and appends "Observation" to the context. Simple prompt engineering + output parsing = agent.

---

**💡 Aha #2 — Tools are just function schemas**

Modern function calling (GPT-4, Claude) works by providing the model with JSON schemas describing available tools:

```python
tools = [{
    "name": "search_web",
    "description": "Search the web for current information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}]
```

The model outputs structured JSON when it wants to call a tool. The framework executes it and feeds results back. The entire agent loop is just: generate → parse → execute → append → repeat.

---

**💡 Aha #3 — Multi-agent systems fail gracefully (or catastrophically)**

Multi-agent systems decompose complex tasks across specialised agents: a planner, a researcher, a coder, a critic. The key challenge: agents can get stuck in loops, generate conflicting information, or amplify early errors across the pipeline.

The most robust multi-agent pattern I found: **supervisor + workers with explicit state**. The supervisor maintains a shared task state and routes sub-tasks to specialised workers. Workers report structured results back. If a worker fails, the supervisor can retry with a different worker or strategy.

---

The thing that surprised me most: agents are not more powerful than the underlying model — they're a way of applying the model's capability *iteratively*. A task that's too complex for one pass becomes tractable as a sequence of simpler steps. That's both the power and the limitation.

**Do you think we're in the "rule-based chatbot to LLM" transition point for agents — heading toward agents that actually complete complex, long-horizon tasks reliably? Or is reliable agency a fundamentally harder problem?**

---

#AIAgents #LangChain #ReAct #FunctionCalling #MultiAgent #LLM #GenerativeAI #BuildingInPublic #10WeekChallenge
