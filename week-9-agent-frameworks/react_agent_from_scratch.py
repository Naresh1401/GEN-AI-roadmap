"""
react_agent_from_scratch.py
============================
Implements a ReAct (Reasoning + Acting) agent from scratch without any
agent framework dependencies.

ReAct (Yao et al., 2022) interleaves reasoning traces with tool calls:
    Thought → Action → Observation → Thought → ... → Answer

The key insight: by generating "Thought:" tokens before each tool call,
the model can plan, reconsider, and chain multiple tool calls in a
coherent reasoning trace — all within a single context window.

This implementation works with any LLM API that supports chat completions.
A mock LLM is provided so you can run this without API keys.

Usage:
    python react_agent_from_scratch.py
    
    # To use with a real LLM, set your API key:
    export ANTHROPIC_API_KEY=your_key
"""

import json
import re
from typing import List, Dict, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Tool Definitions
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """
    A tool the agent can use.
    
    In production, tools call external APIs, databases, code executors.
    Here we use simple Python functions as mock implementations.
    """
    name:        str
    description: str
    function:    Callable
    
    def to_schema(self) -> Dict:
        """JSON schema for function calling (OpenAI/Anthropic format)."""
        return {
            "name":        self.name,
            "description": self.description,
        }
    
    def call(self, **kwargs) -> str:
        """Execute the tool and return string result."""
        try:
            result = self.function(**kwargs)
            return str(result)
        except Exception as e:
            return f"Tool error: {str(e)}"


# Mock tool implementations
def web_search(query: str) -> str:
    """Mock web search returning canned results."""
    results = {
        "population of France":   "France has a population of approximately 68 million (2024 estimate).",
        "area of France":         "France covers 551,695 km² — the largest country in the EU.",
        "capital of France":      "The capital of France is Paris, population ~2.1 million.",
        "eiffel tower height":    "The Eiffel Tower is 330 metres (1,083 feet) tall including the broadcast antenna.",
        "gpt-4 parameters":       "GPT-4's parameter count has not been officially disclosed. Estimates range from 1-2 trillion.",
        "transformer architecture": "Transformers use self-attention mechanisms. Introduced by Vaswani et al. in 2017.",
    }
    # Fuzzy match
    query_lower = query.lower()
    for key, val in results.items():
        if any(word in query_lower for word in key.split()):
            return val
    return f"No results found for '{query}'. Try a different query."


def calculator(expression: str) -> str:
    """Safe calculator using eval with limited namespace."""
    try:
        allowed = {"__builtins__": {}}
        import math as _math
        allowed.update({k: getattr(_math, k) for k in dir(_math) if not k.startswith('_')})
        result = eval(expression, allowed, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"


def get_current_date() -> str:
    """Return current date."""
    from datetime import date
    return f"Today is {date.today().strftime('%B %d, %Y')}."


def lookup_fact(topic: str) -> str:
    """Quick fact lookup."""
    facts = {
        "python": "Python was created by Guido van Rossum. First released in 1991.",
        "transformer": "The Transformer architecture was introduced by Vaswani et al. in 'Attention Is All You Need' (2017).",
        "backpropagation": "Backpropagation computes gradients via the chain rule. It was popularised for neural networks by Rumelhart et al. in 1986.",
        "attention":  "Attention mechanisms compute a weighted sum of values, with weights determined by query-key similarity.",
    }
    for key in facts:
        if key in topic.lower():
            return facts[key]
    return f"No quick fact available for '{topic}'."


AVAILABLE_TOOLS = [
    Tool("web_search",     "Search the web for current information. Use for recent events, statistics, or general knowledge.", lambda **kw: web_search(kw.get("query", ""))),
    Tool("calculator",     "Evaluate a mathematical expression. Pass 'expression' as a Python math expression string.", lambda **kw: calculator(kw.get("expression", ""))),
    Tool("get_date",       "Get the current date.", lambda **kw: get_current_date()),
    Tool("lookup_fact",    "Look up a specific technical fact. Pass 'topic'.", lambda **kw: lookup_fact(kw.get("topic", ""))),
]


# ---------------------------------------------------------------------------
# 2. ReAct Prompt Builder
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to the following tools:

{tool_descriptions}

To answer the user's question, use this format EXACTLY:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON object with tool arguments]

When you have enough information to answer:
Thought: [your reasoning about the final answer]
Answer: [your complete answer to the user's question]

Rules:
- Always start with a Thought
- Always provide Action Input as valid JSON
- Only use tools listed above
- After each Observation, continue with another Thought
- If a tool returns an error, try a different approach
"""

def build_react_prompt(tools: List[Tool]) -> str:
    """Build the ReAct system prompt with tool descriptions."""
    tool_descs = "\n".join(
        f"- {t.name}: {t.description}" for t in tools
    )
    return REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descs)


# ---------------------------------------------------------------------------
# 3. Output Parser
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """A parsed agent action from LLM output."""
    thought:      str
    action:       Optional[str]    = None  # Tool name, or None if "Answer"
    action_input: Optional[Dict]   = None  # Tool arguments
    answer:       Optional[str]    = None  # Final answer


def parse_react_output(text: str) -> AgentAction:
    """
    Parse the LLM's ReAct-formatted output.
    
    Handles both action outputs (Thought + Action + Action Input)
    and final answer outputs (Thought + Answer).
    """
    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Answer:|$)", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    
    # Check for final Answer
    answer_match = re.search(r"Answer:\s*(.+?)$", text, re.DOTALL)
    if answer_match:
        return AgentAction(thought=thought, answer=answer_match.group(1).strip())
    
    # Check for Action
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match  = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    
    if action_match:
        action = action_match.group(1).strip()
        try:
            action_input = json.loads(input_match.group(1)) if input_match else {}
        except json.JSONDecodeError:
            action_input = {}
        return AgentAction(thought=thought, action=action, action_input=action_input)
    
    # Fallback: treat entire output as answer
    return AgentAction(thought=thought, answer=text.strip())


# ---------------------------------------------------------------------------
# 4. Mock LLM (so demo works without API keys)
# ---------------------------------------------------------------------------

class MockReActLLM:
    """
    Mock LLM that simulates ReAct reasoning for pre-defined questions.
    Replace with a real LLM API for production use.
    """
    
    def __init__(self):
        self.step = 0
        self.current_task = None
    
    def _make_response(self, thought: str, action: str = None, 
                        action_input: dict = None, answer: str = None) -> str:
        if answer:
            return f"Thought: {thought}\nAnswer: {answer}"
        ai_json = json.dumps(action_input or {})
        return f"Thought: {thought}\nAction: {action}\nAction Input: {ai_json}"
    
    def generate(self, messages: List[Dict]) -> str:
        """Generate next ReAct step based on conversation history."""
        # Count observations so far
        observations = [m for m in messages if m["role"] == "user" and "Observation:" in m["content"]]
        n_obs = len(observations)
        
        # Get the original task
        user_messages = [m for m in messages if m["role"] == "user"]
        original_query = user_messages[0]["content"] if user_messages else ""
        
        # Simple rule-based mock responses
        if "population density" in original_query.lower() and n_obs == 0:
            return self._make_response(
                "I need the population and area of France to compute population density.",
                "web_search", {"query": "population of France"}
            )
        elif "population density" in original_query.lower() and n_obs == 1:
            return self._make_response(
                "Now I need the area of France.",
                "web_search", {"query": "area of France"}
            )
        elif "population density" in original_query.lower() and n_obs == 2:
            return self._make_response(
                "I have both values. Population=68M, Area=551695 km². Let me calculate.",
                "calculator", {"expression": "68000000 / 551695"}
            )
        elif "population density" in original_query.lower() and n_obs >= 3:
            return self._make_response(
                "The calculation gives approximately 123 people per km².",
                answer="France's population density is approximately 123 people per km² (68 million people / 551,695 km²)."
            )
        elif "eiffel tower" in original_query.lower() and n_obs == 0:
            return self._make_response(
                "I'll look up the height of the Eiffel Tower.",
                "web_search", {"query": "eiffel tower height"}
            )
        elif "eiffel tower" in original_query.lower() and n_obs >= 1:
            last_obs = observations[-1]["content"]
            return self._make_response(
                "I found the answer in the search result.",
                answer="The Eiffel Tower is 330 metres (1,083 feet) tall, including the broadcast antenna."
            )
        elif "current date" in original_query.lower() and n_obs == 0:
            return self._make_response(
                "I'll get the current date.",
                "get_date", {}
            )
        elif "current date" in original_query.lower() and n_obs >= 1:
            last_obs = observations[-1]["content"].replace("Observation:", "").strip()
            return self._make_response(
                "I retrieved the current date.",
                answer=last_obs
            )
        else:
            return self._make_response(
                "I'll search for information to answer this question.",
                "web_search", {"query": original_query[:50]}
            )


# ---------------------------------------------------------------------------
# 5. ReAct Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    ReAct agent: orchestrates the Thought → Action → Observation loop.
    
    Args:
        llm: Any callable that takes List[Dict] messages and returns str
        tools: List of Tool objects
        max_steps: Maximum number of tool calls before forcing an answer
    """
    
    def __init__(
        self,
        llm,
        tools:      List[Tool],
        max_steps:  int = 8,
        verbose:    bool = True,
    ):
        self.llm       = llm
        self.tool_map  = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.verbose   = verbose
        self.system_prompt = build_react_prompt(tools)
    
    def run(self, query: str) -> str:
        """Execute the ReAct loop for a given query."""
        messages = [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": query},
        ]
        
        if self.verbose:
            print(f"\n{'='*55}")
            print(f"Query: {query}")
            print(f"{'='*55}")
        
        for step in range(self.max_steps):
            # Generate next action
            raw_output = self.llm.generate(messages)
            messages.append({"role": "assistant", "content": raw_output})
            
            # Parse output
            parsed = parse_react_output(raw_output)
            
            if self.verbose:
                print(f"\nStep {step+1}:")
                print(f"  Thought: {parsed.thought}")
            
            # Check for final answer
            if parsed.answer:
                if self.verbose:
                    print(f"  Answer: {parsed.answer}")
                return parsed.answer
            
            # Execute tool
            if parsed.action and parsed.action in self.tool_map:
                tool    = self.tool_map[parsed.action]
                kwargs  = parsed.action_input or {}
                
                if self.verbose:
                    print(f"  Action: {parsed.action}({kwargs})")
                
                observation = tool.call(**kwargs)
                
                if self.verbose:
                    print(f"  Observation: {observation}")
                
                # Append observation to conversation
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            else:
                # Unknown tool or parsing failure
                if self.verbose:
                    print(f"  [Invalid action: {parsed.action}]")
                messages.append({
                    "role": "user",
                    "content": "Observation: Tool not found. Please try a different approach."
                })
        
        return "I was unable to answer this question within the step limit."


# ---------------------------------------------------------------------------
# 6. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("ReAct Agent Demo")
    print("=" * 55)
    
    llm   = MockReActLLM()
    agent = ReActAgent(llm, AVAILABLE_TOOLS, max_steps=8, verbose=True)
    
    queries = [
        "What is the population density of France? (Hint: population / area)",
        "How tall is the Eiffel Tower?",
        "What is today's date?",
    ]
    
    results = []
    for query in queries:
        answer = agent.run(query)
        results.append({"query": query, "answer": answer})
    
    print("\n" + "=" * 55)
    print("Summary of Results:")
    for r in results:
        print(f"\n  Q: {r['query']}")
        print(f"  A: {r['answer'][:100]}...")
    
    print("\n" + "=" * 55)
    print("✅ ReAct agent demo complete!")
    print("\nTo use with Claude API:")
    print("  import anthropic")
    print("  client = anthropic.Anthropic()")
    print("  # Use client.messages.create() with tool definitions")


# ============================================================
# SECTION 4 & 5 — Deep Dive & Interview Prep
# ============================================================
"""
WEEK 9: AGENT FRAMEWORKS — Deep Dive & Interview Prep

CONCEPTS:

1. REACT (REASONING + ACTING)
ReAct adds "Thought:" prefixes before tool calls, creating a visible reasoning trace.
The model generates thought tokens, then action tokens, then sees the observation —
and this full trace is in context for the next step. The key insight: the reasoning
trace is not just for interpretability — it conditions the next action and helps the
model plan. Models without reasoning traces make tool call choices less coherently.

2. FUNCTION CALLING / TOOL USE
Modern LLMs (GPT-4, Claude) support structured tool calls natively. You provide a
JSON schema describing each tool. The model outputs structured JSON when it wants to
call a tool (rather than embedding "Action: " in free text). The orchestrator parses
the JSON, calls the function, and appends the result. Structured tool calling is more
reliable than free-text parsing (ReAct), especially for complex argument types.

3. PLANNING AND TASK DECOMPOSITION
Complex tasks require multi-step planning: "Write a report on X" requires (1) research,
(2) outline, (3) draft, (4) refine. Agents handle this via:
- Sequential planning: decompose upfront, execute in order (brittle — early mistakes compound)
- Reactive planning: decide next step based on current context (robust but may loop)
- ReAct: hybrid — reason about what to do next, then act

4. MULTI-AGENT PATTERNS
Supervisor + Workers: a supervisor LLM manages a shared task state, routing subtasks
to specialist agents (researcher, coder, writer). Workers return structured results.
Supervisor checks completion and retries failures.
This pattern is robust to individual worker failures and allows parallelism.

5. AGENT MEMORY
In-context (short): the entire conversation history. Limited by context window (~128K tokens).
Fast but expensive for long sessions.
External (vector store): past interactions embedded and retrieved. Scalable, but fuzzy —
may miss exact facts. Good for "remember when we discussed X."
External (key-value): exact fact retrieval. "User prefers French." No ambiguity.

MISCONCEPTIONS:
1. "Agents are just chatbots with tools." Agents execute multi-step tasks autonomously,
making decisions, recovering from errors, and calling tools without human confirmation
for each step. Chatbots ask for clarification at each turn. The distinction is in
autonomy and multi-step execution.
2. "More tools = better agent." Too many tools overwhelms the model's context and creates
ambiguity about which tool to use. 5-10 well-defined, clearly documented tools typically
outperform 50 overlapping tools. Tool design (clear names, descriptions, examples)
is as important as model capability.
3. "Agents are reliable enough for autonomous production deployment." As of 2024,
LLM agents make mistakes, get stuck in loops, and can hallucinate tool calls.
Production deployments use human-in-the-loop for high-stakes actions, automated
tests for common paths, and retry limits with graceful fallback.

INTERVIEW QUESTIONS (5):

Q1: What is the difference between ReAct and function calling?
A: Both allow LLMs to use tools. ReAct embeds tool calls as free text within the
LLM's output, requiring a parser (regex/rules) to extract the action. Function
calling (OpenAI, Anthropic) extends the model to output structured JSON for tool
calls as a first-class capability, separate from the text output. Function calling
is more reliable (schema validation, structured output) but requires model-side support.
ReAct can be applied to any instruction-following model with good formatting fidelity.

Q2: How do you prevent infinite loops in an agent?
A: (1) Step limit: hard cap on the number of tool calls per session. (2) Observation
deduplication: if the agent calls the same tool with the same arguments twice, detect
this and inject a meta-observation ("You've already tried this — try a different approach").
(3) Progress check: after N steps, inject a meta-prompt asking the agent whether it
has enough information to answer. (4) Tool call validation: reject structurally invalid
tool calls immediately rather than letting the agent spiral.

Q3: How would you evaluate the quality of an agent's tool usage?
A: (1) Task success rate: did the agent achieve the goal? (2) Tool efficiency: did it
use the minimum number of tool calls? (3) Tool relevance: were called tools appropriate
for the subtask? (4) Error recovery: when a tool fails, does the agent try a reasonable
alternative? For benchmark evaluation, use environments with known ground truth (e.g.,
WebArena, AgentBench) and measure success rate + efficiency jointly.

Q4: What is the difference between single-agent and multi-agent systems?
A: Single-agent: one LLM orchestrates all reasoning and tool calls. Simpler, lower latency,
no coordination overhead. Bottlenecked by one context window.
Multi-agent: multiple specialised LLM instances. Can run in parallel (faster for
independent subtasks), each with a focused context window (better for specialised domains).
Requires a coordination protocol (supervisor or explicit message passing).
Multi-agent is not always better — the communication overhead can outweigh the benefits
for simple tasks.

Q5: How does LangGraph differ from LangChain for agent development?
A: LangChain provides tool wrappers, prompt templates, and sequential chains. It's
imperative: you define a linear pipeline. LangGraph models agents as state machines
(directed graphs): nodes are LLM or tool calls, edges are transitions. The agent
state is a typed dataclass. This enables: cycles (retry loops), branching (conditional
routing), parallel execution (multiple nodes at once), and persistence (pause/resume).
LangGraph is more expressive and production-ready for complex, non-linear agent workflows.

SYSTEM DESIGN: Design an autonomous research assistant agent for writing technical reports.
Architecture: Supervisor agent orchestrates; Researcher agent (web search, arxiv),
Writer agent (drafting), Reviewer agent (fact-checking, citations).
State: report sections (dict), sources (list), current task, revision count.
Tools: web_search, arxiv_search, extract_text (scrape URL), write_section, check_facts.
Memory: source documents in vector store; report draft in state.
Safeguards: max 20 tool calls per report; human approval before final publish;
citation verification (every claim traced to a source); time limit (5 min max).
Monitoring: per-agent token usage, tool call success rate, fact-check pass rate.
"""
