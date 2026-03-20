# 🔷 SECTION 1 — LinkedIn Post: Week 3 — Prompt Engineering & RAG

---

A language model that can't remember anything beyond its context window just got a long-term memory upgrade.

That's what I built this week: a Retrieval-Augmented Generation (RAG) pipeline that lets a 7B parameter model answer questions about documents it has never seen during training.

**Week 3 of my 10-week GenAI roadmap: Prompt Engineering & RAG.**

---

**💡 Aha #1 — Few-shot prompting is in-context gradient descent (sort of)**

Before RAG, I spent time with the fundamentals of prompt engineering. The one that stopped me cold:

Few-shot prompting works not because the model "learns" from the examples in the prompt — it can't update its weights. It works because the examples shift the *distribution* of likely completions.

```
# Zero-shot: relies entirely on pretrained knowledge
"Classify sentiment: 'The movie was dull.' → "

# Few-shot: examples constrain the output distribution
"positive: 'Amazing film!'
negative: 'Completely boring.'
Classify: 'The movie was dull.' → "
```

Show the model the pattern once, and it pattern-matches to it. Show it five times, and it virtually can't deviate. This is why formatting your examples consistently matters more than the specific examples themselves.

---

**💡 Aha #2 — Chain-of-thought is scratchpad memory, not reasoning**

I implemented chain-of-thought (CoT) prompting and benchmarked it against direct answering on multi-step math. CoT improved accuracy by ~30%.

But here's the uncomfortable truth: the model doesn't *reason* step by step. It *generates tokens that look like* step-by-step reasoning, and those intermediate tokens become context that steers subsequent tokens toward a correct answer.

```python
# Direct: model must retrieve answer in one pass
prompt = "If a bat costs $1.10 more than a ball, and together cost $1.20, what does the ball cost?"

# CoT: intermediate tokens guide the generation
prompt = """... Let me think step by step.
Total = $1.20. Let ball = x. Bat = x + 1.10.
x + (x + 1.10) = 1.20 → 2x = 0.10 → x = $0.05"""
```

The scratchpad matters. It's not window dressing.

---

**💡 Aha #3 — RAG is a retrieval problem dressed as a generation problem**

Building the RAG pipeline forced me to realise: the quality of a RAG system is almost entirely determined by retrieval quality, not by the LLM.

The architecture:

```
Query → Embed → Vector Search → Top-K Chunks
                                     ↓
                          LLM + Context → Answer
```

If the retrieved chunks are irrelevant, the LLM hallucinates confidently with those irrelevant chunks as support. If retrieval is perfect, even a 3B model answers correctly.

I evaluated with RAGAS — measuring Faithfulness (does the answer stay within the retrieved context?) and Answer Relevancy (does it actually address the question?). The single biggest lever: chunk size and overlap strategy. 512-token chunks with 64-token overlap outperformed 256-token chunks by 18 RAGAS points on my test set.

---

The most counterintuitive thing I learned: making your retrieval *too precise* can hurt RAG performance. Fetching the single most similar chunk misses surrounding context. Fetching the top 5 chunks and letting the LLM synthesise them dramatically improves faithfulness.

More context, not less — up to a limit.

**Question for the thread: Do you think RAG is a temporary solution until context windows are large enough to fit entire knowledge bases? Or is there something fundamentally better about retrieval even at scale?**

---

#GenerativeAI #RAG #PromptEngineering #LLM #VectorSearch #MachineLearning #NLP #BuildingInPublic #RAGAS #10WeekChallenge
