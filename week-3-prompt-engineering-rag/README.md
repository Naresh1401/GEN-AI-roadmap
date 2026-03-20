# Week 3: Prompt Engineering & RAG

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

This week covers the two primary ways to steer LLM behaviour without updating weights: prompt engineering (few-shot, chain-of-thought, structured prompts) and Retrieval-Augmented Generation (RAG), which gives a model access to external knowledge at inference time.

**Core concepts:**
- Few-shot and zero-shot prompting
- Chain-of-thought (CoT) reasoning
- Retrieval-Augmented Generation (RAG)
- Embedding models and vector similarity
- RAGAS evaluation framework

---

## Architecture Diagrams

### RAG Pipeline

```
┌─────────────────────────────────────────────────────┐
│                  INDEXING (offline)                  │
│                                                       │
│  Documents → Chunking → Embedding Model → Vectors   │
│                                   │                   │
│                              Vector DB                │
│                           (FAISS/Chroma)              │
└─────────────────────────────────────────────────────┘
                          │
                          │ (query time)
                          ▼
┌─────────────────────────────────────────────────────┐
│                  RETRIEVAL (online)                  │
│                                                       │
│  User Query → Embed → Nearest Neighbour Search      │
│                             │                         │
│                         Top-K Chunks                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  GENERATION (online)                 │
│                                                       │
│  Prompt = System + Retrieved Chunks + User Query    │
│                       │                              │
│                    LLM Call                          │
│                       │                              │
│                    Answer                            │
└─────────────────────────────────────────────────────┘
```

### RAGAS Evaluation Metrics

```
               ┌─────────────────────────────────┐
               │         RAGAS Metrics            │
               └───────────────┬─────────────────┘
          ┌────────────────────┼───────────────────┐
          ▼                    ▼                   ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
  │ Faithfulness │   │   Answer     │   │    Context       │
  │              │   │  Relevancy   │   │   Precision      │
  │ Does answer  │   │              │   │                  │
  │ stay within  │   │ Does answer  │   │ How much of the  │
  │ the context? │   │ address the  │   │ retrieved context│
  │              │   │ question?    │   │ is actually used?│
  │ 0–1 (LLM     │   │              │   │                  │
  │ verified)    │   │ Cosine sim   │   │ Precision@K      │
  └──────────────┘   └──────────────┘   └──────────────────┘
```

---

## Mathematical Intuition

### Cosine Similarity (Retrieval Core)

```
similarity(q, d) = (q · d) / (||q|| × ||d||)
```

Projects both query and document into the same embedding space, then measures the angle between them. `1.0` = identical direction (most relevant), `0.0` = orthogonal (unrelated), `-1.0` = opposite.

**Why cosine over Euclidean distance?** Embedding magnitude encodes intensity, not meaning. "great" and "GREAT GREAT GREAT" might have different magnitudes but similar directions. Cosine normalises this out.

### Chunking Trade-off

```
Chunk size ↑  →  More context per chunk, fewer chunks retrieved
              →  Risk of diluting relevance signal

Chunk size ↓  →  More precise retrieval, more chunks needed
              →  Risk of missing surrounding context
```

Optimal chunk size is empirically determined — typically 256–1024 tokens with 10–20% overlap.

---

## Setup

```bash
cd week-3-prompt-engineering-rag

python -m venv venv
source venv/bin/activate

pip install torch transformers sentence-transformers faiss-cpu chromadb \
            datasets ragas langchain openai python-dotenv tqdm
```

---

## Running the Implementations

```bash
# 1. RAG Pipeline
python rag_pipeline.py

# 2. Retrieval methods comparison
python retrieval_comparison.py

# 3. RAGAS evaluation
python ragas_evaluation.py
```

---

## Key Takeaways

| Concept | Core Insight |
|---|---|
| Few-shot prompting | Examples constrain output distribution — format consistency matters more than content |
| Chain-of-thought | Intermediate tokens are working memory — they steer subsequent generation |
| Chunking strategy | 512 tokens + 64 overlap is a strong baseline; evaluate on your domain |
| Retrieval quality | The LLM is not the bottleneck in RAG — retrieval is |
| RAGAS Faithfulness | LLM-verified: does the answer contain only claims from the retrieved context? |

---

## References

- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903) — Wei et al., 2022
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Es et al., 2023
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych, 2019
