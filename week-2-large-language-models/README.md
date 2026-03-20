# Week 2: Large Language Models

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

This week goes from raw text to autoregressive generation — covering the full LLM pipeline: tokenization with BPE, the GPT decoder-only architecture, fine-tuning a pretrained model on custom data, and the KV cache optimisation that makes inference practical.

**Core concepts covered:**
- Byte-Pair Encoding (BPE) tokenization
- GPT architecture (decoder-only Transformer)
- Causal (autoregressive) masking
- Fine-tuning GPT-2 with Hugging Face
- KV cache and its role in efficient inference

---

## Architecture Diagrams

### BPE Tokenization Pipeline

```
Raw Text
   │
   ▼
┌────────────────────────────────────────┐
│  Pre-tokenization (whitespace/punct)   │
│  "Hello, world!" → ["Hello", ",",      │
│                     " world", "!"]     │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Character-level initialisation        │
│  "Hello" → ['H','e','l','l','o']       │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Iterative BPE merges                  │
│  Count pairs → merge most frequent    │
│  Repeat until vocab_size reached       │
│                                        │
│  Merge 1: ('l','l') → 'll'             │
│  Merge 2: ('H','e') → 'He'             │
│  Merge 3: ('He','ll') → 'Hell'         │
│  ...                                   │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Final vocabulary + merge rules        │
│  Encode: text → token ID sequence      │
│  Decode: token IDs → text              │
└────────────────────────────────────────┘
```

### GPT (Decoder-Only) Architecture

```
Token IDs
    │
    ▼
┌───────────────────────────┐
│  Token Embedding          │  (vocab_size → d_model)
│  + Positional Embedding   │  (learned, not sinusoidal)
└─────────────┬─────────────┘
              │
    ┌─────────▼─────────┐
    │  Decoder Block ×N  │  (N = 12 for GPT-2 small)
    │                    │
    │  ┌──────────────┐  │
    │  │ LayerNorm     │  │
    │  └──────┬───────┘  │
    │         │           │
    │  ┌──────▼───────┐  │
    │  │ Masked Multi- │  │  ← CAUSAL mask: no future tokens
    │  │ Head Attention│  │
    │  └──────┬───────┘  │
    │         │           │
    │  ┌──────▼───────┐  │
    │  │ Residual Add  │  │
    │  └──────┬───────┘  │
    │         │           │
    │  ┌──────▼───────┐  │
    │  │ LayerNorm     │  │
    │  └──────┬───────┘  │
    │         │           │
    │  ┌──────▼───────┐  │
    │  │ Feed-Forward  │  │
    │  └──────┬───────┘  │
    │         │           │
    │  ┌──────▼───────┐  │
    │  │ Residual Add  │  │
    │  └──────┬───────┘  │
    └─────────┼──────────┘
              │
    ┌─────────▼─────────┐
    │   LayerNorm        │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  LM Head (Linear)   │  (d_model → vocab_size)
    └─────────┬───────────┘
              │
    ┌─────────▼──────────┐
    │  Softmax → logits   │
    └────────────────────┘
              │
        Next Token
```

### KV Cache During Inference

```
Step 1 (token "The"):
  Compute Q₁, K₁, V₁
  Cache: K_cache=[K₁], V_cache=[V₁]
  Output: token "cat"

Step 2 (token "cat"):
  Compute Q₂, K₂, V₂
  Cache: K_cache=[K₁,K₂], V_cache=[V₁,V₂]
  Q₂ attends to [K₁,K₂] — NO recomputation of K₁,V₁
  Output: token "sat"

Step N:
  Only compute Qₙ, Kₙ, Vₙ
  Append to cache → attend over full history
  Cost per step: O(n) not O(n²) repeated
```

---

## Mathematical Intuition

### 1. BPE Merge Score

```
score(pair) = frequency(pair)
```

BPE simply picks the most frequent adjacent pair of symbols and merges them. No probability theory — just counting. After `V - V₀` merges (where V₀ is character vocabulary size), you have your target vocabulary.

**Why it works:** Language has hierarchical structure. Common substrings (morphemes, word roots) appear together frequently, so BPE discovers them automatically.

### 2. Causal Language Modelling Objective

```
L = -Σ log P(xₜ | x₁, x₂, ..., xₜ₋₁)
```

**Plain English:** At every position, predict the next token given all previous tokens. The loss is the negative log probability of the correct next token — averaged across all positions in all sequences. Lower loss = better next-token prediction.

**Why negative log?** We want to *maximise* probability of the correct token. Maximising `log P` is the same as minimising `-log P`. Convex optimisers minimise, so we negate.

### 3. Causal Attention Mask

```
M[i,j] = 0  if j ≤ i  (token i can attend to token j)
M[i,j] = -∞ if j > i  (token i cannot see future token j)
```

The upper triangle of the attention score matrix gets `-∞` before softmax, making those positions contribute 0 weight. This single constraint transforms the encoder into an autoregressive model.

### 4. KV Cache Memory Cost

```
Memory per token = 2 × num_layers × num_heads × d_head × bytes_per_element
```

For GPT-2 small (12 layers, 12 heads, 64 head dim, FP16):
```
= 2 × 12 × 12 × 64 × 2 bytes = 36,864 bytes ≈ 36 KB per token
```

For a 4K context: ~144 MB. For GPT-4 scale with 128K context: this is why KV cache is a primary memory constraint in LLM serving.

---

## Setup

```bash
# Navigate to week folder
cd week-2-large-language-models

# Create environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets tiktoken matplotlib tqdm

# Optional: for GPU monitoring
pip install nvitop
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- transformers 4.40+
- datasets (Hugging Face)
- tiktoken (OpenAI's BPE reference implementation)

---

## Running the Implementations

### 1. BPE Tokenizer from Scratch

```bash
python bpe_tokenizer.py
```

**What it does:** Implements BPE from scratch — initialises a character vocabulary, iteratively merges the most frequent adjacent pair, builds encode/decode functions, and compares against `tiktoken`'s `cl100k_base` (the GPT-4 tokenizer).

**Expected output:**
```
=== BPE Tokenizer Demo ===
Training on 5 documents...
Merge   1: ('e', 's') → 'es'   (freq: 47)
Merge   2: ('i', 'n') → 'in'   (freq: 39)
...
Merge 500: ('ation', 's') → 'ations' (freq: 3)

Vocabulary size: 556

Encode: "lowest" → [23, 41, 18]  → tokens: ['low', 'es', 't']
Decode: [23, 41, 18] → "lowest"

Compression ratio: 3.21x  (chars per token)
vs tiktoken cl100k: 4.18x compression
```

### 2. Fine-tune GPT-2 on Custom Data

```bash
python finetune_gpt2.py
```

**What it does:** Fine-tunes `gpt2` (117M parameters) on a custom text dataset using Hugging Face `Trainer`. Includes data preprocessing, tokenization with truncation, training loop, and text generation comparison before/after fine-tuning.

**Expected output:**
```
=== GPT-2 Fine-Tuning ===
Dataset: 1,247 training examples
Model: GPT-2 (117M parameters)
Device: cuda

Epoch 1/3: loss=3.241  ppl=25.5
Epoch 2/3: loss=2.847  ppl=17.2
Epoch 3/3: loss=2.591  ppl=13.3

--- Generation (before fine-tuning) ---
Prompt: "The transformer architecture"
Output: "The transformer architecture has been used in many..."

--- Generation (after fine-tuning) ---
Output: "The transformer architecture allows the model to..."
Model saved to ./gpt2-finetuned/
```

### 3. KV Cache Explorer

```bash
python kv_cache_explorer.py
```

**What it does:** Implements a GPT-style decoder with and without KV cache, benchmarks generation speed, measures memory usage, and plots latency vs sequence length.

**Expected output:**
```
=== KV Cache Benchmark ===
Generating 100 tokens from prompt (10 tokens)...

Without KV cache:  4.82s  | 20.7 tok/s
With KV cache:     0.61s  | 163.4 tok/s
Speedup: 7.9×

Memory (KV cache for 512 tokens, 6 layers): 18.87 MB
Latency plot saved → kv_cache_latency.png
```

---

## Key Takeaways

| Concept | Why It Matters |
|---|---|
| BPE tokenization | Handles unseen words gracefully; controls vocabulary size |
| Causal masking | The one constraint that makes autoregressive generation possible |
| Decoder-only vs Encoder-only | GPT (generation) vs BERT (understanding) — different objectives |
| Weight tying | LM head and embedding share weights → fewer params, better generalisation |
| KV cache | Makes real-time streaming generation feasible at inference |
| Perplexity (PPL) | Language model's primary training metric: `exp(cross-entropy loss)` |

---

## What to Study Next (Week 3)

- Prompt engineering: few-shot, chain-of-thought, structured outputs
- Retrieval-Augmented Generation (RAG) — giving LLMs external memory
- RAG evaluation with RAGAS
- Vector databases: FAISS, ChromaDB, Pinecone

---

## References

- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Sennrich et al. (original BPE paper)
- [Efficient Memory Management for LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023
- [Hugging Face GPT-2 docs](https://huggingface.co/gpt2)
