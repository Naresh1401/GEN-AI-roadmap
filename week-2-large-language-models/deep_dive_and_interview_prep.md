# Week 2 — Concept Deep Dive & Interview Prep

---

# 🔷 SECTION 4 — Concept Deep Dive

> For a strong software engineer new to ML. No assumed background beyond Week 1.

---

## Core Concepts

---

### 1. GPT Architecture (Decoder-Only Transformer)

**What it is:**
A Transformer that uses *only the decoder half* — stacked causal self-attention layers, no encoder, no cross-attention. The only structural difference from Week 1's encoder is a triangular attention mask that prevents each position from seeing future tokens.

**The analogy:**
An encoder reads a whole sentence and builds a complete understanding of it (like re-reading a book with full context). A decoder writes one word at a time, only ever looking back at what it has already written — like a human typing, unable to revise until the sentence is complete. GPT is a decoder: it reads left-to-right, always predicting what comes next.

**Why decoder-only dominates:**
The encoder-decoder architecture (original Transformer) is powerful for tasks with explicit input/output sequences (translation, summarisation). But for scaling to general intelligence, the decoder-only approach is elegantly simple: one objective (predict next token), one architecture, unlimited scalability. GPT-3, LLaMA, Claude, Gemini — all decoder-only.

**The pretraining objective:**
```
L = -Σ log P(xₜ | x₁, ..., xₜ₋₁)
```
At every token position, predict the next token. Train on 1T tokens. Watch the model discover grammar, facts, code, reasoning — as emergent consequences of getting very good at this simple task.

**Common misconception:**
> "GPT reads the whole input before generating."

No. During training with teacher forcing, yes — the full sequence is available and the causal mask blocks future positions. But during *generation*, it truly produces one token at a time, feeding each output back as the next input. This is autoregressive decoding.

---

### 2. Tokenization & BPE

**What it is:**
The process of converting raw text into integer IDs that the model can process. BPE (Byte-Pair Encoding) builds a subword vocabulary by iteratively merging the most frequent adjacent character pairs.

**The analogy:**
Think of tokenization as Morse code for text. The common letters (e, t, a, o) get short codes; rare ones get long ones. BPE does the same thing for text: "the" gets one token (it's everywhere); "desynchronisation" gets split into known pieces. The dictionary (vocabulary) is learned from data, not hand-crafted.

**The algorithm, plain English:**
1. Start with every character as its own symbol
2. Count every adjacent pair of symbols across the whole training corpus
3. Merge the most frequent pair into a new single symbol
4. Repeat steps 2–3 until vocabulary size is reached

**Why it matters:**
- Handles any language and any new word without failing
- Controls the trade-off between vocabulary size and sequence length
- Common words → single token (efficiency); rare words → subword pieces (coverage)
- The tokenizer is the first layer of the model — a bad tokenizer creates a hard problem for the neural network

**Common misconception:**
> "Tokenization is trivial — just split on spaces."

Enormously wrong. Tokenization choices affect: arithmetic ability (how numbers are split), code understanding, multilingual performance, and prompt length. Tokens like ` Paris` (space + Paris) vs `Paris` are different IDs — which is why prompt formatting matters so much.

---

### 3. Causal Masking

**What it is:**
A triangular boolean mask applied to the attention score matrix that sets all upper-triangle entries to −∞ before softmax. This ensures each token can only attend to itself and all *previous* tokens, never future ones.

**The analogy:**
Imagine a panel of judges scoring contestants, but each judge can only see contestants who have already performed — not the ones waiting backstage. The "mask" is the curtain. During training, all contestants are already there (the full sequence), but the curtain enforces the same left-to-right constraint anyway.

**Why it's necessary:**
Without causal masking, the model would "cheat" during training — using future tokens to predict the current one (trivially easy, zero signal). By masking future tokens, every position must predict based only on its actual available context, the same constraint it faces at inference time.

**The zero-cost constraint:**
The mask is applied to the *attention scores* before softmax, not the tokens themselves. This means all positions are computed in parallel during training (the efficiency of the Transformer). The causal constraint is enforced mathematically, not structurally.

---

### 4. Inference Optimisation & KV Cache

**What it is:**
KV cache is a technique that stores the Key and Value tensors computed during previous autoregressive steps, so they don't need to be recomputed for the new token. Only the Query for the new token is computed fresh; it attends over the cached history.

**The analogy:**
You're live-commenting a sports game. Every minute you write a new sentence — but you need to remember everything said before. Without KV cache, you re-read the entire transcript from the start every minute. With KV cache, you keep the transcript *already highlighted* with the important notes, and only read your latest sentence plus the pre-processed summary.

**The math of why it matters:**
Without cache: generating n tokens costs O(1² + 2² + ... + n²) = O(n³) attention ops.
With cache: generating n tokens costs O(n) (one forward pass of 1 token × n steps) = O(n²) total — and with pipelining, latency per token is O(1).

**The trade-off:**
KV cache consumes memory proportional to: `2 × layers × heads × head_dim × context_length`. For a 70B LLaMA model with 128K context in FP16: ~hundreds of GBs. This is why batching and context length are the primary constraints in LLM serving — not compute.

**Common misconception:**
> "KV cache only helps for long contexts."

It helps from the very first generated token. Even with a 10-token prompt, generating token 11 would otherwise recompute all 10 previous K and V values. The benefit is proportional to prompt length and grows quadratically.

---

### 5. Perplexity

**What it is:**
The standard intrinsic evaluation metric for language models. Mathematically: the exponential of the average cross-entropy loss per token.

```
PPL = exp( -1/N × Σ log P(xₜ | x₁...xₜ₋₁) )
```

**The analogy:**
Perplexity answers: "If the model had to randomly guess the next token, how many choices would it be effectively choosing between?" PPL=10 means the model is as confused as if choosing uniformly among 10 options at every step. PPL=50,000 means it's essentially guessing from the entire vocabulary.

GPT-2 on WebText: ~35 PPL. A random model over 50k vocab: 50,000 PPL. A perfect model: 1 PPL (knows exactly what comes next).

**Why it matters as a training signal:**
Perplexity is directly optimised during pretraining (it's just a function of cross-entropy loss). Lower PPL = better next-token prediction = better language modelling = better downstream performance — usually.

**Common misconception:**
> "Lower perplexity always means better model."

Not always. A model can overfit to its evaluation domain and show low PPL but fail on out-of-distribution tasks. Also, PPL cannot be compared across models with different tokenizers (different vocabulary sizes lead to incomparable PPL values). This is why benchmark suites (MMLU, HellaSwag) exist alongside PPL.

---

## ⚠️ Top Misconceptions Summary

1. **"The model learns facts from pretraining."** It learns *token co-occurrence patterns*. "Paris" follows "capital of France is" not because the model knows geography, but because that sequence appears billions of times in training data. The distinction matters enormously for understanding hallucination and knowledge cutoffs.

2. **"Tokenization is language-agnostic."** BPE is trained on data. If your training corpus is 95% English, the tokenizer will be highly efficient for English (3–4 chars/token) and inefficient for other scripts like Thai or Arabic (potentially 1 char/token). This explains why multilingual models underperform on low-resource languages — they get fewer tokens' worth of training data for the same compute budget.

3. **"KV cache is a compression technique."** No — it's purely a *memoization* optimization. The K and V tensors in the cache are identical to what would be computed without it. There is no approximation or information loss. KV quantization (storing cache in INT8/INT4) *is* a lossy compression, and a separate optimization.

---

# 🔷 SECTION 5 — Interview Prep

---

## 5 Conceptual Questions

---

**Q1. What is the difference between causal language modelling (GPT-style) and masked language modelling (BERT-style)?**

**Model Answer:**
Causal LM (GPT): Each token predicts the next token using only its left context. The objective is `P(xₜ | x₁...xₜ₋₁)`. This makes it naturally suited for generation. The causal mask ensures no information leakage from future positions.

Masked LM (BERT): A random subset (~15%) of tokens are replaced with `[MASK]`, and the model predicts those masked tokens using bidirectional context (both left and right). This doesn't support direct generation (predicting `[MASK]` tokens doesn't compose into a generative model) but produces richer contextual representations for classification and retrieval.

**Trade-off:** Causal LM sees only half the context (left side only) at each position → less efficient use of each token for representation learning. Masked LM sees full context → better representations, but cannot generate autoregressively. This is why GPT is used for generation and BERT for embeddings/classification.

---

**Q2. Why does GPT-2 use learned positional embeddings rather than sinusoidal ones?**

**Model Answer:**
The original "Attention Is All You Need" used fixed sinusoidal encodings; GPT-2 uses learned absolute position embeddings — simply an `nn.Embedding(max_seq_len, d_model)` table. Empirically, both perform similarly in the original paper. Learned embeddings give the model more flexibility but cap hard at `max_seq_len` (you can't extrapolate beyond positions seen during training). Sinusoidal encodings can theoretically extrapolate because `PE(pos+k)` is a linear function of `PE(pos)`, but in practice both struggle with long extrapolation. Modern LLMs use RoPE (Rotary Position Embedding) or ALiBi specifically to address this length generalisation problem.

---

**Q3. What is teacher forcing, and why is it used during training?**

**Model Answer:**
Teacher forcing is the training strategy where, at each step, the model receives the *ground-truth previous token* as input (not its own prediction from the previous step). Without it, early in training when the model makes bad predictions, errors compound — a wrong token at step 3 leads to a wrong distribution at step 4, poisoning the entire sequence. Teacher forcing decouples each step from previous errors, making the training signal clean and stable. The downside is a "train-test discrepancy": at inference the model sees its *own* (potentially wrong) outputs, not ground truth. This is called exposure bias and motivates techniques like scheduled sampling (mixing real and predicted tokens) or reinforcement learning fine-tuning (RLHF, Week 4).

---

**Q4. What happens to the KV cache when you switch from FP32 to FP16? Can you go further?**

**Model Answer:**
Switching from FP32 to FP16 cuts KV cache memory exactly in half (32 → 16 bits per element). You can go further with KV quantization: storing cache in INT8 cuts memory to 1/4 of FP32, INT4 to 1/8. Research shows that INT8 KV cache incurs minimal quality degradation (< 0.5% on most benchmarks). INT4 is used in production systems (e.g., Mistral's inference stack, LMDeploy) with acceptable quality trade-offs. The key insight: K and V are intermediate activations, not model weights. Their quantization error affects attention weights but is averaged over a full softmax — making them more tolerant to quantization noise than weights.

---

**Q5. How does nucleus (top-p) sampling differ from temperature scaling, and when would you use each?**

**Model Answer:**
**Temperature:** Scales all logits by `1/T` before softmax. `T < 1` sharpens the distribution (more deterministic); `T > 1` flattens it (more random). Temperature affects the *shape* of the distribution — it doesn't truncate the vocabulary, just reweights it.

**Top-p (Nucleus):** Sorts tokens by probability, takes the smallest set whose cumulative probability ≥ p, and samples only from that set. This adapts the candidate set dynamically — when the model is confident (one token has 95% prob), it effectively becomes greedy; when uncertain (probability spread across 1000 tokens), it samples from many options.

**When to use each:**
- Factual Q&A: Low temperature (0.1–0.3), no top-p — want high-confidence outputs
- Creative writing: Higher temperature (0.7–1.0) + top-p=0.9 — want diversity without incoherence
- Code generation: Temperature=0.2, top-p=0.95 — moderately deterministic but some flexibility
- In practice, temperature and top-p are often combined.

---

## 2 System Design Prompts

---

**Design Prompt 1: GPT-Based Autocomplete API**

> Design a real-time autocomplete API serving 10,000 requests/second with p99 latency < 200ms. Each request provides a text prefix and expects 5 token completions.

**Key discussion points:**
- Model: Small, distilled GPT (e.g., GPT-2 small or custom 50M) for latency budget. Larger models go through a routing layer only for premium tier.
- Serving: TorchServe or Triton with dynamic batching — aggregate requests within a 5ms window into batches of 32+.
- KV cache: Disabled for this use case (5 tokens only; cache overhead exceeds benefit). Enable for longer-context use cases.
- Caching layer: Cache LRU for popular prefixes — hash(prompt) → completions. Cover the long tail.
- GPU fleet: Auto-scaling GPU pool behind a load balancer. Each GPU runs multiple model replicas.
- Fallback: If GPU pool is saturated, fall back to n-gram statistical autocomplete (deterministic, CPU-only, < 1ms).
- Monitoring: Token/sec throughput, p50/p95/p99 latency per endpoint, error rate, GPU utilisation.

---

**Design Prompt 2: Private Code Completion Service (GitHub Copilot Clone)**

> Design an on-premise code completion service for an enterprise that cannot send source code to external APIs. Must support 500 concurrent users, 12+ programming languages, and produce suggestions within 1 second.

**Key discussion points:**
- Model: Code-specific LLM (CodeLlama, StarCoder2) fine-tuned on internal codebase for domain adaptation.
- On-premise deployment: Air-gapped Kubernetes cluster with A100/H100 GPUs. No data leaves the network.
- Fill-in-the-middle (FIM): Unlike standard causal LM, code completion needs context from both before AND after the cursor. FIM training (prefix + suffix → middle) is essential.
- KV cache: Prefix prompt (file context above cursor) is often static per session → aggressive prefix caching to eliminate prompt recomputation.
- Speculative decoding: Smaller draft model generates K tokens; large model verifies in one pass. Provides 2–4× speedup.
- Multi-language: Single model trained on all languages; language detected from file extension and injected as a special token.
- IDE integration: VS Code / JetBrains plugins using streaming (SSE/WebSocket) to show partial completions as they generate.
- Security: All prompts/completions logged internally for abuse monitoring, never exfiltrated.

---

## 3 "Gotcha" Questions

---

**Gotcha 1: "GPT is trained with teacher forcing, but generated with its own outputs. Doesn't this break things?"**

**Why it's a gotcha:** Most people know teacher forcing exists but haven't thought through the train-test gap.

**Expected answer:** Yes, this is a known problem called *exposure bias*. The model trains on gold token sequences but at inference receives its own (potentially wrong) generations. Early errors compound — a slightly wrong token shifts the distribution for the next token, and the model has never been trained to recover from its own mistakes. 
The practical mitigations: (1) Scheduled sampling — gradually replace gold tokens with model predictions during training. (2) RLHF/DPO — fine-tune with RL on actual generated sequences (Week 4). (3) In practice, modern large models are robust to this because they're trained on so much diverse data that their own generations are usually "in distribution." But it's a real issue for small models trained on narrow datasets.

---

**Gotcha 2: "Two models both report perplexity of 20. Are they equally good?"**

**Why it's a gotcha:** Perplexity is routinely misused in ML interviews and papers.

**Expected answer:** Not necessarily — in fact, you cannot directly compare them if they use different tokenizers. Perplexity is measured per *token*, and if one tokenizer produces an average of 3 characters per token while another produces 5 characters per token, the models are being evaluated on different granularities. The model with the coarser tokenizer has an "easier" per-token prediction problem. The correct comparison is to evaluate both on the same character-level or byte-level metric. Additionally, perplexity on the training domain doesn't predict performance on other domains — a model can have low PPL on Wikipedia text and struggle badly on medical or legal text.

---

**Gotcha 3: "What limits the context window of a GPT model? Can you just increase max_seq_len?"**

**Why it's a gotcha:** The obvious answer (just change a number) misses 3 separate failure modes.

**Expected answer:** Three distinct limitations:
1. **Positional embedding coverage:** Models with absolute learned position embeddings (like GPT-2) have never seen positions beyond `max_seq_len` during training. Extrapolating beyond this fails — the model has no representation for those positions. RoPE/ALiBi help here.
2. **KV cache memory:** At 128K tokens with a 70B model in FP16, the KV cache alone is ~320GB per sequence. This is why long-context serving requires memory management strategies like PagedAttention (vLLM).
3. **Attention degradation:** Even with position encodings that support long contexts mathematically, models struggle to *use* information from very early in the context (lost-in-the-middle problem). Research shows that performance peaks at the beginning and end of a long context, with a U-shaped degradation in between. Simply supporting a long context window ≠ the model can reliably use all of it.
