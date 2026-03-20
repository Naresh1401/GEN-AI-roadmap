# Week 1 — Concept Deep Dive & Interview Prep

---

# 🔷 SECTION 4 — Concept Deep Dive

> For a strong software engineer new to ML. No assumed ML background.

---

## Core Concepts Explained

---

### 1. Self-Attention

**What it is:**
Every token in a sequence computes a weighted average of *all other tokens* to update its own representation. The weights represent "how relevant is each other token to understanding me right now?"

**The analogy:**
Imagine reading a mystery novel and keeping a running mental model of every character. When you hit the word "she", your brain instantly scans back and identifies which character "she" refers to — weighing context, earlier mentions, grammatical clues. Self-attention does this for every token, simultaneously, for every other token.

**Why it matters:**
Before Transformers, RNNs processed sequences *step by step*, like reading one word at a time and keeping a single "memory". If the relevant context was 500 tokens back, the gradient signal was nearly gone by the time it needed to update that early memory. Self-attention has no "distance penalty" — it connects any two positions in one step, regardless of sequence length.

**Common misconception:**
> "Self-attention just looks at nearby words, like a convolutional filter."

Wrong. Self-attention is *global* — every position attends to every other position with equal maximum reach. Local patterns emerge only because the model *learns* to weight them higher, not because the architecture enforces it.

---

### 2. Scaled Dot-Product Attention

**What it is:**
The specific computation used to implement one "round" of attention:
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**The analogy:**
Think of a search engine. You type a query (Q). Every web page has a key (its meta-description, K) and a value (its actual content, V). The engine scores your query against all keys, softmaxes those scores into a probability distribution, then returns a weighted blend of all the page values — biased heavily toward the best-matching pages.

In Transformers, *every token* simultaneously acts as a query, a key, and a value for every other token.

**Why the scaling by √d_k?**
This is the most commonly asked "why" in interviews. If `d_k = 512`, dot products can reach magnitudes of ~512 (variance grows linearly with dimension). Large dot products push softmax into regions where one logit dominates and all others get near-zero probability — the gradient of softmax goes to zero there (saturation). Dividing by `√d_k` brings variance back to ~1.0.

**Common misconception:**
> "The Q, K, V are the raw input tokens."

No. Q, K, V are *learned linear projections* of the same input embedding. The three projection matrices (Wq, Wk, Wv) are learned parameters that allow the model to extract different aspects of the same input for each role.

---

### 3. Multi-Head Attention

**What it is:**
Run scaled dot-product attention `h` times in parallel, each with its own set of learned projection matrices, then concatenate and project back to the original dimension.

**The analogy:**
A doctor looking at a patient's chart might simultaneously consider: cardiovascular risk (one "head"), medication interactions (another head), and family history (a third head). Each expert asks a different question of the same data. Multi-head attention does exactly this — different heads can specialise in different relationship types (syntax, coreference, semantic similarity), and the final projection blends their perspectives.

**Why it's efficient:**
Each head works on dimension `d_model / h`, so the total compute is roughly the same as single-head attention on the full dimension — but with much richer representational power.

**Common misconception:**
> "Each head sees different input tokens."

No. All heads receive the *same input*. What differs is the *learned projection* that transforms the shared input before attention is computed. Heads see different *views* of the same sequence, not different sequences.

---

### 4. Positional Encoding

**What it is:**
A deterministic signal added to each token's embedding to tell the model *where* in the sequence that token appears.

**Why it's necessary:**
Self-attention is a *set operation* — it produces the same output regardless of input order (if you shuffle the tokens, you get the same result, just shuffled). To make the model sensitive to word order (which is crucial for language), you have to explicitly encode position.

**The analogy:**
Imagine GPS coordinates. A city doesn't change its nature based on its coordinates, but the coordinates are crucial context. Positional encodings are like GPS for token position — a unique fingerprint injected into each token's representation.

**The sinusoid trick:**
Using sin/cos at different frequencies means the model can infer relative distances:
- `PE(pos+1)` is predictably related to `PE(pos)` via a rotation matrix
- The model can generalise to sequences longer than those seen during training (within limits)

**Common misconception:**
> "Positional encodings are learned like embeddings."

In the original paper, they are *fixed* (not learned). Interestingly, the authors tried both and found similar performance. Modern LLMs like GPT-2 use *learned* position embeddings; others like LLaMA use RoPE (Rotary Position Embedding), a more sophisticated variant.

---

### 5. The Encoder Block Structure

**What it is:**
Each encoder layer has two sub-layers:
1. Multi-head self-attention
2. Position-wise feed-forward network (MLP)

Each sub-layer has a residual connection and layer normalisation.

**The analogy:**
Think of assembly line workers and quality control. In each station (encoder block):
- The attention step is the "team meeting" — tokens discuss what they mean in context of each other
- The FFN is the "individual thinking time" — each token updates itself based on what it learned in the meeting
- The residual connection is the "don't change what isn't broken" principle — if the sub-layer hasn't learned anything useful yet, the original information flows through unchanged

**Why residual connections matter:**
In a 12-layer Transformer, gradients must flow backward through 24 sub-layers. Without residuals, gradients would vanish or explode. The residual `x = x + sublayer(x)` creates a "gradient highway" directly from output to input.

**Pre-LN vs Post-LN:**
The original paper applies LayerNorm *after* the residual: `LayerNorm(x + sublayer(x))`. Modern models apply it *before*: `x + sublayer(LayerNorm(x))`. Pre-LN is more stable at initialisation and the standard choice today.

---

## ⚠️ Top Misconceptions Summary

1. **"Attention = soft lookup table"** — Partially true but misses the key insight: Q, K, V all come from *the same input* (in self-attention). It's a sequence attending to itself, not an external memory.

2. **"Transformers solve the long-range dependency problem completely"** — Attention is O(n²) in sequence length. At 1M tokens, it's impractical. This is why FlashAttention, Mamba, and linear attention variants exist.

3. **"Residual connections just prevent vanishing gradients"** — They also allow the model to be interpreted as an ensemble of shallow networks (Veit et al., 2016). And practically: they allow near-zero initialisation of sub-layers, which is key for deep model stability.

---

# 🔷 SECTION 5 — Interview Prep

---

## 5 Conceptual Questions (with Model Answers)

---

**Q1. Why do we divide by √d_k in scaled dot-product attention?**

**Model Answer:**
The dot product of two random vectors has variance proportional to their dimension. For vectors of dimension `d_k`, the variance of their dot product is `d_k`. When `d_k` is large (e.g., 512), raw dot products can be very large in magnitude, which pushes softmax into saturation — the gradient of softmax becomes near zero everywhere, making learning very slow. Dividing by `√d_k` normalises the variance back to 1.0, keeping gradients healthy regardless of model dimension.

---

**Q2. What is the difference between self-attention and cross-attention?**

**Model Answer:**
In **self-attention**, Query, Key, and Value all come from the same sequence — every token attends to every other token in the same input. This is used in both encoder and decoder.

In **cross-attention**, the Query comes from one sequence (e.g., the decoder's current state), while Key and Value come from another sequence (e.g., the encoder's output). This is how the decoder "looks at" the source sequence in a translation model. It's also the mechanism behind CLIP and other cross-modal architectures.

---

**Q3. Why doesn't the Transformer architecture inherently encode sequence order?**

**Model Answer:**
Self-attention is a *permutation-equivariant* operation — if you permute the input tokens, the output is permuted identically. This is because each attention score depends only on the content of two tokens (their query/key similarity), not their position. This means shuffling tokens produces exactly the same representations, just reordered. Positional encodings break this symmetry by injecting explicit position information *before* the attention computation, making the model position-sensitive.

---

**Q4. How do residual connections and layer normalisation interact in Transformers?**

**Model Answer:**
Residual connections (`x = x + sublayer(x)`) provide gradient highways through deep networks and allow sub-layers to implement small perturbations rather than wholesale transformations — which is easier to learn. Layer normalisation stabilises the distribution of activations across the feature dimension (not the batch dimension like BatchNorm). 

In Pre-LN (modern standard): `x = x + sublayer(LayerNorm(x))` — the normalised input is sent to the sub-layer, and the result is added back to the *unnormalised* residual. This prevents the common training instability seen in deep Post-LN models.

---

**Q5. Why is the FFN dimension typically 4× d_model?**

**Model Answer:**
This is an empirical hyperparameter from the original paper — the expansion to `4 × d_model` was found to provide sufficient capacity without being wasteful. The intuition: attention is a relatively "cheap" operation (it's just weighted averaging), so the FFN needs to do the heavy lifting for learning complex transformations. The expansion factor gives it room. More recent work (e.g., Chinchilla scaling laws) suggests the optimal ratio depends on depth and parameter budget. Many modern LLMs use 8/3 × d_model with SwiGLU activation instead.

---

## 2 System Design Prompts

---

**Design Prompt 1: Real-Time Document Similarity Search**

> Design a system that uses Transformer-based embeddings to find the top-10 most similar documents to a user query in < 100ms from a corpus of 50M documents.

**Key discussion points:**
- Encoder-only Transformer (e.g., BERT-style) for producing fixed-size document embeddings
- Offline batch encoding of all 50M docs → store in vector database (FAISS, Pinecone, Weaviate)
- Approximate nearest neighbour (ANN) search with HNSW or IVF index for speed
- Trade-off: ANN vs exact search (recall vs latency)
- Embedding model choice: domain-specific fine-tuning vs general-purpose (e.g., sentence-transformers)
- Freshness problem: how do you handle new documents? Incremental indexing strategy.
- Scaling: sharding the vector index across multiple machines

---

**Design Prompt 2: Multi-Language Text Classification at Scale**

> Design an ML system to classify customer support tickets (from 20 languages) into 150 categories in real time, with < 50ms latency at 10,000 requests/second.

**Key discussion points:**
- Multilingual Transformer encoder (mBERT, XLM-RoBERTa) for language-agnostic embeddings
- Classification head: linear layer over [CLS] token
- Latency budget: model inference dominates — consider ONNX export, quantisation (INT8)
- Serving: TorchServe / Triton with dynamic batching (batch incoming requests to exploit GPU parallelism)
- Caching: hash-based cache for duplicate/near-duplicate tickets
- Monitoring: prediction confidence distribution (detect distribution shift), per-language accuracy dashboards
- Fallback: route low-confidence predictions to human review queue

---

## 3 "Gotcha" Questions

---

**Gotcha 1: "Attention is O(n²). How does GPT-4 handle 128k token context windows?"**

**Why it's a gotcha:** Most people know attention is O(n²) but don't know the workarounds.

**Expected answer:** Several techniques: (1) **FlashAttention** — a hardware-aware implementation that tiles the attention computation to stay in GPU SRAM, reducing memory I/O (still O(n²) compute, but fast in practice). (2) **Sparse attention** (Longformer, BigBird) — attend only to local windows + global tokens, reducing to O(n). (3) **ALiBi / RoPE** — position encodings that generalise beyond training length. (4) Modern SSMs (Mamba) replace attention entirely with O(n) recurrence.

---

**Gotcha 2: "If you train a Transformer without positional encodings, what happens?"**

**Why it's a gotcha:** Forces understanding of what attention actually computes and what PE actually contributes.

**Expected answer:** The model becomes a *bag-of-words* model — it learns to classify/predict based on *which* tokens appear, but not *where* or in *what order* they appear. For some tasks (e.g., set classification), this would work fine. For language modelling, "the dog bit the man" and "the man bit the dog" would produce identical intermediate representations. Performance would be poor on order-sensitive tasks. Interestingly, models without PE do learn some implicit order signals from training dynamics — but it's far weaker than explicit PE.

---

**Gotcha 3: "Explain why LayerNorm is used instead of BatchNorm in Transformers."**

**Why it's a gotcha:** Feels like implementation trivia but reveals deep understanding of normalisation.

**Expected answer:** BatchNorm normalises across the *batch* dimension — it computes mean/variance over a batch of examples at each feature position. This has two problems for Transformers: (1) **Variable sequence lengths** — padding makes the batch dimension noisy; normalising across it would mix real and padded positions. (2) **Autoregressive inference** — during generation, you process one token at a time (batch size = 1). BatchNorm statistics are meaningless with a single example and must use running statistics from training, introducing a train/inference mismatch. LayerNorm normalises across the *feature* dimension for each individual token independently, making it batch-size-agnostic and compatible with both training and single-token inference.
