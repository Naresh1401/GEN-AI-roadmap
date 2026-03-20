# Week 1: Transformer Foundations

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

This week covers the architectural DNA of every modern large language model: the **Transformer**. Introduced in *Attention Is All You Need* (Vaswani et al., 2017), the Transformer replaced recurrence with self-attention — enabling parallelism, longer-range dependencies, and the scale that makes GPT-4, Claude, and Gemini possible.

**Core concepts covered:**
- Self-attention and scaled dot-product attention
- Multi-head attention
- Positional encodings (sinusoidal)
- The Transformer encoder block

---

## Architecture Diagrams

### Scaled Dot-Product Attention

```
          Q         K         V
          │         │         │
          ▼         ▼         │
       ┌──────────────┐       │
       │  MatMul(Q,Kᵀ) │       │
       └──────┬───────┘       │
              │ Scale (÷ √d_k) │
              ▼               │
       ┌──────────┐           │
       │  Softmax  │           │
       └──────┬───┘           │
              │               │
              ▼               ▼
          ┌─────────────────────┐
          │     MatMul          │
          └──────────┬──────────┘
                     │
                  Output
```

### Multi-Head Attention

```
Input ──┬──────────────────────────────────┐
        │                                  │
   ┌────▼────┐  ┌────────┐   ┌────────┐   │
   │ Linear   │  │ Linear │   │ Linear │   │ (repeat h times)
   │ Wq, Wk,  │  │  Wq_2  │   │  Wq_h  │   │
   │   Wv_1  │  │ Wk_2   │   │  Wk_h  │   │
   └────┬────┘  └───┬────┘   └───┬────┘   │
        │           │            │         │
   ┌────▼────┐  ┌───▼────┐  ┌───▼────┐   │
   │ Attn_1  │  │ Attn_2 │  │ Attn_h │   │
   └────┬────┘  └───┬────┘  └───┬────┘   │
        └───────────┴────────────┘        │
                    │                     │
               ┌────▼────┐               │
               │ Concat   │               │
               └────┬────┘               │
                    │                     │
               ┌────▼────┐               │
               │  Linear  │ ◄────────────┘
               │    Wo    │
               └────┬────┘
                    │
                 Output
```

### Transformer Encoder Block

```
Input Embeddings + Positional Encoding
              │
              ▼
    ┌─────────────────────┐
    │   Multi-Head         │
    │   Self-Attention     │
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │   Add & LayerNorm   │  ◄── Residual connection
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │   Feed-Forward       │
    │   Network (FFN)      │
    │   Linear → ReLU → Linear
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │   Add & LayerNorm   │  ◄── Residual connection
    └────────┬────────────┘
             │
           Output
```

---

## Mathematical Intuition

### 1. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

**Plain English:**
- `Q` (Queries): "What am I looking for?" — one vector per token
- `K` (Keys): "What do I contain?" — one vector per token
- `V` (Values): "What do I actually contribute?" — one vector per token
- `QKᵀ`: Similarity score between every query-key pair — shape `[seq_len, seq_len]`
- `÷ √d_k`: Prevents scores from growing large with dimension, keeping gradients healthy
- `softmax`: Converts raw scores into a probability distribution (they sum to 1)
- `· V`: Weighted sum of values — each output token is a blend of all value vectors

**Why scale by √d_k?**
If `d_k = 64`, the dot products have variance ~64. Without scaling, softmax would saturate (output near 0 or 1 everywhere), making gradients vanish. Dividing by `√64 = 8` normalises the variance back to 1.

### 2. Positional Encoding

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

**Plain English:**
Each dimension of the positional encoding oscillates at a different frequency. Low dimensions oscillate fast (every 2 positions); high dimensions oscillate very slowly (over thousands of positions). Together they create a unique "fingerprint" for each position — like reading a binary clock where each bit column ticks at a different rate.

**Why sinusoids?** The model can compute relative positions: `PE(pos + k)` is a linear transformation of `PE(pos)`. This lets the model generalise to sequence lengths not seen during training.

### 3. Feed-Forward Network (within each encoder block)

```
FFN(x) = max(0, x · W₁ + b₁) · W₂ + b₂
```

**Plain English:** A simple 2-layer MLP applied independently to each position. Think of attention as the communication step (tokens talk to each other) and FFN as the computation step (each token thinks on its own). The hidden dimension is typically 4× the model dimension, giving the network room to learn complex transformations.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/genai-mastery-roadmap
cd genai-mastery-roadmap/week-1-transformer-foundations

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib

# Optional: for notebook exploration
pip install jupyter ipywidgets
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA)
- NumPy, Matplotlib

---

## Running the Implementations

### 1. Scaled Dot-Product Attention

```bash
python scaled_dot_product_attention.py
```

**What it does:** Implements attention from scratch, visualises attention weights as a heatmap, and verifies output shapes match expectations.

**Expected output:**
```
Input shape:  torch.Size([2, 10, 64])   # (batch, seq_len, d_model)
Output shape: torch.Size([2, 10, 64])
Attention weights shape: torch.Size([2, 10, 10])
Attention weights sum to 1.0: True
```

### 2. Multi-Head Attention

Included in `scaled_dot_product_attention.py` as `MultiHeadAttention` class.

### 3. Mini Transformer Encoder

```bash
python mini_transformer_encoder.py
```

**What it does:** Builds a complete Transformer encoder (N stacked encoder blocks), runs a forward pass with random tokens, and prints intermediate shapes at each stage so you can trace the data flow.

**Expected output:**
```
=== Mini Transformer Encoder Forward Pass ===
Token IDs:           torch.Size([2, 20])
After embedding:     torch.Size([2, 20, 128])
After pos encoding:  torch.Size([2, 20, 128])
After encoder[0]:    torch.Size([2, 20, 128])
After encoder[1]:    torch.Size([2, 20, 128])
Final output:        torch.Size([2, 20, 128])
Total parameters:    419,200
```

---

## Key Takeaways

| Concept | Why It Matters |
|---|---|
| Self-attention | Relates every token to every other in O(1) steps — no forgetting |
| Scaling by √d_k | Keeps softmax gradients healthy regardless of dimension |
| Multi-head attention | Learns multiple relationship types in parallel |
| Positional encoding | Injects order without breaking parallelism |
| Residual connections | Enables gradient flow through deep stacks |
| Layer normalisation | Stabilises training — applied *before* sub-layers (Pre-LN) in modern variants |

---

## What to Study Next (Week 2)

- GPT architecture (decoder-only Transformers)
- Byte-Pair Encoding (BPE) tokenization
- KV cache and inference optimisation
- Autoregressive generation

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP
- [PyTorch nn.MultiheadAttention docs](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
