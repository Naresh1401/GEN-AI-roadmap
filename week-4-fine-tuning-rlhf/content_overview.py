# ============================================================
# WEEK 4: Fine-Tuning & RLHF
# ============================================================

# ============================================================
# 🔷 SECTION 1 — linkedin_post.md
# ============================================================
LINKEDIN_POST = """
# Week 4 LinkedIn Post: Fine-Tuning & RLHF

---

I fine-tuned a 7 billion parameter language model on a single GPU this week.
Three months ago I would have told you that was impossible without a data centre.

**Week 4: Fine-Tuning & RLHF — LoRA, QLoRA, DPO, and alignment.**

---

**💡 Aha #1 — LoRA is a massive cheat code (in the best possible way)**

The insight behind LoRA is profound: when you fine-tune a model, the weight
updates form a *low-rank* matrix. Instead of updating the full d×d weight
matrix, you can approximate the update as the product of two small matrices:

```python
# Full fine-tuning: update all ~134M params in one attention layer
W_updated = W + ΔW  # ΔW is d_model × d_model

# LoRA: ΔW ≈ A @ B, where rank r << d_model
# A: d_model × r   (2048 × 16 = 32,768 params)
# B: r × d_model   (16 × 2048 = 32,768 params)
# Total: 65,536 params vs 4,194,304 — a 64× reduction
```

The entire pretrained model stays frozen. You're only training 0.1–1% of parameters.
And it works *remarkably well* — in most benchmarks, LoRA matches full fine-tuning
within 1–2 percentage points.

---

**💡 Aha #2 — QLoRA makes GPU memory irrelevant**

QLoRA combines LoRA with 4-bit quantisation of the base model. The frozen base
model sits in NF4 (4-bit Normal Float) on GPU — taking ~4× less memory than FP16.
The tiny LoRA adapters train in BF16.

This means: a 7B model that would require 14GB in FP16 now fits in ~4GB.
I fine-tuned LLaMA on a 16GB GPU that was previously limited to 3B models.
The quality trade-off was negligible.

---

**💡 Aha #3 — DPO killed the RLHF complexity cliff**

Classic RLHF requires three models simultaneously: the reference model, the reward
model, and the policy being trained — plus a PPO training loop that is notoriously
unstable.

DPO (Direct Preference Optimisation) derived that you can *skip the reward model
entirely*. Given pairs of (preferred, rejected) responses, DPO directly optimises
the policy to increase the probability of preferred responses relative to rejected ones:

```python
# DPO loss (simplified):
# Increase log P(y_win) - Decrease log P(y_lose)
# Normalised against a reference model to prevent reward hacking
loss = -log_sigmoid(β * (log_prob_win - log_prob_lose - ref_win + ref_lose))
```

The training stability is dramatically better than PPO. DPO is now the default
alignment recipe for most open-source LLMs.

---

The thing that surprised me most: after LoRA fine-tuning, the adapted model weights
can be *merged back* into the base model and distributed as a standalone model —
no LoRA runtime overhead at inference. You fine-tune cheap, serve fast.

**Question: LoRA fine-tuning changes the model's *style* far more readily than its
*factual knowledge*. Do you think fundamental knowledge injection will ever be
as efficient as stylistic adaptation?**

---

#FineTuning #LoRA #QLoRA #DPO #RLHF #LLM #GenerativeAI #MachineLearning #BuildingInPublic
"""

# ============================================================
# README content (printed as docstring, saved separately)
# ============================================================
README = """# Week 4: Fine-Tuning & RLHF

## Overview
Fine-tuning adapts pretrained LLMs to specific tasks or styles.
This week covers parameter-efficient fine-tuning (LoRA, QLoRA),
and alignment methods (DPO, PPO-based RLHF).

## Architecture: LoRA

```
Base Model (frozen)          LoRA Adapter (trainable)
┌─────────────────┐          ┌──────────────────────┐
│  W₀ ∈ ℝ^(d×d)  │    +     │  A ∈ ℝ^(d×r)         │
│                 │          │  B ∈ ℝ^(r×d)         │
│  Frozen         │          │  ΔW = BA (rank r)    │
└─────────────────┘          └──────────────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
                 h = W₀x + BAx
                 (scaled by α/r)
```

## Architecture: DPO Training Flow

```
Dataset: (prompt, y_win, y_lose) pairs
         │
         ▼
Reference Model (frozen copy of base)
Policy Model    (fine-tuned — starts as copy of base)
         │
         ▼
DPO Loss = -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x)
                      - log π_ref(y_w|x) + log π_ref(y_l|x)))
         │
         ▼
Update only policy weights → aligned model
```

## Setup
```bash
pip install torch transformers peft datasets bitsandbytes accelerate trl
```

## Key Takeaways
| Method | Trainable Params | Memory Reduction | Stability |
|---|---|---|---|
| Full FT | 100% | None | High |
| LoRA | 0.1–1% | 4–8× | High |
| QLoRA | 0.1–1% | 8–16× | High |
| PPO (RLHF) | 100% | None | Low |
| DPO | 100% (or LoRA) | Moderate | High |

## References
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Direct Preference Optimisation](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
"""

print(LINKEDIN_POST[:200] + "...")
print(README[:200] + "...")
