# Week 4: Fine-Tuning & RLHF

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

This week covers parameter-efficient fine-tuning (LoRA, QLoRA) and alignment techniques (DPO, RLHF with PPO). You will implement a LoRA adapter from scratch, a DPO training loop, and compare alignment approaches.

**Core concepts:**
- Low-Rank Adaptation (LoRA) and QLoRA
- Supervised fine-tuning (SFT) as a prerequisite for alignment
- RLHF: reward model + PPO
- Direct Preference Optimisation (DPO)
- Catastrophic forgetting and regularisation

---

## Architecture Diagrams

### LoRA Adapter

```
Input x ∈ ℝ^d
      │
      ├─────────────────────────────────────────┐
      │                                         │
      ▼                                         ▼
┌──────────────┐                     ┌──────────────────┐
│  W₀ (frozen) │                     │  A ∈ ℝ^(d × r)  │ ← randomly init
│  d × d       │                     │  B ∈ ℝ^(r × d)  │ ← zero init
└──────┬───────┘                     └────────┬─────────┘
       │                                      │
       │  W₀x                                 │  BAx × (α/r)
       │                                      │  (scaling factor α/r)
       └──────────────────┬───────────────────┘
                          ▼
                    h = W₀x + (α/r) BAx
```

### DPO vs PPO-RLHF Comparison

```
RLHF (PPO):                          DPO:
───────────────────────               ─────────────────────
1. Collect human preferences          1. Collect human preferences
2. Train reward model R(x, y)         2. (No reward model needed)
3. RL loop:                           3. Direct optimisation:
   - Sample from policy               
   - Score with R                        loss = -log σ(β * log-ratio)
   - PPO update                       
   - KL penalty vs reference          4. Single backward pass
4. 3 models in memory                    2 models in memory
   → Unstable, complex                → Stable, simple
```

---

## Mathematical Intuition

### LoRA Scaling Factor

```
h = W₀x + (α/r) * B*A*x
```

`α` is a hyperparameter (typically 16 or 32); `r` is the rank. The ratio `α/r` scales the adapter's contribution. Setting `α = r` means no scaling. Using `α = 2r` doubles the adapter's effective learning rate — a common practical choice.

### DPO Loss

```
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                  - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))]
```

β controls the deviation from the reference model — higher β = smaller deviation = less alignment, less forgetting. Typical values: 0.1–0.5.

---

## Setup

```bash
pip install torch transformers peft datasets bitsandbytes accelerate trl
```

---

## Running the Implementations

```bash
python lora_from_scratch.py         # LoRA implementation + integration
python dpo_training_loop.py         # DPO with preference dataset
python rlhf_vs_dpo_comparison.py    # Side-by-side comparison
```

---

## Key Takeaways

| Method | When to Use |
|---|---|
| LoRA | Domain adaptation, style transfer, task-specific tuning |
| QLoRA | Consumer GPU fine-tuning (< 24GB VRAM) |
| SFT | Teaching instruction-following before alignment |
| DPO | Aligning with human preferences, safety fine-tuning |
| PPO-RLHF | Complex reward shaping, multi-objective alignment |

---

## References

- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Direct Preference Optimisation](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [Training Language Models to Follow Instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022
