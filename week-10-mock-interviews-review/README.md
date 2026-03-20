# Week 10: Mock Interviews & Comprehensive Review

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

This final week consolidates all 9 weeks of material through mock interviews, system design practice, and a comprehensive concept review. The goal is to identify knowledge gaps and solidify understanding through active recall.

**Format:**
- 20 conceptual questions spanning all 9 weeks
- 5 system design challenges (with model answers)
- 10 "gotcha" questions that catch common misconceptions
- Oral explanation exercises

---

## How to Use This Material

1. **Flashcard mode:** Cover the answer, attempt to recall, then check
2. **Whiteboard mode:** Draw architecture diagrams from memory
3. **Timer mode:** 3-minute timer per conceptual question, 20-minute timer per system design
4. **Teach-back mode:** Explain each concept to someone else (or to yourself out loud)

---

## Quick Reference: All 9 Weeks

| Week | Topic | Key Formula/Concept |
|---|---|---|
| 1 | Transformer Foundations | Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V |
| 2 | Large Language Models | KV cache: O(n) vs O(n²) generation |
| 3 | Prompt Engineering & RAG | Hybrid retrieval: BM25 + dense → RRF |
| 4 | Fine-Tuning & RLHF | LoRA: ΔW ≈ BA, (α/r) scaling |
| 5 | Diffusion Models | x_t = √ā_t · x_0 + √(1-ā_t) · ε |
| 6 | ML Systems Design | PSI > 0.25 → retrain; A/B power analysis |
| 7 | Multimodal Models | CLIP: InfoNCE loss, zero-shot classification |
| 8 | Deployment & MLOps | INT8: 4× smaller, 2-4× faster, < 1% accuracy loss |
| 9 | Agent Frameworks | ReAct: Thought → Action → Observation → ... |

---

## References (Full Reading List)

**Foundational Papers:**
- Attention Is All You Need (2017) — Vaswani et al.
- GPT-2 (2019), GPT-3 (2020) — OpenAI
- BERT (2018) — Devlin et al.
- DDPM (2020) — Ho et al.
- LoRA (2021) — Hu et al.
- QLoRA (2023) — Dettmers et al.
- DPO (2023) — Rafailov et al.
- CLIP (2021) — Radford et al.
- ViT (2020) — Dosovitskiy et al.
- Whisper (2022) — Radford et al.
- ReAct (2022) — Yao et al.
- RAGAS (2023) — Es et al.
- Stable Diffusion (2022) — Rombach et al.

**Books:**
- Designing Machine Learning Systems — Chip Huyen
- Natural Language Processing with Transformers — Lewis Tunstall et al.
- Deep Learning — Goodfellow, Bengio, Courville

**Courses:**
- CS224N: NLP with Deep Learning (Stanford)
- Fast.ai Practical Deep Learning
- Andrej Karpathy's Neural Networks: Zero to Hero (YouTube)
