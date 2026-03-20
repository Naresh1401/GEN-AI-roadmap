# 10-Week GenAI Mastery Roadmap

A complete, hands-on curriculum for mastering Generative AI — from Transformer fundamentals to production agent systems.

## Structure

Each week contains 5 deliverables:
- `linkedin_post.md` — public learning post with 3 key insights
- `README.md` — architecture diagrams, math intuition, setup instructions
- `*.py` — fully documented Python implementations
- `deep_dive_and_interview_prep.md` — concept explanations + 5 Q&As + 2 system design prompts + 3 gotcha questions

## Curriculum

| Week | Topic | Key Implementations |
|------|-------|-------------------|
| 1 | Transformer Foundations | Scaled dot-product attention, Multi-head attention, Positional encoding, Encoder block |
| 2 | Large Language Models | BPE tokeniser, GPT-2 fine-tuning, KV cache explorer |
| 3 | Prompt Engineering & RAG | RAG pipeline (FAISS + sentence-transformers), BM25 vs dense vs hybrid retrieval, RAGAS evaluation |
| 4 | Fine-Tuning & RLHF | LoRA from scratch (with weight merging), DPO training loop |
| 5 | Diffusion Models | DDPM from scratch (U-Net + MNIST), ControlNet + DDIM sampling |
| 6 | ML Systems Design | Two-tower recommendation system, PSI drift monitoring, A/B testing framework |
| 7 | Multimodal Models | CLIP from scratch (ViT + text encoder + contrastive training + zero-shot) |
| 8 | Deployment & MLOps | ONNX export, INT8 quantisation, latency benchmarking |
| 9 | Agent Frameworks | ReAct agent from scratch (no framework dependencies), tool execution loop |
| 10 | Mock Interviews & Review | 20 conceptual Q&As, 5 system design challenges, 10 gotcha questions |

## Setup

```bash
pip install torch torchvision transformers datasets sentence-transformers \
            faiss-cpu peft trl bitsandbytes accelerate einops tqdm matplotlib
```

## Usage

Each week is self-contained. Navigate to any week and run the Python scripts directly:

```bash
cd week-3-prompt-engineering-rag
python rag_pipeline.py
python retrieval_comparison.py
```

For the Week 10 interactive mock interview:
```bash
cd week-10-mock-interviews-review
python mock_interview_questions.py
```
