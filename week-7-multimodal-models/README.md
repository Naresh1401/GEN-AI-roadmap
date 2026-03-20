# Week 7: Multimodal Models

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

Multimodal models learn representations that span multiple modalities — text, images, and audio. This week covers CLIP (contrastive image-text pre-training), Vision Transformers (ViT), and Whisper (audio transcription).

**Core concepts:**
- Contrastive learning (CLIP)
- Vision Transformers: patch embeddings, class tokens
- Cross-modal alignment and zero-shot transfer
- Whisper: encoder-decoder for speech recognition
- Multimodal fusion strategies

---

## Architecture Diagrams

### CLIP Architecture

```
Images ──────────────────────────────────────────────┐
                                                       │
[img_1] → ViT Encoder → [e_1]                        │
[img_2] → ViT Encoder → [e_2]   ← L2 Normalised      │
...                               Embeddings           │
[img_N] → ViT Encoder → [e_N]                        │
                                                       │
                    Similarity Matrix (N × N)          │
                    [e_i · t_j / τ]                   │
                                                       │
[txt_1] → Text Encoder → [t_1]                       │
[txt_2] → Text Encoder → [t_2]                       │
...                                                    │
[txt_N] → Text Encoder → [t_N]                       │
                                                       │
Texts ───────────────────────────────────────────────┘

Loss: cross-entropy on diagonal (matching pairs) vs off-diagonal
```

### Vision Transformer (ViT)

```
Input Image (224×224×3)
        │
        ▼ Split into patches
[P1][P2]...[P196]     16×16 pixel patches, 196 total
        │
        ▼ Linear projection
[E1][E2]...[E196]     Each patch → 768-dim vector
        │
        ▼ Prepend [CLS] token + positional embeddings
[CLS][E1][E2]...[E196]
        │
        ▼ Transformer Encoder (12 layers)
[H_CLS][H1][H2]...[H196]
        │
        ▼ [CLS] token → classification head
     Class logits
```

---

## Setup

```bash
pip install torch torchvision transformers datasets Pillow torchaudio
```

---

## Running the Implementations

```bash
python clip_from_scratch.py     # CLIP architecture + contrastive training
python vit_implementation.py    # Vision Transformer on CIFAR-10
python whisper_exploration.py   # Audio transcription pipeline
```

---

## Key Takeaways

| Model | Core Insight |
|---|---|
| CLIP | Zero-shot transfer: classify any category by embedding its text description |
| ViT | Patches = tokens: self-attention over image regions, no CNNs needed |
| Whisper | Log-mel spectrogram → encoder-decoder: audio transcription as translation |
| Contrastive loss | InfoNCE: pull matching pairs together, push non-matching apart |
| Cross-modal alignment | Text and image live in the same embedding space after training |

---

## References

- [CLIP: Learning Transferable Visual Models from Natural Language](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Radford et al., 2022
- [Flamingo: a Visual Language Model](https://arxiv.org/abs/2204.14198) — Alayrac et al., 2022
