# Week 5: Diffusion Models

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

Diffusion models are generative models that learn to reverse a Gaussian noise process. They now dominate image generation (Stable Diffusion, DALL-E 3, Midjourney) and are expanding into audio and video.

**Core concepts:**
- DDPM forward process (noise scheduling)
- DDPM reverse process (learned denoising)
- U-Net architecture with timestep conditioning
- Latent Diffusion Models (Stable Diffusion)
- ControlNet for conditional generation

---

## Architecture Diagrams

### DDPM Forward & Reverse Process

```
Forward (fixed, analytical):
x_0 ─────────────────────────────────────────────────────▶ x_T
(clean)  +ε at step 1  +ε at step 2  ...  +ε at step T  (pure noise)

Closed-form: x_t = √ā_t · x_0 + √(1-ā_t) · ε,  ε ~ N(0,I)

Reverse (learned, iterative):
x_T ─────────────────────────────────────────────────────▶ x_0
(noise)  -ε_θ(x_T,T)  -ε_θ(x_{T-1},T-1)  ...          (clean image)

The model ε_θ(x_t, t) predicts the noise ε added at step t.
```

### U-Net Architecture (Denoiser)

```
Input: x_t (noisy image) + t (timestep embedding)

Encoder:                      Decoder:
Conv 3×3                      Conv 3×3
↓                             ↑  ←────── skip connection
ResBlock + AttnBlock          ResBlock + AttnBlock
↓ (downsample)                ↑  (upsample)
ResBlock + AttnBlock          ResBlock + AttnBlock
↓ (downsample)                ↑  (upsample)
ResBlock + AttnBlock          ResBlock + AttnBlock
         ↓                    ↑
         ──── Bottleneck ──────
              (ResBlock + AttnBlock)

Output: predicted noise ε̂ (same shape as input)
```

### Latent Diffusion (Stable Diffusion)

```
Image (512×512×3)
      │
      ▼ VAE Encoder (frozen)
Latent (64×64×4)
      │
      ▼ DDPM (runs here — 64× cheaper!)
Denoised Latent
      │
      ▼ VAE Decoder (frozen)
Generated Image (512×512×3)
```

---

## Mathematical Intuition

### Noise Schedule

```
β_t: variance of noise added at step t (increases from β_1≈0.0001 to β_T≈0.02)
α_t = 1 - β_t
ā_t = Π_{s=1}^{t} α_s  (cumulative product — decreases toward 0)

Signal-to-noise ratio at step t: ā_t / (1 - ā_t)
  t=0:   SNR → ∞      (pure signal)
  t=T:   SNR → 0      (pure noise)
```

### Training Objective

```
L_simple = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]
```
The model minimises mean squared error between the *actual noise added* and its *prediction of the noise*. Simpler than the full ELBO and empirically works better.

---

## Setup

```bash
pip install torch torchvision matplotlib tqdm einops
```

---

## Running the Implementations

```bash
python ddpm_from_scratch.py        # Full DDPM implementation + MNIST training
python stable_diffusion_explore.py  # ControlNet and SD pipeline exploration
```

---

## Key Takeaways

| Concept | Insight |
|---|---|
| Forward process closed form | Sample any x_t directly without simulating t steps |
| Noise prediction vs score | Equivalent mathematically; noise prediction is simpler |
| Timestep conditioning | Same U-Net handles all noise levels via t embedding |
| Latent diffusion | ~64× cheaper by running in compressed space |
| DDIM sampling | Deterministic, 10–50 steps instead of 1000 |

---

## References

- [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
- [Latent Diffusion Models (Stable Diffusion)](https://arxiv.org/abs/2112.10752) — Rombach et al., 2022
- [ControlNet](https://arxiv.org/abs/2302.05543) — Zhang et al., 2023
- [DDIM: Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al., 2020
