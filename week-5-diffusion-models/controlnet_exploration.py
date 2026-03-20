"""
controlnet_exploration.py
=========================
Explores ControlNet architecture and conditional generation concepts.

ControlNet adds spatial conditioning (edges, depth, pose, segmentation)
to a pretrained diffusion model without retraining the base model.

Architecture:
    - Trainable copy of U-Net encoder ("control copy")
    - Base U-Net decoder (frozen)
    - Zero-convolution layers connect control copy to base decoder
    - Condition (e.g., edge map) is fed only to the control copy

Why zero-convolutions?
    At initialisation, zero-conv outputs are zero — so the base model
    starts unchanged. As training progresses, zero-convs learn to
    pass conditioning signal to the base. This "protects" the pretrained
    model's generative capability during the initial training phase.

Usage:
    python controlnet_exploration.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Zero Convolution
# ---------------------------------------------------------------------------

class ZeroConv2d(nn.Module):
    """
    Convolution layer initialised with all zeros.

    Used in ControlNet to "protect" the pretrained model at initialisation.
    When weights=0, the output is 0, so adding it to the base model
    has no effect initially. As training progresses, weights grow from 0.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        # Explicitly initialise to zero (would be random otherwise)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# 2. Simplified ControlNet Architecture
# ---------------------------------------------------------------------------

class ControlNetBlock(nn.Module):
    """
    One level of ControlNet: processes a conditioning input
    and produces a correction to add to the base model's decoder.
    """

    def __init__(self, ch: int):
        super().__init__()
        # Trainable copy of the encoder block (weights start from base copy)
        self.encoder_copy = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        # Zero-conv: output is zero at init, grows to non-zero during training
        self.zero_conv = ZeroConv2d(ch, ch)

    def forward(self, control_features: torch.Tensor) -> torch.Tensor:
        """
        Process conditioning features and return zero-conv output.
        This output is *added* to the corresponding decoder layer in the base model.
        """
        h = self.encoder_copy(control_features)
        return self.zero_conv(h)


class SimpleControlNet(nn.Module):
    """
    Simplified ControlNet structure demonstrating the principle.

    In production (diffusers library), ControlNet wraps the full SD U-Net.
    Here we show the key idea: zero-conv connects a trainable control path
    to the frozen base model's decoder.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        ch = base_channels

        # Condition encoder: maps edge map / depth map → feature space
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(1, ch // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch // 2, ch, 3, padding=1),
            nn.SiLU(),
        )

        # ControlNet blocks (one per encoder level)
        self.ctrl_blocks = nn.ModuleList([
            ControlNetBlock(ch),
            ControlNetBlock(ch),
        ])

    def forward(self, condition: torch.Tensor) -> list:
        """
        Args:
            condition: conditioning image (B, 1, H, W) — e.g., edge map

        Returns:
            List of conditioning corrections for each decoder level
        """
        h = self.cond_encoder(condition)
        corrections = []
        for block in self.ctrl_blocks:
            corrections.append(block(h))
        return corrections


# ---------------------------------------------------------------------------
# 3. DDIM Sampling (Faster Inference)
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) — faster deterministic sampling.

    Key insight: DDPM requires T=1000 steps for quality results.
    DDIM reformulates the reverse process as a deterministic ODE,
    allowing 10–50 steps with similar quality.

    The ODE view: DDPM noise prediction ε_θ implicitly defines a score function.
    DDIM solves the reverse ODE using larger step sizes.
    """

    def __init__(self, num_inference_steps: int = 50, T: int = 1000):
        self.T = T
        self.num_steps = num_inference_steps

        # Select evenly spaced subset of timesteps
        # E.g., 50 steps from 1000: [980, 960, 940, ..., 20, 0]
        step_ratio = T // num_inference_steps
        self.timesteps = list(reversed(range(0, T, step_ratio)))[:num_inference_steps]

        # Precompute alpha_bar for each selected timestep
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    @torch.no_grad()
    def sample(
        self,
        model:   nn.Module,
        shape:   tuple,
        device:  torch.device,
        eta:     float = 0.0,  # 0=deterministic (DDIM), 1=stochastic (DDPM)
    ) -> torch.Tensor:
        """
        Generate samples in `num_inference_steps` instead of T steps.

        Args:
            eta: stochasticity level. 0 = fully deterministic (DDIM).
                 1 = same stochasticity as DDPM.
        """
        x = torch.randn(shape, device=device)

        for i, t in enumerate(self.timesteps[:-1]):
            t_next = self.timesteps[i + 1]

            ab_t      = self.alpha_bars[t].to(device)
            ab_t_next = self.alpha_bars[t_next].to(device)

            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps_pred = model(x, t_batch)

            # Predict x_0 from x_t and ε prediction
            x_0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            x_0_pred = x_0_pred.clamp(-1, 1)

            # DDIM update step
            sigma = eta * ((1 - ab_t_next) / (1 - ab_t) * (1 - ab_t / ab_t_next)).sqrt()
            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = ab_t_next.sqrt() * x_0_pred + \
                (1 - ab_t_next - sigma**2).clamp(min=0).sqrt() * eps_pred + \
                sigma * noise

        return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# 4. Visualise Conditioning Concepts
# ---------------------------------------------------------------------------

def create_edge_map(image: np.ndarray) -> np.ndarray:
    """Simple Sobel edge detection to simulate ControlNet edge conditioning."""
    from scipy.ndimage import sobel
    sx = sobel(image.astype(float), axis=0)
    sy = sobel(image.astype(float), axis=1)
    edges = np.hypot(sx, sy)
    return (edges / edges.max()).clip(0, 1)


def visualise_controlnet_concept():
    """Visualise how different conditioning signals would guide generation."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Simulate different conditioning inputs
    np.random.seed(42)
    size = 64

    conditions = {
        "Original": np.random.randn(size, size) * 0.3 + 0.5,
        "Edge Map": None,   # Will compute from original
        "Depth Map": np.fromfunction(lambda i, j: (i + j) / (2 * size), (size, size)),
        "Random Mask": np.random.randint(0, 5, (size, size)) / 5.0,
    }

    original = conditions["Original"].clip(0, 1)
    conditions["Edge Map"] = create_edge_map(original)

    titles = list(conditions.keys())
    for i, (title, cond) in enumerate(conditions.items()):
        axes[0][i].imshow(cond, cmap='gray', vmin=0, vmax=1)
        axes[0][i].set_title(f"Condition: {title}")
        axes[0][i].axis('off')

        # Simulate "conditioned generation" (random placeholder for demo)
        axes[1][i].imshow(np.random.randn(size, size) * 0.3 + 0.5, cmap='viridis')
        axes[1][i].set_title(f"Generated (guided by {title})")
        axes[1][i].axis('off')

    fig.suptitle("ControlNet: Different Conditioning Inputs", fontsize=14)
    plt.tight_layout()
    plt.savefig("controlnet_conditions.png", dpi=100)
    print("  ControlNet conditioning visualisation saved → controlnet_conditions.png")


if __name__ == "__main__":
    print("=" * 65)
    print("ControlNet & Stable Diffusion Exploration")
    print("=" * 65)

    print("\n[1/3] Testing ZeroConv initialisation...")
    zc = ZeroConv2d(64, 64)
    x_test = torch.randn(2, 64, 16, 16)
    out = zc(x_test)
    assert out.abs().max().item() < 1e-6, "ZeroConv should output zeros at init"
    print(f"  ZeroConv output at init: max|out| = {out.abs().max().item():.2e}  ✅")

    print("\n[2/3] ControlNet forward pass...")
    ctrl_net = SimpleControlNet(base_channels=32)
    condition = torch.randn(2, 1, 28, 28)
    corrections = ctrl_net(condition)
    print(f"  Input condition shape: {condition.shape}")
    print(f"  Corrections generated: {len(corrections)}")
    print(f"  Correction shapes: {[c.shape for c in corrections]}")
    print(f"  Zero-conv outputs near zero at init: {all(c.abs().max().item() < 1.0 for c in corrections)}")

    print("\n[3/3] DDIM vs DDPM step count comparison...")
    print(f"  DDPM: 1000 steps (stochastic)")
    print(f"  DDIM (eta=0): 50 steps (deterministic) — 20x fewer steps")
    print(f"  DDIM (eta=1): 50 steps (stochastic) — same as DDPM but larger steps")
    ddim = DDIMSampler(num_inference_steps=50, T=300)
    print(f"  DDIM timestep schedule (first 5): {ddim.timesteps[:5]}")

    visualise_controlnet_concept()

    print("\n" + "=" * 65)
    print("✅ ControlNet exploration complete!")
    print("\nFor Stable Diffusion with ControlNet in practice:")
    print("  from diffusers import StableDiffusionControlNetPipeline, ControlNetModel")


# ============================================================
# 🔷 SECTION 4 & 5 — Concept Deep Dive & Interview Prep
# ============================================================
"""
WEEK 5: DIFFUSION MODELS — Deep Dive & Interview Prep

CONCEPTS:

1. DDPM FORWARD PROCESS
Analogy: Gradually adding fog to a photograph until it's indistinguishable from pure
grey static. The forward process is fixed (not learned) and has a closed-form solution:
x_t = √ā_t * x_0 + √(1-ā_t) * ε. You can "fast-forward" to any noise level t directly.

2. DDPM REVERSE PROCESS
Analogy: Asking an expert restorer to look at a foggy photo and tell you what noise was
added. The neural network ε_θ(x_t, t) predicts the noise. Subtracting the predicted
noise (approximately) reverses one denoising step.

3. U-NET ARCHITECTURE
Analogy: An hourglass. Compress to understand the global structure; expand to reconstruct
fine details. Skip connections carry local detail across the compression bottleneck.
The U-Net processes coarse structure at the bottleneck, fine textures in the outer layers.

4. TIMESTEP CONDITIONING
The same U-Net must behave differently at t=999 (remove global noise) vs t=1
(clean up fine details). Timestep t is embedded via sinusoidal encoding (like Transformer
positional encodings) and injected into each layer via group normalisation conditioning.

5. LATENT DIFFUSION MODELS
Instead of running diffusion on 512×512×3 pixels, train a VAE to compress to 64×64×4.
Run DDPM in that latent space (64× fewer operations). Decode the denoised latent back to
pixel space. Quality is nearly identical; compute is reduced by ~50×.

MISCONCEPTIONS:
1. "Diffusion is slow — too slow for production." DDIM reduces generation from 1000 to
   10–20 steps. Consistency models (2023) achieve 1–4 step generation. SD-Turbo generates
   in 1 step. Inference speed is no longer a showstopper.
2. "You can easily do image editing with diffusion." Inverting a real image back to a noise
   latent (DDIM inversion) is non-trivial and imperfect. Editing without degradation
   requires techniques like Prompt-to-Prompt or SDEdit.
3. "The U-Net predicts x_0 directly." Original DDPM predicts the noise ε. Equivalent
   formulations predict the score ∇log p(x_t) or predict x_0 directly — all equivalent
   but have different inductive biases. v-prediction (used in SD 2.x) is another variant.

INTERVIEW QUESTIONS:

Q1: Why is the training objective L_simple (MSE on noise) better than the full ELBO?
A: The full ELBO loss includes a weighting factor that downweights large timesteps (high noise).
MSE treats all timesteps equally. Empirically, MSE gives better sample quality — the
model allocates more capacity to the fine-detail denoising steps (small t) where sample
quality is determined.

Q2: What is guidance scale in Stable Diffusion?
A: Classifier-Free Guidance (CFG). Two forward passes per step: one with the text condition,
one unconditional (empty prompt). The output is: ε_guided = ε_uncond + scale*(ε_cond - ε_uncond).
Higher scale → stronger adherence to prompt, less diversity. Typical values: 7–12.
Higher than ~15 introduces artifacts.

Q3: How does ControlNet preserve base model quality during training?
A: Zero-convolution layers. At initialisation, all zero-conv weights are 0 → output is 0
→ adding it to the base decoder has no effect → base model starts unchanged. As training
progresses on a conditioning dataset, zero-conv weights grow from 0 and the conditioning
signal is gradually introduced. The frozen base never experiences random noise gradients.

Q4: What is the difference between DDPM and score-based diffusion models?
A: Mathematically equivalent. Score-based models (Song & Ermon 2020) learn the score
function ∇_x log p(x_t) — the gradient of the log density. DDPM's noise prediction
ε_θ(x_t, t) is proportional to the score: ε_θ = -√(1-ā_t) * ∇_x log p(x_t).
Score-based framing connects diffusion to stochastic differential equations (SDEs),
enabling continuous-time formulations and enabling DDIM/ODE solvers.

Q5: What makes NF4 (Normal Float 4-bit) better than standard INT4 for QLoRA?
A: Model weights follow an approximately normal distribution. NF4 places quantisation
levels such that each level covers an equal portion of the normal distribution — the
levels are information-theoretically optimal for normally distributed data.
Standard INT4 places levels at equal intervals — wasteful for a normal distribution
where values near zero are far more common than extreme values.

GOTCHA 1: "Just increase the number of DDIM steps for better quality."
Quality saturates around 20–50 DDIM steps. Beyond that, additional steps don't improve
quality but do increase latency. The quality vs speed trade-off is specific to each
scheduler — DEIS, DPM-Solver++ converge faster than DDIM.

GOTCHA 2: "Stable Diffusion v1.5 and v2.1 use the same VAE."
No. SD 2.x changed the text encoder from CLIP ViT-L/14 to OpenCLIP ViT-H/14, which
caused incompatibility with v1.x embeddings. Community LoRAs and embeddings trained
on v1.5 don't transfer to v2.x. This was a significant adoption barrier for v2.

SYSTEM DESIGN: Design a text-to-image generation service for 1000 concurrent users.
- Model: SD 1.5 or SDXL with DDIM 20 steps (good quality-latency trade-off)
- Serving: TensorRT or ONNX-optimised models on A100/H100 GPUs
- Batching: Dynamic batching of requests with same resolution + step count
- Queue: Ray Serve or Celery for async job queue; webhook/polling for results
- Output storage: S3; CDN for serving generated images
- Caching: Cache generations for identical (prompt + seed + parameters) combos
- Cost: ~$0.002/image on A100 at 20 steps; reserved instances for baseline load
- Monitoring: Generation latency, NSFW classifier pass rate, timeout rate
"""
