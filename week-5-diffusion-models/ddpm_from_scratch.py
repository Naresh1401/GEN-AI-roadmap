"""
ddpm_from_scratch.py
====================
Implements DDPM (Denoising Diffusion Probabilistic Models) from scratch
and trains on MNIST.

DDPM overview (Ho et al., 2020):
    Forward process: gradually add Gaussian noise over T steps
    Reverse process: train a neural network to denoise step by step
    Training: predict the added noise ε given (x_t, t)
    Inference: start from x_T ~ N(0,I), iteratively denoise to x_0

Key trick: the forward process has a closed-form solution, so we can
compute x_t at any timestep t directly from x_0 and a noise sample.

Usage:
    python ddpm_from_scratch.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Noise Schedule
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """
    Linear noise schedule for DDPM.

    Precomputes all the constants needed for forward process sampling
    and reverse process denoising.
    """

    def __init__(
        self,
        T:       int   = 1000,
        beta_1:  float = 1e-4,
        beta_T:  float = 0.02,
        device:  torch.device = torch.device('cpu'),
    ):
        self.T = T

        # Linear schedule: β increases from β_1 to β_T
        self.beta = torch.linspace(beta_1, beta_T, T, device=device)

        # Precompute all derived quantities for efficiency
        self.alpha      = 1.0 - self.beta                       # α_t
        self.alpha_bar  = torch.cumprod(self.alpha, dim=0)       # ā_t = Π α_s
        self.sqrt_alpha_bar        = self.alpha_bar.sqrt()
        self.sqrt_one_minus_ab     = (1.0 - self.alpha_bar).sqrt()

        # For reverse process update step
        self.sqrt_recip_alpha      = (1.0 / self.alpha).sqrt()
        self.alpha_bar_prev        = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_variance    = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t:   torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: sample x_t given x_0 and timestep t.

        x_t = sqrt(ā_t) * x_0 + sqrt(1 - ā_t) * ε,  ε ~ N(0, I)

        This is the closed-form solution — no need to simulate t steps.

        Args:
            x_0: clean image (B, C, H, W)
            t:   timestep indices (B,)
            eps: optional pre-sampled noise

        Returns:
            x_t: noisy image at timestep t
            eps: the noise that was added (needed for loss computation)
        """
        if eps is None:
            eps = torch.randn_like(x_0)

        # Extract scalar coefficients for each sample in the batch
        # t is (B,) — we need to index into the schedule and broadcast
        sqrt_ab    = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab  = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)

        x_t = sqrt_ab * x_0 + sqrt_1mab * eps
        return x_t, eps

    @torch.no_grad()
    def p_sample(
        self,
        model:    nn.Module,
        x_t:      torch.Tensor,
        t:        int,
    ) -> torch.Tensor:
        """
        One step of the reverse process: denoise from x_t to x_{t-1}.

        x_{t-1} = (1/√α_t) * (x_t - β_t/√(1-ā_t) * ε_θ(x_t, t)) + σ_t * z

        Args:
            model: the noise-predicting U-Net
            x_t:   current noisy image (B, C, H, W)
            t:     current timestep (scalar int)

        Returns:
            x_{t-1}: slightly less noisy image
        """
        B     = x_t.shape[0]
        t_vec = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # Predict noise ε_θ(x_t, t)
        eps_pred = model(x_t, t_vec)

        beta_t   = self.beta[t]
        sqrt_1mab = self.sqrt_one_minus_ab[t]
        recip_sqrt_alpha = self.sqrt_recip_alpha[t]

        # Compute mean of p(x_{t-1} | x_t)
        mean = recip_sqrt_alpha * (x_t - beta_t / sqrt_1mab * eps_pred)

        if t == 0:
            return mean  # Last step: no noise
        else:
            # Variance for the reverse process
            var = self.posterior_variance[t]
            z   = torch.randn_like(x_t)
            return mean + var.sqrt() * z

    @torch.no_grad()
    def p_sample_loop(
        self,
        model:    nn.Module,
        shape:    Tuple,
        device:   torch.device,
    ) -> torch.Tensor:
        """
        Full reverse process: generate images from pure noise.

        Args:
            shape: (B, C, H, W) — shape of images to generate

        Returns:
            x_0: generated images
        """
        model.eval()
        x = torch.randn(shape, device=device)  # Start from pure noise

        for t in tqdm(reversed(range(self.T)), desc="Sampling", total=self.T, leave=False):
            x = self.p_sample(model, x, t)

        return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# 2. Timestep Sinusoidal Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Embeds scalar timestep t into a vector for conditioning the U-Net.

    Uses sinusoidal encoding (same idea as Transformer position encodings)
    followed by an MLP to produce the final embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) scalar timesteps
        half    = self.dim // 2
        freqs   = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args    = t[:, None].float() * freqs[None]
        emb     = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# 3. U-Net Denoiser
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with group normalisation and optional timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Add timestep embedding to each spatial position
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """
    Simplified U-Net denoiser for MNIST.

    Architecture:
        Encoder: 3 levels of ResBlocks + downsampling
        Bottleneck: ResBlock
        Decoder: 3 levels of ResBlocks + upsampling, with skip connections
        Output: 1×1 conv to match input channels
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch:     int = 64,
        t_dim:       int = 128,
    ):
        super().__init__()
        self.t_emb = SinusoidalTimestepEmbedding(t_dim)

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Encoder
        self.down1 = ResBlock(base_ch,     base_ch * 2, t_dim)
        self.down2 = ResBlock(base_ch * 2, base_ch * 4, t_dim)
        self.pool  = nn.MaxPool2d(2)

        # Bottleneck
        self.mid   = ResBlock(base_ch * 4, base_ch * 4, t_dim)

        # Decoder
        self.up2   = ResBlock(base_ch * 4 + base_ch * 4, base_ch * 2, t_dim)
        self.up1   = ResBlock(base_ch * 2 + base_ch * 2, base_ch,     t_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Output
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise ε given noisy image x_t and timestep t.

        Args:
            x: noisy image (B, C, H, W)
            t: timestep indices (B,)

        Returns:
            Predicted noise (B, C, H, W)
        """
        t_emb = self.t_emb(t)

        x0 = self.init_conv(x)        # (B, 64, H, W)

        # Encoder
        d1 = self.down1(x0, t_emb)    # (B, 128, H, W)
        d2 = self.down2(self.pool(d1), t_emb)  # (B, 256, H/2, W/2)

        # Bottleneck
        m  = self.mid(self.pool(d2), t_emb)    # (B, 256, H/4, W/4)

        # Decoder with skip connections
        u2 = self.up2(torch.cat([self.upsample(m), d2], dim=1), t_emb)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(u1)))


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def train_ddpm(
    model:      nn.Module,
    schedule:   NoiseSchedule,
    dataloader: DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    n_epochs:   int = 5,
) -> list:
    """Train the DDPM denoiser with the simple noise-prediction objective."""
    model.train()
    loss_history = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            images = images.to(device)
            B      = images.shape[0]

            # Sample random timesteps for each image in the batch
            t = torch.randint(0, schedule.T, (B,), device=device)

            # Forward process: add noise
            x_t, eps = schedule.q_sample(images, t)

            # Predict noise and compute MSE loss
            eps_pred = model(x_t, t)
            loss     = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents unstable early training steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / n_batches
        loss_history.append(avg)
        print(f"  Epoch {epoch+1}: loss = {avg:.4f}")

    return loss_history


# ---------------------------------------------------------------------------
# 5. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("DDPM from Scratch — MNIST Training")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    T         = 300   # Fewer timesteps for fast demo (paper uses 1000)
    IMG_SIZE  = 28
    BATCH     = 128
    EPOCHS    = 3     # Increase to 20+ for quality results
    LR        = 2e-4

    # Noise schedule
    schedule = NoiseSchedule(T=T, device=device)

    # Dataset: MNIST, normalised to [-1, 1]
    print("\n[1/4] Loading MNIST...")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    dataset    = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(dataset)} images")

    # Model
    print("\n[2/4] Building U-Net denoiser...")
    model = SimpleUNet(in_channels=1, base_ch=32, t_dim=64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Test forward noise schedule
    print("\n[3/4] Verifying noise schedule...")
    test_images, _ = next(iter(dataloader))
    test_images = test_images[:8].to(device)
    t_test = torch.tensor([0, 50, 100, 200, 299], device=device)
    print("  Timestep → signal-to-noise ratio:")
    for t in t_test:
        snr = (schedule.alpha_bar[t] / (1 - schedule.alpha_bar[t])).item()
        print(f"    t={int(t):4d}: SNR={snr:.4f}")

    # Quick forward process visualisation
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    sample = test_images[0:1]
    axes[0].imshow(sample.cpu().squeeze(), cmap='gray')
    axes[0].set_title('Original')
    for i, t_val in enumerate([50, 100, 150, 200, 299]):
        t_t = torch.tensor([t_val], device=device)
        noisy, _ = schedule.q_sample(sample, t_t)
        axes[i+1].imshow(noisy.cpu().squeeze(), cmap='gray')
        axes[i+1].set_title(f't={t_val}')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("forward_process.png", dpi=100)
    print("  Forward process visualisation saved → forward_process.png")

    # Training
    print(f"\n[4/4] Training DDPM for {EPOCHS} epochs...")
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_history = train_ddpm(model, schedule, dataloader, optimizer, device, EPOCHS)
    print(f"  Final loss: {loss_history[-1]:.4f}")

    # Generate samples
    print("\n  Generating samples...")
    gen = schedule.p_sample_loop(model, shape=(16, 1, IMG_SIZE, IMG_SIZE), device=device)
    grid = torchvision.utils.make_grid(gen * 0.5 + 0.5, nrow=4)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.cpu().permute(1,2,0).clamp(0,1), cmap='gray')
    plt.axis('off')
    plt.title(f'Generated samples (epoch {EPOCHS})')
    plt.savefig("generated_samples.png", dpi=100)
    print("  Generated samples saved → generated_samples.png")

    torch.save(model.state_dict(), "ddpm_mnist.pt")
    print("  Model saved → ddpm_mnist.pt")

    print("\n" + "=" * 65)
    print("✅ DDPM training complete!")
    print("Train for 20+ epochs for recognisable digit shapes.")
