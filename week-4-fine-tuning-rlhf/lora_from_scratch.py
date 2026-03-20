"""
lora_from_scratch.py
====================
Implements LoRA (Low-Rank Adaptation) from scratch and applies it to a
GPT-2 style transformer. Demonstrates parameter count reduction and
shows how adapters are merged back into base weights after training.

LoRA key insight:
    Pre-trained weight matrices W₀ have low "intrinsic rank" during adaptation.
    Rather than updating W₀ directly, we learn ΔW ≈ BA where:
        B ∈ ℝ^(d_out × r),  A ∈ ℝ^(r × d_in),  rank r << min(d_in, d_out)

    During the forward pass: h = W₀x + (α/r) * B(Ax)
    At merge time: W_merged = W₀ + (α/r) * BA  (standard forward pass, no overhead)

Usage:
    python lora_from_scratch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. LoRA Linear Layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    A Linear layer augmented with a LoRA adapter.

    The base weight W₀ is frozen. Adapters A and B are the only
    trainable parameters.

    Initialisation convention:
        A: random Gaussian (provides signal from the start)
        B: zeros (ensures ΔW = 0 at initialisation, preserving base model behaviour)
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        rank:         int   = 16,
        alpha:        float = 32.0,
        dropout:      float = 0.05,
        bias:         bool  = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.alpha        = alpha
        self.scaling      = alpha / rank   # The (α/r) factor

        # Base weight — frozen after loading pretrained weights
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.bias = None

        # LoRA adapters — the ONLY trainable parameters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Optional dropout on the adapter input for regularisation
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Standard linear init for base; Kaiming for A, zeros for B."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # A: small random init (paper uses Kaiming)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: zeros — ΔW = BA = 0 at init, base model is unchanged
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base computation (frozen)
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA adapter computation
        # x → A (project down) → B (project back up) → scale
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),  # (batch, r)
            self.lora_B,                                    # (batch, d_out)
        ) * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA adapter into base weights and return a standard Linear layer.

        After merging: W_merged = W₀ + (α/r) * BA
        The merged layer has NO runtime overhead — identical to a standard Linear.
        This is called before serving the fine-tuned model.
        """
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        merged.weight.data = self.weight.data + self.scaling * (self.lora_B @ self.lora_A)
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged

    def trainable_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def total_parameters(self) -> int:
        base = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
        return base + self.trainable_parameters()


# ---------------------------------------------------------------------------
# 2. Apply LoRA to a Model
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank:             int   = 16
    alpha:            float = 32.0
    dropout:          float = 0.05
    target_modules:   List[str] = None  # None = apply to all Linear layers

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "query", "value",
                                   "c_attn", "c_proj"]


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Replace targeted Linear layers in a model with LoRALinear layers.

    Strategy: only apply LoRA to attention projections (Q, V are most impactful).
    Applying to FFN layers adds more parameters with diminishing returns.

    Args:
        model:  Any PyTorch module
        config: LoRA hyperparameters and target layer names

    Returns:
        Modified model with LoRA layers (base weights frozen, adapters trainable)
    """
    replaced_count = 0

    for name, module in list(model.named_modules()):
        # Check if this module's name matches a target pattern
        should_replace = any(target in name for target in config.target_modules)

        if should_replace and isinstance(module, nn.Linear):
            # Find the parent module to swap the child
            parts   = name.split('.')
            parent  = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]

            # Create LoRA layer with same dimensions
            lora_layer = LoRALinear(
                in_features  = module.in_features,
                out_features = module.out_features,
                rank         = config.rank,
                alpha        = config.alpha,
                dropout      = config.dropout,
                bias         = module.bias is not None,
            )
            # Copy pretrained weights into the frozen base
            lora_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.bias.data = module.bias.data.clone()

            setattr(parent, child_name, lora_layer)
            replaced_count += 1

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print(f"  Replaced {replaced_count} Linear layers with LoRA")
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and LoRA parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora      = sum(p.numel() for n, p in model.named_parameters()
                    if 'lora_A' in n or 'lora_B' in n)
    return {"total": total, "trainable": trainable, "lora": lora}


# ---------------------------------------------------------------------------
# 3. Simple Model for Demo
# ---------------------------------------------------------------------------

class SimpleTransformerBlock(nn.Module):
    """Minimal transformer block to demonstrate LoRA application."""
    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.ffn1   = nn.Linear(d_model, d_model * 4)
        self.ffn2   = nn.Linear(d_model * 4, d_model)
        self.ln1    = nn.LayerNorm(d_model)
        self.ln2    = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.d_model // self.n_heads
        # Simplified attention (no masking for demo)
        q = self.q_proj(x).view(B, T, self.n_heads, h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, h).transpose(1, 2)
        attn = F.softmax(q @ k.transpose(-2,-1) / math.sqrt(h), dim=-1) @ v
        attn = attn.transpose(1,2).contiguous().view(B, T, C)
        x = self.ln1(x + self.o_proj(attn))
        x = self.ln2(x + self.ffn2(F.gelu(self.ffn1(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([SimpleTransformerBlock(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# 4. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("LoRA from Scratch — Demo")
    print("=" * 65)

    # --- Single LoRA layer test ---
    print("\n[1/4] Single LoRALinear layer test...")
    layer = LoRALinear(in_features=512, out_features=512, rank=16, alpha=32)
    x = torch.randn(4, 10, 512)
    out = layer(x)
    assert out.shape == (4, 10, 512)

    trainable = layer.trainable_parameters()
    total     = layer.total_parameters()
    print(f"  Total params:     {total:,}")
    print(f"  Trainable (LoRA): {trainable:,}  ({100*trainable/total:.2f}%)")

    # Verify B=zeros → output identical to base linear at init
    base_linear = nn.Linear(512, 512)
    base_linear.weight.data = layer.weight.data.clone()
    base_linear.bias.data   = layer.bias.data.clone()
    with torch.no_grad():
        base_out = base_linear(x)
        lora_out = layer(x)
    max_diff = (base_out - lora_out).abs().max().item()
    print(f"  Δ(LoRA, base) at init: {max_diff:.2e}  (should be ~0)")

    # --- Apply LoRA to TinyGPT ---
    print("\n[2/4] Apply LoRA to TinyGPT...")
    model = TinyGPT(vocab_size=1000, d_model=256, n_layers=4)
    before = count_parameters(model)
    print(f"  Before LoRA: {before['total']:,} total, {before['trainable']:,} trainable")

    config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    model  = apply_lora(model, config)
    after  = count_parameters(model)

    print(f"  After LoRA:  {after['total']:,} total, {after['trainable']:,} trainable")
    print(f"  LoRA params: {after['lora']:,}  ({100*after['lora']/after['total']:.2f}% of total)")
    print(f"  Training efficiency: {after['trainable']/before['total']*100:.2f}% of params need gradients")

    # --- Forward pass with LoRA active ---
    print("\n[3/4] Forward pass + mock training step...")
    token_ids = torch.randint(0, 1000, (2, 16))
    logits    = model(token_ids)
    labels    = token_ids.roll(-1)
    loss      = F.cross_entropy(logits[:, :-1].reshape(-1, 1000), labels[:, :-1].reshape(-1))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4
    )
    loss.backward()
    optimizer.step()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients flow to LoRA params only: ✅")

    # Verify frozen params have no gradients
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            assert param.grad is None, f"{name} should be frozen!"
    print(f"  Base weights remain frozen: ✅")

    # --- Merge LoRA back into base weights ---
    print("\n[4/4] Merge LoRA adapters into base weights...")
    x_test    = torch.randn(2, 8, 256)
    # Find a LoRA layer and demonstrate merge
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out_before = module(x_test)
            merged     = module.merge_weights()
            out_after  = merged(x_test)
            max_diff   = (out_before - out_after).abs().max().item()
            print(f"  Layer '{name}': merge error = {max_diff:.2e}  (should be ~1e-6)")
            break

    print("\n  Merged model:")
    print("  - No LoRA layers → same speed as original at inference")
    print("  - Knowledge from fine-tuning baked into weights")
    print("  - Can distribute as a standalone model")

    print("\n" + "=" * 65)
    print("✅ LoRA implementation complete!")
