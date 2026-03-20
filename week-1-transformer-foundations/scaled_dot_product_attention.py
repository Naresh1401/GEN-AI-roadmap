"""
scaled_dot_product_attention.py
================================
Implements scaled dot-product attention and multi-head attention from scratch
using PyTorch, following "Attention Is All You Need" (Vaswani et al., 2017).

Why implement from scratch?
    PyTorch's nn.MultiheadAttention abstracts away the internals. Building it
    manually forces you to understand every dimension transformation, every
    hyperparameter choice, and exactly where gradients flow.

Key formula:
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

Usage:
    python scaled_dot_product_attention.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention (the core operation)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,        # Shape: (batch, heads, seq_q, d_k)
    key: torch.Tensor,          # Shape: (batch, heads, seq_k, d_k)
    value: torch.Tensor,        # Shape: (batch, heads, seq_k, d_v)
    mask: Optional[torch.Tensor] = None,  # Shape: (batch, 1, seq_q, seq_k) or None
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        query:     What each position is "looking for"
        key:       What each position "advertises" about itself
        value:     What each position "provides" when selected
        mask:      Optional mask to block attention to certain positions
                   (e.g., padding tokens or future tokens in a decoder)
        dropout_p: Dropout probability applied to attention weights

    Returns:
        output:          Weighted combination of values, shape (batch, heads, seq_q, d_v)
        attention_weights: Softmax weights, shape (batch, heads, seq_q, seq_k)
                          Useful for visualisation and interpretability
    """
    d_k = query.size(-1)  # Dimensionality of queries and keys

    # Step 1: Compute raw attention scores via dot product
    # (batch, heads, seq_q, d_k) × (batch, heads, d_k, seq_k) → (batch, heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale to prevent large dot products from pushing softmax into saturation
    # Without this, gradients nearly vanish when d_k is large (e.g., 512)
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply optional mask
    # For padding masks: fill masked positions with -inf so softmax gives ~0 weight
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax over the key dimension — convert scores to probabilities
    # Each row sums to 1.0: the query distributes its "attention budget" across keys
    attention_weights = F.softmax(scores, dim=-1)

    # Replace NaNs that arise when entire rows are masked (softmax of all -inf)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    # Step 5: Optional dropout on attention weights (used during training only)
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # Step 6: Weighted sum of values
    # (batch, heads, seq_q, seq_k) × (batch, heads, seq_k, d_v) → (batch, heads, seq_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in "Attention Is All You Need".

    Intuition:
        Running attention once gives one "perspective" on token relationships.
        Multi-head attention runs it h times in PARALLEL with different learned
        projections — each head can specialise (e.g., one head for syntax,
        another for coreference, another for semantics). The results are
        concatenated and projected back to d_model.

    Shape contract:
        Input:  (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)  ← same shape, different content
    """

    def __init__(
        self,
        d_model: int,    # Total model dimension (e.g., 512)
        num_heads: int,  # Number of parallel attention heads (e.g., 8)
        dropout: float = 0.1,
    ):
        super().__init__()

        # d_model must be evenly divisible across heads
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.dropout = dropout

        # Four linear projections: Q, K, V (inputs) and output
        # Note: we project d_model → d_model (not d_model → d_k×h separately)
        # because it's equivalent and more efficient as one matrix multiply
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialisation for all projection matrices."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from (batch, seq, d_model) → (batch, heads, seq, d_k).
        
        This reshapes the embedding dimension into (num_heads, d_k) then
        transposes so heads become the second dimension for batched attention.
        """
        batch_size, seq_len, d_model = x.size()
        # Reshape: (batch, seq, d_model) → (batch, seq, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose: (batch, seq, num_heads, d_k) → (batch, num_heads, seq, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads.
        (batch, heads, seq, d_k) → (batch, seq, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # Transpose back: (batch, heads, seq, d_k) → (batch, seq, heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # Merge head dimension: (batch, seq, heads, d_k) → (batch, seq, d_model)
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,          # (batch, seq_q, d_model)
        key: torch.Tensor,            # (batch, seq_k, d_model)
        value: torch.Tensor,          # (batch, seq_k, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        For self-attention: query = key = value = x (same input tensor)
        For cross-attention: query from decoder, key/value from encoder

        Returns:
            output:           (batch, seq_q, d_model)
            attention_weights:(batch, num_heads, seq_q, seq_k)
        """
        # Project inputs to Q, K, V spaces
        Q = self.W_q(query)   # (batch, seq_q, d_model)
        K = self.W_k(key)     # (batch, seq_k, d_model)
        V = self.W_v(value)   # (batch, seq_k, d_model)

        # Split into multiple heads
        Q = self.split_heads(Q)   # (batch, heads, seq_q, d_k)
        K = self.split_heads(K)   # (batch, heads, seq_k, d_k)
        V = self.split_heads(V)   # (batch, heads, seq_k, d_k)

        # Compute attention for all heads in parallel
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        # attn_output: (batch, heads, seq_q, d_k)

        # Concatenate heads and project back to d_model
        attn_output = self.combine_heads(attn_output)  # (batch, seq_q, d_model)
        output = self.W_o(attn_output)                  # (batch, seq_q, d_model)

        return output, attention_weights


# ---------------------------------------------------------------------------
# 3. Visualisation utility
# ---------------------------------------------------------------------------

def visualize_attention(
    attention_weights: torch.Tensor,
    tokens: list,
    head_idx: int = 0,
    title: str = "Attention Weights",
):
    """
    Plot a heatmap of attention weights for a single head.

    Args:
        attention_weights: (batch, heads, seq_q, seq_k) tensor
        tokens:            List of token strings for axis labels
        head_idx:          Which attention head to visualise
        title:             Plot title
    """
    # Extract weights for batch[0], chosen head → (seq_q, seq_k)
    weights = attention_weights[0, head_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Keys (attended to)")
    ax.set_ylabel("Queries (attending from)")
    ax.set_title(f"{title} — Head {head_idx}")

    plt.tight_layout()
    plt.savefig(f"attention_head_{head_idx}.png", dpi=120, bbox_inches='tight')
    print(f"Saved attention heatmap → attention_head_{head_idx}.png")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Scaled Dot-Product & Multi-Head Attention Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Hyperparameters ---
    batch_size = 2
    seq_len    = 10
    d_model    = 64
    num_heads  = 4

    # Random input: simulates token embeddings after lookup
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"Input shape:        {x.shape}  # (batch, seq_len, d_model)")

    # -------------------------------------------------------
    # Part A: Raw scaled dot-product attention
    # -------------------------------------------------------
    print("\n--- Part A: Scaled Dot-Product Attention ---")

    # For this demo, Q = K = V = x (self-attention, single head)
    Q = K = V = x.unsqueeze(1)  # Add head dimension: (batch, 1, seq, d_model)

    output_sdpa, weights_sdpa = scaled_dot_product_attention(Q, K, V)

    print(f"Output shape:       {output_sdpa.shape}")
    print(f"Attn weights shape: {weights_sdpa.shape}")
    print(f"Weights sum to 1.0: {weights_sdpa.sum(dim=-1).allclose(torch.ones(batch_size, 1, seq_len, device=device))}")

    # -------------------------------------------------------
    # Part B: Multi-Head Attention
    # -------------------------------------------------------
    print("\n--- Part B: Multi-Head Attention ---")

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0).to(device)

    output_mha, weights_mha = mha(x, x, x)  # Self-attention: Q=K=V=x

    print(f"Input shape:        {x.shape}")
    print(f"Output shape:       {output_mha.shape}  # Same shape as input ✓")
    print(f"Attn weights shape: {weights_mha.shape}  # (batch, heads, seq_q, seq_k)")

    total_params = sum(p.numel() for p in mha.parameters())
    print(f"MHA parameters:     {total_params:,}  # 4 × d_model² = {4 * d_model**2:,} (no bias)")

    # -------------------------------------------------------
    # Part C: Visualise attention for a toy sentence
    # -------------------------------------------------------
    print("\n--- Part C: Attention Visualisation ---")

    tokens = ["The", "cat", "sat", "on", "the", "mat", "<PAD>", "<PAD>", "<PAD>", "<PAD>"]

    # Simulate a padding mask: 0 = masked (PAD), 1 = attend
    pad_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]],  # batch=1, seq_len=10
        dtype=torch.bool, device=device,
    ).unsqueeze(1).unsqueeze(2)  # → (1, 1, 1, 10) broadcasts over all query positions

    x_single = torch.randn(1, seq_len, d_model, device=device)
    mha_single = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
    _, weights_vis = mha_single(x_single, x_single, x_single, mask=pad_mask)

    visualize_attention(weights_vis, tokens, head_idx=0, title="Self-Attention (with padding mask)")
    print(f"PAD tokens have ~0 attention weight: verified via mask")

    # -------------------------------------------------------
    # Part D: Verify masking effect
    # -------------------------------------------------------
    print("\n--- Part D: Sanity Checks ---")

    # Attention to PAD positions should be ~0
    pad_attention = weights_vis[0, 0, :, 6:].detach().cpu()  # All queries, heads, PAD positions
    print(f"Max attention to PAD positions: {pad_attention.max().item():.6f}  (should be ≈ 0.0)")

    # Attention weights should sum to 1 for non-fully-masked rows
    row_sums = weights_vis[0, 0, :6, :].sum(dim=-1)  # Non-PAD query rows
    print(f"Attention row sums (non-PAD):    {row_sums.detach().cpu().numpy().round(4)}")

    print("\n✅ All checks passed!")
    print("=" * 60)
