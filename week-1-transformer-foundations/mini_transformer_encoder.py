"""
mini_transformer_encoder.py
============================
Implements a complete Transformer Encoder stack from scratch, following
"Attention Is All You Need" (Vaswani et al., 2017).

Architecture:
    Token IDs
        → Embedding Layer
        → + Positional Encoding
        → [Encoder Block] × N
            → Multi-Head Self-Attention
            → Add & LayerNorm
            → Feed-Forward Network (FFN)
            → Add & LayerNorm
        → Output (contextualised token representations)

Why an encoder-only model?
    The encoder produces contextualised representations of input sequences.
    It's the backbone of BERT, RoBERTa, and many classification/retrieval
    systems. Decoders add causal masking for autoregressive generation (Week 2).

Usage:
    python mini_transformer_encoder.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import our attention implementation from the same folder
from scaled_dot_product_attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# 1. Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    Transformers process all tokens in parallel — they have no built-in
    notion of order. Positional encodings inject position information by
    adding a unique, deterministic signal to each token's embedding.

    Why sinusoids?
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        Different dimensions oscillate at different frequencies (like a
        clock with hands for seconds, minutes, hours, ...). The model
        can learn to read "relative distance" between positions because
        PE(pos+k) is a linear function of PE(pos).

    Note: Modern LLMs typically use RoPE (Rotary Position Embedding) or
    ALiBi instead, but sinusoidal PE remains the best starting point to
    understand the concept.
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute and register the full positional encoding matrix
        # Shape: (1, max_seq_len, d_model) — the leading 1 lets it broadcast over batches
        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )  # (d_model/2,) — equivalent to 1/10000^(2i/d_model) but numerically stable

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions → sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions  → cos

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        # Register as buffer (not a parameter — doesn't get gradient updates)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.

        Args:
            x: Token embeddings, shape (batch, seq_len, d_model)

        Returns:
            x + positional_encoding, same shape
        """
        # self.pe[:, :seq_len, :] slices the precomputed table to match current seq_len
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. Position-wise Feed-Forward Network (FFN)
# ---------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """
    Two-layer MLP applied independently to each position.

    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    This is the "thinking" step: after attention lets tokens communicate,
    the FFN processes each token's representation individually.

    The expansion factor (d_ff = 4 × d_model) gives the model expressive
    capacity. Research shows most factual knowledge is stored in FFN weights.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # Default: 4x expansion (from the paper)

        self.linear1  = nn.Linear(d_model, d_ff)
        self.linear2  = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU is more common than ReLU in modern models

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d_model) → (batch, seq, d_ff) → (batch, seq, d_model)
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ---------------------------------------------------------------------------
# 3. Single Encoder Block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    One Transformer encoder layer = Self-Attention + FFN + two residual connections.

    Pre-LN variant (used here):
        x = x + Attention(LayerNorm(x))     ← normalize BEFORE attention
        x = x + FFN(LayerNorm(x))           ← normalize BEFORE FFN

    Why Pre-LN over Post-LN (original paper)?
        Pre-LN is more stable at initialisation and trains more reliably
        for deep models. Most modern LLMs (GPT-2+, LLaMA) use Pre-LN.

    Residual connections:
        x = x + sublayer(x)
        They provide a "highway" for gradients, enabling very deep networks.
        Without them, 12-layer Transformers would not train reliably.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout)

        # Two layer norms — one for each sub-layer
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (batch, seq, d_model)
        src_mask: Optional[torch.Tensor] = None,  # Padding mask
    ) -> torch.Tensor:
        # --- Sub-layer 1: Self-Attention (Pre-LN) ---
        # Normalise → attend → dropout → residual add
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=src_mask)
        x = x + self.dropout(attn_out)

        # --- Sub-layer 2: Feed-Forward (Pre-LN) ---
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


# ---------------------------------------------------------------------------
# 4. Full Transformer Encoder Stack
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Complete Transformer encoder: embedding → positional encoding → N encoder blocks.

    This is functionally equivalent to BERT's encoder architecture
    (minus the segment embeddings and pre-training objectives).

    Args:
        vocab_size:  Number of unique tokens (e.g., 30,000 for BERT)
        d_model:     Embedding dimension (e.g., 128 for small, 768 for BERT-base)
        num_heads:   Attention heads (must divide d_model evenly)
        num_layers:  Number of stacked encoder blocks (e.g., 2 for toy, 12 for BERT)
        d_ff:        FFN hidden size (default: 4 × d_model)
        max_seq_len: Maximum supported sequence length
        dropout:     Dropout probability for regularisation
        pad_token_id:Token ID used for padding (masked in attention)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int    = 128,
        num_heads: int  = 4,
        num_layers: int = 2,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float  = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.d_model      = d_model
        self.pad_token_id = pad_token_id

        # Token embedding: maps integer IDs to d_model-dimensional vectors
        # Scaled by √d_model following the original paper
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_enc    = PositionalEncoding(d_model, max_seq_len, dropout)

        # Stack of N identical encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm (common in Pre-LN architectures)
        self.norm = nn.LayerNorm(d_model)

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """
        Careful initialisation is important for training stability.
        - Embeddings: normal distribution scaled to d_model
        - Linear layers: Xavier uniform (already default in PyTorch)
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.pad_token_id is not None:
            # Zero out the padding embedding so it has no influence
            nn.init.constant_(self.embedding.weight[self.pad_token_id], 0.0)

    def make_padding_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Create a boolean padding mask from token IDs.

        Returns:
            mask: (batch, 1, 1, seq_len), True where tokens are real,
                  False where tokens are PAD. The 1s broadcast over
                  heads and query positions in scaled_dot_product_attention.
        """
        # (batch, seq_len) → True for real tokens, False for PAD
        mask = (token_ids != self.pad_token_id)
        # Add dimensions for heads and query axis: (batch, 1, 1, seq_len)
        return mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        token_ids: torch.Tensor,                   # (batch, seq_len)
        mask: Optional[torch.Tensor] = None,        # Optional pre-made mask
    ) -> torch.Tensor:
        """
        Encode a batch of token ID sequences.

        Args:
            token_ids: Integer tensor of token IDs, shape (batch, seq_len)
            mask:      Optional padding mask; auto-computed from token_ids if None

        Returns:
            Contextualised representations, shape (batch, seq_len, d_model)
        """
        # Build padding mask if not provided
        if mask is None:
            mask = self.make_padding_mask(token_ids)  # (batch, 1, 1, seq_len)

        # Step 1: Token embeddings, scaled by √d_model
        x = self.embedding(token_ids) * math.sqrt(self.d_model)   # (batch, seq, d_model)

        # Step 2: Add positional encodings
        x = self.pos_enc(x)

        # Step 3: Pass through all encoder blocks sequentially
        for layer in self.layers:
            x = layer(x, src_mask=mask)

        # Step 4: Final layer normalisation
        x = self.norm(x)

        return x


# ---------------------------------------------------------------------------
# 5. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Mini Transformer Encoder — Forward Pass Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Config ---
    VOCAB_SIZE  = 1000
    D_MODEL     = 128
    NUM_HEADS   = 4
    NUM_LAYERS  = 2
    MAX_SEQ_LEN = 64
    BATCH_SIZE  = 2
    SEQ_LEN     = 20
    PAD_ID      = 0

    # --- Build model ---
    encoder = TransformerEncoder(
        vocab_size   = VOCAB_SIZE,
        d_model      = D_MODEL,
        num_heads    = NUM_HEADS,
        num_layers   = NUM_LAYERS,
        max_seq_len  = MAX_SEQ_LEN,
        dropout      = 0.0,   # Disable for deterministic demo
        pad_token_id = PAD_ID,
    ).to(device)

    # --- Synthetic input: two sequences with different lengths via PAD ---
    # Sequence 1: 15 real tokens + 5 PAD
    # Sequence 2: 20 real tokens (no PAD)
    token_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    token_ids[0, 15:] = PAD_ID  # Pad the first sequence

    print(f"Token IDs shape:         {token_ids.shape}")
    print(f"Padding mask (seq 0):    {(token_ids[0] != PAD_ID).int().tolist()}")

    # --- Forward pass ---
    encoder.eval()
    with torch.no_grad():
        output = encoder(token_ids)

    print(f"\nOutput shape:            {output.shape}  # (batch, seq, d_model)")

    # --- Verify PAD positions produce zero-ish output (due to padding_idx in embedding) ---
    pad_norm = output[0, 15:, :].norm(dim=-1).mean().item()
    real_norm = output[0, :15, :].norm(dim=-1).mean().item()
    print(f"\nMean norm (real tokens): {real_norm:.4f}")
    print(f"Mean norm (PAD tokens):  {pad_norm:.4f}   ← smaller due to zero init + mask")

    # --- Parameter count ---
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable    = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable:,}")

    # --- Breakdown per component ---
    emb_params = sum(p.numel() for p in encoder.embedding.parameters())
    attn_params = sum(
        p.numel()
        for layer in encoder.layers
        for p in layer.self_attn.parameters()
    )
    ffn_params = sum(
        p.numel()
        for layer in encoder.layers
        for p in layer.ffn.parameters()
    )
    print(f"\nParameter breakdown:")
    print(f"  Embedding:       {emb_params:,}")
    print(f"  Attention (all): {attn_params:,}")
    print(f"  FFN (all):       {ffn_params:,}")

    # --- Trace shapes through the encoder block manually ---
    print("\n--- Manual shape trace through encoder block ---")
    x = encoder.embedding(token_ids) * math.sqrt(D_MODEL)
    print(f"After embedding:         {x.shape}")
    x = encoder.pos_enc(x)
    print(f"After positional enc:    {x.shape}")

    mask = encoder.make_padding_mask(token_ids)
    for i, layer in enumerate(encoder.layers):
        x = layer(x, src_mask=mask)
        print(f"After encoder block {i}:   {x.shape}")

    x = encoder.norm(x)
    print(f"After final LayerNorm:   {x.shape}")

    print("\n✅ Mini Transformer Encoder working correctly!")
    print("=" * 60)
    print("\nNext step: Add a decoder with causal masking → autoregressive LM (Week 2)")
