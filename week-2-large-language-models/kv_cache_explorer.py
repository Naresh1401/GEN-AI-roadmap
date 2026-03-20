"""
kv_cache_explorer.py
====================
Implements KV (Key-Value) caching for a GPT-style decoder and benchmarks
the speedup compared to naive recomputation.

The problem without KV cache:
    Autoregressive generation produces one token per forward pass. Each pass
    recomputes attention over the ENTIRE growing context — O(n) attention
    operations per step → O(n²) total for n generated tokens.

The solution (KV cache):
    Store K and V tensors from previous forward passes. At step t:
        - Only compute Q_t, K_t, V_t for the NEW token
        - Append K_t to K_cache, V_t to V_cache
        - Attend over full K_cache / V_cache without recomputation
    Cost: O(1) per step → O(n) total. Trade-off: O(n) memory.

This file:
    1. Implements a mini GPT decoder with optional KV cache
    2. Benchmarks both modes across varying sequence lengths
    3. Measures GPU memory occupied by the KV cache
    4. Plots latency vs sequence length

Usage:
    python kv_cache_explorer.py
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Minimal GPT decoder configuration."""
    vocab_size:  int   = 1000
    d_model:     int   = 128
    num_heads:   int   = 4
    num_layers:  int   = 6
    max_seq_len: int   = 512
    dropout:     float = 0.0   # Off for benchmarking


# ---------------------------------------------------------------------------
# 2. Attention with KV Cache
# ---------------------------------------------------------------------------

@dataclass
class KVCache:
    """
    Stores Key and Value tensors for all layers, one entry per generation step.

    Shape convention per layer:
        keys:   (batch, num_heads, seq_len_so_far, head_dim)
        values: (batch, num_heads, seq_len_so_far, head_dim)
    """
    keys:   List[Optional[torch.Tensor]] = field(default_factory=list)
    values: List[Optional[torch.Tensor]] = field(default_factory=list)

    def __post_init__(self):
        # Initialise as empty (will be populated per layer)
        pass

    def memory_bytes(self) -> int:
        """Estimate total GPU memory occupied by this KV cache."""
        total = 0
        for k, v in zip(self.keys, self.values):
            if k is not None:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total


class CachedSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional KV caching.

    Two modes:
        use_cache=False (training):  Standard causal attention over full sequence
        use_cache=True  (inference): Only compute new token's Q,K,V; reuse cached K,V
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,                            # (batch, seq_len, d_model)
        layer_idx: int          = 0,
        cache: Optional[KVCache] = None,             # None = no caching
        use_cache: bool         = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.shape

        # Project to Q, K, V all at once (efficient: single matmul)
        qkv = self.qkv_proj(x)                       # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)    # each (B, T, d_model)

        # Reshape to multi-head format: (B, heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # ---- KV Cache Logic ----
        if use_cache and cache is not None:
            # Extend cache lists to cover this layer if needed
            while len(cache.keys) <= layer_idx:
                cache.keys.append(None)
                cache.values.append(None)

            if cache.keys[layer_idx] is not None:
                # Append new K, V to the cached history
                # Cached K shape: (B, heads, past_seq, head_dim)
                k = torch.cat([cache.keys[layer_idx],   k], dim=2)
                v = torch.cat([cache.values[layer_idx], v], dim=2)

            # Update cache with the latest K, V
            cache.keys[layer_idx]   = k
            cache.values[layer_idx] = v

        # ---- Attention Computation ----
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # (B, heads, T_q, T_k)

        # Apply causal mask ONLY during full-sequence (non-cached) forward pass
        # During cached inference, T_q=1 (new token only), so no masking needed
        if not use_cache or (cache is None):
            T_k = k.shape[2]
            T_q = q.shape[2]
            # Upper triangle (future positions) → -inf
            causal_mask = torch.triu(
                torch.ones(T_q, T_k, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)                            # (B, heads, T_q, head_dim)

        # Merge heads: (B, heads, T_q, head_dim) → (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(out), cache


# ---------------------------------------------------------------------------
# 3. GPT Decoder Block
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    """Single GPT decoder block (Pre-LN variant)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = CachedSelfAttention(cfg.d_model, cfg.num_heads)
        self.ffn  = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # Attention sub-layer with residual
        attn_out, cache = self.attn(self.ln1(x), layer_idx, cache, use_cache)
        x = x + attn_out

        # FFN sub-layer with residual
        x = x + self.ffn(self.ln2(x))
        return x, cache


# ---------------------------------------------------------------------------
# 4. Mini GPT Model
# ---------------------------------------------------------------------------

class MiniGPT(nn.Module):
    """
    Minimal GPT-style decoder-only language model.

    Key design choices vs the original GPT-2:
        - Pre-LN (more stable)
        - No dropout (for benchmarking clarity)
        - Shared weight tying between embedding and LM head
          → reduces parameters and improves generalisation
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks    = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: LM head shares weights with token embedding
        # Rationale: the model should assign high logits to tokens whose
        # embedding is similar to the current hidden state
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,              # (B, T)
        use_cache: bool = False,
        cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T = token_ids.shape

        # During cached inference, token_ids is just the new token (T=1)
        # Position = length of cached history + 0 (just one position)
        if use_cache and cache is not None and cache.keys and cache.keys[0] is not None:
            # Current position = past cached length
            past_len = cache.keys[0].shape[2]
            positions = torch.arange(past_len, past_len + T, device=token_ids.device)
        else:
            positions = torch.arange(T, device=token_ids.device)

        x = self.token_emb(token_ids) + self.pos_emb(positions)

        if use_cache and cache is None:
            cache = KVCache()

        for i, block in enumerate(self.blocks):
            x, cache = block(x, layer_idx=i, cache=cache, use_cache=use_cache)

        x      = self.ln_f(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        return logits, cache

    @torch.no_grad()
    def generate_no_cache(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Naive autoregressive generation WITHOUT KV cache.
        
        Every step re-runs the full forward pass over all tokens seen so far.
        O(n²) total attention operations.
        """
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            logits, _ = self.forward(generated, use_cache=False)
            # Take logits at the last position, sample greedily
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated  = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def generate_with_cache(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Efficient autoregressive generation WITH KV cache.

        Prefill phase: Run full forward pass on prompt → populate KV cache.
        Decode phase:  Each step processes only 1 token; appends to cache.
        O(n) total attention operations.
        """
        # Prefill: process the full prompt to populate the cache
        _, cache = self.forward(prompt_ids, use_cache=True, cache=None)

        generated   = prompt_ids.clone()
        # Last token of the prompt → first input for decode phase
        current_ids = prompt_ids[:, -1:]

        for _ in range(max_new_tokens):
            logits, cache = self.forward(current_ids, use_cache=True, cache=cache)
            next_token    = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated     = torch.cat([generated, next_token], dim=1)
            current_ids   = next_token   # Feed only the new token

        return generated


# ---------------------------------------------------------------------------
# 5. Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    model: MiniGPT,
    prompt_lengths: List[int],
    new_tokens: int = 50,
    device: torch.device = torch.device('cpu'),
    num_runs: int = 3,
):
    """
    Measure generation latency with and without KV cache.

    Args:
        prompt_lengths: List of prompt lengths to test
        new_tokens:     How many tokens to generate per run
        num_runs:       Average over multiple runs to reduce noise
    """
    results_no_cache   = []
    results_with_cache = []

    for prompt_len in prompt_lengths:
        prompt = torch.randint(1, model.cfg.vocab_size, (1, prompt_len), device=device)

        # --- Without cache ---
        times_no_cache = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model.generate_no_cache(prompt, max_new_tokens=new_tokens)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_no_cache.append(time.perf_counter() - t0)

        # --- With cache ---
        times_with_cache = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model.generate_with_cache(prompt, max_new_tokens=new_tokens)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_with_cache.append(time.perf_counter() - t0)

        avg_no    = sum(times_no_cache)   / num_runs
        avg_with  = sum(times_with_cache) / num_runs

        results_no_cache.append(avg_no)
        results_with_cache.append(avg_with)

        speedup = avg_no / max(avg_with, 1e-9)
        tok_s_no   = new_tokens / avg_no
        tok_s_with = new_tokens / avg_with

        print(f"  Prompt {prompt_len:4d} tokens | "
              f"No cache: {avg_no:.3f}s ({tok_s_no:6.1f} tok/s) | "
              f"Cache: {avg_with:.3f}s ({tok_s_with:6.1f} tok/s) | "
              f"Speedup: {speedup:.1f}×")

    return results_no_cache, results_with_cache


def estimate_kv_memory(cfg: GPTConfig, seq_len: int, dtype_bytes: int = 2) -> int:
    """
    Estimate KV cache memory for a given sequence length.

    KV cache size = 2 (K+V) × num_layers × num_heads × head_dim × seq_len × bytes
    """
    head_dim  = cfg.d_model // cfg.num_heads
    bytes_per = 2 * cfg.num_layers * cfg.num_heads * head_dim * seq_len * dtype_bytes
    return bytes_per


def plot_benchmark(
    prompt_lengths: List[int],
    results_no_cache: List[float],
    results_with_cache: List[float],
):
    """Save a latency comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("KV Cache Benchmark", fontsize=14, fontweight='bold')

    # Latency vs prompt length
    ax1.plot(prompt_lengths, results_no_cache,   'r-o', label='Without KV cache', linewidth=2)
    ax1.plot(prompt_lengths, results_with_cache, 'g-o', label='With KV cache',    linewidth=2)
    ax1.set_xlabel("Prompt Length (tokens)")
    ax1.set_ylabel("Generation Time (s)")
    ax1.set_title("Latency vs Prompt Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Speedup ratio
    speedups = [
        nc / max(wc, 1e-9)
        for nc, wc in zip(results_no_cache, results_with_cache)
    ]
    ax2.bar(prompt_lengths, speedups, color='steelblue', alpha=0.8)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1×)')
    ax2.set_xlabel("Prompt Length (tokens)")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("KV Cache Speedup Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("kv_cache_latency.png", dpi=120, bbox_inches='tight')
    print("  Latency plot saved → kv_cache_latency.png")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("KV Cache Explorer & Benchmark")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    cfg = GPTConfig(
        vocab_size  = 500,
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 6,
        max_seq_len = 512,
    )

    model = MiniGPT(cfg).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MiniGPT | {total_params:,} params | {cfg.num_layers} layers\n")

    # -------------------------------------------------------
    # Part 1: Correctness check — both modes must produce same output
    # -------------------------------------------------------
    print("[1/4] Correctness verification (same output, different efficiency)...")
    prompt = torch.randint(1, cfg.vocab_size, (1, 10), device=device)

    out_no_cache   = model.generate_no_cache(prompt.clone(), max_new_tokens=20)
    out_with_cache = model.generate_with_cache(prompt.clone(), max_new_tokens=20)

    # Both should produce identical token sequences (same greedy argmax)
    match = torch.equal(out_no_cache, out_with_cache)
    print(f"  Output match (no cache vs cache): {'✅ Identical' if match else '❌ MISMATCH'}")
    print(f"  Sequence (no cache):  {out_no_cache[0].tolist()}")
    print(f"  Sequence (cached):    {out_with_cache[0].tolist()}")

    # -------------------------------------------------------
    # Part 2: KV cache memory analysis
    # -------------------------------------------------------
    print("\n[2/4] KV Cache Memory Analysis...")
    print(f"  {'Seq Length':>12} | {'KV Cache Size':>15} | {'Per-Layer':>12}")
    print("  " + "-" * 45)

    for seq_len in [64, 128, 256, 512]:
        total_bytes = estimate_kv_memory(cfg, seq_len, dtype_bytes=4)  # FP32
        per_layer   = total_bytes // cfg.num_layers
        print(f"  {seq_len:>12} | {total_bytes/1024:>12.1f} KB | {per_layer/1024:>9.1f} KB")

    # Measure actual memory with real cache
    prompt_mem = torch.randint(1, cfg.vocab_size, (1, 64), device=device)
    _, actual_cache = model.forward(prompt_mem, use_cache=True, cache=None)
    actual_bytes = actual_cache.memory_bytes()
    print(f"\n  Actual cache (64 tokens, {cfg.num_layers} layers): {actual_bytes/1024:.1f} KB")

    # -------------------------------------------------------
    # Part 3: Speed benchmark
    # -------------------------------------------------------
    print("\n[3/4] Speed Benchmark...")
    prompt_lengths = [10, 32, 64, 128, 256]
    results_no, results_with = benchmark(
        model, prompt_lengths, new_tokens=30, device=device, num_runs=2
    )

    # -------------------------------------------------------
    # Part 4: Plot
    # -------------------------------------------------------
    print("\n[4/4] Saving benchmark plot...")
    plot_benchmark(prompt_lengths, results_no, results_with)

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print("\n--- Summary ---")
    avg_speedup = sum(nc / max(wc, 1e-9) for nc, wc in zip(results_no, results_with)) / len(results_no)
    print(f"  Average speedup with KV cache: {avg_speedup:.1f}×")
    print(f"  Speedup grows with prompt length (O(n²) vs O(n))")
    print(f"  Cost: linear memory growth — {estimate_kv_memory(cfg, 512)/1024:.0f} KB for 512 tokens")

    print("\n✅ KV cache exploration complete!")
    print("=" * 65)
    print("\nKey insight: KV cache trades memory for compute.")
    print("In production (vLLM, TGI), PagedAttention manages this memory")
    print("like virtual memory in an OS — see Week 8 for deployment details.")
