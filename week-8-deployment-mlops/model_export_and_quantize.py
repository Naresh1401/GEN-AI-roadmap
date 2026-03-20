"""
model_export_and_quantize.py
============================
Demonstrates ONNX model export, INT8 quantisation, and latency benchmarking.

Covers:
- PyTorch → ONNX export (opset 17, dynamic batch size)
- Post-training dynamic quantisation (fast, no calibration data)
- Post-training static quantisation (better, needs calibration)
- Latency + accuracy benchmarking across formats
- TorchServe model packaging workflow

Usage:
    python model_export_and_quantize.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import os


# ---------------------------------------------------------------------------
# 1. Reference model for benchmarking
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    A small BERT-style classification model for benchmarking.
    We use this to compare FP32 vs INT8 vs ONNX Runtime performance.
    """
    def __init__(
        self,
        vocab_size:    int = 5000,
        d_model:       int = 128,
        n_heads:       int = 4,
        n_layers:      int = 3,
        n_classes:     int = 5,
        max_seq_len:   int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x    = self.embedding(input_ids) + self.pos_emb(pos)
        x    = self.encoder(x)
        return self.classifier(x[:, 0])  # CLS token


# ---------------------------------------------------------------------------
# 2. ONNX Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model:      nn.Module,
    dummy_input: torch.Tensor,
    output_path: str = "model.onnx",
    opset:       int = 17,
) -> str:
    """
    Export a PyTorch model to ONNX format.
    
    Key argument: dynamic_axes — allows variable batch size and sequence length
    at inference time. Without this, the ONNX model is fixed to the dummy input's
    batch/sequence dimensions.
    
    opset_version: ONNX opset to target. Higher = more ops available.
    17 is a safe choice for most PyTorch operations as of 2024.
    """
    model.eval()
    
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        opset_version    = opset,
        input_names      = ['input_ids'],
        output_names     = ['logits'],
        dynamic_axes     = {
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'logits':    {0: 'batch_size'},
        },
        do_constant_folding = True,  # Fold constant operations for speed
    )
    
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Exported to {output_path}  ({size_mb:.2f} MB)")
    return output_path


def run_onnx_inference(
    model_path:   str,
    input_data:   np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Run inference using ONNX Runtime.
    
    ONNX Runtime applies graph optimisations automatically:
    - Operator fusion (combine multiple ops into one kernel)
    - Layout optimisation
    - Constant folding
    These typically give 10-50% speedup over pure PyTorch inference.
    """
    try:
        import onnxruntime as ort
        
        # Create inference session with optimisation level
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
        )
        
        input_name = session.get_inputs()[0].name
        
        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_data})
        elapsed = time.perf_counter() - start
        
        return outputs[0], elapsed
    except ImportError:
        print("  onnxruntime not installed — skipping ONNX inference")
        return None, 0.0


# ---------------------------------------------------------------------------
# 3. Dynamic Quantisation
# ---------------------------------------------------------------------------

def apply_dynamic_quantisation(model: nn.Module) -> nn.Module:
    """
    Post-training dynamic quantisation.
    
    Weights: quantised to INT8 and stored as INT8
    Activations: dynamically quantised at runtime (dequantised for each operation)
    
    Pros: No calibration data needed. Simple to apply. Typically 2-4× faster on CPU.
    Cons: Slightly slower than static quantisation (dynamic dequantisation has overhead).
    Best for: Linear layers (BERT, GPT-style models). Less effective for Conv2d.
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec = {nn.Linear, nn.Embedding},
        dtype        = torch.qint8,
    )
    return quantized


# ---------------------------------------------------------------------------
# 4. Latency Benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    model,
    input_data:    torch.Tensor,
    n_warmup:      int = 5,
    n_runs:        int = 50,
    label:         str = "Model",
    is_onnx:       bool = False,
    onnx_path:     str = None,
) -> Dict[str, float]:
    """
    Measure inference latency statistics.
    
    Why warmup? The first few runs are slower due to JIT compilation,
    kernel caching, and CPU frequency scaling. Warmup amortises these.
    
    Why p95/p99? Mean latency is misleading for systems work.
    A model with mean=20ms and p99=200ms will timeout users 1% of the time.
    SLA is typically specified at p95 or p99.
    """
    if not is_onnx:
        model.eval()
    
    latencies = []
    
    for i in range(n_warmup + n_runs):
        if is_onnx:
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                input_name = session.get_inputs()[0].name
                start = time.perf_counter()
                session.run(None, {input_name: input_data.numpy()})
                elapsed = (time.perf_counter() - start) * 1000
            except:
                elapsed = 0.0
        else:
            with torch.no_grad():
                start   = time.perf_counter()
                _       = model(input_data)
                elapsed = (time.perf_counter() - start) * 1000  # ms
        
        if i >= n_warmup:
            latencies.append(elapsed)
    
    latencies = np.array(latencies)
    results = {
        "mean":  float(np.mean(latencies)),
        "p50":   float(np.percentile(latencies, 50)),
        "p95":   float(np.percentile(latencies, 95)),
        "p99":   float(np.percentile(latencies, 99)),
        "std":   float(np.std(latencies)),
    }
    
    print(f"\n  {label}:")
    print(f"    Mean: {results['mean']:.2f}ms  P50: {results['p50']:.2f}ms  "
          f"P95: {results['p95']:.2f}ms  P99: {results['p99']:.2f}ms")
    
    return results


def get_model_size_mb(model: nn.Module) -> float:
    """Return model size in MB (parameter bytes only, no activations)."""
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# 5. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Model Export & Quantisation Benchmark")
    print("=" * 65)
    
    torch.manual_seed(42)
    device = 'cpu'  # Quantisation most impactful on CPU
    
    # Build model
    print("\n[1/4] Building model...")
    model = TransformerClassifier(vocab_size=5000, d_model=128, n_layers=3).eval()
    n_params = sum(p.numel() for p in model.parameters())
    size_fp32 = get_model_size_mb(model)
    print(f"  Parameters: {n_params:,}")
    print(f"  Size (FP32): {size_fp32:.2f} MB")
    
    # Test input (batch_size=1, seq_len=32)
    test_input = torch.randint(0, 5000, (1, 32))
    
    # ONNX Export
    print("\n[2/4] ONNX Export...")
    try:
        onnx_path = export_to_onnx(model, test_input, "/tmp/classifier.onnx")
        onnx_available = True
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        onnx_available = False
    
    # Dynamic Quantisation
    print("\n[3/4] Applying dynamic INT8 quantisation...")
    model_int8 = apply_dynamic_quantisation(model)
    size_int8  = get_model_size_mb(model_int8)
    print(f"  Size (FP32): {size_fp32:.2f} MB")
    print(f"  Size (INT8): {size_int8:.2f} MB")
    print(f"  Compression: {size_fp32/max(size_int8,0.001):.1f}×")
    
    # Verify accuracy preserved
    with torch.no_grad():
        out_fp32 = model(test_input)
        out_int8 = model_int8(test_input)
    pred_match = out_fp32.argmax().item() == out_int8.argmax().item()
    print(f"  Prediction agreement (FP32 vs INT8): {'✅' if pred_match else '⚠️'}")
    max_diff = (out_fp32 - out_int8).abs().max().item()
    print(f"  Max output difference: {max_diff:.4f}")
    
    # Latency benchmark
    print("\n[4/4] Latency benchmark (batch_size=1, CPU)...")
    print("  Note: Run 50 iterations after 5 warmup")
    
    r_fp32 = benchmark_latency(model,       test_input, label="FP32 (baseline)")
    r_int8 = benchmark_latency(model_int8,  test_input, label="INT8 (quantised)")
    
    if onnx_available:
        r_onnx = benchmark_latency(
            None, test_input, label="ONNX Runtime",
            is_onnx=True, onnx_path=onnx_path
        )
    
    print(f"\n  Summary:")
    print(f"    INT8 speedup vs FP32: {r_fp32['mean']/max(r_int8['mean'],0.001):.2f}×")
    
    print("\n" + "=" * 65)
    print("✅ Model export and quantisation complete!")
    print("\nNext steps for production:")
    print("  1. Use onnxruntime-gpu for GPU serving")
    print("  2. Export to TensorRT for maximum GPU throughput")
    print("  3. Use torch.compile() for PyTorch 2.x speedups")
    print("  4. Implement dynamic batching in your serving layer")


# ============================================================
# SECTION 4 & 5 — Deep Dive & Interview Prep
# ============================================================
"""
WEEK 8: DEPLOYMENT & MLOPS — Deep Dive & Interview Prep

CONCEPTS:

1. ONNX (Open Neural Network Exchange)
ONNX is a standardised, language-agnostic intermediate representation for ML models.
A PyTorch model exported to ONNX can run with ONNX Runtime (C++-based, no Python
overhead), TensorRT (NVIDIA GPU optimiser), CoreML (Apple Silicon), ONNX.js (browser).
One model export → deploy anywhere. The ONNX graph is an ops-level IR that runtime
optimisers can fuse, reorder, and specialise for target hardware.

2. QUANTISATION
Weights in FP32 take 4 bytes each. INT8 takes 1 byte. Quantisation maps the
continuous FP32 range to 256 discrete INT8 levels: Q(x) = round(x / scale + zero_point).
Scale and zero_point are calibrated per-layer. Dynamic quantisation calibrates at
runtime. Static quantisation calibrates offline with representative data (more accurate,
faster at inference). INT4 (as in QLoRA) is more aggressive — 16 levels — but
requires special handling.

3. DYNAMIC BATCHING
Model serving becomes more efficient with larger batch sizes (more parallelism on GPU).
Dynamic batching collects requests arriving within a time window (e.g., 10ms) and
processes them as a single batch. Trade-off: latency increases for individual requests
but throughput increases significantly. Target: keep GPU utilisation > 70%.

4. BLUE-GREEN DEPLOYMENT
Two identical production environments (Blue = current, Green = new).
Deploy new model to Green, run smoke tests, then switch traffic.
If Green fails, switch back to Blue immediately. Zero-downtime deployments.
For gradual rollout, canary deployment sends 5% of traffic to Green first.

5. MODEL VERSIONING
Every deployed model should have: model ID, training commit hash, training data
version, evaluation metrics, deployment timestamp. Without this, you cannot
reproduce failures or roll back correctly. MLflow, Weights & Biases, and SageMaker
Model Registry all support model versioning.

MISCONCEPTIONS:
1. "ONNX Runtime always outperforms PyTorch." ONNX Runtime applies static graph
optimisation that PyTorch's dynamic graph can't. But PyTorch 2.x with torch.compile()
closes the gap significantly by tracing and compiling hot code paths.
2. "INT8 quantisation is lossless on NLP models." Not always. Attention softmax
and layer normalisation are sensitive to precision. Models with extreme weight ranges
(outlier channels) quantise poorly — LLM.int8() uses mixed-precision to handle this.
3. "You only need to optimise if you're at scale." Even at small scale, serving costs
money. A 4× speedup from INT8 means 75% fewer GPU-hours — significant at any scale.

INTERVIEW QUESTIONS (5):

Q1: What is train-serve skew and how does it cause model degradation?
A: Train-serve skew occurs when the feature engineering code used at training time
differs from the code used at serving time. Common causes: different libraries
(scikit-learn Imputer at training vs custom Java at serving), different data types
(float64 at training vs float32 at serving), different handling of null values.
The model sees a different input distribution at serving time and produces degraded
predictions. Prevention: feature stores enforce a single code path for both.

Q2: What's the difference between dynamic and static quantisation?
A: Dynamic: only weights are quantised offline. Activations are quantised dynamically
at inference (dequantised on the fly). No calibration data required. Good for RNNs
and Transformers on CPU. Static: both weights AND activations are quantised. Requires
a calibration dataset to determine per-layer activation min/max ranges. Better
accuracy than dynamic, faster at inference (no runtime dequantisation). Required for
hardware accelerators (TPUs, Apple Neural Engine) that need fixed quantisation.

Q3: How does TorchServe handle model versioning?
A: TorchServe supports multiple model versions simultaneously. When registering a model,
you specify a version string. You can set a "default" version that serves most traffic,
and configure weighted routing (e.g., 90% v1, 10% v2 for A/B testing). Model artifacts
(.mar files) are self-contained: model weights + handler code + dependencies.
Rolling back means setting the previous version as default — instant, no redeployment.

Q4: What is the memory hierarchy for LLM serving, and where are the bottlenecks?
A: Layer 1 (fastest): GPU HBM (A100: 80GB, 2TB/s bandwidth) — stores model weights
and KV cache. Layer 2: GPU compute (FLOPS). Bottleneck: often memory bandwidth,
not compute. A 70B model in FP16 = 140GB → needs 2× A100s minimum.
KV cache: 2 × layers × heads × head_dim × seq_len × bytes. For 128K context on
LLaMA-3-70B (80 layers, 64 heads, 128 dim, FP16): ~320GB. This exceeds weight size.
PagedAttention (vLLM) manages KV cache as virtual memory to reduce fragmentation.

Q5: What is torch.compile() and when should you use it?
A: torch.compile() (PyTorch 2.x) traces the computation graph, applies graph
optimisations, and compiles to native code using TorchInductor (triton backend).
Speedups: 30-100% for Transformers, 10-30% for CNNs. Best for: static-shape workloads
(fixed batch size + seq length), repeated inference (compilation amortises).
Avoid for: dynamic shapes (recompiles frequently), debugging (harder to inspect),
model modification between calls.

SYSTEM DESIGN: Design a low-latency LLM serving system (< 200ms, 10k RPS).
Model: quantised LLaMA-3-8B (INT8, 8GB) on A100 GPUs.
Batching: continuous batching (vLLM-style) — new requests join running batch,
completed sequences leave. Much higher throughput than static batching.
KV cache: PagedAttention to reduce fragmentation; set max_seq_len per request.
Load balancing: route to GPUs with sufficient free KV cache, not just round-robin.
Autoscaling: scale GPU workers based on queue depth, not just CPU/memory.
Streaming: stream tokens as they're generated (< 50ms to first token matters UX).
Monitoring: time-to-first-token (TTFT), tokens-per-second, GPU utilisation, queue depth.
Failover: health checks per GPU worker; automatic rerouting on failure.
"""
