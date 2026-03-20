# Week 8: Deployment & MLOps

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

This week covers the engineering required to deploy ML models in production: ONNX export for cross-platform serving, quantisation for speed and memory reduction, TorchServe for model management, and production monitoring.

**Core concepts:**
- ONNX model export and runtime
- INT8 post-training quantisation (PTQ) and quantisation-aware training (QAT)
- TorchServe model serving
- Model versioning and blue-green deployments
- Inference optimisation: batching, caching, hardware selection

---

## Architecture Diagrams

### Deployment Pipeline

```
Training (PyTorch)
       │
       ▼ torch.onnx.export
    model.onnx
       │
       ├──── ONNX Runtime (CPU serving) ──── REST API
       │
       ├──── TensorRT (GPU serving) ──────── REST API
       │
       └──── CoreML (iOS/macOS) ────────── Mobile App
```

### Quantisation (FP32 → INT8)

```
FP32 Model (4 bytes per weight)
         │
         ▼ Calibration (run representative dataset)
         Determine per-layer min/max → scale factor
         │
         ▼ Quantise weights + activations
INT8 Model (1 byte per weight)
         │
  4× smaller   2-4× faster (CPU)   ~1% accuracy loss
```

---

## Setup

```bash
pip install torch torchvision onnx onnxruntime
```

---

## Running the Implementations

```bash
python model_export_and_quantize.py    # ONNX export + INT8 quantisation
python serving_benchmarks.py           # Latency and throughput comparison
```

---

## Key Takeaways

| Optimisation | Size Reduction | Speedup | Quality Loss |
|---|---|---|---|
| FP32 → FP16 | 2× | 1.5–2× (GPU) | < 0.1% |
| FP32 → INT8 (dynamic) | 4× | 2–4× (CPU) | < 1% |
| FP32 → INT8 (static) | 4× | 3–5× (CPU) | < 0.5% |
| ONNX Runtime | None | 1.1–1.5× | None |
| TensorRT | None | 2–5× (GPU) | < 0.1% |
| Pruning (80% sparse) | 5× | 1.5–2× | 1–3% |

---

## References

- [ONNX Documentation](https://onnx.ai/onnx/)
- [PyTorch Quantisation](https://pytorch.org/docs/stable/quantization.html)
- [TorchServe](https://pytorch.org/serve/)
- [BitsAndBytes INT8 Training](https://arxiv.org/abs/2208.07339)
