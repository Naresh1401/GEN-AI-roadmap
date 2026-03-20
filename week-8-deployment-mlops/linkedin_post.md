# Week 8 LinkedIn Post: Deployment & MLOps

---

The model worked perfectly on my laptop.

Then I deployed it and immediately got 4 types of errors I had never seen before.

**Week 8: Deployment & MLOps — TorchServe, ONNX, quantisation, and production monitoring.**

---

**💡 Aha #1 — ONNX is a universal model format, and it matters**

PyTorch models are Python objects. Production serving systems speak C++, Triton, TensorRT. ONNX (Open Neural Network Exchange) is the common format that bridges them:

```python
import torch.onnx
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}},  # Variable batch size
)
```

Once in ONNX, you can run on CPU (ONNX Runtime), GPU (TensorRT), mobile (CoreML), or browser (ONNX.js). One export, everywhere.

---

**💡 Aha #2 — INT8 quantisation is 4× memory + 2-4× faster, with < 1% accuracy loss**

Quantisation converts model weights from FP32 (32-bit float) to INT8 (8-bit integer). The trick is calibration: run representative data through the model to determine the optimal min/max clipping range per layer.

```python
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantise only linear layers
    dtype=torch.qint8,
)
# Model size: 25MB → 6MB. Inference: 2-4x faster on CPU.
```

For NLP models on CPU-based serving (latency-sensitive, GPU too expensive), INT8 quantisation is essentially free performance.

---

**💡 Aha #3 — Model monitoring is harder than model training**

Training has a clear objective and a visible loss curve. Production monitoring requires catching:
- Data drift (input distribution has shifted)
- Concept drift (the relationship between inputs and outputs has changed)
- Model degradation (accuracy dropping on live data)
- Prediction distribution drift (model outputs changing even if inputs haven't)

And you can only do this with proxies — you rarely have ground truth labels for live predictions in real time.

---

The hardest lesson from this week: deployment is where assumptions go to die. Batch sizes that work in training don't match production patterns. Input preprocessing that was "obvious" turns out to differ between training and serving. The engineering work of deployment is about systematically eliminating these assumption gaps.

**What's the most unexpected deployment failure you've encountered — model-related or infrastructure-related?**

---

#MLOps #ModelDeployment #ONNX #Quantisation #TorchServe #MachineLearning #ProductionML #BuildingInPublic #10WeekChallenge
