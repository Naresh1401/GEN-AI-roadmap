# 🔷 SECTION 1 — LinkedIn Post: Week 4 — Fine-Tuning & RLHF

---

I fine-tuned a 7 billion parameter language model on a single GPU this week. Three months ago I would have told you that was impossible without a data centre.

**Week 4 of my 10-week GenAI roadmap: Fine-Tuning, LoRA, QLoRA, and DPO.**

---

**💡 Aha #1 — LoRA is a mathematical cheat code (in the best way)**

The insight behind LoRA is genuinely elegant: when you fine-tune a large model, the weight updates form a *low-rank* matrix. Full fine-tuning allocates memory for the entire gradient of every weight. LoRA approximates those updates as a product of two tiny matrices:

```python
# Full fine-tuning: update 4M params per layer
ΔW ∈ ℝ^(2048 × 2048)  # 4,194,304 parameters

# LoRA: ΔW ≈ B @ A
A ∈ ℝ^(16 × 2048)     # 32,768 params
B ∈ ℝ^(2048 × 16)     # 32,768 params
# Total: 65,536 params — 64× reduction
```

The entire pretrained model stays frozen. You're training 0.1–1% of parameters. In most benchmarks, LoRA matches full fine-tuning within 1–2 points.

---

**💡 Aha #2 — QLoRA makes GPU memory irrelevant**

QLoRA stores the frozen base model in 4-bit NF4 quantisation (4× less memory than FP16). The tiny LoRA adapters train in BF16. A 7B model that requires 14GB in FP16 now fits in ~4GB.

I fine-tuned LLaMA-2-7B on a 16GB GPU that was previously limited to 3B models. The quality gap was negligible — under 1% on standard benchmarks.

The best part: after fine-tuning, LoRA adapters can be **merged back** into the base model weights and served at full speed with no runtime overhead.

---

**💡 Aha #3 — DPO quietly killed RLHF complexity**

Classic RLHF requires three models: reference model, reward model, and policy — plus a PPO training loop that is notoriously unstable (reward hacking, mode collapse, sensitive hyperparameters).

DPO (Direct Preference Optimisation) derived mathematically that you can *skip the reward model entirely*. Given pairs of (preferred, rejected) responses, DPO directly optimises the policy:

```python
# DPO loss (simplified):
loss = -log_sigmoid(β * (
    log_prob(y_win | policy) - log_prob(y_win | reference) -
    log_prob(y_lose | policy) + log_prob(y_lose | reference)
))
```

One model. Standard cross-entropy-style training. Dramatically more stable than PPO. DPO is now the default alignment recipe in most open-source instruction-tuned LLMs.

---

The uncomfortable insight from this week: fine-tuning changes a model's *style* far more readily than its *factual knowledge*. After 500 steps on a custom dataset, the model adopts your tone almost immediately — but "injecting" new facts that weren't in pretraining is unreliable. Facts still mainly come from the 1T+ token pretraining corpus.

This is why the industry consensus is: use RAG for knowledge, use fine-tuning for behaviour.

**Question: Do you think this distinction — RAG for knowledge, fine-tuning for style — will hold as models get larger? Or will fine-tuning eventually become a reliable knowledge injection method?**

---

#FineTuning #LoRA #QLoRA #DPO #RLHF #LLM #GenerativeAI #MachineLearning #BuildingInPublic #PyTorch #Alignment #10WeekChallenge
