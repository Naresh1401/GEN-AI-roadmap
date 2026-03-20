# Week 4 — Concept Deep Dive & Interview Prep

---

# 🔷 SECTION 4 — Concept Deep Dive

---

## 1. LoRA (Low-Rank Adaptation)

**What it is:** A parameter-efficient fine-tuning method that adds small trainable matrices alongside frozen pretrained weights. The update ΔW is constrained to be a product of two low-rank matrices: ΔW = BA where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×d), rank r << d.

**The analogy:** Imagine editing a Wikipedia article. "Full fine-tuning" is rewriting the entire encyclopaedia from scratch. LoRA is leaving the encyclopaedia intact and writing margin notes — small annotations that modify meaning without replacing the base content.

**Why the low-rank assumption holds:** Research on fine-tuning dynamics shows that the intrinsic dimensionality of task adaptation is surprisingly small. A 7B parameter model adapts to a new task primarily in a low-dimensional subspace of weight space. LoRA exploits this — the adaptation can be captured in r=8 or r=16 rank updates.

**The scaling factor α/r:** You don't want the LoRA adapter to dominate too early. The α/r factor controls how aggressively LoRA modifies the output. α=r means no scaling; α=2r doubles the effective learning rate of the adapter.

---

## 2. QLoRA

**What it is:** LoRA fine-tuning where the frozen base model is quantised to 4-bit (NF4 format) to reduce GPU memory. Adapters train in BF16. A dequantisation step (4-bit → BF16) happens during the forward pass.

**The analogy:** Storing a high-resolution photo in compressed JPEG to save disk space. When you need to display it, you decompress on the fly. The quality loss is minimal; the storage savings are enormous. NF4 is a quantisation format specifically designed for normally-distributed neural network weights — it is more accurate than standard INT4 for this data distribution.

**Memory savings:** A 7B model in FP16 = 14GB. Same model in 4-bit NF4 = ~3.5GB. QLoRA adapter adds ~0.1GB. Total: fine-tune a 7B model in ~4GB — now possible on consumer GPUs.

---

## 3. RLHF (Reinforcement Learning from Human Feedback)

**What it is:** A three-stage alignment pipeline:
1. Supervised Fine-Tuning (SFT): fine-tune on (instruction, good response) pairs
2. Reward Model (RM): train a scalar-output model to score responses based on human preference labels
3. PPO: use the reward model as a reward signal to optimise the policy via RL

**The analogy:** Training a dog. SFT is showing it videos of good behaviour. The reward model is the trainer who knows what "good behaviour" looks like and gives scores. PPO is the trial-and-error learning loop where the dog gets rewards and learns to maximise them.

**Why it's hard:** PPO in LLM training requires the base model (reference), reward model, and policy to coexist in memory. The training is sensitive to KL coefficient (how much the policy can deviate from the reference), reward hacking (policy exploiting loopholes in the reward model), and mode collapse (policy converging to a narrow set of high-scoring responses).

---

## 4. DPO (Direct Preference Optimisation)

**What it is:** An equivalent of RLHF that eliminates the reward model by reparameterising the RLHF objective in closed form. Given pairs (y_win, y_lose) for prompt x, DPO maximises the likelihood that the policy prefers y_win, normalised against the reference model.

**The analogy:** A salary negotiation. RLHF is: hire a consultant (reward model) to score every possible offer, then use that score function to guide your negotiation. DPO is: skip the consultant — just look at the two offers directly and move toward the better one.

**Key hyperparameter β:** Controls deviation from the reference. β=0 → no KL constraint, the policy can go anywhere (unstable). β=1 → very close to the reference, minimal alignment. Typical range: 0.1–0.5.

---

## ⚠️ Common Misconceptions

1. **"LoRA can fully replicate full fine-tuning quality."** For most tasks it does, but low-rank adapters can miss fine-grained weight updates that full fine-tuning captures. Rank selection matters — too low (r=1) underfits the adaptation; too high (r=256) loses the parameter efficiency benefit.

2. **"DPO replaces SFT."** DPO requires a decently instruction-following model as a starting point (the reference model). Without SFT first, preference learning has nothing to build on — the reference model is too incoherent to form useful log-probability ratios.

3. **"Higher β in DPO is always safer."** High β pins the policy close to the reference model, which limits how much alignment actually happens. β is a trade-off: too low = reward hacking, too high = no change. Empirical tuning is required.

---

# 🔷 SECTION 5 — Interview Prep

---

## 5 Conceptual Questions

**Q1. Why does LoRA initialise B to zeros?**

The goal of fine-tuning initialisation is to preserve the pretrained model's behaviour at the start of training. If B is initialised randomly, the output h = W₀x + BAx immediately shifts from the pretrained output — the model is no longer "at" its pretrained checkpoint. By initialising B=0, ΔW = BA = 0 at step 0, so the model starts identical to the pretrained model and gradually adapts. A is initialised randomly to break symmetry and allow the gradient to signal which directions to adapt.

**Q2. What is catastrophic forgetting and how does LoRA mitigate it?**

Catastrophic forgetting occurs when fine-tuning on task B causes the model to lose performance on task A (from pretraining or earlier fine-tuning). The base model weights "overwrite" general knowledge. LoRA mitigates this by keeping W₀ frozen — all task-specific adaptation lives in the low-rank adapters, which add to rather than replace the pretrained weights. The base model's knowledge is preserved by construction. Full fine-tuning requires learning rate scheduling, early stopping, or elastic weight consolidation (EWC) to mitigate forgetting.

**Q3. What is the reward hacking problem in RLHF, and how does DPO address it?**

Reward hacking (also called Goodhart's Law in ML): when you optimise a proxy (the reward model) rather than the true objective (human values), the policy finds behaviours that score highly on the reward model but are undesirable in practice. Classic example: a model learns to produce very long responses because the reward model has a slight length bias. DPO uses the human preference data directly without an intermediary reward model, reducing (but not eliminating) the proxy-optimisation problem. The reference model KL constraint in DPO also limits how extreme the optimisation can go.

**Q4. What is the rank r in LoRA and how do you choose it?**

r is the bottleneck dimension of the adapter. r=1 is the most compressed (single direction of adaptation); r=d is equivalent to full rank fine-tuning. Empirically: r=8 or r=16 covers most fine-tuning tasks with minimal quality loss. r=64 or r=128 is used for challenging tasks where more expressivity is needed. A practical approach: start with r=16, run a small benchmark, and increase only if the quality gap vs full fine-tuning is unacceptable. Increasing r beyond 64 rarely improves results but multiplies the adapter parameter count.

**Q5. What is the difference between SFT, RLHF, and DPO in the LLM training pipeline?**

SFT (Supervised Fine-Tuning): teach the base model to follow instructions. Input: instruction, output: desired response. Loss: standard cross-entropy. Output: an instruction-following model that is coherent but not yet aligned with human values.

RLHF: align the SFT model with human preferences using human rankings. Requires training a separate reward model, then optimising with PPO against it. Output: aligned model, but complex and unstable to train.

DPO: align the SFT model using preference pairs directly, without a reward model. Simpler, more stable, matches RLHF quality on most tasks. Now the standard for open-source alignment.

Typical pipeline: pretrain → SFT → DPO. RLHF/PPO reserved for complex reward specification (e.g., safety with multiple objectives).

---

## 2 System Design Prompts

**Design 1: Fine-tuning pipeline for enterprise customer service LLM**

Target: Fine-tune a 13B LLM on internal customer service transcripts to respond in company tone. 50k training examples. Must run on-premise on 4× A100 80GB.

Key points: QLoRA to fit 13B in available GPU memory (4-bit base + LoRA adapters across 4 GPUs with FSDP sharding). SFT first on (question, good_answer) pairs, then DPO on (question, preferred_response, rejected_response). Data pipeline: clean transcripts, filter PII, construct preference pairs from human annotator ratings. Evaluation: BLEU/ROUGE vs baseline, human preference eval on held-out test set. Serving: merge LoRA adapters post-training, serve with vLLM (no adapter overhead). Monitoring: response quality scores, hallucination rate, escalation rate.

**Design 2: Continuous alignment system**

Target: Deploy a LLM assistant with continuous learning from user feedback (thumbs up/down). Must not degrade on previously aligned behaviours.

Key points: Collect preference pairs from implicit (thumbs up/down) and explicit (which response is better) feedback. Buffer new preferences; retrain DPO periodically (weekly or when buffer reaches N samples). Use the previous aligned model as the new reference model (iterative DPO). Evaluate before deployment: regression suite on aligned behaviours, red-team adversarial prompts. Canary deployments: roll new model to 5% of traffic, monitor alignment metrics before full rollout. Store all model checkpoints for rollback capability.

---

## 3 "Gotcha" Questions

**Gotcha 1: "LoRA adapters are small, so fine-tuning is fast, right?"**

Not necessarily. The forward pass still runs through the full base model — the base model must fit in memory. LoRA reduces the *gradient memory* and *optimizer state memory* (no Adam moments for frozen params) but not the *activation memory* or *base model memory*. QLoRA additionally addresses base model memory. The actual training speed is bottlenecked by the full forward pass through the frozen base. LoRA's speed benefit is in optimizer steps (fewer params to update) and memory, not fundamentally in compute.

**Gotcha 2: "DPO uses no reward model, so it can't reward-hack. Correct?"**

DPO eliminates the *learned* reward model but introduces its own form of reward hacking. The implicit reward in DPO is: r(x, y) = β * log(π_θ(y|x) / π_ref(y|x)). A trained DPO model can still over-optimise this implicit reward — increasing the probability of chosen responses while decreasing probability of rejected responses to an extreme degree, leading to degenerate outputs. β controls this but doesn't eliminate it. The reference model KL constraint is the primary guard.

**Gotcha 3: "What happens if you apply LoRA to every layer in a large model?"**

Applying LoRA to all weight matrices (Q, K, V, O projections + FFN layers) does produce better results than applying only to Q, V — but with diminishing returns. More importantly, it multiplies the adapter parameter count. A practical finding from the original LoRA paper: applying to Q and V projections only gives ~95% of the benefit of applying to all attention projections, at a fraction of the adapter size. Some work shows that applying LoRA to FFN layers is important for knowledge-intensive tasks (since FFN layers store factual knowledge) but less important for style/format adaptation.
