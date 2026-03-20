# 🔷 SECTION 1 — LinkedIn Post

---

Your autocomplete doesn't know English. It knows math.

That was the visceral realisation I had this week building a GPT-style language model from the tokenizer all the way up through inference — and figuring out why it's *fast*.

**Week 2 of my 10-week GenAI roadmap: Large Language Models.**

Here are the three insights that genuinely changed how I think about LLMs:

---

**💡 Aha #1 — Tokens are not words. They're compression artifacts.**

Before any neural network sees your text, a tokenizer chops it into subword units using Byte-Pair Encoding (BPE). I implemented BPE from scratch this week, and it's beautifully simple:

```
"low lower lowest" →
merges: ('l','o')→'lo', ('lo','w')→'low' ...
result: ['low', 'lower', 'low', 'est']
```

The algorithm repeatedly merges the most frequent adjacent pair until you hit a target vocabulary size. The result: common words get their own token, rare words get split into recognisable pieces.

"Unbelievable" might become ["un", "believ", "able"]. The model has never seen *that* word — but it's seen all three pieces, in many contexts. That's the magic. It's not understanding — it's pattern compression that *looks* like understanding.

---

**💡 Aha #2 — GPT is just a Transformer decoder with one crucial constraint**

The architecture is almost identical to last week's encoder — except every attention layer is **causally masked**. Each token can only attend to itself and everything *before* it, never what comes after.

```python
# Causal mask: upper triangle is -inf
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

That single constraint — blocking future tokens — turns an encoder into an autoregressive language model. Training objective: given all previous tokens, predict the next one. Do this on a trillion tokens and "understanding" emerges.

The decoder-only architecture (GPT, LLaMA, Claude) now dominates because it scales cleanly and the pretraining objective is elegantly simple.

---

**💡 Aha #3 — KV Cache is the reason inference isn't 100× slower than you'd expect**

During text generation, naively re-running the full forward pass for every new token would recompute Keys and Values for all previous tokens — every single step. That's O(n²) *per token generated*.

KV cache stores those Keys and Values from previous steps. For each new token, you only compute the new K, V pair and append it:

```python
# Instead of recomputing all past K, V:
new_k = compute_key(new_token)
new_v = compute_value(new_token)
K_cache = torch.cat([K_cache, new_k], dim=-2)  # Just append
```

This turns generation from O(n²) per token to O(n) total — and it's the core reason streaming responses are possible in production. Without KV cache, generating 500 tokens would require 500 full forward passes over the growing context.

---

This week I fine-tuned GPT-2 on a custom dataset, which forced me to confront something uncomfortable: the model does not learn "facts" — it learns *token transition probabilities*. Given "The capital of France is", it outputs "Paris" not because it *knows* Paris, but because "Paris" reliably follows that sequence in its training corpus.

That distinction matters enormously for everything from hallucination to alignment.

---

**The question I can't stop thinking about:**

Fine-tuning GPT-2 on 10k examples dramatically shifted its output style — but the model still has the same weights mostly from pretraining. How much of what we call "fine-tuning" is actually *knowledge acquisition* vs *style transfer*?

Where do you think the line is?

---

#GenerativeAI #LLM #GPT #MachineLearning #NLP #Tokenization #DeepLearning #BuildingInPublic #MLEngineering #PyTorch #10WeekChallenge
