# Week 10 LinkedIn Post: Mock Interviews & Review

---

10 weeks ago I could barely implement a dot-product attention layer.

This week I completed 8 mock technical interviews across ML systems design, coding, and conceptual questions. Here's what I learned.

**Week 10 of my 10-week GenAI roadmap: Mock Interviews & Review.**

---

**The three questions I got wrong the most:**

**1. "How does KV cache memory scale, and when does it become a problem?"**

I knew *what* KV cache is. I couldn't derive the memory formula on the spot:

```
Memory = 2 × layers × heads × head_dim × seq_len × bytes_per_element
```

For LLaMA-3-70B with 128K context in FP16: ~320GB per sequence.
That's why KV cache, not weights, is the primary memory constraint for long-context LLM serving.

**2. "Design a recommendation system that handles 100M users."**

My first answer was technically correct but completely ignored cold start (new users, no history), exploration vs exploitation (when to recommend novel content vs reliable history), and freshness (item popularity decays). Real systems balance all three — I was only thinking about the ML model.

**3. "What's wrong with using accuracy as your training metric for fraud detection?"**

Class imbalance: if 0.1% of transactions are fraud, a model predicting "not fraud" for everything gets 99.9% accuracy. Precision-recall trade-off, F1, ROC-AUC — all more meaningful for imbalanced classification. I knew this — I just panicked and forgot.

---

**The three things I got consistently right:**

- Transformer architecture questions (Week 1 fundamentals pay dividends everywhere)
- KV cache, attention complexity, scaling questions
- System design fundamentals: load balancing, caching, async queues

---

**What I'd tell someone starting this roadmap:**

Build everything. Don't just read about attention — implement it. Don't just read about DDPM — train it on MNIST. The interview question you'll struggle with most is not the hardest concept — it's the one you only read about and never implemented.

And write publicly. The act of explaining something clearly enough for a LinkedIn post forces a depth of understanding that reading alone never achieves.

10 weeks. 50+ implementations. 40+ papers.

What's your current week in GenAI?

---

#GenAI #MLEngineering #MachineLearning #BuildingInPublic #CareerDevelopment #TechInterviews #10WeekChallenge #LLM #DeepLearning #AI
