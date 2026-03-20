# Week 6 LinkedIn Post: ML Systems Design

---

Most ML courses teach you to build models. Nobody teaches you what happens after the model works.

This week I studied the infrastructure that separates a research demo from a production system.

**Week 6: ML Systems Design — serving, monitoring, A/B testing.**

---

**💡 Aha #1 — Model accuracy is not a system metric**

Production ML systems are measured on: throughput (requests/sec), latency (p50/p95/p99), availability (uptime), and cost per inference — not model accuracy. Accuracy is a prerequisite, not a deployment target.

A 98% accurate model serving 1000 RPS at 50ms p99 is infinitely more valuable than a 99% accurate model that takes 2 seconds and costs 40× more to run.

---

**💡 Aha #2 — Data drift is the silent killer**

Models degrade because the world changes, not because the model is wrong. A fraud detection model trained on 2022 data sees a completely different distribution by 2024.

```python
# PSI (Population Stability Index) — quantifies feature drift
PSI = Σ (actual% - expected%) * ln(actual% / expected%)
PSI < 0.1: no significant shift
PSI 0.1-0.25: moderate shift, investigate
PSI > 0.25: major shift, retrain
```

Production ML requires statistical drift monitoring on every feature in every model — in real time.

---

**💡 Aha #3 — A/B testing for ML is not A/B testing for UI**

UI A/B tests converge in days. ML A/B tests need to account for novelty effects (users initially behave differently with any new model), position bias (recommendations shown first get more clicks regardless of quality), and delayed outcomes (did the recommended product get purchased 2 weeks later?).

A production ML system without proper experiment infrastructure is flying blind. Feature flags, traffic splitting, holdout groups, and delayed reward attribution are all first-class engineering concerns.

---

Building a recommendation system end-to-end this week made clear why "ML Engineering" is its own discipline. The model is maybe 20% of the total system. The rest is feature pipelines, serving infrastructure, monitoring, and experiment frameworks.

**What do you think is the hardest part of ML in production — model quality, data freshness, or serving reliability?**

---

#MLEngineering #ProductionML #SystemDesign #RecommendationSystem #DataScience #MLOps #BuildingInPublic #10WeekChallenge
