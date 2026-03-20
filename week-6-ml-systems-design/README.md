# Week 6: ML Systems Design

> Part of the [10-Week GenAI Mastery Roadmap](../README.md)

## Overview

This week covers the infrastructure required to take ML models from research to production: model serving architectures, monitoring for data and concept drift, A/B testing frameworks, and recommendation system design.

**Core concepts:**
- Model serving patterns (online vs batch vs streaming)
- Feature stores and data pipelines
- Statistical monitoring: PSI, KS-test, drift detection
- A/B testing for ML models
- Recommendation system design

---

## Architecture Diagrams

### ML System Components

```
┌─────────────────────────────────────────────────────┐
│                TRAINING PIPELINE                     │
│  Raw Data → Feature Engineering → Training →        │
│  Validation → Model Registry                        │
└──────────────────────────┬──────────────────────────┘
                           │ (model artifact)
                           ▼
┌─────────────────────────────────────────────────────┐
│                SERVING PIPELINE                      │
│                                                       │
│  Request → Feature Store → Model Server → Response  │
│                                │                     │
│                         Prediction Log               │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                MONITORING PIPELINE                   │
│                                                       │
│  Prediction Logs → Drift Detection → Alerts         │
│  Ground Truth Labels → Performance Tracking         │
│  A/B Experiment Results → Statistical Tests         │
└─────────────────────────────────────────────────────┘
```

### Recommendation System Architecture

```
User Request
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  Retrieval  │ → │  Ranking     │ → │  Re-ranking   │
│  (Candidate │    │  (ML Model)  │    │  (Business    │
│  Generation)│    │              │    │  Rules/Filters│
│             │    │  1000 items  │    │  → Top 20)    │
│ 10M → 1000  │    │  → 100 items │    │               │
└─────────────┘    └──────────────┘    └───────────────┘
     │
     Embedding similarity / BM25 / Heuristics
```

---

## Setup

```bash
pip install torch scikit-learn pandas numpy scipy matplotlib evidently
```

---

## Running the Implementations

```bash
python recommendation_system.py    # Two-tower + ranking pipeline
python ml_monitoring.py            # Drift detection + alerting
python ab_testing_framework.py     # Statistical A/B testing for ML
```

---

## Key Takeaways

| Topic | Core Insight |
|---|---|
| Model serving | Latency SLA trumps accuracy; p99 matters more than mean |
| Feature stores | Prevent train-serve skew — use the same code path for both |
| PSI monitoring | PSI > 0.25 → trigger retrain; monitor ALL input features |
| A/B testing | Power analysis first — underpowered experiments waste resources |
| Recommendation | Retrieval + ranking + business rules, not a single ML model |

---

## References

- [Machine Learning Systems Design (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Feature Stores for ML](https://www.featurestore.org/)
- [Evidently AI (drift monitoring)](https://evidentlyai.com/)
