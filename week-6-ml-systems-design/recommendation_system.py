"""
recommendation_system.py + ml_monitoring.py + ab_testing_framework.py
======================================================================
Week 6: ML Systems Design

This file contains three implementations:
1. Two-tower recommendation system (retrieval + ranking)
2. ML monitoring (drift detection with PSI and KS-test)
3. A/B testing framework for ML models

Run each section by calling the respective demo function.

Usage:
    python recommendation_system.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
import random


# ============================================================
# PART 1: TWO-TOWER RECOMMENDATION SYSTEM
# ============================================================

class UserTower(nn.Module):
    """
    User encoder: maps user features to an embedding vector.
    
    In production, input features include:
    - User demographics (age, location)
    - Interaction history (viewed items, ratings)
    - Session context (device, time of day)
    """
    def __init__(self, user_feature_dim: int = 32, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(user_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 normalise so cosine similarity = dot product
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    """
    Item encoder: maps item features to an embedding vector.
    
    Input features: title embedding, category, price, popularity, etc.
    """
    def __init__(self, item_feature_dim: int = 32, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(item_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-tower retrieval model for recommendations.
    
    Training: contrastive loss — push (user, interacted_item) embeddings together,
              push (user, random_item) embeddings apart.
    Serving:  user tower runs at request time. Item tower runs offline
              (item embeddings pre-computed and indexed with FAISS/ANN).
    """
    def __init__(self, user_dim: int = 32, item_dim: int = 32, emb_dim: int = 64):
        super().__init__()
        self.user_tower = UserTower(user_dim, emb_dim)
        self.item_tower = ItemTower(item_dim, emb_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))  # Learnable temperature
    
    def forward(
        self,
        user_features: torch.Tensor,   # (B, user_dim)
        item_features: torch.Tensor,   # (B, item_dim) — positive items
    ) -> torch.Tensor:
        user_emb = self.user_tower(user_features)  # (B, emb_dim)
        item_emb = self.item_tower(item_features)  # (B, emb_dim)
        
        # Compute similarity matrix for all (user, item) pairs in batch
        # Diagonal = positive pairs, off-diagonal = negatives (in-batch negatives)
        similarity = (user_emb @ item_emb.T) / self.temperature.exp()
        return similarity
    
    def contrastive_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE (contrastive) loss.
        Diagonal of similarity matrix = positive pairs.
        Row i: user i vs all items. Ground truth = item i (diagonal).
        """
        B = similarity.shape[0]
        labels = torch.arange(B, device=similarity.device)
        # Cross-entropy over rows (user → correct item) + columns (item → correct user)
        loss = (F.cross_entropy(similarity, labels) + 
                F.cross_entropy(similarity.T, labels)) / 2
        return loss


# ============================================================
# PART 2: ML MONITORING (DRIFT DETECTION)
# ============================================================

class DriftMonitor:
    """
    Monitors model inputs and outputs for distribution drift.
    
    Two key drift types:
    - Data drift: P(X) changed — inputs look different from training
    - Concept drift: P(Y|X) changed — relationship between inputs and output changed
    
    Metrics:
    - PSI (Population Stability Index): industry standard, sensitive to large shifts
    - KS test (Kolmogorov-Smirnov): non-parametric, good for continuous features
    """
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str] = None):
        """
        Args:
            reference_data: Training/baseline data (N_ref, n_features)
            feature_names:  Names for each feature column
        """
        self.reference = reference_data
        self.n_features = reference_data.shape[1] if len(reference_data.shape) > 1 else 1
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features)]
        
        # Precompute reference histograms for PSI
        self.ref_histograms = []
        for i in range(self.n_features):
            col = reference_data[:, i] if len(reference_data.shape) > 1 else reference_data
            hist, edges = np.histogram(col, bins=10)
            self.ref_histograms.append((hist / hist.sum(), edges))
    
    def psi(self, current_data: np.ndarray, feature_idx: int = 0) -> float:
        """
        Population Stability Index.
        
        PSI = Σ (actual% - expected%) × ln(actual% / expected%)
        
        Thresholds (industry standard):
            PSI < 0.10: no significant change
            PSI 0.10-0.25: moderate shift — investigate
            PSI > 0.25: major shift — retrain model
        """
        ref_hist, edges = self.ref_histograms[feature_idx]
        
        col = current_data[:, feature_idx] if len(current_data.shape) > 1 else current_data
        cur_hist, _ = np.histogram(col, bins=edges)
        cur_hist = cur_hist / max(cur_hist.sum(), 1)
        
        # Avoid log(0): add small epsilon
        eps = 1e-6
        ref_hist = np.clip(ref_hist, eps, None)
        cur_hist = np.clip(cur_hist, eps, None)
        
        psi_value = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        return float(psi_value)
    
    def ks_test(self, current_data: np.ndarray, feature_idx: int = 0) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test: non-parametric test of whether two distributions are the same.
        Returns (statistic, p_value). p < 0.05 → significant drift.
        """
        ref_col = self.reference[:, feature_idx] if len(self.reference.shape) > 1 else self.reference
        cur_col = current_data[:, feature_idx] if len(current_data.shape) > 1 else current_data
        ks_stat, p_value = stats.ks_2samp(ref_col, cur_col)
        return float(ks_stat), float(p_value)
    
    def monitor(self, current_data: np.ndarray) -> Dict:
        """Run all drift checks and return a report."""
        report = {}
        alerts = []
        
        for i, name in enumerate(self.feature_names):
            psi_val      = self.psi(current_data, i)
            ks_stat, ks_p = self.ks_test(current_data, i)
            
            status = "OK"
            if psi_val > 0.25:
                status = "ALERT"
                alerts.append(f"{name}: PSI={psi_val:.3f} (major drift)")
            elif psi_val > 0.10:
                status = "WARN"
                alerts.append(f"{name}: PSI={psi_val:.3f} (moderate drift)")
            
            report[name] = {
                "psi":    round(psi_val, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_p":    round(ks_p, 4),
                "status":  status,
            }
        
        return {"features": report, "alerts": alerts}


# ============================================================
# PART 3: A/B TESTING FRAMEWORK
# ============================================================

@dataclass
class ABTestResult:
    """Results from an A/B test."""
    metric:      str
    control_mean: float
    treatment_mean: float
    relative_lift: float
    p_value:     float
    significant: bool
    sample_size: int
    power:       float


class ABTestFramework:
    """
    Statistical A/B testing framework for ML model comparisons.
    
    Key components:
    - Power analysis: determine required sample size before starting
    - Significance testing: t-test for continuous metrics, chi-squared for rates
    - Effect size: practical significance beyond statistical significance
    """
    
    def required_sample_size(
        self,
        baseline_rate:   float,
        minimum_effect:  float,  # Minimum detectable effect (relative)
        alpha:           float = 0.05,
        power:           float = 0.80,
    ) -> int:
        """
        Calculate sample size per variant required to detect a given effect.
        
        Formula from two-proportion z-test power analysis.
        
        Args:
            baseline_rate:  Current conversion/click rate (e.g., 0.03 = 3%)
            minimum_effect: Relative improvement to detect (e.g., 0.10 = 10% lift)
            alpha:          Type I error rate (false positive)
            power:          1 - Type II error rate (false negative)
        
        Returns:
            Number of samples needed per variant (control and treatment each)
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_effect)
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
        z_beta  = stats.norm.ppf(power)
        
        pooled_p = (p1 + p2) / 2
        
        n = ((z_alpha * math.sqrt(2 * pooled_p * (1 - pooled_p)) +
              z_beta  * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) /
             (p2 - p1)) ** 2
        
        return int(math.ceil(n))
    
    def evaluate(
        self,
        control_outcomes:   List[float],
        treatment_outcomes: List[float],
        metric_name:        str = "metric",
        alpha:              float = 0.05,
    ) -> ABTestResult:
        """
        Evaluate an A/B test result using Welch's t-test.
        
        Welch's t-test (not equal-variance Student's) is preferred because
        treatment and control often have different variance after a model change.
        """
        ctrl   = np.array(control_outcomes)
        treat  = np.array(treatment_outcomes)
        
        t_stat, p_value = stats.ttest_ind(treat, ctrl, equal_var=False)
        
        ctrl_mean  = float(np.mean(ctrl))
        treat_mean = float(np.mean(treat))
        rel_lift   = (treat_mean - ctrl_mean) / max(abs(ctrl_mean), 1e-9)
        
        # Observed power (post-hoc, for reporting — do power analysis BEFORE running)
        pooled_std = np.sqrt((np.var(ctrl) + np.var(treat)) / 2)
        effect_size = abs(treat_mean - ctrl_mean) / max(pooled_std, 1e-9)
        power = 1 - stats.t.cdf(
            stats.t.ppf(1 - alpha/2, df=len(ctrl)+len(treat)-2) - effect_size * math.sqrt(len(ctrl)/2),
            df=len(ctrl)+len(treat)-2
        )
        
        return ABTestResult(
            metric        = metric_name,
            control_mean  = ctrl_mean,
            treatment_mean = treat_mean,
            relative_lift = rel_lift,
            p_value       = float(p_value),
            significant   = bool(p_value < alpha),
            sample_size   = len(ctrl) + len(treat),
            power         = float(min(power, 1.0)),
        )


# ============================================================
# DEMOS
# ============================================================

def demo_recommendation():
    print("=" * 55)
    print("PART 1: Two-Tower Recommendation System")
    print("=" * 55)
    
    torch.manual_seed(42)
    model = TwoTowerModel(user_dim=16, item_dim=16, emb_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Simulate training data
    print("\nTraining on synthetic interaction data (5 epochs)...")
    for epoch in range(5):
        B = 64
        user_feats = torch.randn(B, 16)
        item_feats = torch.randn(B, 16)
        
        similarity = model(user_feats, item_feats)
        loss = model.contrastive_loss(similarity)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")
    
    # Retrieval
    print("\nRetrieval demo:")
    user_emb = model.user_tower(torch.randn(1, 16))
    item_embs = model.item_tower(torch.randn(100, 16))
    sims = (user_emb @ item_embs.T).squeeze()
    top5 = sims.topk(5).indices.tolist()
    print(f"  Top 5 items for user: {top5} (from 100 candidates)")


def demo_monitoring():
    print("\n" + "=" * 55)
    print("PART 2: ML Monitoring (Drift Detection)")
    print("=" * 55)
    
    np.random.seed(42)
    # Reference: training data distribution
    n_ref = 10000
    n_feat = 3
    reference = np.random.randn(n_ref, n_feat)
    feature_names = ["age_scaled", "purchase_count", "session_duration"]
    
    monitor = DriftMonitor(reference, feature_names)
    
    # Scenario 1: No drift
    current_no_drift = np.random.randn(2000, n_feat)
    report_nd = monitor.monitor(current_no_drift)
    print("\nNo-drift scenario:")
    for name, stats_ in report_nd["features"].items():
        print(f"  {name}: PSI={stats_['psi']:.4f}  Status={stats_['status']}")
    
    # Scenario 2: Major drift on feature 0 (e.g., age distribution shifted)
    current_drifted = np.random.randn(2000, n_feat)
    current_drifted[:, 0] += 3.0  # Age shifted by 3 std devs
    report_d = monitor.monitor(current_drifted)
    print("\nDrift scenario (feature 0 shifted by 3σ):")
    for name, stats_ in report_d["features"].items():
        print(f"  {name}: PSI={stats_['psi']:.4f}  Status={stats_['status']}")
    if report_d["alerts"]:
        print(f"\n  🚨 ALERTS:")
        for alert in report_d["alerts"]:
            print(f"    {alert}")


def demo_ab_testing():
    print("\n" + "=" * 55)
    print("PART 3: A/B Testing Framework")
    print("=" * 55)
    
    np.random.seed(42)
    framework = ABTestFramework()
    
    # Power analysis before running
    baseline = 0.035  # 3.5% click-through rate
    mde      = 0.10   # 10% relative lift
    n_needed = framework.required_sample_size(baseline, mde)
    print(f"\nPower analysis (α=0.05, power=0.80):")
    print(f"  Baseline CTR: {baseline:.1%}")
    print(f"  Minimum detectable effect: {mde:.0%} relative lift")
    print(f"  Required sample size (per variant): {n_needed:,}")
    
    # Simulate experiment: treatment is 12% better
    n = 5000
    true_lift = 0.12
    ctrl_conv  = np.random.binomial(1, baseline,              n).astype(float)
    treat_conv = np.random.binomial(1, baseline * (1 + true_lift), n).astype(float)
    
    result = framework.evaluate(ctrl_conv.tolist(), treat_conv.tolist(), "CTR")
    print(f"\nExperiment results (N={result.sample_size:,}):")
    print(f"  Control mean:   {result.control_mean:.4f}")
    print(f"  Treatment mean: {result.treatment_mean:.4f}")
    print(f"  Relative lift:  {result.relative_lift:.1%}")
    print(f"  p-value:        {result.p_value:.4f}")
    sig = "✅ Significant" if result.significant else "❌ Not significant"
    print(f"  Result:         {sig} (α=0.05)")


if __name__ == "__main__":
    demo_recommendation()
    demo_monitoring()
    demo_ab_testing()
    print("\n" + "=" * 55)
    print("✅ Week 6: ML Systems Design demos complete!")


# ============================================================
# SECTION 4 & 5 — Deep Dive & Interview Prep
# ============================================================
"""
WEEK 6: ML SYSTEMS DESIGN — Deep Dive & Interview Prep

CONCEPTS:

1. TRAIN-SERVE SKEW
The most common production failure: training and serving use different feature engineering
code. One team writes the training pipeline in Python/Pandas; another writes serving
in Java/C++. Drift occurs by accident. Solution: feature stores (Feast, Tecton) ensure
the exact same transformation runs at both training time and serving time.

2. PSI (POPULATION STABILITY INDEX)
Measures how much an input feature distribution has shifted from reference.
PSI = Σ (current_freq - reference_freq) × log(current_freq / reference_freq).
Rule of thumb: PSI>0.25 = major shift, retrain. PSI 0.1-0.25 = investigate.
PSI<0.1 = stable. Run PSI on every model feature, every day, in production.

3. A/B TESTING FOR ML MODELS
Unlike UI tests (days to converge), ML model tests need: (1) power analysis upfront —
underpowered tests waste traffic; (2) holdout groups — never expose 100% to the new model;
(3) delayed outcome handling — purchases may happen days after a recommendation;
(4) novelty effect control — users click on new things regardless of quality initially.

4. TWO-TOWER RETRIEVAL
A scalable architecture where user and item are encoded separately.
Training uses contrastive loss (InfoNCE) to push matching (user, item) pairs together.
At serving: item embeddings are pre-computed offline and indexed (FAISS, ScaNN).
User embedding is computed at request time; top-K retrieved by ANN search.
This decoupling is what makes recommendation systems scale to billions of items.

5. EXPLORATION VS EXPLOITATION
Pure ranking models exploit known preferences and create filter bubbles.
Production recommendation systems use exploration strategies:
- ε-greedy: with probability ε, show a random item
- Upper Confidence Bound (UCB): weight items by (mean + exploration_bonus)
- Thompson Sampling: Bayesian approach, sample from posterior
Exploration rate often decreases as user history grows.

MISCONCEPTIONS:
1. "Deploy the model with highest offline accuracy." Offline accuracy on a test set
often doesn't predict online performance. The test set may not reflect the live
distribution. Always validate with live A/B tests before full rollout.
2. "Monitoring is optional for low-stakes applications." All production ML models
degrade silently. Without monitoring you find out from users, not dashboards.
3. "More features always improve recommendations." More features increase training
complexity, serving latency, and infrastructure cost. Feature importance analysis
often shows 5-10 features explain 90% of the variance. Start simple.

INTERVIEW QUESTIONS (5):
Q1: What is the cold start problem and how do you address it?
A: New users/items have no interaction history. Solutions: (1) Content-based fallback
— recommend based on item attributes, not user history; (2) Ask new users for explicit
preferences during onboarding; (3) Popularity-based fallback for new items;
(4) Transfer learning from similar users (user clustering); (5) Side information
(demographics, device type) for user cold start.

Q2: How do you handle the delay between recommendation and outcome?
A: Delayed reward attribution. A product click may lead to a purchase 3 days later.
Solutions: (1) Use session-level signals (clicks, dwell time) as proxies for
immediate feedback; (2) Store recommendation logs with session IDs; join with
downstream conversions; (3) Use survival analysis or hazard models for time-to-event;
(4) Weight recent rewards more heavily to account for recency.

Q3: Why is cosine similarity used for two-tower retrieval, not Euclidean distance?
A: Cosine similarity is scale-invariant — it measures the *direction* of embedding
vectors, not their magnitude. After L2 normalisation, cosine similarity = dot product,
which is fast to compute and optimised by FAISS's IndexFlatIP. Euclidean distance is
sensitive to embedding magnitude (a user with many interactions might have a large-
magnitude embedding, biasing retrieval toward similar users rather than similar items).

Q4: What is an offline/online evaluation mismatch, and why does it happen?
A: An offline evaluation (e.g., precision@k on a test set) measures how often the model
correctly ranks items from logged interactions. But logged data has survivorship bias:
you only have interactions for items that were *previously shown* to users. The model
never had a chance to be evaluated on items that were never shown. Online A/B tests
evaluate the model on a live, unbiased traffic sample — always the gold standard.

Q5: How would you monitor a ranking model for concept drift?
A: Concept drift in a ranking model is hard to detect directly (you need ground truth).
Proxies: (1) Prediction score distribution — if scores shift, the model sees different
inputs; (2) Rank distribution of interacted items — if CTR@1 is dropping, top-1
predictions are getting worse; (3) Feature PSI on model inputs; (4) Online A/B
comparison of rolling model vs. a frozen "reference" model — divergence indicates drift.

SYSTEM DESIGN: Design a real-time content ranking system for a social feed.
Retrieval: candidate pool from social graph (friends), interest graph (followed topics),
trending content (popularity scores). Feature computation: real-time user features
from streaming (Kafka → Flink); item features pre-computed daily. Ranking: GBDT or
neural ranker on (user, item, context) features. Business rules post-processing:
diversity constraints, spam filtering, ad insertion. Serving: < 100ms SLA — use
precomputed item features, fast feature lookup via Redis, model on GPU. Monitoring:
per-hour CTR, position-weighted CTR, novelty/serendipity metrics, diversity.
"""
