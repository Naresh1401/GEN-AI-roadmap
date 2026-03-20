"""
dpo_training_loop.py
====================
Implements a Direct Preference Optimisation (DPO) training loop from scratch.

DPO Background:
    RLHF requires training a reward model R(x, y) then using PPO to optimise
    the policy against that reward model — complex, memory-intensive, unstable.

    DPO shows that the optimal RLHF policy can be expressed in closed form,
    and the reward model can be eliminated. Given a preference dataset of
    (prompt x, winning response y_w, losing response y_l), DPO minimises:

    L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

    β: temperature controlling deviation from reference model
    π_θ: policy model (being trained)
    π_ref: reference model (frozen copy of starting checkpoint)

Usage:
    python dpo_training_loop.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. DPO Loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_chosen_logps:   torch.Tensor,   # log probs of chosen response under policy
    policy_rejected_logps: torch.Tensor,   # log probs of rejected response under policy
    ref_chosen_logps:      torch.Tensor,   # log probs of chosen response under reference
    ref_rejected_logps:    torch.Tensor,   # log probs of rejected response under reference
    beta:                  float = 0.1,
    reference_free:        bool  = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DPO loss computation.

    The loss pushes the policy to:
        1. Increase log prob of y_win relative to the reference model
        2. Decrease log prob of y_lose relative to the reference model

    The reference model acts as a KL regulariser — without it, the policy
    would "reward hack" by assigning near-zero probability to rejected responses
    (even when they're just slightly worse than chosen).

    Args:
        *_logps:         Sequence-level log probabilities (sum over tokens)
        beta:            Deviation temperature. Higher β = closer to reference.
        reference_free:  If True, set ref log probs to 0 (effectively no KL reg)

    Returns:
        loss:    Scalar DPO loss
        chosen_rewards:   Per-sample reward for chosen responses (for logging)
        rejected_rewards: Per-sample reward for rejected responses (for logging)
    """
    if reference_free:
        ref_chosen_logps   = torch.zeros_like(policy_chosen_logps)
        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

    # Log ratio = log(π_θ / π_ref) = log π_θ - log π_ref
    chosen_log_ratio   = policy_chosen_logps   - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    # The "rewards" the model assigns (useful for monitoring training)
    # These should diverge as training progresses: chosen_rewards > rejected_rewards
    chosen_rewards   = beta * chosen_log_ratio.detach()
    rejected_rewards = beta * rejected_log_ratio.detach()

    # DPO loss: negative log sigmoid of the reward gap
    # When chosen reward >> rejected reward: loss → 0
    # When they're equal or reversed: loss is large
    loss = -F.logsigmoid(beta * (chosen_log_ratio - rejected_log_ratio)).mean()

    return loss, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# 2. Sequence Log Probability
# ---------------------------------------------------------------------------

def get_sequence_logps(
    model:     nn.Module,
    input_ids: torch.Tensor,      # (batch, seq_len)
    labels:    torch.Tensor,      # (batch, seq_len)  -100 = ignore
) -> torch.Tensor:
    """
    Compute per-sequence log probabilities from a language model.

    For a sequence [t1, t2, ..., tN], the model predicts each next token.
    The log probability of the sequence is the sum of log P(t_i | t_<i).

    We mask out prompt tokens (label=-100) so only the response tokens
    contribute to the log probability — we're measuring how well the model
    generates the *response*, not the prompt.
    """
    logits = model(input_ids).logits if hasattr(model(input_ids), 'logits') else model(input_ids)

    # Shift: model predicts t_{i+1} from t_i
    shift_logits = logits[:, :-1, :]          # (B, T-1, vocab)
    shift_labels = labels[:, 1:]              # (B, T-1)

    # Compute per-token log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, vocab)

    # Gather log prob of the actual next token
    # (B, T-1) — log P(t_{i+1} | t_{≤i})
    token_logps = log_probs.gather(
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    # Zero out ignored positions (prompt tokens)
    mask       = (shift_labels != -100).float()
    token_logps = token_logps * mask

    # Sum over sequence → scalar per sample
    return token_logps.sum(dim=-1)


# ---------------------------------------------------------------------------
# 3. Preference Dataset
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """A single (prompt, chosen, rejected) preference triple."""
    prompt:   str
    chosen:   str
    rejected: str


def make_preference_dataset() -> List[PreferencePair]:
    """
    Sample alignment preference dataset for demonstration.
    In production: use Anthropic HH-RLHF, OpenAI WebGPT, or domain-specific labels.
    """
    return [
        PreferencePair(
            prompt   = "Explain what a transformer is.",
            chosen   = "A transformer is a neural network architecture based on self-attention mechanisms. It processes all input tokens simultaneously rather than sequentially, allowing for efficient parallelisation and strong long-range dependency modelling.",
            rejected = "A transformer is a thing. It does stuff with text. I think it uses attention or something like that.",
        ),
        PreferencePair(
            prompt   = "What is the capital of France?",
            chosen   = "The capital of France is Paris.",
            rejected = "I think it might be Paris, but I'm not totally sure. It could also be Lyon.",
        ),
        PreferencePair(
            prompt   = "How do I improve my code quality?",
            chosen   = "Write unit tests, use a linter, add clear docstrings, review code with peers, and break large functions into smaller, focused ones.",
            rejected = "Just write better code and try harder.",
        ),
        PreferencePair(
            prompt   = "Summarise the key idea of attention mechanisms.",
            chosen   = "Attention mechanisms allow a model to weigh the importance of different input tokens when computing a representation. Each output position is a weighted average of all input positions, with weights learned from the query-key compatibility.",
            rejected = "Attention is when the model pays attention to the input. Very important in modern AI.",
        ),
    ]


# ---------------------------------------------------------------------------
# 4. Simplified DPO Trainer
# ---------------------------------------------------------------------------

class SimpleDPOTrainer:
    """
    Minimal DPO trainer demonstrating the training loop structure.

    This uses a tiny model for demonstration. For real fine-tuning,
    use HuggingFace TRL's DPOTrainer which handles tokenisation,
    batching, gradient accumulation, and distributed training.
    """

    def __init__(
        self,
        policy_model:    nn.Module,
        ref_model:       nn.Module,
        tokenizer,
        beta:            float = 0.1,
        learning_rate:   float = 1e-6,
    ):
        self.policy  = policy_model
        self.ref     = ref_model
        self.beta    = beta

        # Freeze reference model completely
        for param in self.ref.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate
        )

    def compute_dpo_loss_from_batch(
        self,
        chosen_input_ids:   torch.Tensor,
        chosen_labels:      torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels:    torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """One DPO training step."""
        # Policy log probs
        policy_chosen_logps   = get_sequence_logps(self.policy, chosen_input_ids,   chosen_labels)
        policy_rejected_logps = get_sequence_logps(self.policy, rejected_input_ids, rejected_labels)

        # Reference log probs (no grad needed)
        with torch.no_grad():
            ref_chosen_logps   = get_sequence_logps(self.ref, chosen_input_ids,   chosen_labels)
            ref_rejected_logps = get_sequence_logps(self.ref, rejected_input_ids, rejected_labels)

        loss, chosen_rew, rejected_rew = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps,    ref_rejected_logps,
            beta = self.beta,
        )

        return {
            "loss":             loss,
            "chosen_rewards":   chosen_rew.mean(),
            "rejected_rewards": rejected_rew.mean(),
            "reward_margin":    (chosen_rew - rejected_rew).mean(),  # Should increase during training
        }

    def train_step(self, batch: Dict) -> Dict[str, float]:
        self.optimizer.zero_grad()
        metrics = self.compute_dpo_loss_from_batch(**batch)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        return {k: v.item() for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# 5. Demo using synthetic logits (no full LLM needed)
# ---------------------------------------------------------------------------

class FakeLM(nn.Module):
    """Minimal LM stub for DPO demonstration without downloading a real model."""
    def __init__(self, vocab_size=100, d=32, seq_len=20):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d)
        self.linear  = nn.Linear(d, vocab_size)
        self.seq_len = seq_len

    def forward(self, input_ids):
        return self.linear(self.embed(input_ids))


def simulate_dpo_training(n_steps: int = 20):
    """
    Demonstrate DPO training dynamics with synthetic data.
    Shows that reward_margin increases as training progresses.
    """
    torch.manual_seed(42)
    vocab_size = 100
    seq_len    = 10

    policy = FakeLM(vocab_size)
    ref    = FakeLM(vocab_size)
    ref.load_state_dict(policy.state_dict())  # Start from same point

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-3)

    # Synthetic preference batch
    chosen_ids   = torch.randint(0, vocab_size, (4, seq_len))
    rejected_ids = torch.randint(0, vocab_size, (4, seq_len))

    # Labels: -100 for prompt (first 3 tokens), actual ids for response
    chosen_labels   = chosen_ids.clone()
    chosen_labels[:, :3] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, :3] = -100

    print(f"  Step  | Loss   | Chosen Rew | Rejected Rew | Margin")
    print(f"  " + "-" * 58)

    history_margins = []

    for step in range(n_steps):
        optimizer.zero_grad()

        policy_chosen_logps   = get_sequence_logps(policy, chosen_ids,   chosen_labels)
        policy_rejected_logps = get_sequence_logps(policy, rejected_ids, rejected_labels)
        with torch.no_grad():
            ref_chosen_logps   = get_sequence_logps(ref, chosen_ids,   chosen_labels)
            ref_rejected_logps = get_sequence_logps(ref, rejected_ids, rejected_labels)

        loss, chosen_rew, rejected_rew = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps,    ref_rejected_logps,
            beta=0.1
        )
        margin = (chosen_rew - rejected_rew).mean().item()
        history_margins.append(margin)

        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == n_steps - 1:
            print(f"  {step:4d}  | {loss.item():.4f} | {chosen_rew.mean().item():10.4f} | "
                  f"{rejected_rew.mean().item():12.4f} | {margin:.4f}")

    # Verify margin increased (training is working)
    margin_start = history_margins[0]
    margin_end   = history_margins[-1]
    print(f"\n  Reward margin: {margin_start:.4f} → {margin_end:.4f}")
    if margin_end > margin_start:
        print(f"  ✅ Margin increased — policy prefers chosen responses more than reference")
    else:
        print(f"  ⚠️ Margin did not increase — try more steps or higher LR")

    return history_margins


if __name__ == "__main__":
    print("=" * 65)
    print("DPO Training Loop Demo")
    print("=" * 65)

    print("\n[1/3] DPO loss function test...")
    B = 4
    # Mock log probs — chosen should score higher
    policy_chosen_logps   = torch.tensor([-2.1, -1.8, -2.5, -1.9])
    policy_rejected_logps = torch.tensor([-3.2, -3.5, -3.1, -3.8])
    ref_chosen_logps      = torch.tensor([-2.5, -2.5, -2.5, -2.5])
    ref_rejected_logps    = torch.tensor([-2.5, -2.5, -2.5, -2.5])

    loss, c_rew, r_rew = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps, beta=0.1
    )
    print(f"  Loss:             {loss.item():.4f}")
    print(f"  Chosen rewards:   {c_rew.mean().item():.4f}  (should be positive)")
    print(f"  Rejected rewards: {r_rew.mean().item():.4f}  (should be negative)")
    print(f"  Reward margin:    {(c_rew - r_rew).mean().item():.4f}  (should be positive)")

    print("\n[2/3] Sample preference dataset...")
    dataset = make_preference_dataset()
    for pair in dataset[:2]:
        print(f"  Prompt:   '{pair.prompt[:50]}'")
        print(f"  Chosen:   '{pair.chosen[:60]}'...")
        print(f"  Rejected: '{pair.rejected[:60]}'...")
        print()

    print("[3/3] Simulating DPO training dynamics...")
    margins = simulate_dpo_training(n_steps=25)

    print("\n" + "=" * 65)
    print("✅ DPO training loop complete!")
    print("For real LLM fine-tuning: use HuggingFace TRL DPOTrainer")
    print("  from trl import DPOTrainer, DPOConfig")
