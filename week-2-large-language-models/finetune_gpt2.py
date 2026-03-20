"""
finetune_gpt2.py
================
Fine-tunes GPT-2 (117M parameters) on a custom text dataset using
Hugging Face Transformers.

What fine-tuning does:
    GPT-2 is pretrained on ~40GB of internet text (WebText). Fine-tuning
    updates all (or some) weights to adapt the model's output distribution
    toward a target domain. The training objective is identical to pretraining:
    causal language modelling — predict the next token given all previous tokens.

Key architectural note:
    GPT-2 uses a DECODER-ONLY Transformer (no cross-attention, only causal
    self-attention). Unlike BERT's encoder, GPT-2 reads left-to-right only,
    making it naturally suited for text generation.

This script demonstrates:
    - Dataset preparation and tokenization for CLM
    - Trainer API with sensible defaults
    - Generation before and after fine-tuning
    - Perplexity calculation

Usage:
    python finetune_gpt2.py                    # Uses built-in sample data
    python finetune_gpt2.py --data my_text.txt # Use your own .txt file
"""

import os
import math
import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore', category=UserWarning)

# Hugging Face imports
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TextGenerationPipeline,
)


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    A simple CLM dataset that tokenizes raw text and splits into fixed-length blocks.

    Why fixed-length blocks?
        The Trainer batches by block size — all sequences must be the same length.
        We concatenate all text, then chunk into blocks of `block_size` tokens.
        This avoids padding overhead and maximises token utilisation.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: GPT2Tokenizer,
        block_size: int = 128,
    ):
        self.block_size = block_size

        # Tokenize all texts and concatenate into one long sequence
        # GPT-2 uses EOS between documents to signal document boundaries
        all_ids: List[int] = []
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            all_ids.extend(encoded)
            all_ids.append(tokenizer.eos_token_id)  # Document separator

        # Chunk into non-overlapping blocks of `block_size` tokens
        # Drop the final incomplete block
        self.examples: List[List[int]] = []
        for i in range(0, len(all_ids) - block_size, block_size):
            block = all_ids[i : i + block_size]
            self.examples.append(block)

        print(f"  Total tokens:        {len(all_ids):,}")
        print(f"  Block size:          {block_size}")
        print(f"  Number of examples:  {len(self.examples):,}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        # For CLM, labels = input_ids (shifted inside the model)
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {'input_ids': ids, 'labels': ids.clone()}


# ---------------------------------------------------------------------------
# 2. Generation utility
# ---------------------------------------------------------------------------

def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.92,
    num_sequences: int = 1,
) -> List[str]:
    """
    Autoregressive text generation with top-p (nucleus) sampling.

    Args:
        temperature: Controls randomness. < 1.0 = more focused/deterministic.
                     temperature → 0 approaches greedy decoding.
        top_p:       Nucleus sampling — sample from the smallest set of tokens
                     whose cumulative probability exceeds top_p. Avoids sampling
                     from the long tail of unlikely tokens.
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens   = max_new_tokens,
            do_sample        = True,
            temperature      = temperature,
            top_p            = top_p,
            num_return_sequences = num_sequences,
            pad_token_id     = tokenizer.eos_token_id,
            # Prevent the model from repeating n-grams
            no_repeat_ngram_size = 3,
        )

    texts = []
    for seq in outputs:
        # Decode only the newly generated tokens (not the prompt)
        new_tokens = seq[input_ids.shape[-1]:]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return texts


# ---------------------------------------------------------------------------
# 3. Perplexity evaluation
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: List[str],
    block_size: int = 128,
) -> float:
    """
    Compute perplexity on a held-out set.

    Perplexity = exp(average negative log-likelihood per token)

    A lower perplexity means the model assigns higher probability to the
    ground-truth next tokens — it's more "surprised" by fewer things.
    Random (uniform) over 50,000 vocab: PPL = 50,000.
    Good GPT-2 on WebText: PPL ≈ 18–35. After domain fine-tuning: lower.
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors='pt').to(device)
            if ids.shape[1] < 2:
                continue
            # Truncate to block_size for memory safety
            ids = ids[:, :block_size]

            outputs = model(input_ids=ids, labels=ids)
            # outputs.loss is already mean cross-entropy over non-masked tokens
            total_loss   += outputs.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# 4. Sample dataset (ML/NLP domain)
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    """The transformer architecture revolutionised natural language processing by replacing recurrent networks with self-attention mechanisms. Every token attends to every other token simultaneously, enabling parallelism and long-range dependency modelling.""",

    """Large language models are trained using the causal language modelling objective: given a sequence of tokens, predict the next token. This simple objective, applied at scale, leads to emergent capabilities like few-shot learning and chain-of-thought reasoning.""",

    """Byte-pair encoding is the tokenization algorithm used by GPT-2 and GPT-3. It starts with a character vocabulary and iteratively merges the most frequent adjacent pairs until reaching the target vocabulary size. This produces subword tokens that balance vocabulary size with sequence length.""",

    """Self-attention computes relationships between all pairs of tokens in a sequence. The query, key, and value projections allow the model to learn different aspects of these relationships. Scaling by the square root of the key dimension prevents gradient saturation.""",

    """Fine-tuning adapts a pretrained language model to a specific domain or task. The model's weights, initialised from pretraining, are updated using a smaller domain-specific dataset. The learning rate should be much lower than during pretraining to avoid catastrophic forgetting.""",

    """The KV cache stores key and value tensors from previous autoregressive steps. During generation, only the new token's query needs to be computed. The cached keys and values allow attention to the full context without recomputation, dramatically speeding up inference.""",

    """Residual connections add the input of a layer directly to its output, creating gradient highways through deep networks. Layer normalisation stabilises training by normalising activations across the feature dimension. Together they enable training of Transformers with dozens of layers.""",

    """Positional encodings provide the Transformer with information about token order. GPT-2 uses learned absolute position embeddings, adding a position-specific vector to each token embedding. More recent models use rotary position embeddings for better length generalisation.""",

    """The feed-forward network within each Transformer block applies a two-layer MLP independently to each token position. The hidden dimension is typically four times the model dimension. This component is believed to store factual knowledge learned during pretraining.""",

    """Perplexity measures how well a language model predicts a held-out sequence. It equals the exponential of the average cross-entropy loss per token. A lower perplexity indicates the model assigns higher probability to the ground-truth next tokens.""",

    """Temperature scaling controls the randomness of language model generation. Dividing logits by a temperature less than one sharpens the distribution toward the most likely token. Temperature greater than one flattens the distribution, increasing diversity.""",

    """Top-p sampling, also called nucleus sampling, selects from the smallest set of tokens whose cumulative probability exceeds a threshold. This avoids the extremes of greedy decoding and pure random sampling, producing fluent and diverse text.""",
]


# ---------------------------------------------------------------------------
# 5. Main fine-tuning script
# ---------------------------------------------------------------------------

def main(data_path: Optional[str] = None):
    print("=" * 65)
    print("GPT-2 Fine-Tuning on Custom Data")
    print("=" * 65)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------
    # Load tokenizer and model
    # -------------------------------------------------------
    print("\n[1/5] Loading GPT-2 tokenizer and model...")
    model_name = "gpt2"  # 117M parameters — smallest GPT-2

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT-2 has no pad token by default; set EOS as pad
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT-2 ({total_params/1e6:.0f}M parameters)")

    # -------------------------------------------------------
    # Prepare dataset
    # -------------------------------------------------------
    print("\n[2/5] Preparing dataset...")

    if data_path and Path(data_path).exists():
        raw_texts = Path(data_path).read_text().split('\n\n')
        raw_texts = [t.strip() for t in raw_texts if len(t.strip()) > 50]
        print(f"  Loaded {len(raw_texts)} paragraphs from {data_path}")
    else:
        raw_texts = SAMPLE_TEXTS
        print(f"  Using built-in ML domain sample corpus ({len(raw_texts)} texts)")

    # 80/20 train/eval split
    split   = int(len(raw_texts) * 0.8)
    train_texts = raw_texts[:split]
    eval_texts  = raw_texts[split:]

    block_size = 128

    print("  Building train dataset:")
    train_dataset = TextDataset(train_texts, tokenizer, block_size)
    print("  Building eval dataset:")
    eval_dataset  = TextDataset(eval_texts,  tokenizer, block_size)

    # Data collator: handles dynamic padding within a batch
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,  # CLM, not masked LM (MLM is BERT-style)
    )

    # -------------------------------------------------------
    # Generation BEFORE fine-tuning
    # -------------------------------------------------------
    print("\n[3/5] Generating text BEFORE fine-tuning...")
    prompts = [
        "The transformer architecture",
        "Large language models learn",
        "Tokenization is the process",
    ]

    for prompt in prompts:
        output = generate_text(model, tokenizer, prompt, max_new_tokens=50)
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: {output[0][:120]}...\n")

    ppl_before = compute_perplexity(model, tokenizer, eval_texts)
    print(f"  Perplexity (before): {ppl_before:.2f}")

    # -------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------
    print("\n[4/5] Fine-tuning GPT-2...")

    output_dir = "./gpt2-finetuned"

    training_args = TrainingArguments(
        output_dir          = output_dir,
        overwrite_output_dir= True,
        num_train_epochs    = 3,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        # Low learning rate — we're fine-tuning, not training from scratch
        # High LR causes "catastrophic forgetting" of pretrained knowledge
        learning_rate       = 2e-5,
        weight_decay        = 0.01,    # L2 regularisation on weights
        warmup_steps        = 20,      # Linearly ramp LR from 0 to peak
        eval_strategy       = "epoch",
        save_strategy       = "epoch",
        load_best_model_at_end = True,
        logging_steps       = 5,
        fp16                = torch.cuda.is_available(),  # Mixed precision on GPU
        report_to           = "none",  # Disable wandb for standalone use
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        data_collator   = data_collator,
        tokenizer       = tokenizer,
    )

    train_result = trainer.train()
    print(f"\n  Training complete!")
    print(f"  Final training loss:  {train_result.training_loss:.4f}")

    # -------------------------------------------------------
    # Generation AFTER fine-tuning
    # -------------------------------------------------------
    print("\n[5/5] Generating text AFTER fine-tuning...")

    for prompt in prompts:
        output = generate_text(model, tokenizer, prompt, max_new_tokens=50)
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: {output[0][:120]}...\n")

    ppl_after = compute_perplexity(model, tokenizer, eval_texts)
    print(f"  Perplexity (before): {ppl_before:.2f}")
    print(f"  Perplexity (after):  {ppl_after:.2f}")
    print(f"  Improvement:         {(ppl_before - ppl_after):.2f} PPL reduction")

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved → {output_dir}/")

    print("\n✅ Fine-tuning complete!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on custom text data")
    parser.add_argument('--data', type=str, default=None,
                        help="Path to a .txt file with paragraphs separated by blank lines")
    args = parser.parse_args()
    main(data_path=args.data)
