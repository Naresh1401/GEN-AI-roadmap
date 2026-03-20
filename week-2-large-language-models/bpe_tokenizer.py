"""
bpe_tokenizer.py
=================
Implements Byte-Pair Encoding (BPE) tokenization from scratch.

BPE was introduced for NMT in Sennrich et al. (2016) and became the
foundation of GPT-2/3/4 tokenizers (via tiktoken) and BERT's WordPiece.

Algorithm:
    1. Start with a character-level vocabulary
    2. Count all adjacent symbol pairs across the corpus
    3. Merge the most frequent pair into a single new symbol
    4. Repeat steps 2-3 until vocabulary size is reached

Why BPE matters:
    - Fixed vocabulary size, open vocabulary coverage (never OOV)
    - Frequent words get their own token; rare words split into subwords
    - Compression: reduces sequence length vs character-level
    - The tokenizer is language-agnostic (works on any byte sequence)

Usage:
    python bpe_tokenizer.py
"""

import re
import json
import collections
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Core BPE Training
# ---------------------------------------------------------------------------

def get_vocab_from_corpus(corpus: List[str]) -> Dict[str, int]:
    """
    Pre-tokenize corpus (split on whitespace) and build initial character vocab.

    Each word is represented as a tuple of characters plus a special end-of-word
    marker '</w>'. The marker is crucial: it lets the tokenizer know where word
    boundaries are, so "low" and "lower" don't produce ambiguous merges.

    Args:
        corpus: List of training text strings

    Returns:
        vocab: Dict mapping space-separated character sequences to their frequency
               e.g., {'l o w </w>': 5, 'l o w e r </w>': 3, ...}
    """
    vocab = collections.defaultdict(int)
    for text in corpus:
        # Simple whitespace pre-tokenization (production BPE uses regex patterns)
        for word in text.lower().split():
            # Strip punctuation from word boundaries
            word = re.sub(r'[^a-z0-9]', '', word)
            if word:
                # Represent each word as space-separated characters + end marker
                char_word = ' '.join(list(word)) + ' </w>'
                vocab[char_word] += 1
    return dict(vocab)


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """
    Count frequency of every adjacent symbol pair across the vocabulary.

    Args:
        vocab: Dict of {character-sequence: frequency}

    Returns:
        pairs: Dict of {(sym1, sym2): total_count}
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            # Each pair occurrence is weighted by word frequency
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def merge_vocab(
    pair: Tuple[str, str],
    vocab: Dict[str, int],
) -> Dict[str, int]:
    """
    Apply a single BPE merge rule to the entire vocabulary.

    Replaces all occurrences of `pair` with the merged symbol.

    Args:
        pair:  The symbol pair to merge, e.g., ('l', 'o')
    
    Returns:
        new_vocab: Updated vocabulary with merge applied
    """
    new_vocab = {}
    # Build regex that matches the pair only as adjacent tokens (with space boundaries)
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    merged = ''.join(pair)  # 'l' + 'o' → 'lo'

    for word, freq in vocab.items():
        new_word = pattern.sub(merged, word)
        new_vocab[new_word] = freq

    return new_vocab


def train_bpe(
    corpus: List[str],
    vocab_size: int = 500,
    verbose: bool = True,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE on a text corpus.

    Args:
        corpus:     List of training strings
        vocab_size: Target vocabulary size (including initial character vocab)
        verbose:    Print merge operations as they happen

    Returns:
        vocab:        Final vocabulary (character sequences → frequencies)
        merge_rules:  Ordered list of merge operations (applied at encode time)
    """
    # Step 1: Build initial character-level vocabulary
    vocab = get_vocab_from_corpus(corpus)

    # Initial symbol set: all unique characters + </w>
    base_symbols = set()
    for word in vocab:
        for sym in word.split():
            base_symbols.add(sym)
    base_vocab_size = len(base_symbols)

    if verbose:
        print(f"Initial character vocabulary: {base_vocab_size} symbols")
        print(f"Target vocabulary size:       {vocab_size}")
        print(f"Number of merges to perform:  {vocab_size - base_vocab_size}\n")

    merge_rules: List[Tuple[str, str]] = []
    num_merges = vocab_size - base_vocab_size

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        # Select the most frequent pair (ties broken by sort order for reproducibility)
        best_pair = max(pairs, key=lambda p: (pairs[p], p))
        best_freq = pairs[best_pair]

        vocab = merge_vocab(best_pair, vocab)
        merge_rules.append(best_pair)

        if verbose and (i < 10 or i % 50 == 0):
            merged = ''.join(best_pair)
            print(f"Merge {i+1:4d}: {best_pair} → '{merged}'  (freq: {best_freq})")

    return vocab, merge_rules


# ---------------------------------------------------------------------------
# 2. BPE Tokenizer (Encode / Decode)
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    BPE Tokenizer with encode and decode methods.

    Special tokens:
        <PAD>  = 0   Padding to equal-length batches
        <UNK>  = 1   Unknown tokens (shouldn't occur after BPE)
        <BOS>  = 2   Beginning of sequence
        <EOS>  = 3   End of sequence

    After training, call .encode(text) → List[int] and .decode(ids) → str.
    """

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(self):
        self.merge_rules: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def train(self, corpus: List[str], vocab_size: int = 500, verbose: bool = True):
        """Train BPE on corpus and build token ↔ ID mappings."""
        _, self.merge_rules = train_bpe(corpus, vocab_size, verbose)

        # Build vocabulary from merge rules + base characters
        # Start with special tokens
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

        # Collect all symbols that appear in the trained vocab
        trained_vocab_words, _ = train_bpe(corpus, vocab_size, verbose=False)
        all_symbols: set = set()
        for word in trained_vocab_words:
            for sym in word.split():
                all_symbols.add(sym)

        # Add merged symbols from merge rules
        for pair in self.merge_rules:
            all_symbols.add(''.join(pair))

        all_tokens = special_tokens + sorted(all_symbols)

        self.token_to_id = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.vocab_size  = len(self.token_to_id)

        if verbose:
            print(f"\nFinal vocabulary size: {self.vocab_size}")

    def _apply_merges(self, word_chars: List[str]) -> List[str]:
        """Apply all merge rules to a list of characters (greedy, ordered)."""
        symbols = list(word_chars)

        for pair in self.merge_rules:
            merged = ''.join(pair)
            i = 0
            while i < len(symbols) - 1:
                if (symbols[i], symbols[i + 1]) == pair:
                    symbols = symbols[:i] + [merged] + symbols[i + 2:]
                    # Don't advance i — new merge might be possible at same position
                else:
                    i += 1

        return symbols

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode a string to a list of token IDs.

        Steps:
            1. Whitespace pre-tokenization
            2. Character split + </w> end marker per word
            3. Apply all merge rules (greedy, in training order)
            4. Map symbols to IDs
        """
        ids = [self.token_to_id[self.BOS_TOKEN]] if add_special_tokens else []
        unk_id = self.token_to_id[self.UNK_TOKEN]

        for word in text.lower().split():
            word_clean = re.sub(r'[^a-z0-9]', '', word)
            if not word_clean:
                continue
            chars = list(word_clean) + ['</w>']
            tokens = self._apply_merges(chars)
            for token in tokens:
                ids.append(self.token_to_id.get(token, unk_id))

        if add_special_tokens:
            ids.append(self.token_to_id[self.EOS_TOKEN])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to text."""
        special = {self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}
        tokens = []

        for id_ in ids:
            token = self.id_to_token.get(id_, self.UNK_TOKEN)
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)

        # Join tokens and clean up end-of-word markers
        text = ''.join(tokens)
        text = text.replace('</w>', ' ').strip()
        return text

    def save(self, path: str):
        """Serialize tokenizer to JSON."""
        data = {
            'merge_rules': self.merge_rules,
            'token_to_id': self.token_to_id,
            'vocab_size':  self.vocab_size,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from JSON."""
        data = json.loads(Path(path).read_text())
        tok = cls()
        tok.merge_rules = [tuple(pair) for pair in data['merge_rules']]
        tok.token_to_id = data['token_to_id']
        tok.id_to_token = {int(k): v for k, v in data.items() if k.isdigit()} 
        tok.id_to_token = {idx: tok_ for tok_, idx in tok.token_to_id.items()}
        tok.vocab_size   = data['vocab_size']
        return tok


# ---------------------------------------------------------------------------
# 3. Compression Analysis
# ---------------------------------------------------------------------------

def analyze_compression(tokenizer: BPETokenizer, test_texts: List[str]):
    """Compare character length vs token count to measure compression ratio."""
    print("\n--- Compression Analysis ---")
    print(f"{'Text':<40} {'Chars':>6} {'Tokens':>7} {'Ratio':>7}")
    print("-" * 64)

    total_chars  = 0
    total_tokens = 0

    for text in test_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        chars  = len(text.replace(' ', ''))  # chars without spaces
        tokens = len(ids)
        ratio  = chars / max(tokens, 1)
        total_chars  += chars
        total_tokens += tokens
        print(f"{text[:38]:<40} {chars:>6} {tokens:>7} {ratio:>7.2f}x")

    overall = total_chars / max(total_tokens, 1)
    print(f"\nOverall compression: {overall:.2f}x chars per token")


# ---------------------------------------------------------------------------
# 4. Demo / Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("BPE Tokenizer — Training & Demo")
    print("=" * 65)

    # Training corpus: a small ML-focused text sample
    CORPUS = [
        "the transformer architecture uses self attention mechanisms to process sequences",
        "attention is all you need is a seminal paper in natural language processing",
        "byte pair encoding is used for tokenization in large language models",
        "the encoder processes the input sequence and the decoder generates the output",
        "lower lower lowest lower lower lower lower",
        "newer newer newest newer newer newer newer",
        "the neural network learns to predict the next token given previous tokens",
        "language models are trained on large corpora of text from the internet",
        "fine tuning adapts a pretrained model to a specific downstream task",
        "tokenization converts raw text into integer token identifiers",
        "the vocabulary size determines how many unique tokens the model can represent",
        "subword tokenization handles rare words by splitting them into known pieces",
        "the embedding layer maps token identifiers to dense vector representations",
        "positional encodings provide the model with information about token order",
        "multi head attention allows the model to attend to different positions simultaneously",
        "residual connections help gradients flow through deep neural networks",
        "layer normalization stabilizes training by normalizing activations",
        "the feed forward network applies a two layer mlp to each position independently",
        "beam search is a common decoding strategy for sequence to sequence models",
        "greedy decoding selects the highest probability token at each step",
    ]

    # Train tokenizer
    print("\n[1/4] Training BPE tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300, verbose=True)

    # Encode / decode round-trip
    print("\n[2/4] Encode → Decode round-trip tests")
    test_cases = [
        "lowest",
        "transformer architecture",
        "tokenization",
        "language models learn",
        "newer and newest",
    ]

    for text in test_cases:
        ids    = tokenizer.encode(text, add_special_tokens=False)
        tokens = [tokenizer.id_to_token[i] for i in ids]
        decoded = tokenizer.decode(ids)
        match  = "✅" if decoded.strip() == text.strip() else "⚠️"
        print(f"  {match} '{text}' → {tokens} → '{decoded}'")

    # Compression analysis
    analyze_compression(tokenizer, [
        "the transformer architecture",
        "fine tuning large language models",
        "byte pair encoding tokenization",
        "multi head self attention mechanisms",
        "positional encoding in transformers",
    ])

    # Demonstrate unknown subword handling
    print("\n[3/4] Out-of-vocabulary handling")
    unk_text = "cryptography"  # Likely not in tiny corpus
    ids = tokenizer.encode(unk_text, add_special_tokens=False)
    tokens = [tokenizer.id_to_token.get(i, '<UNK>') for i in ids]
    print(f"  'cryptography' → {tokens}")
    print(f"  (subword pieces, not <UNK> — BPE never fully fails)")

    # Save and reload
    print("\n[4/4] Save / reload")
    tokenizer.save("bpe_tokenizer.json")
    loaded = BPETokenizer.load("bpe_tokenizer.json")
    assert loaded.encode("transformer") == tokenizer.encode("transformer"), "Round-trip failed!"
    print("  Save/load round-trip: ✅")

    print("\n" + "=" * 65)
    print("✅ BPE Tokenizer complete!")
    print("Next: See finetune_gpt2.py for how GPT-2 uses its tokenizer during training.")
    print("=" * 65)
