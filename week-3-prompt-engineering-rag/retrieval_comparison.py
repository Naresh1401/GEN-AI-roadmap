"""
retrieval_comparison.py
=======================
Compares three retrieval methods for RAG:
    1. Dense retrieval  — embedding cosine similarity (semantic)
    2. Sparse retrieval — BM25 (keyword-based, no embeddings)
    3. Hybrid retrieval — reciprocal rank fusion of both

Each method has different strengths:
    Dense:  Catches semantic similarity ("automobile" ≈ "car")
    Sparse: Precise keyword matching, excels at rare terms, IDs, codes
    Hybrid: Best of both — robust across query types

Usage:
    python retrieval_comparison.py
"""

import math
import re
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 1. BM25 (Sparse Retrieval)
# ---------------------------------------------------------------------------

class BM25:
    """
    BM25 (Best Match 25) — the gold standard sparse retrieval algorithm.

    BM25 extends TF-IDF with two key improvements:
        1. Term frequency saturation: adding more occurrences of a term
           yields diminishing returns (controlled by k1)
        2. Document length normalisation: longer documents are penalised
           for having more term occurrences by chance (controlled by b)

    Score formula:
        score(q, d) = Σ IDF(t) * (TF(t,d) * (k1+1)) / (TF(t,d) + k1*(1-b+b*|d|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation (typical: 1.2–2.0)
            b:  Length normalisation (0=no norm, 1=full norm, typical: 0.75)
        """
        self.k1 = k1
        self.b  = b
        self.corpus_tokens: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_freq: Dict[str, int] = defaultdict(int)  # Number of docs containing term
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        self.N: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return re.findall(r'\b[a-z]+\b', text.lower())

    def fit(self, documents: List[str], doc_ids: List[str]):
        """Build the BM25 index from a list of documents."""
        self.N = len(documents)
        self.doc_ids = doc_ids

        self.corpus_tokens = []
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc)
            self.corpus_tokens.append(tokens)
            total_len += len(tokens)

            # Track which docs contain each unique term
            for term in set(tokens):
                self.doc_freq[term] += 1

        self.avg_doc_len = total_len / self.N

        # Precompute IDF for all terms
        # IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        # The +1 prevents negative IDF for very common terms
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k documents for a query."""
        query_terms = self._tokenize(query)
        scores = np.zeros(self.N)

        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc_tokens in enumerate(self.corpus_tokens):
                tf = doc_tokens.count(term)
                dl = len(doc_tokens)
                # BM25 scoring formula
                numerator   = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
                scores[i]  += idf * (numerator / denominator)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]


# ---------------------------------------------------------------------------
# 2. Dense Retrieval
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Embedding-based semantic retrieval using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray = None
        self.doc_ids: List[str] = []

    def fit(self, documents: List[str], doc_ids: List[str]):
        self.doc_ids   = doc_ids
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)

    def score(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        qe    = self.model.encode([query], normalize_embeddings=True)[0]
        sims  = self.embeddings @ qe
        idx   = np.argsort(sims)[::-1][:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]


# ---------------------------------------------------------------------------
# 3. Hybrid Retrieval — Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for a document d:
        RRF(d) = Σ 1 / (k + rank(d, list_i))

    k=60 is the canonical default from the original paper (Cormack 2009).
    It dampens the advantage of very high-ranked documents, making the
    fusion more robust to noise in individual rankers.

    Why RRF instead of score averaging?
        Scores from different systems (e.g., BM25 scores vs cosine similarity)
        are not on the same scale. RRF only uses *ranks*, making it
        scale-agnostic and very robust.
    """
    doc_scores: Dict[str, float] = defaultdict(float)

    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, start=1):
            doc_scores[doc_id] += 1.0 / (k + rank)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


# ---------------------------------------------------------------------------
# 4. Evaluation: Recall@K and MRR
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k results."""
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & set(relevant)) / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """MRR: 1/rank of the first relevant document (0 if none found)."""
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    method_name: str,
    results: List[Tuple[str, float]],
    relevant_ids: List[str],
    k_values: List[int] = [1, 3, 5],
):
    """Print recall@k and MRR for a retrieval result."""
    retrieved = [doc_id for doc_id, _ in results]
    print(f"\n  {method_name}:")
    for k in k_values:
        r = recall_at_k(retrieved, relevant_ids, k)
        print(f"    Recall@{k}: {r:.2f}")
    mrr = mean_reciprocal_rank(retrieved, relevant_ids)
    print(f"    MRR:      {mrr:.3f}")
    print(f"    Top-3: {retrieved[:3]}")


# ---------------------------------------------------------------------------
# 5. Demo / Main
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "doc_attention":    "The scaled dot-product attention computes softmax(QK^T / sqrt(d_k)) * V. Multi-head attention applies this h times with different learned projections.",
    "doc_gpt":          "GPT uses a decoder-only Transformer with causal masking. It is trained with the causal language modelling objective on large text corpora.",
    "doc_bert":         "BERT uses a bidirectional encoder trained with masked language modelling. It is well-suited for classification and embedding tasks.",
    "doc_lora":         "LoRA fine-tunes language models by adding trainable low-rank matrices to attention layers. This reduces trainable parameters by over 90%.",
    "doc_rag":          "RAG systems combine document retrieval with language model generation. The retrieval quality determines the grounding quality of generated answers.",
    "doc_tokenizer":    "BPE tokenization iteratively merges the most frequent adjacent character pairs. It produces subword tokens that balance vocabulary size with coverage.",
    "doc_kvcache":      "KV cache stores key and value tensors between generation steps. It eliminates redundant computation, making autoregressive generation practical.",
    "doc_diffusion":    "Diffusion models learn to reverse a Gaussian noise process. DDPM trains a neural network to predict the noise added at each diffusion step.",
    "doc_clip":         "CLIP trains a vision encoder and text encoder jointly on image-text pairs. It learns a shared embedding space where similar images and captions are nearby.",
    "doc_quantisation": "Post-training quantisation reduces model precision from FP32 to INT8 or INT4. This reduces memory and increases throughput with minimal accuracy loss.",
}

QUERIES_AND_RELEVANT = [
    {
        "query":    "How does attention compute similarity between tokens?",
        "relevant": ["doc_attention"],
    },
    {
        "query":    "reducing memory during fine-tuning with low-rank matrices",
        "relevant": ["doc_lora"],
    },
    {
        "query":    "KV cache kvcache key value generation efficiency",  # Keyword-heavy query
        "relevant": ["doc_kvcache"],
    },
    {
        "query":    "training image and text in a shared embedding space",
        "relevant": ["doc_clip"],
    },
]


if __name__ == "__main__":
    print("=" * 65)
    print("Retrieval Methods Comparison: Sparse vs Dense vs Hybrid")
    print("=" * 65)

    doc_ids   = list(DOCUMENTS.keys())
    doc_texts = list(DOCUMENTS.values())

    # Fit all retrievers
    print("\n[1/3] Fitting retrievers...")

    print("  Fitting BM25...")
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(doc_texts, doc_ids)

    print("  Fitting Dense Retriever (all-MiniLM-L6-v2)...")
    dense = DenseRetriever()
    dense.fit(doc_texts, doc_ids)

    print("  Done.")

    # Evaluate on test queries
    print("\n[2/3] Evaluating retrieval quality...")

    aggregate: Dict[str, List[float]] = defaultdict(list)

    for qa in QUERIES_AND_RELEVANT:
        query    = qa["query"]
        relevant = qa["relevant"]
        print(f"\n  Query: '{query}'")
        print(f"  Expected: {relevant}")

        bm25_results   = bm25.score(query, top_k=5)
        dense_results  = dense.score(query, top_k=5)
        hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results], k=60)[:5]

        for name, results in [("BM25 (sparse)", bm25_results),
                               ("Dense (semantic)", dense_results),
                               ("Hybrid (RRF)", hybrid_results)]:
            retrieved = [d for d, _ in results]
            mrr = mean_reciprocal_rank(retrieved, relevant)
            r1  = recall_at_k(retrieved, relevant, 1)
            print(f"    {name:22s}: R@1={r1:.1f}  MRR={mrr:.3f}  Top1={retrieved[0] if retrieved else '-'}")
            aggregate[name].append(mrr)

    # Aggregate MRR
    print("\n[3/3] Aggregate Mean MRR across all queries:")
    for name, mrrs in aggregate.items():
        avg = sum(mrrs) / len(mrrs)
        print(f"  {name:25s}: MRR = {avg:.3f}")

    print("\n  Note: Hybrid typically wins on mixed query sets.")
    print("  Dense wins on paraphrase/semantic queries.")
    print("  Sparse wins on exact keyword / ID queries.")

    print("\n" + "=" * 65)
    print("✅ Retrieval comparison complete!")
