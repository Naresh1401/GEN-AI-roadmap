"""
ragas_evaluation.py
===================
Implements simplified versions of RAGAS metrics to evaluate RAG pipelines.

RAGAS (Retrieval-Augmented Generation Assessment) provides LLM-graded metrics:
    - Faithfulness:       Are all claims in the answer supported by the context?
    - Answer Relevancy:   Does the answer address the question?
    - Context Precision:  Is retrieved context relevant to the question?
    - Context Recall:     Does the context contain all info needed for the answer?

This file implements lightweight approximations (no LLM calls needed).
For production RAGAS, pip install ragas and plug in your LLM API key.

Usage:
    python ragas_evaluation.py
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 1. RAGAS Metric Implementations
# ---------------------------------------------------------------------------

@dataclass
class RAGASResult:
    """Holds all RAGAS metric scores for a single Q&A evaluation."""
    question:           str
    answer:             str
    context:            str
    faithfulness:       float  # 0–1
    answer_relevancy:   float  # 0–1
    context_precision:  float  # 0–1


class RAGASEvaluator:
    """
    Approximates RAGAS metrics using embedding similarity.

    Production RAGAS calls an LLM to judge faithfulness and relevancy.
    These approximations use cosine similarity as a proxy — cheaper,
    deterministic, but less nuanced than LLM judgment.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"  Loading evaluation model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: Is the answer grounded in the provided context?

        Approximation: Split answer into sentences, measure max cosine
        similarity of each sentence to the context. Average across sentences.

        Production approach: LLM generates all factual claims from the answer,
        then checks each claim against the context. Faithfulness = fraction
        of claims that can be verified from context.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
        if not sentences:
            sentences = [answer]

        ctx_emb  = self.model.encode([context])[0]
        sent_embs = self.model.encode(sentences)

        scores = [self._cosine(se, ctx_emb) for se in sent_embs]
        return float(np.mean(scores))

    def answer_relevancy(self, answer: str, question: str) -> float:
        """
        Answer Relevancy: Does the answer address the question?

        Approximation: Cosine similarity between answer and question embeddings.

        Production approach: LLM generates N hypothetical questions that
        the answer would address. Average cosine sim to the original question.
        High similarity → answer is directly addressing the question.
        """
        q_emb = self.model.encode([question])[0]
        a_emb = self.model.encode([answer])[0]
        return self._cosine(q_emb, a_emb)

    def context_precision(self, context_chunks: List[str], question: str) -> float:
        """
        Context Precision: How relevant is each retrieved chunk to the question?

        Approximation: Average cosine similarity of each retrieved chunk to the question.
        High precision → every retrieved chunk is relevant, no noise.
        """
        if not context_chunks:
            return 0.0
        q_emb   = self.model.encode([question])[0]
        c_embs  = self.model.encode(context_chunks)
        scores  = [self._cosine(ce, q_emb) for ce in c_embs]
        return float(np.mean(scores))

    def evaluate(
        self,
        question:       str,
        answer:         str,
        context:        str,
        context_chunks: List[str],
    ) -> RAGASResult:
        """Run all RAGAS metrics for a single QA pair."""
        return RAGASResult(
            question          = question,
            answer            = answer,
            context           = context,
            faithfulness      = self.faithfulness(answer, context),
            answer_relevancy  = self.answer_relevancy(answer, question),
            context_precision = self.context_precision(context_chunks, question),
        )

    def evaluate_dataset(
        self,
        dataset: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate a full dataset of QA pairs.

        Args:
            dataset: List of dicts with keys: question, answer, context, context_chunks

        Returns:
            Aggregate metric scores (mean across all examples)
        """
        results = []
        for ex in dataset:
            r = self.evaluate(
                question       = ex['question'],
                answer         = ex['answer'],
                context        = ex['context'],
                context_chunks = ex.get('context_chunks', [ex['context']]),
            )
            results.append(r)

        return {
            "faithfulness":       np.mean([r.faithfulness      for r in results]),
            "answer_relevancy":   np.mean([r.answer_relevancy  for r in results]),
            "context_precision":  np.mean([r.context_precision for r in results]),
            "ragas_score":        np.mean([
                np.mean([r.faithfulness, r.answer_relevancy, r.context_precision])
                for r in results
            ]),
            "individual_results": results,
        }


# ---------------------------------------------------------------------------
# 2. Demo / Main
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    {
        "question": "What is the formula for scaled dot-product attention?",
        "context":  "Scaled dot-product attention computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V. The scaling by sqrt(d_k) prevents gradient vanishing when d_k is large.",
        "answer":   "The formula is softmax(QK^T divided by the square root of d_k) multiplied by V. The square root scaling prevents very small gradients.",
        "context_chunks": [
            "Scaled dot-product attention computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.",
            "The scaling by sqrt(d_k) prevents gradient vanishing when d_k is large.",
        ],
    },
    {
        "question": "How does LoRA reduce memory requirements?",
        "context":  "LoRA (Low-Rank Adaptation) freezes the original model weights and adds small trainable matrices A and B to each attention layer. The update is W + AB^T where A and B have rank r << d. This reduces trainable parameters by over 90%.",
        "answer":   "LoRA adds small low-rank matrices to attention layers while freezing the original weights. Since rank r is much smaller than the full dimension, the number of trainable parameters is dramatically reduced.",
        "context_chunks": [
            "LoRA (Low-Rank Adaptation) freezes the original model weights and adds small trainable matrices A and B.",
            "The update is W + AB^T where A and B have rank r << d. This reduces trainable parameters by over 90%.",
        ],
    },
    {
        "question": "What is RLHF used for in language models?",
        "context":  "Positional encodings add information about token order to embeddings. GPT-2 uses learned absolute position embeddings, adding a position-specific vector to each token embedding.",
        # Deliberately bad context — tests faithfulness (answer should not claim more than context says)
        "answer":   "RLHF stands for Reinforcement Learning from Human Feedback. It is used to align language models with human preferences by training a reward model on human preference labels.",
        "context_chunks": [
            "Positional encodings add information about token order to embeddings.",
        ],
    },
    {
        "question": "What does RAGAS stand for and what does it measure?",
        "context":  "RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG pipelines. It measures faithfulness, answer relevancy, context precision, and context recall. Faithfulness checks if the answer is grounded in the context.",
        "answer":   "RAGAS stands for Retrieval-Augmented Generation Assessment. It evaluates RAG systems on faithfulness (grounding), answer relevancy, context precision, and context recall.",
        "context_chunks": [
            "RAGAS is a framework for evaluating RAG pipelines.",
            "It measures faithfulness, answer relevancy, context precision, and context recall.",
        ],
    },
]


if __name__ == "__main__":
    print("=" * 65)
    print("RAGAS Evaluation Demo")
    print("=" * 65)

    print("\n[1/3] Initialising evaluator...")
    evaluator = RAGASEvaluator()

    print("\n[2/3] Evaluating individual examples...")
    for ex in EVAL_DATASET:
        r = evaluator.evaluate(
            question       = ex['question'],
            answer         = ex['answer'],
            context        = ex['context'],
            context_chunks = ex.get('context_chunks', [ex['context']]),
        )
        print(f"\n  Q: {r.question[:60]}...")
        print(f"  Faithfulness:       {r.faithfulness:.3f}  (≥0.8 = good)")
        print(f"  Answer Relevancy:   {r.answer_relevancy:.3f}  (≥0.7 = good)")
        print(f"  Context Precision:  {r.context_precision:.3f}  (≥0.6 = good)")
        avg = (r.faithfulness + r.answer_relevancy + r.context_precision) / 3
        flag = "✅" if avg > 0.65 else "⚠️ "
        print(f"  {flag} Average: {avg:.3f}")

    print("\n[3/3] Aggregate evaluation across dataset...")
    agg = evaluator.evaluate_dataset(EVAL_DATASET)
    print(f"\n  Faithfulness:       {agg['faithfulness']:.3f}")
    print(f"  Answer Relevancy:   {agg['answer_relevancy']:.3f}")
    print(f"  Context Precision:  {agg['context_precision']:.3f}")
    print(f"  ─────────────────────────────")
    print(f"  RAGAS Score:        {agg['ragas_score']:.3f}")

    print("\n  Note: Example 3 (wrong context) should score low faithfulness")
    print("  — the answer claims facts not present in the retrieved context.")
    ex3 = agg['individual_results'][2]
    print(f"  Example 3 faithfulness: {ex3.faithfulness:.3f}  ({'⚠️ low as expected' if ex3.faithfulness < 0.5 else '✅'})")

    print("\n" + "=" * 65)
    print("✅ RAGAS evaluation complete!")


# ============================================================
# 🔷 SECTION 4 — Concept Deep Dive
# ============================================================
"""
CONCEPT DEEP DIVE: Week 3 — Prompt Engineering & RAG
------------------------------------------------------

1. FEW-SHOT PROMPTING
   What: Providing input-output examples directly in the prompt before your actual query.
   Analogy: A style guide for a contractor. You don't explain the rules — you just show
   5 samples of work done correctly. The contractor infers the pattern.
   Why it works: The examples shift the model's prior distribution. Given "positive → 5 stars,
   negative → 1 star, mixed → ?", the model is constrained to use your format.
   Misconception: "More examples are always better." False. 4–8 well-chosen, diverse examples
   typically outperform 20+ examples, and longer prompts increase latency and cost.

2. CHAIN-OF-THOUGHT (CoT)
   What: Appending "Let's think step by step" or including step-by-step worked examples
   in the prompt. The model generates intermediate reasoning tokens before the final answer.
   Analogy: Working memory for the model. A person can't multiply 847 × 293 in their head —
   but writing down intermediate steps makes it tractable. Generated tokens ARE working memory.
   Why it works: The model's next-token prediction is conditioned on ALL previous tokens.
   Intermediate reasoning tokens update the context toward a correct final answer.
   Misconception: "CoT means the model is actually reasoning." The model is generating tokens
   that look like reasoning. The accuracy improvement is real, but the mechanism is
   statistical, not symbolic. Models can produce plausible-looking CoT with a wrong answer.

3. RETRIEVAL-AUGMENTED GENERATION (RAG)
   What: Augmenting LLM generation with documents retrieved from an external store at query time.
   Analogy: The difference between asking a doctor to recall a fact from memory vs. handing
   them the patient's full medical record before they answer. Same doctor, much better answer.
   Why it works: LLMs have static weights (knowledge cutoff). RAG provides dynamic,
   updatable knowledge without retraining. Retrieval quality determines generation quality.
   Misconception: "RAG solves hallucination." RAG reduces hallucination when the retrieved
   context is relevant. If retrieval fails (returns wrong chunks), the LLM will hallucinate
   *confidently* using the bad context — sometimes worse than no context at all.

4. VECTOR EMBEDDINGS
   What: Dense numerical representations of text where semantic similarity corresponds
   to geometric proximity (typically cosine similarity).
   Analogy: A map where similar concepts are geographically close. "King" and "Queen"
   are nearby; "automobile" and "car" are nearly identical; "banana" and "attention
   mechanism" are far apart.
   Why they matter: Keyword search fails on paraphrases ("car" ≠ "automobile"). Embeddings
   capture meaning independent of exact wording, enabling semantic search.

5. RAGAS FAITHFULNESS
   What: The fraction of claims in the generated answer that can be attributed to
   the retrieved context. Measured by LLM-based claim extraction and verification.
   Analogy: A journalist's citation check. Every specific claim must be traceable
   to a source document. Unsourced claims = hallucinations.
   Why it matters: A model with access to perfect context can still invent unsupported
   claims. Faithfulness specifically measures this failure mode.

COMMON MISCONCEPTIONS:
1. "Larger context windows make RAG obsolete." Even GPT-4 with 128K context can't
   fit a company's entire document library. Also, attention degrades on very long
   contexts (lost-in-the-middle effect) — retrieval is still better than stuffing
   everything in.
2. "Prompt engineering is just about wording." Prompt structure, example selection,
   delimiter choice, and output format specification are all engineering decisions
   with measurable impact on downstream task performance.
3. "BM25 is outdated." BM25 consistently beats dense retrieval on keyword-heavy queries
   (medical codes, product IDs, technical terms). Production systems almost always
   use hybrid retrieval. SPLADE and other learned sparse models extend BM25 with
   neural capabilities.

INTERVIEW PREP (Week 3):

Q1: What is the difference between sparse and dense retrieval?
A: Sparse retrieval (BM25, TF-IDF) represents documents as sparse bags of words —
mostly zeros with nonzero values for present terms. Fast, interpretable, excellent
for exact keyword matching. Dense retrieval (embedding models) maps text to dense
vectors using neural networks — captures semantic similarity across paraphrases.
Hybrid search (RRF of both) is the production standard.

Q2: How does RAGAS measure faithfulness?
A: Faithfulness = (# answer claims supported by context) / (# total answer claims).
RAGAS uses an LLM to extract all factual claims from the answer as atomic statements,
then uses an LLM to verify each claim against the retrieved context. Score = fraction
of verified claims. This requires two LLM calls per evaluation sample.

Q3: What is Reciprocal Rank Fusion and why is it used in hybrid search?
A: RRF combines multiple ranked lists by summing 1/(k + rank) for each document
across all lists. k=60 dampens rank sensitivity. RRF uses only *ranks*, not raw
scores — making it robust to score scale differences between BM25 (can be >100)
and cosine similarity (always 0–1). It consistently outperforms score averaging.

Q4: What is "lost in the middle" and how does it affect RAG design?
A: Research shows LLMs perform worse on information placed in the middle of a long
context compared to the beginning or end — a U-shaped performance curve. In RAG,
this means your most relevant chunks should go at the start or end of the context
block, not sandwiched in the middle. It also means simply using larger top-k is not
always better.

Q5: What is the difference between an embedding model and a generative model?
A: Embedding models (SBERT, E5, BGE) map text to a fixed-size vector for similarity
comparison. They're typically encoder-only (BERT-style) and never generate text.
Generative models (GPT, LLaMA) produce new text token by token. RAG pipelines use
embedding models for retrieval and generative models for synthesis. They can overlap
(LLM embeddings via mean pooling) but are optimised for different objectives.

SYSTEM DESIGN Q1: Design a RAG system for a legal firm's 50TB document archive.
Key points: Chunking strategy for long legal documents (hierarchical: section → paragraph);
metadata filtering (jurisdiction, date, case type) to pre-filter before embedding search;
hybrid retrieval (BM25 for case numbers/citations + dense for semantic search); reranker
model (cross-encoder) for top-20 → top-5 reranking; source attribution in generated
answers for auditability; human-in-the-loop for high-stakes answers.

SYSTEM DESIGN Q2: You notice your RAG system has 0.4 faithfulness. How do you improve it?
Diagnosis first: are retrieved chunks relevant? If not → fix retrieval (chunk size, overlap,
reranking). If yes but faithfulness is low → prompt engineering (be more explicit:
"answer ONLY from context, if not present say I don't know"), add self-consistency checks,
reduce temperature, or add a post-generation faithfulness verifier that scores and rejects
unfaithful outputs.

GOTCHA Q1: "RAG context always improves answers, right?"
Wrong — contextual distraction. LLMs are susceptible to context manipulation. If you
inject adversarial or irrelevant context, the model often uses it. Studies show that
adding wrong-but-plausible context can reduce accuracy below no-context baseline.

GOTCHA Q2: "Why not just use keyword search (Elasticsearch) instead of embeddings?"
Keyword search fails on paraphrases, misspellings, and cross-language queries. But BM25
is not obsolete — it handles exact terminology, product codes, and rare proper nouns better
than embeddings. Production systems almost always hybridise.

GOTCHA Q3: "Can you evaluate a RAG system without ground truth answers?"
Yes — reference-free evaluation. RAGAS faithfulness doesn't need ground truth answers,
only question + context + generated answer. You can measure faithfulness and answer
relevancy on any generated output. Context recall requires ground truth. Building
synthetic evaluation sets (LLM-generated QA pairs from your documents) is standard practice.
"""
