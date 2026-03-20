"""
mock_interview_questions.py
============================
Week 10: Comprehensive Review & Mock Interviews

This file contains:
1. 20 conceptual Q&A covering all 9 weeks
2. 5 system design challenges with model answers
3. 10 "gotcha" questions (common misconceptions)
4. Oral explanation templates

Run this file for an interactive mock interview session.

Usage:
    python mock_interview_questions.py
    python mock_interview_questions.py --random    # Shuffle questions
    python mock_interview_questions.py --category transformer
"""

import sys
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Question:
    category: str
    question: str
    answer:   str
    hint:     Optional[str] = None


# ---------------------------------------------------------------------------
# 1. Conceptual Questions (20 total, ~2 per week)
# ---------------------------------------------------------------------------

CONCEPTUAL_QUESTIONS = [
    # Week 1: Transformer Foundations
    Question(
        category = "transformer",
        question = "Why is the attention score divided by √d_k before applying softmax?",
        answer   = """
Without scaling, large d_k causes the dot products Q·Kᵀ to have large magnitude.
Large inputs to softmax push the output toward near-one-hot distributions (very peaked),
causing vanishing gradients for all tokens except the argmax.

Dividing by √d_k keeps the variance of the dot products constant regardless of d_k:
  Var(q·k) = d_k × Var(q_i)×Var(k_i)
After scaling: Var((q·k)/√d_k) = Var(q_i)×Var(k_i) = 1 (assuming unit variance init)

This ensures informative gradients can flow through softmax during early training.
        """.strip(),
        hint = "Think about what large inputs do to softmax, and what that means for gradients.",
    ),
    
    Question(
        category = "transformer",
        question = "What does multi-head attention provide over single-head attention?",
        answer   = """
Multi-head attention (h heads) allows the model to attend to different aspects of the
sequence simultaneously, using different learned linear projections:
  - Head 1 might capture syntactic dependencies
  - Head 2 might capture long-range semantic relations
  - Head 3 might capture local contextual patterns

Mathematically: split d_model into h subspaces of dimension d_model/h.
Run attention independently in each subspace. Concatenate and project back.

Single-head must compromise and attend to all patterns in one pass.
Multi-head is an ensemble of attention "perspectives" with no extra parameters
(total QKV dimension stays the same when split across heads).
        """.strip(),
    ),
    
    # Week 2: LLMs
    Question(
        category = "llm",
        question = "What is the KV cache and why does it make autoregressive generation faster?",
        answer   = """
In autoregressive generation, to predict token t_{n+1}, the model runs attention over
all previous tokens [t_1, ..., t_n]. Without caching, this recomputes K and V for all
n previous tokens at every step — O(n) recomputation per step → O(n²) total.

KV cache stores the Key and Value tensors for all previously computed tokens.
At each new step, only the NEW token's Q,K,V are computed. The new K,V are appended
to the cache; attention is run over the cached K,V + new K,V → O(1) per step.

Memory cost: 2 × layers × heads × head_dim × seq_len × bytes_per_element
For LLaMA-3-70B at 128K context in FP16: ~320GB — larger than the model weights!
        """.strip(),
        hint = "Think about which computation is redundant when generating one new token.",
    ),
    
    Question(
        category = "llm",
        question = "What is Byte Pair Encoding (BPE) and why is it used over word-level tokenisation?",
        answer   = """
BPE starts with individual characters as the vocabulary, then iteratively merges
the most frequent adjacent pair of tokens until the target vocabulary size is reached.

Advantages over word-level tokenisation:
1. No <UNK> tokens: any string can be encoded (worst case: character by character)
2. Handles rare words: "unbelievably" → ["un", "believ", "ably"] — all common subwords
3. Multilingual: character overlap across languages
4. Morphologically productive: "running", "runs", "runner" all share "run" subword

Compared to character-level: sequences are shorter (more efficient attention),
and subwords carry more semantic content than individual characters.
        """.strip(),
    ),
    
    # Week 3: RAG & Prompt Engineering
    Question(
        category = "rag",
        question = "What is RAGAS Faithfulness and why is it important?",
        answer   = """
RAGAS Faithfulness measures whether all factual claims in the generated answer
can be attributed to the retrieved context.

Faithfulness = (# answer claims supported by context) / (# total claims in answer)

It's important because an LLM can confidently generate plausible-sounding claims
that are NOT in the retrieved context (hallucinations). A high faithfulness score
means the model is "citing its sources" — the answer is grounded in what was retrieved,
not invented from pretrained parametric memory.

Measured via an LLM judge: extract claims from answer → verify each against context.
Low faithfulness = the RAG system is hallucinating despite having retrieval.
        """.strip(),
    ),
    
    Question(
        category = "rag",
        question = "What is Reciprocal Rank Fusion and why does it work better than score averaging for hybrid search?",
        answer   = """
RRF combines multiple ranked result lists by summing 1/(k + rank_i) for each document
across all lists. k=60 prevents extreme scores for rank-1 items.

Why better than score averaging:
  1. Score scales differ: BM25 scores can be 0-50+; cosine similarity is 0-1.
     Averaging directly would make BM25 dominate. RRF uses only ranks.
  2. Rank-invariance: RRF gives the same result regardless of whether BM25 scores are
     multiplied by 10 or 0.1 — only the ordering matters.
  3. Empirically robust: RRF consistently outperforms score fusion across diverse
     query types and was validated on TREC benchmarks (Cormack et al., 2009).
        """.strip(),
    ),
    
    # Week 4: Fine-Tuning
    Question(
        category = "finetuning",
        question = "Why is LoRA's B matrix initialised to zeros?",
        answer   = """
LoRA's weight update is ΔW = BA (scaled by α/r).

If B is initialised to zeros: ΔW = 0·A = 0 at step 0.
This means the fine-tuned model starts IDENTICAL to the pretrained model.

Why this matters:
- The pretrained model is at a well-optimised point in parameter space.
- Starting with a random ΔW immediately disturbs that optimum before any useful
  gradient information has been observed.
- Zero B ensures the pretrained model's behaviour is preserved at t=0.
  Training then smoothly adapts from this stable starting point.

A is initialised randomly (not zero) to break symmetry — zero A would also give
ΔW=0, but then no gradient could flow through A at all.
        """.strip(),
    ),
    
    Question(
        category = "finetuning",
        question = "What is the difference between SFT, RLHF, and DPO in the LLM training pipeline?",
        answer   = """
SFT (Supervised Fine-Tuning):
  - Input: (instruction, desired_response) pairs
  - Loss: standard cross-entropy (teacher forcing)
  - Output: an instruction-following model
  - Purpose: teach the model to follow instructions and maintain conversation format

RLHF (Reinforcement Learning from Human Feedback):
  - Requires: human-ranked response pairs → train a reward model → PPO against reward
  - 3 models simultaneously in memory; unstable training (sensitive to KL coefficient)
  - Output: aligned model that maximises human preference reward
  - Purpose: align model outputs with human values

DPO (Direct Preference Optimisation):
  - Input: (prompt, winning_response, losing_response) preference pairs
  - Loss: -log σ(β * (log π_θ(y_w|x)/π_ref - log π_θ(y_l|x)/π_ref))
  - No reward model; 2 models (policy + frozen reference)
  - Stable training; standard pipeline: Pretrain → SFT → DPO
        """.strip(),
    ),
    
    # Week 5: Diffusion
    Question(
        category = "diffusion",
        question = "What is the forward process closed-form solution in DDPM and why is it important?",
        answer   = """
x_t = √ā_t · x_0 + √(1-ā_t) · ε,  where ε ~ N(0, I)

ā_t = Π_{s=1}^{t} (1 - β_s)  (cumulative product of noise schedule)

Importance:
1. Training efficiency: to generate a training sample at timestep t,
   you don't need to simulate t noising steps sequentially.
   One matrix multiply produces x_t directly from x_0 and a noise sample.
   Training complexity is O(1) per sample regardless of T.

2. Flexible supervision: you can train on any random t in [1, T] uniformly,
   not just step-by-step. This means the model sees all noise levels equally.

3. Tractable likelihood: the forward process q(x_t|x_0) is a Gaussian with
   known mean and variance — enabling the ELBO derivation.
        """.strip(),
    ),
    
    # Week 6: ML Systems
    Question(
        category = "systems",
        question = "What is train-serve skew and what causes it?",
        answer   = """
Train-serve skew: the model receives different input distributions at serving time
than during training, causing degraded predictions.

Common causes:
1. Different code paths: training uses Pandas/scikit-learn; serving uses Java/C++.
   Identical-looking transformations produce numerically different outputs.
2. Data type mismatches: float64 at training vs float32 at serving for numerics.
3. Null handling: different strategies for missing values at train vs serve time.
4. Feature ordering: feature vector columns appear in different order.
5. Temporal skew: features computed at training time use future data accidentally
   (leakage); serving sees only truly available past data.

Solution: feature stores (Feast, Tecton) enforce a single transformation code path
for both training feature generation and serving feature lookup.
        """.strip(),
    ),
    
    # Week 7: Multimodal
    Question(
        category = "multimodal",
        question = "How does CLIP enable zero-shot classification?",
        answer   = """
After CLIP training, text and image embeddings are aligned:
matching (image, text) pairs have high cosine similarity.

Zero-shot classification for N classes:
1. For each class c_i, create a text prompt: "a photo of a {c_i}"
2. Embed all N prompts → N text embeddings t_1, ..., t_N
3. Embed the query image → image embedding e
4. Compute cosine similarity: score_i = e · t_i
5. Predicted class = argmax(score_i) after softmax with temperature τ

No training examples needed — the model generalises from the semantic
alignment between vision and language learned during pretraining.
This works even for categories not seen during CLIP training, as long as
they can be described in natural language.
        """.strip(),
    ),
    
    # Week 8: Deployment
    Question(
        category = "deployment",
        question = "What is the difference between dynamic and static INT8 quantisation?",
        answer   = """
Dynamic Quantisation:
  - Weights: quantised to INT8 offline, stored as INT8
  - Activations: dequantised at runtime per-layer (no pre-calibration)
  - Pros: no calibration data required; easy to apply
  - Cons: runtime overhead from dynamic quantisation; slightly slower than static
  - Best for: Linear layers in BERT/GPT (activations rarely extreme)

Static Quantisation:
  - Weights AND activations: both quantised to INT8
  - Requires calibration: run representative data to determine per-layer activation ranges
  - Pros: faster inference (no runtime dequantisation); lower latency
  - Cons: requires calibration dataset; more setup
  - Best for: CNNs, hardware accelerators (TPU, Apple Neural Engine)

Both achieve ~4× model size reduction and ~2-4× CPU speedup with <1% accuracy loss.
        """.strip(),
    ),
    
    # Week 9: Agents
    Question(
        category = "agents",
        question = "What is the ReAct pattern and why does the 'Thought:' prefix improve performance?",
        answer   = """
ReAct (Reasoning + Acting) interleaves explicit reasoning traces with tool calls:
  Thought: [plan next step]
  Action: [tool_name]
  Action Input: [arguments]
  Observation: [tool result]
  → repeat until Answer

Why 'Thought:' improves performance:
1. Working memory: generated reasoning tokens are in context for the next token prediction.
   The model conditions action selection on the explicit reasoning trace.
2. Error correction: if the previous observation suggests an error, the Thought
   step allows the model to reason "I got the wrong result — I should try Y instead."
3. Interpretability: the reasoning trace is human-readable, enabling debugging.
4. Planning: multi-step tasks (search, then calculate, then verify) can be planned
   explicitly in Thought steps before executing.

Without Thought, the model must "plan" implicitly in unobservable internal states —
empirically less reliable for complex multi-step tasks.
        """.strip(),
    ),
    
    # Cross-cutting
    Question(
        category = "cross_cutting",
        question = "What is the scaling hypothesis and what evidence supports it?",
        answer   = """
The scaling hypothesis (Kaplan et al., 2020; Hoffmann et al., 2022) states that
model performance improves predictably as a power law with: model size (parameters),
training data (tokens), and compute (FLOPs).

L(N, D) ≈ A/N^α + B/D^β + C_irr

Evidence:
1. GPT-2 → GPT-3 → GPT-4: qualitative capability jumps at each scale
2. Emergent capabilities (arithmetic, chain-of-thought, in-context learning) appear
   at threshold scales and weren't present at smaller scales
3. Chinchilla optimal scaling: 70B model trained on 1.4T tokens outperforms 280B
   model trained on 300B tokens — compute-optimal scaling

Limitations:
- Some capabilities may require architectural innovations, not just scale
- Data quality matters — scaling on low-quality data has diminishing returns
- Inference cost scales with model size — not always practical to deploy largest models
        """.strip(),
    ),
    
    Question(
        category = "cross_cutting",
        question = "What is catastrophic forgetting and how is it mitigated in fine-tuning?",
        answer   = """
Catastrophic forgetting: when fine-tuning on task B, the model's weights shift and
performance on previously learned tasks (from pretraining or earlier fine-tuning)
degrades significantly.

Mitigation strategies:
1. LoRA: freezes base weights entirely — general knowledge in W₀ is preserved by
   construction. Task adaptation lives only in the adapter BA.
2. Low learning rate: a 10-100× lower LR than pretraining limits how far weights shift.
3. Elastic Weight Consolidation (EWC): adds a regularisation term penalising changes
   to weights that were important for previous tasks (measured via Fisher information).
4. Rehearsal / replay: mix a small fraction of previous-task data into the new training set.
5. Gradient projection: project gradients onto the null space of previous task gradients
   (Gradient Episodic Memory).

Practical recommendation: use LoRA for most fine-tuning tasks (forgetting prevention
by construction), with careful learning rate selection.
        """.strip(),
    ),
]

# ---------------------------------------------------------------------------
# 2. System Design Questions (5)
# ---------------------------------------------------------------------------

SYSTEM_DESIGN_QUESTIONS = [
    {
        "question": "Design a real-time GenAI content moderation system that reviews user posts before publication.",
        "requirements": ["< 200ms latency", "99.9% uptime", "Handle 50k posts/second", "Multi-language support"],
        "model_answer": """
ARCHITECTURE:
Serving: Multiple LLM moderation model instances (e.g., fine-tuned LLaMA-3-8B INT8)
behind a load balancer. GPU cluster with dynamic batching (collect requests for 10ms, run as batch).

Pipeline:
1. Tier 1 (< 5ms): Regex + blocked keyword filter catches obvious violations immediately.
2. Tier 2 (< 50ms): Fine-tuned small model (DistilBERT or RoBERTa) for toxicity/spam.
3. Tier 3 (< 200ms): LLM for nuanced content (borderline cases, hate speech, NSFW).
Only ~5% of posts reach Tier 3.

Multi-language: Language detection → route to language-specific model OR
multilingual model (mBERT/XLM-R fine-tuned on multilingual moderation data).

Reliability: 99.9% uptime = 8.7 hours downtime/year.
Blue-green deployment; automatic failover to Tier 1+2 if LLM unavailable.
Circuit breaker: if LLM latency > 150ms, route to Tier 2 only with "requires review" flag.

Monitoring: Per-tier accuracy, false positive rate (legitimate posts blocked),
false negative rate (harmful posts passed), latency per tier, daily calibration on labeled set.

Data: Continuously collect labeled (post, decision, ground_truth) pairs.
Weekly retraining on new data patterns. Human review queue for borderline cases → feedback loop.
    """,
    },
    {
        "question": "Design a RAG system for an enterprise knowledge base with 500k internal documents.",
        "requirements": ["< 3 second response time", "Source attribution required", "Role-based access control", "Daily document updates"],
        "model_answer": """
INDEXING PIPELINE (daily batch):
New/updated documents → chunking (512 tokens, 64-token overlap) → E5-large embeddings (GPU batch)
→ FAISS/Qdrant vector index + metadata store (doc_id, role_permissions, last_updated, source_url).
500k documents × avg 20 chunks = 10M vectors. Storage: ~10M × 768 × 4 bytes ≈ 30GB.

ACCESS CONTROL:
Each chunk stored with ACL (list of roles that can see it).
At query time: filter by user's roles BEFORE or AFTER retrieval.
Pre-filtering (recommended): add ACL as a Qdrant filter condition → only retrieve accessible chunks.
This prevents information leakage even if someone crafts adversarial queries.

RETRIEVAL & GENERATION:
Query → intent detection → hybrid search (BM25 + dense, RRF fusion) with ACL filter
→ Top 20 chunks → cross-encoder reranker → Top 5 chunks → LLM with citation prompt.
LLM prompt: "Answer ONLY from the provided sources. For each claim, cite [Source N]."

LATENCY BUDGET (3s total):
Embedding query: 50ms | Vector search: 100ms | Reranker: 200ms | LLM generation: 2000ms | Overhead: 650ms

DAILY UPDATES:
Incremental indexing: only embed changed documents. Use document hash to detect changes.
Tombstone deleted docs in vector DB (soft delete + filter). Hard delete weekly.

MONITORING: Answer faithfulness (sampled, LLM judge), citation accuracy, user satisfaction rating,
index freshness (time since last update per document), retrieval precision@5.
    """,
    },
    {
        "question": "Design a fine-tuning pipeline to continuously improve a customer service LLM from live conversation feedback.",
        "requirements": ["Run on-premise", "User feedback: thumbs up/down per conversation", "Must not degrade on existing capabilities", "Weekly update cycle"],
        "model_answer": """
DATA COLLECTION:
Log all conversations with metadata (user_id, timestamp, thumbs signal, rating).
Weekly: collect conversations from the past 7 days with explicit feedback.
Construct preference pairs: (prompt, good_response_A, poor_response_B) from:
  a) Same prompt with differing thumbs up/down signals
  b) Human reviewers label borderline cases

WEEKLY TRAINING PIPELINE:
1. Filter: Remove low-confidence pairs (both responses similar, no signal difference).
2. SFT warmup: Fine-tune on new thumbs-up conversations (anchor high-quality responses).
3. DPO alignment: Train on preference pairs. Reference model = previous week's model.
4. Evaluation: Run regression test suite (fixed 500 test cases from Week 1 baseline).
   If regression > 2% absolute on any category → abort deployment, investigate.

ON-PREMISE SETUP:
QLoRA on 4× A100 80GB. 7B base model in NF4 (~4GB) + LoRA adapters (BF16).
Training time: ~4 hours for 50k preference pairs. Infrastructure: SLURM or Ray cluster.
Model artifacts: MLflow. Rollback = redeploy previous week's model artifact.

ITERATIVE DPO:
Use the PREVIOUS aligned model as the reference (not the original SFT model).
This allows progressive alignment improvement without drifting from core capabilities.

DEPLOYMENT:
Blue-green: deploy new model to 10% traffic (Green), monitor for 48h.
If metrics pass: promote Green to 100%. Weekly release cadence aligned with sprint cycle.
    """,
    },
    {
        "question": "Design the ML backend for a multimodal product search (text + image input) for an e-commerce site.",
        "requirements": ["Support text query, image query, and text+image combined query", "10M products", "< 300ms p99", "Daily catalog updates"],
        "model_answer": """
MODALITY ENCODING:
- Text queries → E5-text-large (or CLIP text encoder): 768-dim embedding
- Image queries → CLIP ViT-L/14: 768-dim embedding
- Combined (text + image): weighted combination of text and image embeddings
  combined = α * text_emb + (1-α) * image_emb, where α is a learnable scalar
  (fine-tuned on combined query logs)

All product images + descriptions are pre-indexed as CLIP joint embeddings
→ queries and products live in the same embedding space.

PRODUCT INDEX:
10M products × ~3 images each = 30M image embeddings.
Store in Qdrant/Milvus with product metadata (category, price, stock).
HNSW index: approximate nearest neighbour in < 50ms for 30M vectors.
Hybrid: combine CLIP embedding search with BM25 on product title/description (RRF).

DAILY CATALOG UPDATES:
New products → real-time embedding pipeline (Kafka → embedding service → index upsert).
Product deactivations → soft delete. Hourly full consistency check.

RANKING:
After retrieval (top 100): cross-encoder (query, product) relevance reranker.
Business rules: boost sponsored, filter out-of-stock, category filters.

QUERY TYPE ROUTING:
Text-only: text encoder → CLIP text embedding
Image-only: image encoder → CLIP image embedding
Text+Image: encode both → weighted combination
Product similarity ("find similar"): fetch product's stored embedding directly

LATENCY BUDGET (300ms):
Encoding: 30ms (GPU) | ANN search: 50ms | BM25: 20ms | RRF fusion: 10ms | Reranker: 150ms | Overhead: 40ms
    """,
    },
    {
        "question": "Design an LLM-powered coding assistant that can answer questions about a proprietary codebase.",
        "requirements": ["Codebase: 2M lines across 50k files", "Answers require understanding code context", "Must cite specific file + line numbers", "On-premise, no external API calls"],
        "model_answer": """
CODEBASE INDEXING:
Parse with tree-sitter → extract: functions, classes, methods, docstrings, comments.
Chunking strategy: function-level chunks (not arbitrary character splits).
Augment each chunk with: file path, function name, class hierarchy, called-by graph.
Index: CodeBERT or UniXcoder embeddings (code-specific models) → FAISS flat index.
2M lines ÷ avg 30 lines/function ≈ 65k function-level chunks. Feasible.

HYBRID RETRIEVAL:
BM25 (keyword): excels at finding exact function names, variable names, error codes.
Dense (CodeBERT): excels at semantic search ("where is the authentication logic?").
Hybrid (RRF): robust to both query types.
Metadata filters: file_path, language, last_modified, author.

GENERATION:
On-premise LLM: CodeLlama-34B (fine-tuned on internal code + QA pairs) with QLoRA.
Context: retrieved code chunks (with file:line attribution) + system prompt.
Prompt: "Answer using ONLY the provided code. For each claim cite [file:lines].
         If the information isn't in the provided code, say so."

CITATION FORMAT:
Structured output: JSON with {"answer": str, "citations": [{"file": str, "lines": [int, int], "snippet": str}]}
UI renders file:line as clickable links to internal code browser.

CODEBASE UPDATES:
Git hook → diff files on each commit → re-embed only changed functions.
Full re-index weekly (catches moved code, renamed functions).

SECURITY:
Air-gapped deployment. All models run on-premise.
Audit log: every query and response stored. Regular red-team sessions for prompt injection.
    """,
    },
]

# ---------------------------------------------------------------------------
# 3. Gotcha Questions (10)
# ---------------------------------------------------------------------------

GOTCHA_QUESTIONS = [
    {
        "question": "The Transformer uses Layer Norm. Does it matter whether Layer Norm is applied before or after the sub-layer (Pre-LN vs Post-LN)?",
        "gotcha": "Most people say 'it's just a normalisation detail.' But Pre-LN (normalise before attention/FFN) is dramatically more training stable — gradients at the beginning of training are well-conditioned. Post-LN (original Transformer) requires careful learning rate warmup to avoid early divergence. Modern LLMs (LLaMA, Mistral, GPT-4) all use Pre-LN (RMSNorm variant). Post-LN is essentially deprecated for large-scale training.",
    },
    {
        "question": "Does adding more retrieved chunks in RAG always improve answer quality?",
        "gotcha": "No — the 'lost-in-the-middle' effect. LLMs perform best on information at the start and end of the context window, and worse on information in the middle. Adding too many chunks (top-20, top-50) causes important information to get buried in the middle. Additionally, irrelevant chunks add noise and can distract the model. Optimal top-k is usually 3-7 for most tasks — retrieve more, then rerank to select the best.",
    },
    {
        "question": "LoRA is applied to attention Q and V matrices. Would it work just as well applied only to Q?",
        "gotcha": "No. Ablation studies in the original LoRA paper show that adapting both Q and V is significantly better than Q alone. V contains the actual 'values' that get attended to — adapting only Q changes what gets attended to but not what information is retrieved. Adapting both gives the adapter the ability to modify both the attention pattern (Q, K) and the information being aggregated (V). Some work shows K is less important than V; most production LoRA applies to Q and V at minimum.",
    },
    {
        "question": "Can you improve RAG accuracy by switching from FAISS to a more sophisticated vector database?",
        "gotcha": "The vector database choice has negligible impact on retrieval quality. FAISS (exact search) and any ANN index (HNSW, IVF) give identical or near-identical results. The factors that actually drive retrieval quality are: embedding model quality, chunk size + overlap strategy, hybrid vs dense retrieval, and reranking. Switching from FAISS to Pinecone changes serving cost and scale — not quality. Engineers often focus on the wrong lever.",
    },
    {
        "question": "DPO doesn't need a reward model, so it can't reward hack. Is that correct?",
        "gotcha": "No. DPO's implicit reward is β * log(π_θ/π_ref). A DPO-trained model can over-optimise this implicit reward, increasing log probability of chosen responses while decreasing probability of rejected responses to degenerate extremes. The reference model KL constraint prevents the worst cases, but reward hacking in DPO is documented — the model can learn to produce high-scoring but stilted, unnatural text. β tuning and early stopping matter.",
    },
    {
        "question": "Is DDPM's reverse process deterministic?",
        "gotcha": "No — DDPM's reverse process adds noise at each step: x_{t-1} = mean(x_t) + σ_t * z, where z ~ N(0,I). This is stochastic — different runs from the same x_T produce different images. DDIM (Denoising Diffusion Implicit Models) removes this stochasticity by setting η=0, producing a deterministic reverse ODE. DDIM determinism is useful for: reproducible generation (same seed = same image), latent space interpolation, and image editing (invert to latent, modify, denoise).",
    },
    {
        "question": "A model with 99.5% accuracy on a test set should be deployed confidently. Agree or disagree?",
        "gotcha": "Disagree — accuracy is meaningless without knowing the class distribution. A fraud detection model predicting 'not fraud' for everything achieves 99.9% accuracy if fraud rate is 0.1%. The test set may not represent the live distribution (covariate shift). The test set may have been contaminated with training data. Confidence should be based on: precision/recall/F1/AUC on a carefully held-out test set, A/B testing against the current production model, and monitoring on live traffic after deployment.",
    },
    {
        "question": "Context window size is the key bottleneck for LLM serving. True or false?",
        "gotcha": "Partially false. Context window size affects KV cache memory, not the model weights. For typical API use (< 8K tokens), the bottleneck is memory bandwidth (reading model weights and KV cache) and compute (attention + FFN). For very long contexts (> 32K tokens), KV cache memory becomes the bottleneck — it can exceed model weight memory. Serving throughput is usually bottlenecked by GPU memory bandwidth, not compute (FLOP/s) — models are often memory-bound, not compute-bound.",
    },
    {
        "question": "Can you fine-tune a model on synthetic data generated by a larger model (student-teacher / distillation)?",
        "gotcha": "Yes, and this is widely used (Alpaca, Vicuna, Orca, etc. use GPT-4-generated data for fine-tuning). However: (1) The fine-tuned model is bounded by the quality of the teacher — it can't exceed what the teacher knows. (2) Synthetic data can introduce biases and errors from the teacher model. (3) Some license agreements (OpenAI ToS) explicitly prohibit using GPT-4 outputs to train competing models — legal compliance matters. (4) The student model needs to be large enough to absorb the teacher's behaviour.",
    },
    {
        "question": "Multi-head attention runs h separate attention operations — doesn't that make it h times slower?",
        "gotcha": "No — multi-head attention has the SAME compute as single-head with the same total d_model. Each head uses dimension d_model/h (not d_model). The total Q,K,V projection FLOPs are identical: h heads × (d_model × d_model/h) = d_model². Attention itself is O(n² × d_model/h × h) = O(n² × d_model) — same as single-head. The 'h separate operations' can be executed in parallel as batched matrix multiplies. Multi-head doesn't cost more — it provides more expressivity for free.",
    },
]


# ---------------------------------------------------------------------------
# 4. Interactive Mock Interview Session
# ---------------------------------------------------------------------------

def print_separator(char="─", width=65):
    print(char * width)

def run_conceptual_quiz(questions: List[Question], shuffle: bool = False, category: str = None):
    """Run an interactive conceptual quiz."""
    q_list = [q for q in questions if (category is None or q.category == category)]
    if shuffle:
        random.shuffle(q_list)
    
    print_separator("=")
    print(f"CONCEPTUAL QUIZ — {len(q_list)} questions")
    print_separator("=")
    
    for i, q in enumerate(q_list):
        print(f"\nQ{i+1}/{len(q_list)} [{q.category.upper()}]:")
        print(f"  {q.question}")
        if q.hint:
            print(f"\n  💡 Hint: {q.hint}")
        input("\n  [Press Enter to see answer...]")
        print(f"\n  ANSWER:")
        for line in q.answer.split('\n'):
            print(f"  {line}")
        print_separator()

def run_gotcha_quiz(questions: list, shuffle: bool = False):
    """Run the gotcha questions."""
    q_list = list(questions)
    if shuffle:
        random.shuffle(q_list)
    
    print_separator("=")
    print(f"GOTCHA QUESTIONS — {len(q_list)} questions")
    print_separator("=")
    
    for i, q in enumerate(q_list):
        print(f"\nGotcha Q{i+1}/{len(q_list)}:")
        print(f"  {q['question']}")
        input("\n  [Press Enter to see the gotcha...]")
        print(f"\n  ⚠️  The Gotcha:")
        for line in q['gotcha'].split('. '):
            print(f"  {line.strip()}.")
        print_separator()

def run_system_design_quiz(questions: list):
    """Run system design questions."""
    print_separator("=")
    print(f"SYSTEM DESIGN — {len(questions)} questions (20 min each)")
    print_separator("=")
    
    for i, q in enumerate(questions):
        print(f"\nDesign Q{i+1}/{len(questions)}:")
        print(f"  {q['question']}")
        print(f"\n  Requirements:")
        for req in q['requirements']:
            print(f"    - {req}")
        input("\n  [Sketch your solution, then press Enter for model answer...]")
        print(f"\n  MODEL ANSWER:")
        for line in q['model_answer'].strip().split('\n'):
            print(f"  {line}")
        print_separator()


if __name__ == "__main__":
    args = sys.argv[1:]
    shuffle = '--random' in args
    category = None
    for arg in args:
        if arg.startswith('--category='):
            category = arg.split('=')[1]
    
    print_separator("=", 65)
    print("  WEEK 10: MOCK INTERVIEW — 10-WEEK GENAI MASTERY ROADMAP")
    print_separator("=", 65)
    print("\nAvailable modes:")
    print("  1. Conceptual questions (20 total)")
    print("  2. System design questions (5 total)")
    print("  3. Gotcha questions (10 total)")
    print("  4. Quick summary (no interaction)")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == "1":
        run_conceptual_quiz(CONCEPTUAL_QUESTIONS, shuffle=shuffle, category=category)
    elif choice == "2":
        run_system_design_quiz(SYSTEM_DESIGN_QUESTIONS)
    elif choice == "3":
        run_gotcha_quiz(GOTCHA_QUESTIONS, shuffle=shuffle)
    elif choice == "4":
        print("\n" + "="*55)
        print("10-WEEK GENAI MASTERY: QUICK SUMMARY")
        print("="*55)
        
        summaries = [
            ("Week 1", "Transformer", "Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V"),
            ("Week 2", "LLMs & GPT", "Causal masking + KV cache = generative LLMs"),
            ("Week 3", "RAG", "Retrieval quality → generation quality"),
            ("Week 4", "Fine-Tuning", "LoRA: ΔW ≈ BA (rank r << d) + DPO alignment"),
            ("Week 5", "Diffusion", "x_t = √ā_t·x_0 + √(1-ā_t)·ε (closed form!)"),
            ("Week 6", "ML Systems", "Monitor PSI; feature stores prevent train-serve skew"),
            ("Week 7", "Multimodal", "CLIP: zero-shot via aligned image-text embeddings"),
            ("Week 8", "Deployment", "INT8: 4× smaller, 2-4× faster, <1% accuracy loss"),
            ("Week 9", "Agents", "ReAct: Thought→Action→Observation, iteratively"),
            ("Week 10", "Review", "You are HERE. Go build things."),
        ]
        
        for week, topic, insight in summaries:
            print(f"\n  {week}: {topic}")
            print(f"    → {insight}")
        
        print(f"\n  Total questions available: {len(CONCEPTUAL_QUESTIONS)} conceptual + {len(GOTCHA_QUESTIONS)} gotchas + {len(SYSTEM_DESIGN_QUESTIONS)} design")
        print(f"\n  Good luck. 🚀")
    else:
        print("Invalid choice. Run again with mode 1-4.")
