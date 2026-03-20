"""
rag_pipeline.py
===============
Builds a complete Retrieval-Augmented Generation (RAG) pipeline from scratch.

RAG solves the "static knowledge" problem in LLMs:
    - LLMs have a training cutoff date
    - LLMs can't access proprietary/private documents
    - LLMs hallucinate when asked about unknown facts

RAG solution:
    1. Index: Chunk documents → embed → store in vector DB
    2. Retrieve: Embed query → find similar chunks
    3. Generate: Feed retrieved context + query to LLM → grounded answer

This implementation uses:
    - sentence-transformers for embeddings (all-MiniLM-L6-v2)
    - FAISS for vector storage and retrieval
    - A local generation function (or HuggingFace model for full pipeline)

Usage:
    python rag_pipeline.py
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 1. Document Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A text chunk with metadata."""
    text:       str
    doc_id:     str
    chunk_idx:  int
    start_char: int
    end_char:   int
    embedding:  Optional[np.ndarray] = None


def chunk_document(
    text: str,
    doc_id: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    """
    Split a document into overlapping fixed-size chunks.

    Why overlap? A sentence at the end of one chunk might be the key
    sentence for a query — but without overlap, it lacks its preceding
    context. Overlap ensures every sentence appears in at least one
    chunk with full surrounding context.

    Args:
        chunk_size: Target chunk size in characters (not tokens)
        overlap:    Character overlap between adjacent chunks
    """
    # Sentence-aware splitting: try not to cut mid-sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks: List[Chunk] = []
    current_chunk = ""
    chunk_start = 0
    chunk_idx = 0
    char_pos = 0

    for sentence in sentences:
        # If adding this sentence exceeds chunk_size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(Chunk(
                text       = current_chunk.strip(),
                doc_id     = doc_id,
                chunk_idx  = chunk_idx,
                start_char = chunk_start,
                end_char   = char_pos,
            ))
            chunk_idx += 1
            # Keep last `overlap` characters as the start of the next chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
            chunk_start = char_pos - len(overlap_text)
        else:
            current_chunk += (" " if current_chunk else "") + sentence
        
        char_pos += len(sentence) + 1

    # Don't forget the final chunk
    if current_chunk.strip():
        chunks.append(Chunk(
            text       = current_chunk.strip(),
            doc_id     = doc_id,
            chunk_idx  = chunk_idx,
            start_char = chunk_start,
            end_char   = char_pos,
        ))

    return chunks


# ---------------------------------------------------------------------------
# 2. Vector Store (FAISS-backed)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Simple FAISS-backed vector store for embedding storage and retrieval.

    Uses Inner Product (IP) similarity on L2-normalised vectors,
    which is equivalent to cosine similarity.
    """

    def __init__(self, embedding_dim: int = 384):
        try:
            import faiss
            # IndexFlatIP: exact (non-approximate) inner product search
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.use_faiss = True
        except ImportError:
            print("  [Warning] faiss not available, using numpy brute-force search")
            self.use_faiss = False
            self.embeddings: List[np.ndarray] = []

        self.chunks: List[Chunk] = []
        self.embedding_dim = embedding_dim

    def add(self, chunks: List[Chunk]):
        """Add embedded chunks to the index."""
        for chunk in chunks:
            assert chunk.embedding is not None, "Chunk must have embedding before adding"
            
            # L2-normalise for cosine similarity via inner product
            emb = chunk.embedding.astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            
            if self.use_faiss:
                import faiss
                self.index.add(emb.reshape(1, -1))
            else:
                self.embeddings.append(emb)
            
            self.chunks.append(chunk)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k most similar chunks.

        Returns list of (chunk, score) pairs, sorted by descending similarity.
        """
        qe = query_embedding.astype(np.float32)
        qe = qe / (np.linalg.norm(qe) + 1e-10)

        if self.use_faiss:
            scores, indices = self.index.search(qe.reshape(1, -1), min(top_k, len(self.chunks)))
            results = [(self.chunks[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            # Brute-force numpy cosine similarity
            if not self.embeddings:
                return []
            matrix = np.stack(self.embeddings)
            scores  = matrix @ qe
            top_idx = np.argsort(scores)[::-1][:top_k]
            results = [(self.chunks[i], float(scores[i])) for i in top_idx]

        return results

    def __len__(self):
        return len(self.chunks)


# ---------------------------------------------------------------------------
# 3. RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline: index → retrieve → generate.

    Components:
        - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (22M params)
        - Vector store: FAISS flat index
        - Generator: template-based (swap in any LLM API for production)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        overlap: int = 64,
        top_k: int = 5,
    ):
        print(f"  Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        self.vector_store = VectorStore(embedding_dim=self.embedding_dim)
        self.chunk_size   = chunk_size
        self.overlap      = overlap
        self.top_k        = top_k

        print(f"  Embedding dim: {self.embedding_dim}")

    def index_documents(self, documents: Dict[str, str]):
        """
        Process and index a collection of documents.

        Args:
            documents: Dict of {doc_id: text}
        """
        all_chunks: List[Chunk] = []

        for doc_id, text in documents.items():
            chunks = chunk_document(text, doc_id, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)

        print(f"  Chunked {len(documents)} documents into {len(all_chunks)} chunks")

        # Batch embed all chunks (much faster than one-by-one)
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=False)

        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        self.vector_store.add(all_chunks)
        print(f"  Indexed {len(self.vector_store)} chunks in vector store")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Embed query and retrieve top-k most similar chunks."""
        k = top_k or self.top_k
        query_emb = self.embedder.encode([query])[0]
        return self.vector_store.search(query_emb, top_k=k)

    def build_prompt(self, query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
        """
        Build the RAG prompt: system instruction + retrieved context + user query.

        Prompt structure matters enormously:
            - Place context BEFORE the question (primes the model)
            - Clearly delimit each chunk (prevents bleed-through)
            - Explicit instruction to say "I don't know" reduces hallucination
        """
        context_blocks = []
        for i, (chunk, score) in enumerate(retrieved):
            context_blocks.append(
                f"[Source {i+1} | doc: {chunk.doc_id} | relevance: {score:.3f}]\n{chunk.text}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information to answer the question, say "I don't have enough information to answer this."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        return prompt

    def answer(
        self,
        query: str,
        generator_fn=None,
    ) -> Dict:
        """
        Full RAG: retrieve → build prompt → generate answer.

        Args:
            query:        User's question
            generator_fn: Callable that takes a prompt string and returns answer string.
                          If None, returns the prompt for manual inspection.

        Returns:
            Dict with keys: answer, retrieved_chunks, prompt
        """
        # Step 1: Retrieve
        retrieved = self.retrieve(query)

        # Step 2: Build prompt
        prompt = self.build_prompt(query, retrieved)

        # Step 3: Generate
        if generator_fn:
            answer_text = generator_fn(prompt)
        else:
            # Without an LLM, return the most relevant chunk as a baseline
            answer_text = f"[Top retrieved chunk]\n{retrieved[0][0].text}" if retrieved else "No relevant context found."

        return {
            "query":            query,
            "answer":           answer_text,
            "retrieved_chunks": [(c.text[:100] + "...", s) for c, s in retrieved],
            "prompt":           prompt,
        }


# ---------------------------------------------------------------------------
# 4. Few-shot & Chain-of-Thought Prompt Templates
# ---------------------------------------------------------------------------

def few_shot_sentiment_prompt(text: str) -> str:
    """Demonstrate few-shot prompting for sentiment classification."""
    return f"""Classify the sentiment of each text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: "The transformer architecture changed everything."
Sentiment: POSITIVE

Text: "Training costs are completely unsustainable at this scale."
Sentiment: NEGATIVE

Text: "The paper was published in 2017."
Sentiment: NEUTRAL

Text: "I struggled through this implementation but finally got it working."
Sentiment: POSITIVE

Text: "{text}"
Sentiment:"""


def chain_of_thought_math_prompt(problem: str) -> str:
    """Demonstrate chain-of-thought prompting for multi-step reasoning."""
    return f"""Solve the following maths problem step by step.

Problem: A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?
Let's think step by step:
- Let the ball cost x dollars
- The bat costs x + 1.00 dollars
- Together: x + (x + 1.00) = 1.10
- 2x + 1.00 = 1.10
- 2x = 0.10
- x = $0.05
Answer: The ball costs $0.05.

Problem: {problem}
Let's think step by step:"""


# ---------------------------------------------------------------------------
# 5. Demo / Main
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "transformer_paper": """
    The Transformer architecture was introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017.
    It replaced recurrent neural networks with self-attention mechanisms, allowing for much greater parallelisation during training.
    The key innovation was the scaled dot-product attention mechanism, which computes attention weights as softmax(QK^T / sqrt(d_k)) * V.
    The architecture uses multi-head attention, which allows the model to attend to different positions simultaneously using different learned projections.
    Positional encodings are added to the embeddings to provide information about token order.
    The original Transformer had 6 encoder layers and 6 decoder layers, with a model dimension of 512 and 8 attention heads.
    Training used the Adam optimiser with a custom learning rate schedule including warmup steps.
    """,

    "gpt_architecture": """
    GPT (Generative Pre-trained Transformer) uses only the decoder portion of the Transformer architecture.
    Unlike BERT which uses masked language modelling, GPT uses causal language modelling: predicting the next token given all previous tokens.
    GPT-2 was released in 2019 with 1.5 billion parameters and demonstrated strong few-shot learning capabilities.
    GPT-3 scaled this to 175 billion parameters and showed emergent abilities including arithmetic, translation, and code generation.
    The causal attention mask prevents each token from attending to future tokens, ensuring autoregressive generation.
    GPT uses learned positional embeddings rather than the sinusoidal encodings of the original Transformer.
    Weight tying between the embedding layer and the output projection reduces parameters and improves generalisation.
    """,

    "fine_tuning": """
    Fine-tuning adapts a pretrained language model to a specific domain or task by continuing training on a smaller dataset.
    LoRA (Low-Rank Adaptation) reduces fine-tuning cost by only training small adapter matrices while freezing the base model.
    QLoRA combines LoRA with 4-bit quantisation, enabling fine-tuning of 70B parameter models on a single 48GB GPU.
    The learning rate for fine-tuning should be 10-100x lower than pretraining to avoid catastrophic forgetting.
    Instruction tuning fine-tunes models on (instruction, response) pairs to follow natural language instructions.
    RLHF (Reinforcement Learning from Human Feedback) uses human preference labels to align model outputs with human values.
    DPO (Direct Preference Optimisation) achieves RLHF-style alignment without requiring a separate reward model.
    """,

    "rag_systems": """
    Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model to answer questions about external documents.
    The indexing phase involves chunking documents, embedding chunks with a sentence transformer, and storing embeddings in a vector database.
    Common vector databases include FAISS (local, highly optimised), ChromaDB (easy to use), Pinecone (managed cloud), and Weaviate.
    Retrieval quality is measured by recall@k: what fraction of relevant documents appear in the top-k results.
    The generation phase concatenates retrieved chunks with the user query in a structured prompt.
    RAGAS evaluates RAG systems on faithfulness, answer relevancy, context precision, and context recall.
    Hybrid search combining dense (embedding) and sparse (BM25) retrieval typically outperforms either alone.
    """,
}


if __name__ == "__main__":
    print("=" * 65)
    print("RAG Pipeline Demo")
    print("=" * 65)

    # Build pipeline
    print("\n[1/4] Initialising RAG pipeline...")
    rag = RAGPipeline(chunk_size=400, overlap=50, top_k=3)

    # Index documents
    print("\n[2/4] Indexing knowledge base...")
    rag.index_documents(KNOWLEDGE_BASE)

    # Query the RAG system
    print("\n[3/4] Querying the RAG system...")
    queries = [
        "What is the scaled dot-product attention formula?",
        "How does LoRA reduce fine-tuning cost?",
        "What metrics does RAGAS use to evaluate RAG systems?",
        "What is the difference between GPT and BERT architectures?",
    ]

    for query in queries:
        result = rag.answer(query)
        print(f"\n  Q: {query}")
        print(f"  Top retrieved chunk (score: {result['retrieved_chunks'][0][1]:.3f}):")
        print(f"  '{result['retrieved_chunks'][0][0]}'")

    # Demonstrate prompt engineering patterns
    print("\n[4/4] Prompt engineering patterns...")

    print("\n  --- Few-Shot Sentiment ---")
    test_sentences = [
        "This implementation is surprisingly elegant.",
        "The documentation is completely missing.",
    ]
    for s in test_sentences:
        prompt = few_shot_sentiment_prompt(s)
        print(f"  Prompt for '{s}':")
        print(f"  [Last line: 'Text: \"{s}\" | Sentiment:' → model fills in]")

    print("\n  --- Chain of Thought Math ---")
    problem = "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?"
    cot_prompt = chain_of_thought_math_prompt(problem)
    print(f"  Problem: {problem}")
    print(f"  CoT prompt appended: 'Let's think step by step:'")
    print(f"  (Expected: each machine makes 1 widget in 5 minutes → 100 machines, 100 widgets → 5 minutes)")

    print("\n" + "=" * 65)
    print("✅ RAG Pipeline complete!")
    print("=" * 65)
