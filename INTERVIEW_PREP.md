# Interview Prep — GenAI Projects Portfolio

Master these questions and you can discuss any of the 6 projects confidently in any ML/AI engineering interview.

---

## 🔷 Project 1: Enterprise RAG

**Q: Walk me through your RAG architecture.**

"The pipeline has three stages. Ingestion: documents are loaded (PDF/DOCX/HTML), chunked into 512-character overlapping segments using a recursive sentence-aware splitter, and embedded using BAAI/bge-base-en-v1.5 — a top-performing open-source model. Retrieval: I implemented hybrid search combining BM25 (keyword matching) and dense embeddings, fused via Reciprocal Rank Fusion — this improved recall@5 by ~23% over dense-only. A cross-encoder reranker then re-scores the top 20 candidates to select the best 5. Generation: I built a citation prompt that forces the LLM to attribute every claim to a [Source N], with an explicit instruction to say 'I don't know' if context is insufficient."

**Q: What is Reciprocal Rank Fusion and why did you use it over score averaging?**

"RRF scores each document as Σ 1/(k + rank_i) across all result lists, where k=60. The key insight is that it uses only *ranks*, not raw scores. BM25 scores can be 0–50+, while cosine similarity is always 0–1 — averaging these directly would let BM25 dominate. RRF is scale-invariant, robust, and empirically outperforms score averaging across query types."

**Q: How does your hallucination detection work?**

"I split the generated answer into individual claims (sentences). For each claim, I compute cosine similarity against all retrieved context chunks. Claims with max-similarity below 0.45 are flagged as potentially hallucinated. The overall faithfulness score is the mean similarity across all claims. This runs in ~10ms and requires no API call — it's pure embedding arithmetic. For higher accuracy I have an LLM-based NLI verifier that asks the model to label each claim as SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED — this runs asynchronously for monitoring, not in the hot path."

**Q: What's the 'lost in the middle' problem and how did you handle it?**

"Research shows LLMs perform worse on information placed in the middle of a long context — performance follows a U-shape (best at start and end). In RAG, this means if I return 10 chunks, the 5th chunk might be ignored. I handle this by using a reranker to aggressively cut to the top 5 most relevant chunks, placing the highest-scoring chunk first in the prompt, and limiting total context to ~2000 tokens."

---

## 🔷 Project 2: Intelligent Document Processing

**Q: How do you handle per-field confidence scoring?**

"The extraction prompt explicitly instructs the LLM to return each field as {value, confidence, raw_text}. Confidence is 0–1: 1.0 for explicitly stated values, 0.7 for clearly implied values, 0.4 for uncertain, 0.0 for missing. A ConfidenceAnalyzer then checks: if more than 30% of fields are below 0.50 threshold, the document is routed to the human review queue. This gives business users a clear signal about which documents need human attention."

**Q: Why not just use AWS Textract or Azure Form Recogniser?**

"Managed services charge $0.05–$0.20 per page and require you to define fixed field templates per document type. My LLM-based approach costs ~$0.002 per document (GPT-4o-mini API cost), handles any document layout without templates, and produces richer output including rationale and raw text attribution. The trade-off: it's slower (~2s vs ~200ms for Textract) and requires an LLM API, making it less suitable for high-volume batch processing of perfectly structured forms."

**Q: How do you handle malformed JSON from the LLM?**

"Three-stage repair: first, strip markdown code fences and attempt direct json.loads(). If that fails, use regex to extract the outermost JSON object. If that fails, apply common repairs — remove trailing commas, fix single quotes — and retry. If all three fail, return an empty dict and the document is flagged for review. I also use response_format={'type': 'json_object'} with OpenAI to reduce malformed JSON rate significantly."

---

## 🔷 Project 3: AI Code Review Agent

**Q: What security vulnerabilities does your agent catch?**

"SQL injection (f-string or string concatenation with user input), hardcoded secrets/passwords/API keys (regex patterns for common formats like sk-, ghp_, password=), path traversal (os.path.join with user input without sanitization), insecure authentication (MD5 password hashing, no rate limiting on login endpoints), XSS in template rendering, and use of eval() or exec() with user-controlled input."

**Q: How does the GitHub webhook integration work?**

"GitHub sends an HTTP POST to my /webhook/github endpoint on PR opened/synchronize events. I verify the payload signature using HMAC-SHA256 with a shared webhook secret — this prevents forged webhook calls. The actual review runs as a FastAPI BackgroundTask so the webhook returns 200 immediately (GitHub requires a fast response). The background task fetches PR file diffs via PyGithub, runs the review per file, and posts results back as a GitHub PR review with inline comments."

**Q: How do you prevent hallucinations in code review?**

"I use structured JSON output with a well-defined schema — the model must return a list of {file, line, severity, category, message, suggestion} objects. This constrains the output format. I also instruct the model to only cite specific line numbers from the provided diff, and I validate that cited line numbers exist in the diff before posting. If a comment references line 150 but the diff only has 80 lines, I adjust or skip that comment."

---

## 🔷 Project 4: Text-to-SQL Agent

**Q: Explain your self-correction loop.**

"The engine generates SQL from the question and schema. If execution raises a sqlite3.Error, I feed the error message back to the LLM with the original question, the failed SQL, and the schema — and ask it to fix the query. This repeats up to 3 times. Common errors it fixes: wrong column names (misspelled), missing JOIN conditions, incorrect aggregate syntax, and dialect differences. In testing, this reduced the failure rate from ~25% to under 3% for queries on the sample database."

**Q: How does your safety layer work?**

"Before executing any generated SQL, I run it through is_safe_sql() which checks for BLOCKED_PATTERNS — regex matches for DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, UPDATE, GRANT, REVOKE — case-insensitively. If any match, the query is rejected with a clear error message and the LLM is not given a chance to retry with a destructive query. This means even if the LLM generates 'delete from users' in lowercase, it's caught."

**Q: How would you extend this to PostgreSQL or MySQL?**

"The Text2SQLEngine takes a db_path and dialect parameter. For PostgreSQL, I'd replace sqlite3 with psycopg2 or asyncpg, adjust the schema extraction query from PRAGMA table_info() to information_schema.columns, and update the prompt to mention the dialect. The LLM naturally generates dialect-appropriate SQL when you specify 'PostgreSQL' in the schema prompt. Connection pooling would be needed for production to avoid connection overhead per query."

---

## 🔷 Project 5: Meeting Intelligence Platform

**Q: How do you extract action items accurately?**

"The key is the schema-driven prompt. I define exactly what an action item is: 'a specific task assigned to a named person with an implied or explicit deadline'. The LLM returns structured JSON with description, owner, due_date, and priority fields. The hardest part is owner attribution — meeting transcripts often have implicit ownership ('James, can you...') that requires understanding the conversational context. GPT-4o-mini handles this well because it has strong conversational reasoning."

**Q: How would you scale this to process 1000 meetings/day?**

"Three changes: (1) Async processing — use Celery + Redis to queue transcript analysis jobs, return a job ID immediately, poll for results. (2) Caching — if the same transcript is submitted twice (e.g., re-analysis), cache the result by transcript hash. (3) Batch embedding — if using embeddings for search, batch-embed all new meetings nightly rather than incrementally. At 1000 meetings/day with ~$0.001 per analysis (GPT-4o-mini), that's ~$1/day in API costs — very manageable."

**Q: How does semantic search work across meetings?**

"Each meeting's summary, action items, and decisions are concatenated into a single text string and embedded using sentence-transformers. When a user searches, the query is embedded and cosine similarity is computed against all meeting embeddings. The top-k most similar meetings are returned. For production with thousands of meetings, I'd use FAISS for fast approximate nearest-neighbour search instead of brute-force cosine similarity."

---

## 🔷 Project 6: LLM Safety & Guardrails

**Q: What injection patterns does your system catch?**

"13 patterns including: 'ignore [any words] instructions', 'disregard [any words] instructions', 'you are now', 'pretend you are', 'jailbreak', 'do anything now', 'dan mode', 'override instructions', 'bypass safety/filter/guard'. The patterns use regex with flexible matching — e.g., 'ignore.{0,20}instructions' catches 'ignore all previous instructions', 'ignore your prior instructions', etc. I also watch for system prompt extraction attempts."

**Q: How do you handle PII detection without false positives?**

"I use targeted regex patterns per PII type: emails (RFC 5322 pattern), US phone numbers (multiple formats with/without country code), SSNs (###-##-#### format), credit cards (4 groups of 4 digits with optional separators), IP addresses, and API key prefixes (sk-, ghp_, eyJ for JWTs). The patterns are specific enough to avoid most false positives — for example, '123-456-7890' matches the phone pattern but '12-3456-789' doesn't. For production, I'd add a secondary NER-based classifier for names, addresses, and dates."

**Q: Why did you build this yourself instead of using NeMo Guardrails or Guardrails AI?**

"For the portfolio project, building from scratch demonstrates understanding of the underlying mechanics. In production, I'd evaluate managed solutions. NeMo Guardrails is excellent but requires NVIDIA infrastructure. Guardrails AI has good coverage but adds ~100ms latency. A custom solution lets you tune false positive/negative rates for your specific use case and adds zero additional vendor dependency. I'd benchmark all three on your actual data distribution before committing."

---

## 🔷 Cross-Project System Design Questions

**Q: How would you monitor all these systems in production?**

"Three layers: (1) Infrastructure metrics — Prometheus + Grafana for latency (p50/p95/p99), error rates, queue depth, GPU utilisation. (2) ML-specific metrics — RAGAS scores sampled at 5% of production traffic, hallucination rate trends, IDP field confidence distributions, SQL failure rate before/after self-correction. (3) Business metrics — for RAG: % of queries with answers vs 'I don't know', user satisfaction thumbs up/down; for IDP: % auto-processed vs review queue rate; for code review: % of suggestions accepted by developers. Alert thresholds: p99 latency > 3s, error rate > 1%, hallucination rate > 15%."

**Q: How would you handle a model API outage?**

"Graceful degradation with circuit breaker pattern. Each project has a tenacity @retry decorator with exponential backoff (up to 3 attempts). If all retries fail, the circuit breaker opens and requests are routed to a fallback: for RAG — return top retrieval results without LLM generation; for IDP — flag document for human review; for code review — return 'review pending' status; for text-to-SQL — return error with suggested query templates. Use Redis to store circuit breaker state shared across multiple API instances."

**Q: What would you change if you had to handle 10x more load?**

"(1) Async LLM calls with streaming — use async FastAPI routes + httpx AsyncClient for non-blocking LLM calls. (2) Request batching — collect requests within a 100ms window and batch-embed together (Whisper, embedding models all support batching). (3) Caching — Redis cache for repeated queries with the same context. (4) Model quantisation — switch to INT8 quantised local models (llama.cpp) for latency-sensitive paths to eliminate API cost and latency. (5) Horizontal scaling — stateless FastAPI instances behind a load balancer, with shared Redis for state and S3 for document storage."

---

## 🔷 Behavioural Questions (STAR format)

**"Tell me about a technical challenge you faced building these projects."**

"Building the IDP system, I discovered that LLMs frequently returned malformed JSON for complex documents — sometimes nesting fields incorrectly, sometimes using single quotes instead of double quotes, sometimes adding trailing commas. My initial approach of just calling json.loads() failed ~15% of the time. I implemented a three-stage JSON repair pipeline: strip markdown fences, extract the outermost JSON object with regex, then apply common fix patterns. This brought the parse failure rate from 15% to under 1%. The remaining 1% are flagged for human review — which is actually the right outcome for documents the model genuinely struggled with."

**"How did you decide what to build and prioritise?"**

"I researched the most common ML engineering job postings and identified recurring themes: RAG systems, document processing, LLM safety, and SQL generation appeared in over 60% of postings. I then prioritised based on two factors: real-world ROI (which problems do companies actually pay to solve?) and technical depth (which projects demonstrate the most transferable skills?). RAG ranked highest on both — every enterprise with internal documents wants it, and it requires expertise in embedding models, vector search, and LLM prompting simultaneously."

**"What would you do differently if you built these again?"**

"Three things. First, I'd add proper observability from day one — structured logging with correlation IDs, distributed tracing, and a metrics dashboard before writing any business logic. Second, I'd design the data model for the IDP system to support versioned schema evolution — as document types and fields change, you need backward compatibility. Third, for the RAG system, I'd implement document-level deduplication in the ingestion pipeline — the same document ingested twice creates duplicate chunks that pollute retrieval results."
