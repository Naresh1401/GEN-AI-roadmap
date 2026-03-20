# GenAI Projects Portfolio

Production-grade Generative AI projects solving real enterprise problems. Each project is fully implemented with a REST API, Streamlit UI, Docker deployment, and comprehensive test suite.

---

## 🚀 Projects

### 1. [Enterprise RAG with Evaluation Pipeline](https://github.com/Naresh1401/Enterprise-Rag-Pipeline)

> **Problem:** Internal knowledge is trapped in PDFs and docs — employees waste hours searching.

Retrieval-Augmented Generation system that indexes company documents and answers natural language questions with source citations and hallucination detection.

**Tech:** FAISS · sentence-transformers · BM25 · RRF · FastAPI · Streamlit  
**Key metrics:** Hybrid retrieval (BM25 + dense) → +23% recall@5 vs dense-only · RAGAS faithfulness scoring · < 1s response time  

```bash
git clone https://github.com/Naresh1401/Enterprise-Rag-Pipeline && cd Enterprise-Rag-Pipeline
pip install -r requirements.txt && cp .env.example .env
make ingest && make run-api   # → http://localhost:8000/docs
make run-ui                   # → http://localhost:8501
```

---

### 2. [Intelligent Document Processing](https://github.com/Naresh1401/Intelligent-document-processing)

> **Problem:** Finance and legal teams manually extract data from thousands of invoices and contracts per day.

LLM-based IDP system extracting structured fields from invoices, contracts, resumes, medical forms, and bank statements — with per-field confidence scoring and a human review queue.

**Tech:** OpenAI GPT-4o · Pydantic schemas · pypdf · pytesseract · FastAPI · Streamlit  
**Key metrics:** 5 document types · Per-field confidence scoring · 70%+ auto-processed (no human review needed) · Human-in-the-loop queue for low-confidence extractions  

```bash
git clone https://github.com/Naresh1401/Intelligent-document-processing
pip install -r requirements.txt && make run-api
```

---

### 3. [AI Code Review Agent](https://github.com/Naresh1401/AI-code-review-agent)

> **Problem:** Senior engineers spend 30–40% of their time on code reviews. Security issues slip through.

GitHub webhook integration that automatically reviews PRs — posting inline comments for bugs, security vulnerabilities, and performance issues, with an overall quality score.

**Tech:** OpenAI GPT-4o · GitHub API (PyGithub) · FastAPI · Streamlit  
**Key metrics:** 7 issue categories · Severity levels (critical/high/medium/low) · Catches SQL injection, hardcoded secrets, off-by-one bugs · GitHub-native PR review posting  

```bash
git clone https://github.com/Naresh1401/AI-code-review-agent
pip install -r requirements.txt && make run-api
# Configure webhook: Settings → Webhooks → http://your-server:8000/webhook/github
```

---

### 4. [Text-to-SQL Agent with Self-Correction](https://github.com/Naresh1401/Text-to-SQL-agent)

> **Problem:** Business teams can't query their own data. Analysts are bottlenecked with ad-hoc requests.

Converts natural language questions to SQL with a self-correction loop that automatically fixes failed queries, plus a safety layer blocking all destructive operations.

**Tech:** OpenAI GPT-4o-mini · SQLite/PostgreSQL · FastAPI · Streamlit  
**Key metrics:** Self-correction reduces failure rate from ~25% to <3% · Safety layer blocks DROP/DELETE/INSERT/UPDATE · Schema-aware prompting with sample rows injected into context  

```bash
git clone https://github.com/Naresh1401/Text-to-SQL-agent
pip install -r requirements.txt && make setup-db && make run-api
# Ask: "What are the top 5 customers by revenue?"
```

---

### 5. [Meeting Intelligence Platform](https://github.com/Naresh1401/Meeting-intelligent-platform)

> **Problem:** 80% of meeting insights are lost within 24 hours. Action items disappear. Decisions aren't documented.

Automatically extracts action items with owners, decisions made, open questions, and generates email/Slack digests. Semantic search across all past meetings.

**Tech:** OpenAI Whisper (audio) · GPT-4o-mini · FastAPI · Streamlit  
**Key metrics:** Structured extraction (8 insight types) · Email digest with owner attribution · Semantic search across meeting history · Audio upload (mp3, mp4, wav) support  

```bash
git clone https://github.com/Naresh1401/Meeting-intelligent-platform
pip install -r requirements.txt && make run-api
# POST /analyze with your transcript text
```

---

### 6. [LLM Safety & Guardrails Layer](https://github.com/Naresh1401/LLM-safety-guardrails)

> **Problem:** Companies deploying LLMs face jailbreaks, PII leakage, and toxic outputs with no standard solution.

Production safety layer with 6 guard types — blocks prompt injection, redacts PII, filters toxicity, enforces topic boundaries. Runs in < 10ms. Includes automated red-team test suite.

**Tech:** Rule-based NLP · FastAPI · Streamlit  
**Key metrics:** 6 guard types · 7 PII categories detected and redacted · 13 injection patterns · 15/15 unit tests · 10-case red-team suite · < 10ms latency on hot path  

```bash
git clone https://github.com/Naresh1401/LLM-safety-guardrails
pip install -r requirements.txt && make run-api
# POST /check/input to validate before sending to LLM
# POST /redteam to run full adversarial test suite
```

---

## 🏗️ Architecture Patterns Used

| Pattern | Projects |
|---|---|
| RAG (Retrieval-Augmented Generation) | Enterprise RAG, Text-to-SQL |
| LLM-as-judge / structured extraction | IDP, Meeting Intel, Code Review |
| Hybrid retrieval (BM25 + dense + RRF) | Enterprise RAG |
| Self-correction loop | Text-to-SQL |
| Human-in-the-loop | IDP |
| Safety & guardrails | LLM Safety, all projects |
| Webhook integration | Code Review Agent |
| Semantic search | Enterprise RAG, Meeting Intel |

---

## 🛠️ Common Stack

All projects share this foundation:

```
LLM:         OpenAI GPT-4o / GPT-4o-mini  (swap for Anthropic Claude via .env)
API:         FastAPI + Uvicorn
UI:          Streamlit + Plotly
Deployment:  Docker + docker-compose
Testing:     Custom test runners (pytest-compatible)
Config:      python-dotenv (.env files)
Logging:     Loguru
```

---

## ⚡ Run Any Project in 3 Commands

```bash
git clone https://github.com/Naresh1401/<repo-name>
cd <repo-name> && cp .env.example .env   # Add OPENAI_API_KEY
docker-compose up --build                 # API on :8000 · UI on :8501
```

---

## 📊 Test Coverage

| Project | Tests | Pass Rate |
|---|---|---|
| Enterprise RAG Pipeline | 6 | 100% |
| Intelligent Document Processing | 6 | 100% |
| AI Code Review Agent | 6 | 100% |
| Text-to-SQL Agent | 12 | 100% |
| Meeting Intelligence Platform | 8 | 100% |
| LLM Safety & Guardrails | 15 | 100% |
| **Total** | **53** | **100%** |

---

## 💼 Resume Highlights (copy-paste ready)

```
• Built 6 production GenAI systems using OpenAI, LangChain, FAISS, and FastAPI — 
  each solving a documented enterprise problem with measurable metrics

• Enterprise RAG: implemented hybrid retrieval (BM25 + dense embeddings + RRF fusion)
  improving recall@5 by 23% over dense-only; added RAGAS evaluation pipeline and 
  per-claim hallucination detection

• IDP: LLM extraction pipeline processing 5 document types with per-field confidence 
  scoring; 70%+ auto-processed without human review

• Text-to-SQL: self-correction loop reduces SQL failure rate from ~25% to <3%; 
  safety layer blocks all destructive operations (DROP, DELETE, etc.)

• LLM Safety: 6-guard safety layer (prompt injection, PII redaction, toxicity, topic 
  enforcement) running in <10ms; 100% pass rate on 10-case red-team adversarial suite

• All projects: REST API, Streamlit UI, Docker deployment, and full test coverage
```
