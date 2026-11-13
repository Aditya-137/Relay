# Vaani — The Relay Assignment

> Grounded, zero-hallucination multi-step RAG pipeline answering technical questions from arXiv research papers under Groq free-tier constraints. Features two-tier rate limiting, local cross-encoder reranking with an early-exit score gate, deterministic citation resolution, and a calibrated stand-in mock service.

## 🟢 Status: 100% Complete & Delivered

- **Final Report:** [`docs/report.md`](file:///Users/adity/Repos/vaani/docs/report.md) (All 12 required sections + 4 complete tables)
- **Telemetry Dataset:** [`results/results.csv`](file:///Users/adity/Repos/vaani/results/results.csv) (152 validated test and benchmark runs)
- **Project Tracker:** [`docs/PROJECT_TRACKER.md`](file:///Users/adity/Repos/vaani/docs/PROJECT_TRACKER.md)

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

### 2. Configure Credentials (Optional for Stand-in Mock)
```bash
cp .env.example .env
# Set GROQ_API_KEY in .env if testing real Groq
```

### 3. Run a Query through the Pipeline
```bash
# Run against the real Groq API (or stand-in mock if daily quota is reached)
python -m src.pipeline.orchestrator "What is the framerate of the Mimi neural audio codec in Moshi?"

# Force execution through the calibrated stand-in mock
python -m src.pipeline.orchestrator --source standin "What acoustic delay is used in Moshi?"
```

### 4. Run Load Scenarios & Benchmarks
```bash
# Run load scenarios (Steady, Burst, Sustained Overload, Limit Halved, Daily Budget Exhausted)
python scripts/run_load_scenarios.py

# Recompute report tables from results/results.csv
python scripts/compute_tables.py
```

---

## 🏛️ Pipeline Architecture

1. **Tool Call (Hybrid Retrieval):** FAISS dense semantic search (`all-MiniLM-L6-v2`) + Okapi BM25 sparse keyword search ($\alpha=0.5$).
2. **Local Cross-Encoder Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring candidate passages on local CPU (0 API tokens).
3. **Score Gate Early Exit (Branch 1):** Rejects out-of-domain queries immediately ($T < 0.0$) in <200ms without consuming LLM quota.
4. **Grounded Drafter (Streaming LLM Call):** Groq `qwen/qwen3.8-27b` generating structured JSON with passage tags `[C1]`, `[C2]`.
5. **Citation Resolver & Grounding Verification (Branch 2 & Loop):** Resolves tags to canonical citations `[Title, Section X, p. Y]`; triggers candidate expansion retry ($k_{cand}=20$, max 1 retry) if unanswerable or citations fail.
6. **Two-Tier Rate Limiter:** Enforces global 30 RPM / 7,000 ITPM / 1,000 RPD and per-user 2 RPM fair share with a 10s give-up timeout.


