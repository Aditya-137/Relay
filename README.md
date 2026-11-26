# Relay

Grounded RAG pipeline that answers technical questions from arXiv papers while
staying inside Groq free-tier limits. Hybrid retrieval, local cross-encoder
reranking, an early-exit gate for out-of-domain queries, deterministic citation
resolution, and a two-tier rate limiter.

## Results

| Metric | Value |
|---|---|
| Citation accuracy (N = __ labeled questions) | __% |
| Out-of-domain queries rejected by the gate | __% |
| Gate rejection latency | < 200 ms, 0 LLM tokens |
| End-to-end latency p50 / p99 (real Groq) | __ s / __ s |
| Runs on real Groq vs calibrated stand-in mock | __ / __ (152 total) |

Latency and quota numbers come from the real-Groq runs. The stand-in mock is
used for load scenarios the free-tier quota can't sustain, and those rows are
marked as mock in `results/results.csv`.

Full write-up: [`docs/report.md`](docs/report.md) · Raw runs: [`results/results.csv`](results/results.csv)

## Architecture

1. **Hybrid retrieval:** FAISS dense search (`all-MiniLM-L6-v2`) combined with
   Okapi BM25 keyword search (alpha = 0.5).
2. **Local reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores candidate
   passages on CPU, so it uses no API tokens.
3. **Score gate:** queries whose top rerank score is below 0.0 are rejected
   immediately, before any LLM call.
4. **Grounded drafter:** Groq `qwen/qwen3.8-27b` returns structured JSON with
   passage tags `[C1]`, `[C2]`.
5. **Citation resolver:** maps tags to `[Title, Section X, p. Y]` and verifies
   grounding. If the answer is unanswerable or a citation fails, it retries
   once with a wider candidate pool (k = 20).
6. **Two-tier rate limiter:** global 30 RPM / 7,000 input TPM / 1,000 RPD, plus
   a per-user 2 RPM fair share with a 10 s give-up timeout.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
cp .env.example .env   # set GROQ_API_KEY for real Groq; not needed for the mock

# real Groq (falls back to the mock if the daily quota is hit)
python -m src.pipeline.orchestrator "What is the framerate of the Mimi neural audio codec in Moshi?"

# force the calibrated stand-in mock
python -m src.pipeline.orchestrator --source standin "What acoustic delay is used in Moshi?"
```

## Benchmarks

```bash
# Steady, Burst, Sustained Overload, Limit Halved, Daily Budget Exhausted
python scripts/run_load_scenarios.py

# Recompute report tables from results/results.csv
python scripts/compute_tables.py
```

## Layout

`src/` pipeline · `config/` limits and thresholds · `data/` corpus ·
`scripts/` load scenarios and table generation · `results/` run telemetry ·
`docs/` design report
