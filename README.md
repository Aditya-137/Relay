# Relay

Grounded RAG pipeline over three arXiv papers (Attention Is All You Need, Moshi,
Dynamic Chunking), built to stay inside Groq free-tier limits. Hybrid retrieval,
local cross-encoder reranking, an early-exit gate for out-of-domain queries,
deterministic citation resolution, and a two-tier rate limiter.

## Results

**Answer quality** (15 hand-labeled questions, primary config `preset_512`, k=3, rerank on)

| Outcome | Count |
|---|---|
| Correct and cited | 10 / 15 (66.7%) |
| Wrong or uncited | 1 / 15 (6.7%) |
| Out-of-domain refused (score gate, 0 LLM tokens) | 2 / 2 |
| Answerable question refused | 2 / 13 |

N is small. Treat these as a sanity check, not a benchmark.

**Real Groq limits found by probing:** 30 RPM, 7,000 input TPM, 1,000 RPD,
200,000 TPD (rolling 24 h window). Limits are enforced with HTTP 429 plus `retry-after`.

**Out-of-domain handling:** the score gate refuses in about 0.2 s with zero tokens,
versus 11–26 s when the LLM has to refuse (no-rerank baseline).

**Load scenarios.** Only Steady ran on real Groq; the rest ran on a stand-in
calibrated from 150+ real calls (log-normal TTFT, the real limits above).

| Scenario | Source | Requests | Succeeded | Rate-limited | Timed out |
|---|---|---|---|---|---|
| Steady, 8/min | Real | 6 | 100% | 0% | 0% |
| Burst, 180/min | Stand-in | 15 | 33.3% | 0% | 66.7% |
| Sustained overload, 80/min | Stand-in | 20 | 15% | 15% | 70% |
| Per-minute limit halved | Stand-in | 10 | 40% | 20% | 40% |
| Daily budget exhausted | Stand-in | 4 | instant refusal, 0 ms | | |

Under overload, requests wait at most 10 s in the queue and then time out cleanly.
Details: [`docs/report.md`](docs/report.md) · Raw runs: [`results/results.csv`](results/results.csv)

## Architecture

1. **Hybrid retrieval:** FAISS dense search (`all-MiniLM-L6-v2`) plus Okapi BM25 (alpha = 0.5).
2. **Local reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU, 0 API tokens.
3. **Score gate:** top rerank score below 0.0 means immediate refusal, before any LLM call.
4. **Grounded drafter:** Groq `qwen/qwen3.8-27b` returns structured JSON with passage tags `[C1]`, `[C2]`.
5. **Citation resolver:** maps tags to `[Title, Section X, p. Y]` and validates them.
   One retry with a wider candidate pool (k = 20) if the answer is unanswerable or a citation fails.
6. **Two-tier rate limiter:** global 30 RPM / 7,000 ITPM / 1,000 RPD / 200k TPD, plus
   per-user 2 RPM fair share. At 80% of quota it degrades to `preset_256`, k=2, with a
   shorter prompt (about 45% fewer tokens).

## Known limitations

- 15-question evaluation set, hand-labeled.
- Load scenarios other than Steady are simulated.
- The retry can push prompt size up near the daily token ceiling.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
cp .env.example .env   # set GROQ_API_KEY for real Groq; not needed for the mock

python -m src.pipeline.orchestrator "What is the framerate of the Mimi neural audio codec in Moshi?"
python -m src.pipeline.orchestrator --source standin "What acoustic delay is used in Moshi?"

python scripts/run_load_scenarios.py   # Steady, Burst, Overload, Limit Halved, Daily Exhausted
python scripts/compute_tables.py       # recompute report tables from results/results.csv
```
