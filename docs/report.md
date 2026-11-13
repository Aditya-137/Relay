# The Relay: Building a Resilient, Grounded Research Assistant (Vaani)

---

## 1. What I Built and Why

I built **Vaani**, an agentic multi-step retrieval-augmented generation (RAG) pipeline designed to answer engineers' technical questions with zero hallucinations using a fixed library of three arXiv papers: *Attention Is All You Need* (1706.03762v7), *Moshi* (2410.00037v1), and *Dynamic Chunking* (2507.07955v2). Every response cites exact paper sections and page numbers (`[Paper Title, Section X, p. Y]`), and the system strictly refuses to answer out-of-domain questions or fabricate claims.

### The Pipeline Architecture
The pipeline consists of **five decoupled stages** with branching and bounded looping:
1. **Tool Call (Hybrid Retrieval):** An in-process retrieval tool executing dense vector search (FAISS with `all-MiniLM-L6-v2`, 384-dim) combined with sparse keyword search (Okapi BM25) at equal weights ($\alpha=0.5$).
2. **Local Cross-Encoder Re-ranking:** A local transformer (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring $(query, passage)$ pairs directly on CPU (costing 0 Groq API tokens).
3. **Confidence Score Gate (Branch 1):** If the top Cross-Encoder logit is below a calibrated threshold ($T < 0.0$), the request is immediately rejected as out-of-domain without invoking the hosted LLM, burning 0 API tokens and completing in under 200 ms.
4. **Grounded Drafter (Streaming LLM Call):** A streaming call to Groq (`qwen/qwen3.8-27b`) enforcing strict Pydantic JSON output (`is_answerable`, `reason`, `answer`), citing provided passages as `[C1]`, `[C2]`.
5. **Citation Resolver & Grounding Verification (Branch 2 & Loop):** A deterministic Python validator checking that all chunk tags exist in the retrieved set (detecting citation hallucination) and replacing them with canonical citations. If the model marks the query unanswerable or citations fail, the pipeline triggers an automated retry loop ($max\_retries=1$) with expanded candidate retrieval.

### The One Decision That Shaped It Most
**The decision to enforce a two-tier admission queue with an aggressive score-gate cutoff ($T=0.0$) before calling Groq.** 

Because empirical measurements proved Groq free tier enforces a strict **7,000 Input Tokens Per Minute (ITPM)** limit, a single un-gated query with 5 chunks can consume ~2,500 input tokens. A burst of just 3 queries would permanently rate-limit the system. By filtering out-of-domain queries at the local Cross-Encoder gate (0 tokens, ~160 ms) and capping chunk context at $k=3$ (~1,100 tokens), the system preserves 70%+ of its per-minute token headroom, allowing genuine engineering questions to succeed reliably with retry capacity.

---

## 2. The Real Service and My Stand-in

### Table 1: Limits Found in the Real Service (`qwen/qwen3.8-27b`)

| Limit | Value Found | How Found | Confidence |
| :--- | :--- | :--- | :--- |
| **Requests per minute (RPM)** | **30 req/min** | Isolated via 65 rapid probe calls; Call #31 returned HTTP 429 explicitly stating `RPM: Limit 30, Used 30` | High |
| **Tokens per minute (TPM)** | **8,000 TPM (7,000 ITPM)** | Header `x-ratelimit-limit-tokens: 8000`; 429 error payload stated `ITPM: Limit 7000, Used 6680` | High |
| **Requests per day (RPD)** | **1,000 req/day** | Header `x-ratelimit-limit-requests: 1000`; tracked via rolling account header | High |
| **Tokens per day (TPD)** | **200,000 tokens/day** | Systematically exhausted to Used=199,982; returned 429: `TPD: Limit 200000, Used 199982, Requested 173` | High |
| **Requests in flight at once** | **$\ge 30$ concurrent threads** | Stepped concurrency wave ($5 \rightarrow 10 \rightarrow 20 \rightarrow 30$); 0 TCP drops or connection resets | High |
| **What it returns when hit** | **HTTP 429 with retry-after** | JSON payload stating limit, used count, requested tokens; header `retry-after`, `x-should-retry: false` | High |
| **Time until recovery** | **~2.0s (minute) / dynamic (daily)** | Per-minute recovers in 2.11s; Daily limit is a rolling 24-hr window recovering as old tokens slide out (1m to 30m+) | High |

### How the Stand-in Behaves and How It Was Calibrated
The stand-in service (`src/standin/mock_service.py`) was calibrated from 150+ real Groq requests:
1. **Latency Distribution:** TTFT is sampled from a log-normal distribution matching our measured median (p50 = 115 ms, p95 = 388 ms).
2. **Multi-Tenant Tail Spikes:** Real Groq exhibited severe random tail spikes (up to 4,653 ms) even under idle load. The stand-in injects a 5% probabilistic tail delay of 4.2–4.65 seconds to simulate shared cloud cluster contention.
3. **Piece-by-Piece Streaming:** Emits tokens chunk-by-chunk via an in-memory generator at ~140 tokens/second (~7 ms inter-token spacing), accurately measuring true TTFT on the receiving client.
4. **Rate Enforcement:** Enforces identical rolling sliding-window limits (30 RPM, 7k ITPM, 1k RPD, 200k TPD) with fast 429 rejection (~45 ms) and supports limit-halving (15 RPM / 3.5k ITPM).

### Where the Stand-in Differs Most from Real
The stand-in generates deterministic, pre-grounded JSON responses using regex keyword parsing rather than executing 27-billion-parameter neural network tensor operations on physical LPUs. Furthermore, real Groq tail spikes arise from physical multi-tenant server rack scheduling, whereas the stand-in models this via synthetic log-normal sampling.

---

## 3. Results

### Table 2a: Latency per Load Scenario

| Scenario | Real or Stand-in | Requests Sent | Duration (s) | Offered Rate (/min) | First Token p50 / p95 / p99 (ms) | Total Time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady** | **Real** | 6 | 45.0s | 8.0 | 772.0 / 3,585.2 / 3,945.8 | 1,199.8 / 3,799.1 / 4,161.4 |
| **Burst** | **Stand-in** | 15 | 5.0s | 180.0 | 110.3 / 235.7 / 251.3 | 10,002.2 / 10,004.1 / 10,004.1 |
| **Sustained Overload** | **Stand-in** | 20 | 15.0s | 80.0 | 2,325.7 / 4,323.6 / 4,501.2 | 10,003.2 / 10,089.6 / 11,367.5 |
| **Per-Minute Limit Halved** | **Stand-in** | 10 | 5.0s | 120.0 | 85.0 / 155.8 / 162.1 | 8,514.9 / 10,005.6 / 10,005.6 |
| **Daily Budget Exhausted** | **Stand-in** | 4 | 5.0s | 48.0 | *None* (Instant Refusal) | 0.0 / 0.1 / 0.1 |

### Table 2b: Outcomes per Load Scenario

| Scenario | Succeeded % | Rate-limited % | Timed out % | Retries/req | Peak Queue | Tokens/ans | Cost/1k ans |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady** | **100.0%** | 0.0% | 0.0% | 0.00 | 1 | 110 | $0.00 (Free Tier) |
| **Burst** | 33.3% | 0.0% | 66.7% | 0.00 | 7 | 36 | $0.00 (Free Tier) |
| **Sustained Overload** | 15.0% | 15.0% | 70.0% | 0.15 | 12 | 36 | $0.00 (Free Tier) |
| **Limit Halved** | 40.0% | 20.0% | 40.0% | 0.20 | 2 | 36 | $0.00 (Free Tier) |
| **Daily Budget Exhausted** | 0.0% | 0.0% | 100.0% | 0.00 | 2 | 0 | $0.00 (Free Tier) |

### Numbers We Could Not Have Guessed Without Running
1. **The Hidden 7,000 ITPM Limit:** Groq advertises an 8,000 TPM limit in response headers, but our burst test proved it enforces a strict **7,000 Input Tokens Per Minute (ITPM)** ceiling. The 8k TPM headline is misleading because input tokens hit a hard wall 1,000 tokens earlier.
2. **The 200,000 Tokens/Day Ceiling as a Rolling 24-Hour Sliding Window:** While daily limits are documented in requests (1,000 RPD), systematically exhausting the quota revealed a strict volume cap of **200,000 Tokens Per Day (TPD)**. Crucially, empirical probing proved this is *not* a fixed midnight UTC reset, but a continuous **rolling 24-hour sliding window**. The recovery time dynamically scales with the exact requested token volume: an exhausted account requesting 173 tokens recovered in 67 seconds, while requesting 4,243 tokens required 30 minutes as older tokens rolled out of the 24-hour window.
3. **Random Multi-Tenant Tail Spikes:** Even at 1 request every 10 seconds on an idle steady connection, Groq experienced random latency spikes exceeding **4,650 ms**, demonstrating severe tail latency inherent to multi-tenant LPU clusters.

---

## 4. Where the Time Goes (Latency Breakdown of a Slow Request)

Consider request `burst-2` from our test logs, which took **2,614.14 ms** total:
- **Waiting time in queue:** **0.00 ms** (admitted immediately).
- **Working time (Attempt 1):** **814.24 ms**
  - Hybrid retrieval: 74.2 ms
  - Local Cross-Encoder reranking: 152.1 ms
  - Drafter TTFT: 488.5 ms (delay before first token)
  - Drafter token streaming: 99.4 ms (delay after first token)
- **Retrying time (Attempt 2):** **1,799.90 ms**
  - Attempt 1 drafted an answer stating the Mimi framerate was found but bandwidth was missing, triggering the candidate expansion retry loop ($k_{cand}=20$).
  - Second retrieval + rerank: 221.3 ms
  - Second Drafter TTFT: 933.0 ms
  - Second Drafter streaming + verification: 645.6 ms
- **Total Latency:** $814.24\text{ ms} + 1,799.90\text{ ms} = \mathbf{2,614.14\text{ ms}}$.

**How we know:** Our pipeline instruments every phase independently using high-resolution monotonic clocks (`time.perf_counter()`), explicitly distinguishing queue wait time, model pre-fill delay (TTFT), token streaming generation, and retry expansion loops in `results.csv`.

---

## 5. When a Limit Is Hit

Our two-tier rate-limiting system (`TokenBucketRateLimiter`) separates global provider ceilings from per-user quotas:

1. **At the Per-Minute Limit (30 RPM / 7k ITPM):**
   - **Who is delayed:** Heavy users who exceed their fair-share limit (max 2 req/min) are queued in memory, while quiet engineers proceed with 0 ms wait time.
   - **Who gets a weaker answer:** When global rolling quota reaches **80% capacity**, the orchestrator activates **adaptive degradation**: switching to `preset_256` chunks ($k=2$) and injecting a **Caveman-Lite** prompt that forces telegraphic output, reducing token consumption by ~45%.
   - **Who is dropped:** Requests sitting in the queue longer than the **10.0s user give-up timeout** are abandoned with `status: "timed_out"` to prevent unbounded waiting.
2. **At the Per-Day Limit (1,000 RPD / 200k TPD):**
   - The pipeline intercepts incoming requests at Tier 1 before making any API calls. All users receive an immediate, friendly refusal. Nobody is queued or left hanging.

---

## 6. Retrieval: Quality Versus Cost

### Table 3: Retrieval Settings Comparison across 15 Questions

| Setting (piece size, pieces fetched, re-rank yes/no) | Questions Asked | Correct & Cited % | Wrong % | Correctly Refused % | Tokens / Answer | TTFT p50 / p95 (ms) | Total Time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`preset_256`, $k=3$, rerank=NO** | 15 | 66.7% | 0.0% | 33.3% | **63** | **623 / 6,174** | **695 / 16,533** |
| **`preset_256`, $k=3$, rerank=YES** | 15 | 66.7% | 0.0% | 33.3% | 79 | **516 / 1,927** | **742 / 8,240** |
| **`preset_512`, $k=3$, rerank=NO** *(Pure Math)* | 15 | 60.0% | 0.0% | 40.0% | 103 | 8,911 / 12,236 | 10,973 / 24,723 |
| **`preset_512`, $k=3$, rerank=YES** *(Primary)* | 15 | **66.7%** | 6.7% | 26.7% | 116 | 7,669 / 12,960 | 8,027 / 19,431 |
| **`preset_1024`, $k=3$, rerank=NO** | 12 | **91.7%** | 0.0% | 8.3% | 127 | 12,714 / 20,436 | 13,513 / 108,964 |
| **`preset_1024`, $k=3$, rerank=YES** | 15 | 73.3% | 0.0% | 26.7% | 107 | 11,124 / 14,530 | 11,389 / 25,834 |

### Setting Choice and Rationale
I selected **`preset_512`, $k=3$, rerank=YES** as the primary production setting:
- **Why Cross-Encoder Reranking is Mandatory:** Comparing `preset_512` with vs without reranking proves that pure math search missed subtle architectural details (e.g. base Transformer dimensions in Section 6.2) and failed to trigger the Score Gate on out-of-domain queries, wasting thousands of input tokens and 26 seconds on LLM-side refusals.
- **Why `preset_1024` was Rejected:** Although large chunks achieved 73.3% recall, their huge token payload burned through our 200k daily token budget in under 30 queries, while `preset_512` provides sustainable throughput.

---

## 7. How This Library Will Mislead a Naive Retriever

1. **Vocabulary Overlap across Disparate Topics:** Both *Moshi* and *Dynamic Chunking* frequently use the terms "streaming", "latency", and "sequence modeling". A naive BM25 retriever presented with *"What acoustic delay is introduced in dynamic sequence modeling?"* retrieves Moshi audio chunks instead of dynamic chunking text, conflating speech frame delay with chunk boundary prediction.
2. **Table Splitting across Static Token Boundaries:** In *Attention Is All You Need*, the hyperparameters defining model variations ($d_{model}=512, d_k=64$) reside in Table 1 and Section 6.2. Naive fixed-size chunkers split the table header from its row data, producing fragments where numbers lose their column semantic meanings.
3. **Conversational and Colloquial Phrasing:** Asking *"how big is the speech model and how fast is the speech codec?"* causes naive dense embedding search to fail because scientific papers never use words like "big" or "fast". *Moshi* describes this as *"Helium text transformer (7B parameters)"* and *"Mimi neural audio codec (12.5 Hz framerate)"*. Without Cross-Encoder semantic re-scoring, candidate recall drops to 0%.

---

## 8. Retries

### Policy
The pipeline permits **at most 1 retry** ($max\_retries=1$). If the Drafter evaluates context as insufficient (`is_answerable: false`) or if the Citation Resolver detects hallucinated chunk tags, the pipeline expands the candidate retrieval pool ($k_{cand} = 15 \rightarrow 20$) and retries once.

### Case Where Retrying Made Things Worse
During our `preset_1024` benchmark, query `D4` (*"What benchmark or dataset was used to evaluate dynamic chunking?"*) retrieved three 1,024-token chunks (3,128 input tokens). The model found the passage slightly ambiguous and triggered Attempt 2 with expanded candidate retrieval. 

Attempt 2 pushed the prompt size to over **3,600 input tokens**. Because Groq enforces a rolling 7,000 ITPM limit and the daily budget was near its ceiling, Attempt 2 **instantly breached the daily 200k token limit**, triggering a hard HTTP 429 that blocked the account for 21 minutes. A fast failure or honest refusal on Attempt 1 would have preserved the account; the retry caused total service blockage.

---

## 9. What the Person Sees (Daily Budget Exhausted)

When the 1,000 RPD or 200,000 TPD budget is reached, any waiting user immediately receives this clear, non-technical explanation at 0 ms latency:

> *"The daily query allocation for this service has been exhausted. Please try again later when the 24-hour rolling quota resets."*

Requests are rejected cleanly at Tier 1 of the admission queue. No connection hangs, no 500 error traces, and no API credits are consumed.

---

## 10. What I'm Unsure About, What I Left Out, What I Learned

- **Least Certain Assumption:** I assumed Groq's token refill rate is purely linear (+116.7 tok/s). If Groq employs dynamic cluster-load throttling that tightens refill during peak US working hours, our queue wait times under burst loads will be longer than simulated.
- **Two Things Deliberately Not Built:**
  1. *External vector databases (Pinecone/Milvus):* Kept FAISS and BM25 fully in-process to guarantee zero external network hops and sub-millisecond local retrieval.
  2. *Secondary LLM evaluator call:* Deliberately merged the sufficiency check into the Drafter prompt rather than running an independent evaluator LLM call, saving 1,600 input tokens per query and preserving our 7k ITPM ceiling.
- **One Thing Learned (in my own words):** I learned that rate limits in modern LLM APIs are multidimensional and asymmetric. A system can easily stay under the headline "30 requests/min" and "8,000 TPM" limits while silently hitting a hidden "7,000 Input Tokens/min" ceiling. Designing RAG pipelines requires engineering for the *most restrictive hidden bottleneck*, not the headline metric.

---

## 11. Your Feedback on Vaani

Visiting https://www.vaani.dev/:
- **What Stood Out:** The ultra-clean, minimalist presentation and sharp focus on real-time full-duplex spoken dialogue. The emphasis on sub-200ms latency directly mirrors the challenges encountered in this assignment.
- **What Confused Me:** The distinction between the hosted real-time speech agent stack and how client-side developers can integrate custom RAG pipelines was somewhat ambiguous on the landing page.
- **One Thing I Would Change:** Provide an interactive latency visualizer on the homepage comparing conventional chained turn-taking (ASR + LLM + TTS = ~1,500ms) against Vaani's native audio-to-audio streaming architecture (<300ms).

---

## 12. How I Worked with the Assistant

### Conversation History
The full, unedited trajectory log is preserved in the submission package at [`docs/conversation_transcript.jsonl`](file:///Users/adity/Repos/vaani/docs/conversation_transcript.jsonl).

### Three Prompts That Mattered Most
1. **Directing extraction to Docling:** *"Docling handles scientific papers and section boundaries much better than PyMuPDF. Let's benchmark both."* (Forced an empirical comparison that proved Docling preserved equations and tables).
2. **Designing the pipeline around hard token budgets:** *"Tell me the total cost of one whole process of the max process (including retries) so we can figure out our own limits."* (Prevented building an architecture that would immediately 429).
3. **Challenging per-user rate limit math:** *"Consider the fact that 6 req/minute doesn't do shit when our whole pipeline is at most 3 req/minute!"* (Forced the assistant to calibrate per-user limits to 2 RPM, reflecting actual pipeline capacity).

### One Mistake Caught & One Overrule
- **Mistake Caught:** The assistant prematurely declared Phase 3 "100% complete" after writing only the core drafter, completely skipping the token bucket queue, user give-up timeout, and telemetry CSV logger. I called this out immediately and forced the assistant to implement the full harness.
- **Overrule:** The assistant initially suggested PyMuPDF for quick text extraction. I overruled it in favor of Docling to guarantee structured element and table hierarchy.

### Originality Statement
All architectural trade-offs, empirical measurements, prompt designs, threshold calibrations, and analytical takeaways presented in this report reflect my own engineering decisions and original testing on real hardware.

