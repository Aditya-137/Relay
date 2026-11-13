# Groq Empirical Findings & Engineering Learning Notes

> **Date:** 2026-11-12  
> **Model Tested:** `qwen/qwen3.8-27b` (Groq Free Tier)  
> **Raw Data File:** `results/groq_measurements.json`  
> **Test Script:** `scripts/measure_groq.py`

---

## 1. How the Experiment Was Performed

To avoid burning through our daily quota blindly, we ran a structured 3-part probe:

1. **Part 1 — Streaming Baseline (Spaced Load)**
   - Sent **12 streaming requests** spaced **3.5 seconds apart** using `stream=True`.
   - Prompt: *"Explain scaled dot-product attention in 2 concise sentences."* (`max_tokens=100`, `temperature=0.2`).
   - Recorded **TTFT (Time to First Token)** when the first non-empty delta arrived, and **Total Latency** when the stream closed.

2. **Part 2 — Controlled Rate-Limit Burst & Recovery**
   - Fired up to 25 rapid back-to-back synchronous HTTP POST requests (`httpx.Client`) directly to `https://api.groq.com/openai/v1/chat/completions`.
   - Each request carried ~460 input tokens + `max_tokens=50`.
   - Inspected all `x-ratelimit-*` headers on every single response until HTTP 429 occurred.
   - Upon hitting HTTP 429, began polling every 2 seconds with a minimal `"Reply ok"` payload until the first HTTP 200 succeeded to measure real recovery time.

3. **Part 3 — Concurrency Test**
   - Waited 10 seconds for the sliding window to refill, then launched **5 parallel threads** (`ThreadPoolExecutor`) simultaneously to observe whether Groq throttles concurrent in-flight connections.

---

## 2. Key Engineering Findings (What We Discovered)

### Finding 1: The Hidden Input Token Limit (ITPM) vs Total TPM
- **What the headers said:** `x-ratelimit-limit-tokens: 8000`
- **What the 429 error body actually revealed:**
  ```json
  "message": "Rate limit reached for model `qwen/qwen3.8-27b` ... on input tokens per minute (ITPM): Limit 7000, Used 6680, Requested 473. Please try again in 1.3114s."
  ```
- **Why this matters:** Even though the response header `x-ratelimit-remaining-tokens` showed `1268` tokens remaining, Groq still rejected the call because it enforces a **separate, stricter 7,000 Input Tokens Per Minute (ITPM)** limit! In a RAG system where ~85% of tokens are retrieved input context, **7,000 ITPM is our true bottleneck**, not 8,000 TPM.

### Finding 2: Tail Latency Spikes Under Burst Load ("Succeeding Slowly")
- During the spaced baseline (12 calls):
  - **TTFT:** p50 = `114.74 ms`, p95 = `387.63 ms`
  - **Total Time:** p50 = `239.18 ms`, p95 = `532.73 ms`
- During the rapid burst:
  - Normal calls took `144–207 ms`.
  - **Burst 04 jumped to `4,553.04 ms` (4.55s)** and **Burst 05 took `4,247.81 ms` (4.25s)**—a **20× latency spike** while still returning HTTP 200!
- **Why did Burst 04 & 05 spike to ~4.5 seconds?**
  While Groq's internal infrastructure is a black box, two adjacent calls stalling by ~4.2–4.5s during a burst point to:
  1. **Gateway Soft-Queueing / Scheduler Delay:** API routers may temporarily queue back-to-back calls before reaching hard 429 limits.
  2. **Multi-Tenant Contention / Cold Worker Routing:** On the shared free tier, requests can hit a busy LPU cluster or undergo scheduling delays while the HTTP socket remains open.
- **Takeaway:** Latency degrades drastically before failure ("succeeds slowly"). Our pipeline must enforce timeouts, and our stand-in mock must simulate occasional slow-success tail spikes.

### Finding 3: Continuous Sliding-Window Recovery (Leaky Bucket Refill)
- When Burst 18 hit HTTP 429, Groq returned header `retry-after: 2` and body message `"Please try again in 1.3114s"`.
- Notice that `x-ratelimit-reset-tokens` read `57.257s` (the time to refill the *entire* 8,000 bucket back to 100%), **yet our polling recovered in just `2.11 seconds`**.
- **Takeaway:** You do **not** have to wait 60 seconds after a 429! Groq refills tokens continuously over time (like a sliding window / leaky bucket). Waiting just `retry-after` seconds (1–3s) is enough for small requests to pass through again.

### Finding 4: Request Concurrency (In-Flight Requests)
- All **5 parallel threads** succeeded simultaneously in `254–258 ms`.
- At least 5 in-flight requests are permitted concurrently on the free tier without serialization penalties.

### Finding 5: Daily Request Limit (RPD)
- `x-ratelimit-limit-requests: 1000` (and `x-ratelimit-reset-requests` tracks daily countdown).
- Because we sent ~30 requests during this probe, `rem_req` dropped from `999` down to `971`.
- Notice also: `x-ratelimit-limit-requests` is **1,000 RPD**, NOT 30 RPM! The per-minute request limit (RPM) is either higher than 18 calls/10s or is completely overshadowed by the 7,000 ITPM token cap on RAG workloads.

---

## 3. Experiment #2 Findings (`scripts/measure_groq_deep.py`)

Raw Data File: `results/groq_deep_measurements.json`

### Finding 6: Exact Pure RPM Limit Isolated (`RPM = 30`)
- Fired tiny ~15-token prompts back-to-back so ITPM (7,000) would NOT trip.
- **Calls 1 to 30 all returned HTTP 200** (~97–125 ms each).
- **Call 31 (at t=4.45s) immediately returned HTTP 429** with the explicit payload:
  ```json
  "on requests per minute (RPM): Limit 30, Used 30, Requested 1. Please try again in 2s."
  ```
- **Takeaway:** Groq enforces **four independent limit buckets** simultaneously on `qwen/qwen3.8-27b`:
  - **RPM:** 30 requests / minute
  - **ITPM:** 7,000 input tokens / minute
  - **Total TPM:** 8,000 tokens / minute (header)
  - **RPD:** 1,000 requests / day (header)

### Finding 7: Stepped Concurrency Stress (`5 → 10 → 20 → 30` parallel threads)
Using tiny prompts separated by 15s cooldowns:
| Concurrency Wave | OK / Sent | Rate Limited (429) | Connection Errors | p50 Latency | p95 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N = 5** | 5 / 5 | 0 | 0 | 177.7 ms | 179.4 ms |
| **N = 10** | 9 / 10 | 1 | 0 | 195.6 ms | 230.3 ms |
| **N = 20** | 7 / 20 | 13 | 0 | 198.8 ms | 200.8 ms |
| **N = 30** | 7 / 30 | 23 | 0 | 224.7 ms | 231.4 ms |

- **What this proves:**
  1. Groq **did not drop sockets or throttle parallel execution below ~10–30 threads**—it had 0 connection errors and p50 only ticked up slightly from `177ms → 224ms`.
  2. Notice why `N=20` and `N=30` only allowed **7 requests** through after a 15-second cooldown: $30\text{ RPM} \times \frac{15\text{s}}{60\text{s}} = 7.5\text{ requests}$! That confirms **Groq's RPM limiter also refills via a continuous ~0.5 req/sec sliding window (7–8 tokens every 15s)**, rather than resetting strictly at :00 on the clock!

### Finding 8: Steady-Load Tail Spikes & Smart Retry Recovery (`100%` Success)
- **Steady Safe Pace (6 RAG calls spaced 7.5s apart):**
  - 6/6 succeeded (100%).
  - Calls 1, 3, 4, 5, 6 took `264–371 ms`.
  - **Call 2 spiked to `4,653.25 ms` (4.65s)** even though the system was completely idle for 7.5 seconds before it! This confirms tail latency spikes happen randomly on the shared free tier even under gentle steady traffic.
- **Overload Burst (16 RAG calls ~550 tokens each) + Smart `Retry-After` Fallback:**
  - Calls 1–12 succeeded immediately (`183–438 ms`).
  - Calls 13–16 hit 429 (`ITPM` bucket full).
  - By reading `retry-after` / error wait seconds (`1.15s`, `5.15s`, `4.15s`, `5.15s`) and retrying once, **16/16 (100%) succeeded**!
  - Total latency for retried calls rose from ~190ms p50 up to **5,579ms max** (`wait + work`).

