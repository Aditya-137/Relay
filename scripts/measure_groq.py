"""scripts/measure_groq.py

Empirical measurement script for Groq Free Tier behavior (Phase 2).
Measures:
1. Streaming Latency Distribution (TTFT and Total Time) across realistic calls.
2. Exact rate-limit headers (RPD, TPM, remaining quotas, reset times).
3. Controlled 429 rate-limit triggering (captures exact error JSON, retry-after, and recovery time).
4. Concurrent in-flight request handling.

Outputs raw logs and summary to:
  results/groq_measurements.json
"""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv("/Users/adity/Repos/vaani/.env")
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

OUTPUT_PATH = Path("/Users/adity/Repos/vaani/results/groq_measurements.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_ID = "qwen/qwen3.8-27b"


def probe_streaming_latency(client: Groq, num_requests: int = 12) -> Dict[str, Any]:
    """Measure TTFT and Total generation latency using streaming on modest spaced requests."""
    print(f"\n--- 1. Probing Streaming Latency ({num_requests} requests, ~3.5s spaced) ---")
    prompt = "Explain scaled dot-product attention in 2 concise sentences."
    records: List[Dict[str, Any]] = []

    for i in range(num_requests):
        t_start = time.perf_counter()
        ttft_ms = None
        tokens_out = 0
        reply_parts = []
        status = "ok"
        error_msg = None

        try:
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    reply_parts.append(delta)
                    tokens_out += 1
            total_ms = (time.perf_counter() - t_start) * 1000.0
            if ttft_ms is None:
                ttft_ms = total_ms
        except RateLimitError as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            status = "rate_limited"
            error_msg = str(e)
        except Exception as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            status = "error"
            error_msg = str(e)

        rec = {
            "req_idx": i + 1,
            "status": status,
            "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
            "total_ms": round(total_ms, 2),
            "stream_chunks": tokens_out,
            "error": error_msg,
        }
        records.append(rec)
        print(f"  Req {i+1:02d}: status={status} | TTFT={rec['ttft_ms']}ms | Total={rec['total_ms']}ms | chunks={tokens_out}")

        if i < num_requests - 1:
            time.sleep(3.5)

    ok_ttfts = [r["ttft_ms"] for r in records if r["status"] == "ok" and r["ttft_ms"] is not None]
    ok_totals = [r["total_ms"] for r in records if r["status"] == "ok"]

    def pct(arr, p):
        if not arr:
            return None
        sorted_arr = sorted(arr)
        idx = min(int(len(sorted_arr) * p), len(sorted_arr) - 1)
        return round(sorted_arr[idx], 2)

    return {
        "records": records,
        "summary": {
            "count": len(records),
            "ok_count": len(ok_totals),
            "ttft_p50_ms": round(statistics.median(ok_ttfts), 2) if ok_ttfts else None,
            "ttft_p95_ms": pct(ok_ttfts, 0.95),
            "total_p50_ms": round(statistics.median(ok_totals), 2) if ok_totals else None,
            "total_p95_ms": pct(ok_totals, 0.95),
        },
    }


def probe_headers_and_429_burst() -> Dict[str, Any]:
    """Use raw HTTP requests to inspect headers, trigger a controlled 429 (via TPM/RPM burst), and measure recovery."""
    print("\n--- 2. Probing Rate Limit Headers & Triggering Controlled 429 ---")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Moderate payload (~600 tokens) to hit the 8,000 TPM limit cleanly within ~12-15 rapid calls
    filler_context = "Attention mechanisms allow modeling dependencies without regard to distance. " * 45
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": f"Context: {filler_context}\nSummarize in 5 words."}
        ],
        "max_tokens": 50,
    }

    burst_logs = []
    hit_429 = False
    t_429_hit = None
    first_429_details = None
    recovery_sec = None

    with httpx.Client(timeout=30.0) as http:
        for i in range(25):
            t0 = time.perf_counter()
            resp = http.post(url, headers=headers, json=payload)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            rl_headers = {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
            body = resp.json()

            entry = {
                "burst_idx": i + 1,
                "status_code": resp.status_code,
                "elapsed_ms": elapsed_ms,
                "headers": rl_headers,
                "usage": body.get("usage"),
            }
            burst_logs.append(entry)
            print(
                f"  Burst {i+1:02d}: HTTP {resp.status_code} ({elapsed_ms}ms) | "
                f"rem_req={rl_headers.get('x-ratelimit-remaining-requests')} | "
                f"rem_tok={rl_headers.get('x-ratelimit-remaining-tokens')} | "
                f"retry_after={rl_headers.get('retry-after')}"
            )

            if resp.status_code == 429:
                hit_429 = True
                t_429_hit = time.perf_counter()
                first_429_details = {
                    "status_code": 429,
                    "headers": rl_headers,
                    "body": body,
                }
                print("  >>> Hit 429! Capturing recovery time...")
                break

        # Measure actual recovery time until a minimal request succeeds again
        if hit_429:
            retry_payload = {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Reply ok"}],
                "max_tokens": 5,
            }
            retry_attempts = 0
            while retry_attempts < 20:
                time.sleep(2.0)
                retry_attempts += 1
                r = http.post(url, headers=headers, json=retry_payload)
                if r.status_code == 200:
                    recovery_sec = round(time.perf_counter() - t_429_hit, 2)
                    print(f"  >>> Recovered after {recovery_sec}s (attempt {retry_attempts})")
                    break
                else:
                    print(f"  ... still 429 after {round(time.perf_counter() - t_429_hit, 1)}s (retry-after={r.headers.get('retry-after')})")

    return {
        "burst_logs": burst_logs,
        "hit_429": hit_429,
        "first_429_details": first_429_details,
        "empirical_recovery_seconds": recovery_sec,
    }


def probe_concurrency(client: Groq, n_concurrent: int = 5) -> Dict[str, Any]:
    """Wait for minute token reset, then fire N concurrent requests to check in-flight limits."""
    print(f"\n--- 3. Probing Concurrent In-Flight Requests ({n_concurrent} parallel threads) ---")
    time.sleep(10.0)  # Let token bucket refill

    def call_one(idx):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": f"Say the number {idx}."}],
                max_tokens=10,
            )
            return {"idx": idx, "status": "ok", "ms": round((time.perf_counter() - t0) * 1000.0, 2)}
        except Exception as e:
            return {"idx": idx, "status": "error", "ms": round((time.perf_counter() - t0) * 1000.0, 2), "err": str(e)}

    results = []
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(call_one, i) for i in range(n_concurrent)]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"  Thread {res['idx']}: status={res['status']} ({res['ms']}ms)")

    return {
        "n_concurrent": n_concurrent,
        "ok_count": sum(1 for r in results if r["status"] == "ok"),
        "results": results,
    }


def main():
    client = Groq(api_key=API_KEY)
    print(f"Starting Groq Empirical Measurement on model: {MODEL_ID}")

    streaming_data = probe_streaming_latency(client, num_requests=12)
    burst_data = probe_headers_and_429_burst()
    concurrency_data = probe_concurrency(client, n_concurrent=5)

    final_report = {
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "streaming_latency": streaming_data,
        "rate_limit_burst": burst_data,
        "concurrency_test": concurrency_data,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nMeasurement complete! Full raw data saved to {OUTPUT_PATH}")
    print("\nLatency Summary:", json.dumps(streaming_data["summary"], indent=2))
    if burst_data["first_429_details"]:
        print("\nFirst 429 Body:", json.dumps(burst_data["first_429_details"]["body"], indent=2))
        print("Empirical Recovery Seconds:", burst_data["empirical_recovery_seconds"])


if __name__ == "__main__":
    main()

