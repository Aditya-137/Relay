"""scripts/measure_groq_deep.py

Experiment #2: Deep empirical probing of Groq Free Tier (`qwen/qwen3.8-27b`).
Covers:
1. Pure RPM Isolation: Tiny ~15-token prompts fired continuously up to 65 calls to detect pure Requests-Per-Minute limit without tripping ITPM (7,000).
2. Stepped Concurrency Stress: Parallel waves of 5 -> 10 -> 20 -> 30 simultaneous requests to measure queueing delay and connection stability.
3. Sustainable Pace vs Overload + Smart Retry Verification: Realistic ~600-token RAG payloads comparing safe spaced traffic vs bursty traffic with Retry-After recovery.

Outputs raw data to: results/groq_deep_measurements.json
"""

import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv("/Users/adity/Repos/vaani/.env")
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

OUTPUT_PATH = Path("/Users/adity/Repos/vaani/results/groq_deep_measurements.json")
MODEL_ID = "qwen/qwen3.8-27b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def pct(arr: List[float], p: float) -> float | None:
    if not arr:
        return None
    s = sorted(arr)
    idx = min(int(len(s) * p), len(s) - 1)
    return round(s[idx], 2)


def test_pure_rpm(max_calls: int = 65) -> Dict[str, Any]:
    """Send tiny ~15-token prompts as fast as possible to isolate the pure RPM limit."""
    print(f"\n=== TEST 1: PURE RPM ISOLATION (up to {max_calls} tiny prompts) ===")
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Reply: 1"}],
        "max_tokens": 2,
    }
    logs = []
    t_window_start = time.perf_counter()
    first_429_at = None
    first_429_body = None

    with httpx.Client(timeout=30.0) as http:
        for i in range(1, max_calls + 1):
            t0 = time.perf_counter()
            resp = http.post(API_URL, headers=HEADERS, json=payload)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            since_start = round(time.perf_counter() - t_window_start, 2)

            rl = {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
            body = resp.json()

            entry = {
                "idx": i,
                "t_sec": since_start,
                "status": resp.status_code,
                "ms": elapsed_ms,
                "rem_req": rl.get("x-ratelimit-remaining-requests"),
                "rem_tok": rl.get("x-ratelimit-remaining-tokens"),
                "retry_after": rl.get("retry-after"),
            }
            logs.append(entry)

            if i <= 5 or i % 10 == 0 or resp.status_code != 200:
                print(
                    f"  Call {i:02d} (t={since_start:4.1f}s): HTTP {resp.status_code} ({elapsed_ms:6.1f}ms) | "
                    f"rem_req={entry['rem_req']} | rem_tok={entry['rem_tok']}"
                )

            if resp.status_code == 429:
                first_429_at = i
                first_429_body = body
                print(f"  >>> HIT 429 AT CALL {i} (t={since_start}s)! Body: {json.dumps(body)}")
                break

    ok_latencies = [x["ms"] for x in logs if x["status"] == 200]
    return {
        "total_sent": len(logs),
        "succeeded": len(ok_latencies),
        "hit_429": first_429_at is not None,
        "first_429_call_idx": first_429_at,
        "first_429_body": first_429_body,
        "latency_p50_ms": round(statistics.median(ok_latencies), 2) if ok_latencies else None,
        "latency_p95_ms": pct(ok_latencies, 0.95),
        "latency_max_ms": round(max(ok_latencies), 2) if ok_latencies else None,
        "logs": logs,
    }


def run_concurrency_wave(n_threads: int) -> Dict[str, Any]:
    """Fire n_threads simultaneously with tiny prompts."""
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say ok"}],
        "max_tokens": 2,
    }

    def single_req(tid: int):
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(API_URL, headers=HEADERS, json=payload)
                ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return {
                    "tid": tid,
                    "status": r.status_code,
                    "ms": ms,
                    "error": r.json().get("error", {}).get("message") if r.status_code != 200 else None,
                }
        except Exception as e:
            return {"tid": tid, "status": -1, "ms": round((time.perf_counter() - t0) * 1000.0, 2), "error": str(e)}

    results = []
    t_wave_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(single_req, i) for i in range(n_threads)]
        for fut in as_completed(futures):
            results.append(fut.result())
    wave_wall_ms = round((time.perf_counter() - t_wave_start) * 1000.0, 2)

    ok_ms = [r["ms"] for r in results if r["status"] == 200]
    n_429 = sum(1 for r in results if r["status"] == 429)
    n_err = sum(1 for r in results if r["status"] not in (200, 429))

    summary = {
        "concurrency": n_threads,
        "wall_time_ms": wave_wall_ms,
        "ok_count": len(ok_ms),
        "rate_limited_429": n_429,
        "other_errors": n_err,
        "min_ms": round(min(ok_ms), 2) if ok_ms else None,
        "p50_ms": round(statistics.median(ok_ms), 2) if ok_ms else None,
        "p95_ms": pct(ok_ms, 0.95),
        "max_ms": round(max(ok_ms), 2) if ok_ms else None,
        "sample_error": next((r["error"] for r in results if r["error"]), None),
    }
    print(
        f"  Wave N={n_threads:2d}: OK={len(ok_ms):2d}/{n_threads} | 429={n_429:2d} | Err={n_err} | "
        f"p50={summary['p50_ms']}ms | p95={summary['p95_ms']}ms | max={summary['max_ms']}ms"
    )
    return summary


def test_stepped_concurrency(levels: List[int] = [5, 10, 20, 30]) -> List[Dict[str, Any]]:
    """Run concurrency waves stepping up from 5 to 30 with clean cooldowns."""
    print(f"\n=== TEST 2: STEPPED CONCURRENCY STRESS ({levels}) ===")
    wave_summaries = []
    for lvl in levels:
        print(f"  Cooling down 15s before N={lvl} wave...")
        time.sleep(15.0)
        res = run_concurrency_wave(lvl)
        wave_summaries.append(res)
    return wave_summaries


def parse_wait_seconds(resp: httpx.Response) -> float:
    """Extract wait duration from retry-after header or Groq JSON body message."""
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    try:
        msg = resp.json().get("error", {}).get("message", "")
        m = re.search(r"try again in ([\d\.]+)s", msg)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 2.0


def test_sustainable_vs_overload_retry() -> Dict[str, Any]:
    """Compare safe spaced RAG-sized traffic (~600 tokens) vs overloaded burst + smart Retry-After fallback."""
    print("\n=== TEST 3: SUSTAINABLE PACE VS OVERLOAD + SMART RETRY ===")
    print("  Cooling down 25s for token bucket reset...")
    time.sleep(25.0)

    rag_filler = "Transformer architecture uses multi-head self-attention to capture dependencies. " * 45
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": f"Context:\n{rag_filler}\nAnswer in 10 words: What does it use?"}],
        "max_tokens": 40,
    }

    # Part A: Safe Steady Pace (6 requests spaced 7.5s apart = 8 req/min * ~550 tokens = ~4,400 ITPM)
    print("  [3A] Safe Steady Pace (6 requests, 7.5s apart)...")
    steady_logs = []
    with httpx.Client(timeout=30.0) as http:
        for i in range(1, 7):
            t0 = time.perf_counter()
            r = http.post(API_URL, headers=HEADERS, json=payload)
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
            steady_logs.append({"idx": i, "status": r.status_code, "ms": ms})
            print(f"    Steady {i}/6: HTTP {r.status_code} ({ms}ms)")
            if i < 6:
                time.sleep(7.5)

    # Part B: Overload Burst (16 rapid requests) WITH Smart Retry-After Backoff (max 3 retries)
    print("  Cooling down 20s before Overload + Retry test...")
    time.sleep(20.0)
    print("  [3B] Overload Burst (16 RAG-sized calls) with Retry-After fallback...")
    overload_logs = []
    with httpx.Client(timeout=30.0) as http:
        for i in range(1, 17):
            t_req_start = time.perf_counter()
            retries = 0
            wait_ms_total = 0.0
            final_status = None

            while retries <= 3:
                r = http.post(API_URL, headers=HEADERS, json=payload)
                final_status = r.status_code
                if r.status_code == 200:
                    break
                if r.status_code == 429 and retries < 3:
                    wait_sec = parse_wait_seconds(r) + 0.15  # exact wait + tiny jitter
                    retries += 1
                    wait_ms_total += wait_sec * 1000.0
                    print(f"    Overload {i:02d}: hit 429, sleeping {wait_sec:.2f}s (retry {retries}/3)...")
                    time.sleep(wait_sec)
                else:
                    break

            total_ms = round((time.perf_counter() - t_req_start) * 1000.0, 2)
            overload_logs.append({
                "idx": i,
                "final_status": final_status,
                "retries": retries,
                "wait_ms": round(wait_ms_total, 2),
                "total_ms": total_ms,
            })
            print(f"    Overload {i:02d}: final={final_status} | retries={retries} | wait={wait_ms_total:.0f}ms | total={total_ms:.0f}ms")

    return {
        "steady_pace": {
            "count": len(steady_logs),
            "success_rate": round(sum(1 for x in steady_logs if x["status"] == 200) / len(steady_logs) * 100, 1),
            "logs": steady_logs,
        },
        "overload_with_retry": {
            "count": len(overload_logs),
            "final_success_rate": round(sum(1 for x in overload_logs if x["final_status"] == 200) / len(overload_logs) * 100, 1),
            "total_retries": sum(x["retries"] for x in overload_logs),
            "total_p50_ms": round(statistics.median([x["total_ms"] for x in overload_logs]), 2),
            "total_max_ms": round(max(x["total_ms"] for x in overload_logs), 2),
            "logs": overload_logs,
        },
    }


def main():
    print(f"Starting Deep Groq Probing (Experiment #2) on {MODEL_ID}")
    rpm_res = test_pure_rpm(max_calls=65)
    concurrency_res = test_stepped_concurrency(levels=[5, 10, 20, 30])
    pace_res = test_sustainable_vs_overload_retry()

    full_output = {
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_1_pure_rpm": rpm_res,
        "test_2_stepped_concurrency": concurrency_res,
        "test_3_sustainable_vs_overload": pace_res,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EXPERIMENT #2 COMPLETE! Raw results saved to {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

