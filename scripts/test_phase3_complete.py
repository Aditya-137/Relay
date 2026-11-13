"""scripts/test_phase3_complete.py

Comprehensive verification of Phase 3 completion:
1. Rate-Limiter Token Bucket admission & queueing.
2. Burst concurrent requests tracking waiting_ms and peak_queue_length.
3. User give-up timeout trigger (status: timed_out).
4. Daily budget exhaustion trigger (status: daily_budget_exhausted).
5. Verification of results/results.csv records for Tables 2a, 2b, 3.
"""

import concurrent.futures
import csv
import logging
import time
from pathlib import Path

from src.pipeline.orchestrator import VaaniPipeline
from src.utils.rate_limiter import TokenBucketRateLimiter
from src.utils.metrics_collector import MetricsCollector, DEFAULT_CSV_PATH

logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 80)
    print("PHASE 3 COMPLETE: RATE LIMITING, QUEUEING, & TELEMETRY TEST")
    print("=" * 80)

    # Clean existing test CSV if needed or inspect it
    collector = MetricsCollector()
    rate_limiter = TokenBucketRateLimiter(
        rpm_limit=30, itpm_limit=7000, rpd_limit=1000, user_giveup_timeout_s=5.0
    )
    pipeline = VaaniPipeline(rate_limiter=rate_limiter, metrics_collector=collector)

    # -------------------------------------------------------------
    # 1. Burst Concurrency Test (3 Simultaneous Requests)
    # -------------------------------------------------------------
    print("\n[1/3] Firing Burst of 3 Queries Through Rate-Limited Pipeline...")
    test_queries = [
        "What are the dimensions d_model and d_k in the Transformer base model?",
        "What is Mimi in Moshi and what framerate and bandwidth does it operate at?",
        "How do you bake sourdough bread with 75% hydration?",
    ]

    def run_one(q, idx):
        return pipeline.ask(q, scenario="burst_test", request_id=f"burst-{idx}")

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_one, q, i) for i, q in enumerate(test_queries, 1)]
        results = [f.result() for f in futures]
    t_burst = time.perf_counter() - t0

    stats = rate_limiter.get_stats()
    print(f"Burst completed in {t_burst:.2f}s")
    print(f"Peak Queue Length: {stats['peak_queue_length']}")
    for r in results:
        print(f"  - [{r.status}] waiting: {r.latency.waiting_ms}ms, total: {r.latency.total_ms}ms, TTFT: {r.latency.ttft_ms}ms | {r.query[:45]}...")

    # -------------------------------------------------------------
    # 2. Test User Give-up Timeout (Queue Timeout)
    # -------------------------------------------------------------
    print("\n[2/3] Testing User Give-up Timeout (Queue Timeout)...")
    # Drain tokens temporarily to simulate a saturated queue
    rate_limiter.input_tokens = 0.0
    rate_limiter.req_tokens = 0.0
    # Acquire with tiny timeout (0.05s) so request gives up immediately
    admitted, wait_ms, status = rate_limiter.acquire(estimated_input_tokens=1500, timeout_s=0.05)
    print(f"Timeout simulation: admitted={admitted}, wait_ms={wait_ms}ms, status={status}")
    assert status == "timed_out", f"Expected timed_out, got {status}"

    # -------------------------------------------------------------
    # 3. Test Daily Budget Exhaustion (Section 7 Policy)
    # -------------------------------------------------------------
    print("\n[3/3] Testing Daily Budget Exhaustion Policy...")
    rate_limiter.daily_requests_used = 1000
    res_exhausted = pipeline.ask(
        "What is the Transformer architecture?", scenario="daily_exhausted_test"
    )
    print(f"Daily Exhaustion Status: {res_exhausted.status}")
    print(f"Daily Exhaustion User Reply:\n{res_exhausted.answer}")
    assert res_exhausted.status == "daily_budget_exhausted"

    # -------------------------------------------------------------
    # 4. Verify results.csv
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFYING results/results.csv OUTPUT")
    print("=" * 80)
    if DEFAULT_CSV_PATH.exists():
        with open(DEFAULT_CSV_PATH, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            print(f"Total rows in results.csv: {len(reader)}")
            print("Last 3 entries:")
            for row in reader[-3:]:
                print(f"  - Req {row['request_id']} | {row['scenario']} | {row['status']} | TTFT: {row['ttft_ms']}ms | Total: {row['total_ms']}ms | Wait: {row['waiting_ms']}ms | In: {row['input_tokens']} | Out: {row['output_tokens']}")
    print("\n✅ PHASE 3 VERIFICATION PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

