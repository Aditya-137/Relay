"""scripts/run_load_scenarios.py

Executes the load scenarios required for Tables 2a and 2b in the assignment brief:
1. Steady Load (Real Groq, 8 RPM offered rate)
2. Sudden Burst (Stand-in, 15 concurrent requests)
3. Sustained Overload (Stand-in, 25 requests @ 45 RPM offered rate)
4. Per-Minute Limit Halved (Stand-in halved to 15 RPM / 3.5k ITPM)
5. Daily Budget Exhausted (Stand-in daily limit cap)

All results are automatically appended to results/results.csv.
"""

import concurrent.futures
import logging
import sys
import time
from pathlib import Path

from src.pipeline.orchestrator import VaaniPipeline
from src.standin.mock_service import StandinMockService
from src.utils.rate_limiter import TokenBucketRateLimiter

logging.basicConfig(level=logging.WARNING)

BENCHMARK_QUERIES = [
    "What are the dimensions d_model and d_k in the Transformer base model?",
    "What is Mimi in Moshi and what framerate and bandwidth does it operate at?",
    "How does Moshi model multi-stream audio for full-duplex spoken dialogue?",
    "What is the parameter size of the Helium text transformer in Moshi?",
    "What is the key difference between fixed-size chunking and dynamic chunking?",
    "How does the boundary predictor in dynamic chunking decide when to end a chunk?",
    "How do you bake sourdough bread with 75% hydration?",
    "What are the rules of chess regarding en passant capture and castling?",
]


def run_steady_scenario(pipeline: VaaniPipeline, num_requests: int = 8):
    print("\n" + "=" * 80)
    print("SCENARIO 1: STEADY LOAD (Real Groq, ~8 RPM)")
    print("=" * 80)
    for i in range(num_requests):
        q = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        t0 = time.perf_counter()
        res = pipeline.ask(
            query=q,
            user_id=f"steady_user_{i}",
            scenario="steady",
            source="real",
            request_id=f"steady-{i+1}",
        )
        print(f"  Req {i+1}/{num_requests} [{res.status}]: TTFT={res.latency.ttft_ms or 0:.0f}ms | Total={res.latency.total_ms:.0f}ms | Wait={res.latency.waiting_ms:.0f}ms")
        time.sleep(7.5)  # Steady pacing


def run_burst_scenario(pipeline: VaaniPipeline, burst_size: int = 15):
    print("\n" + "=" * 80)
    print(f"SCENARIO 2: SUDDEN BURST (Stand-in, {burst_size} concurrent requests)")
    print("=" * 80)

    def send_one(idx):
        q = BENCHMARK_QUERIES[idx % len(BENCHMARK_QUERIES)]
        return pipeline.ask(
            query=q,
            user_id=f"burst_user_{idx % 5}",  # 5 users bursting
            scenario="burst",
            source="standin",
            request_id=f"burst-{idx+1}",
        )

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as executor:
        futures = [executor.submit(send_one, i) for i in range(burst_size)]
        results = [f.result() for f in futures]
    t_tot = time.perf_counter() - t0

    stats = pipeline.rate_limiter.get_stats()
    print(f"Burst finished in {t_tot:.2f}s | Peak Queue Length: {stats['peak_queue_length']}")
    statuses = [r.status for r in results]
    print(f"Statuses Breakdown: Succeeded={statuses.count('success')}, Gate Refused={statuses.count('refused_score_gate')}, Timed Out={statuses.count('timed_out')}")


def run_sustained_overload_scenario(pipeline: VaaniPipeline, num_requests: int = 25, offered_rpm: float = 45.0):
    print("\n" + "=" * 80)
    print(f"SCENARIO 3: SUSTAINED OVERLOAD (Stand-in, {num_requests} requests @ {offered_rpm} RPM)")
    print("=" * 80)
    interval = 60.0 / offered_rpm  # ~1.33 seconds between requests

    results = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    futures = []

    def send_req(idx):
        q = BENCHMARK_QUERIES[idx % len(BENCHMARK_QUERIES)]
        return pipeline.ask(
            query=q,
            user_id=f"overload_user_{idx % 4}",
            scenario="sustained_overload",
            source="standin",
            request_id=f"overload-{idx+1}",
        )

    for i in range(num_requests):
        f = pool.submit(send_req, i)
        futures.append(f)
        time.sleep(interval)

    results = [f.result() for f in futures]
    pool.shutdown()

    statuses = [r.status for r in results]
    print(f"Overload Summary: Succeeded={statuses.count('success')}, Timed Out={statuses.count('timed_out')}, Rate Limited={statuses.count('rate_limited')}")


def run_limit_halved_scenario(pipeline: VaaniPipeline, num_requests: int = 12):
    print("\n" + "=" * 80)
    print("SCENARIO 4: PER-MINUTE LIMIT HALVED (Stand-in, 15 RPM / 3.5k ITPM)")
    print("=" * 80)
    # Temporarily install halved standin client
    halved_mock = StandinMockService(halve_limits=True)
    halved_limiter = TokenBucketRateLimiter(rpm_limit=15, itpm_limit=3500)
    halved_pipeline = VaaniPipeline(rate_limiter=halved_limiter, standin_client=halved_mock)

    for i in range(num_requests):
        q = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        res = halved_pipeline.ask(
            query=q,
            user_id=f"halved_user_{i % 3}",
            scenario="per_minute_limit_halved",
            source="standin",
            request_id=f"halved-{i+1}",
        )
        print(f"  Req {i+1}/{num_requests} [{res.status}]: TTFT={res.latency.ttft_ms or 0:.0f}ms | Degraded={res.degraded}")
        time.sleep(2.0)


def run_daily_exhausted_scenario(pipeline: VaaniPipeline, num_requests: int = 5):
    print("\n" + "=" * 80)
    print("SCENARIO 5: DAILY BUDGET EXHAUSTED (Stand-in, 1k RPD limit hit)")
    print("=" * 80)
    exhausted_limiter = TokenBucketRateLimiter()
    exhausted_limiter.daily_requests_used = 1000
    exhausted_pipeline = VaaniPipeline(rate_limiter=exhausted_limiter)

    for i in range(num_requests):
        q = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        res = exhausted_pipeline.ask(
            query=q,
            user_id=f"user_{i}",
            scenario="daily_budget_exhausted",
            source="standin",
            request_id=f"daily-exh-{i+1}",
        )
        print(f"  Req {i+1} Status: {res.status} | User Message Preview: {res.answer[:70]}...")


def main():
    pipeline = VaaniPipeline()

    # 1. Steady Load (Real Groq)
    run_steady_scenario(pipeline, num_requests=6)

    # 2. Sudden Burst (Stand-in)
    run_burst_scenario(pipeline, burst_size=15)

    # 3. Sustained Overload (Stand-in)
    run_sustained_overload_scenario(pipeline, num_requests=20, offered_rpm=45.0)

    # 4. Limit Halved (Stand-in)
    run_limit_halved_scenario(pipeline, num_requests=10)

    # 5. Daily Budget Exhausted (Stand-in)
    run_daily_exhausted_scenario(pipeline, num_requests=4)

    print("\n" + "=" * 80)
    print("✅ ALL LOAD SCENARIOS COMPLETE! Telemetry recorded to results/results.csv.")
    print("=" * 80)


if __name__ == "__main__":
    main()

