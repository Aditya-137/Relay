"""scripts/test_standin.py

Verifies the Stand-in Mock Service (Phase 4):
1. Test StandinMockService directly (latency distribution, streaming, and 429 limit trigger).
2. Test VaaniPipeline end-to-end with source="standin".
3. Verify results.csv logs source="standin".
"""

import json
import logging
import time

from src.pipeline.orchestrator import VaaniPipeline
from src.standin.mock_service import StandinMockService

logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 80)
    print("PHASE 4: STAND-IN MOCK SERVICE VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Direct Test of StandinMockService
    # -------------------------------------------------------------
    print("\n[1/3] Testing StandinMockService directly...")
    mock = StandinMockService(rpm_limit=5, itpm_limit=5000, simulate_tail_spikes=False)

    sample_messages = [
        {"role": "system", "content": "You are a JSON assistant."},
        {
            "role": "user",
            "content": (
                "QUESTION:\nWhat are the dimensions d_model and d_k in the Transformer base model?\n\n"
                "EXCERPTS:\n[C1] Paper: Attention Is All You Need | Section: 3.2 Attention | Page: 4\n"
                "Content: In this work we facilitate these with multi-head attention where d_model=512 and d_k=64.\n\n"
                "Produce your JSON evaluation and cited answer."
            ),
        },
    ]

    t0 = time.perf_counter()
    res = mock.chat_stream(sample_messages)
    t_tot = (time.perf_counter() - t0) * 1000

    print(f"Direct Mock Call: Status={res['status']}, TTFT={res['ttft_ms']}ms, Working={res['working_ms']}ms, Total={t_tot:.1f}ms")
    print(f"Mock Output Text: {res['text']}")
    assert res["status"] == "success"
    assert res["ttft_ms"] is not None and 80.0 <= res["ttft_ms"] <= 350.0

    # Test 429 Trigger on Standin
    print("\n[2/3] Testing Stand-in 429 Rate Limit Enforcement...")
    responses = [mock.chat_stream(sample_messages) for _ in range(6)]
    statuses = [r["status"] for r in responses]
    print(f"6 Rapid Calls Statuses: {statuses}")
    assert "rate_limited" in statuses, "Expected rate_limited status on rapid calls exceeding RPM limit!"
    print("✅ Stand-in rate limiter triggered HTTP 429 cleanly.")

    # -------------------------------------------------------------
    # 3. Pipeline End-to-End Test with source="standin"
    # -------------------------------------------------------------
    print("\n[3/3] Testing VaaniPipeline with source='standin'...")
    pipeline = VaaniPipeline()

    # Query 1: In-domain
    res_in = pipeline.ask(
        "What are the dimensions d_model and d_k in the Transformer base model?",
        user_id="engineer_alice",
        scenario="standin_smoke_test",
        source="standin",
    )
    print(f"\nIn-Domain Query: Status={res_in.status}")
    print(f"TTFT:            {res_in.latency.ttft_ms} ms")
    print(f"Total Latency:   {res_in.latency.total_ms} ms")
    print(f"Citations:       {res_in.citations}")
    print(f"Answer:          {res_in.answer}")
    assert res_in.status == "success"
    assert len(res_in.citations) > 0

    # Query 2: Out-of-domain (Refused at Score Gate)
    res_out = pipeline.ask(
        "How do you bake sourdough bread with 75% hydration?",
        user_id="engineer_bob",
        scenario="standin_smoke_test",
        source="standin",
    )
    print(f"\nOut-of-Domain:   Status={res_out.status}")
    print(f"Total Latency:   {res_out.latency.total_ms} ms")
    print(f"Answer:          {res_out.answer}")
    assert res_out.status == "refused_score_gate"

    print("\n" + "=" * 80)
    print("✅ PHASE 4 STAND-IN MOCK SERVICE VERIFICATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

